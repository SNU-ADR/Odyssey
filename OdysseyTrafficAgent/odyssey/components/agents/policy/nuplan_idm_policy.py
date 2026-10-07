"""Background agents driven by nuPlan's own IDM, as a manager-level batch pre-pass.

WHY. nuPlan's IDMAgents handles the complete reactive background fleet as one observation:
plan_route() extends each rail, _get_relevant_stop_lines() inserts the RED stop-line
polygon into the occupancy map as a virtual lead vehicle, and the lead
search is `own path buffer ∩ STRtree occupancy map`, which finds a car crossing the junction.

It is also much faster. odyssey's own nearest-lane search
(edge_road_network.get_closest_lane_index) walks every lane and every segment on each
localisation and dominates a PDM+IDM rollout's run time; nuPlan indexes the same query with a
shapely STRtree.

SHAPE. nuPlan replaces the whole observation; odyssey has agent OBJECTS whose poses the
render, data and metric managers read off agent_manager.all_agents. So this keeps the object
graph and drives it: IDMAgents propagates every agent at once, and each agent's policy then
returns its own share of that one result as a Trajectory. The batch is computed lazily on the
first agent to ask in a given sim step, so no manager needs a new hook.

FRAMES. nuPlan works in global UTM; odyssey's agents are scene-local. The converter's
initial_ego_center is the one bridge, exactly as in pdm_policy.
"""
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from collections import OrderedDict, defaultdict
from types import MethodType, SimpleNamespace
from typing import Optional

import numpy as np
from shapely.affinity import translate
from shapely.geometry import LineString, Point
from shapely.geometry.base import CAP_STYLE
from shapely.ops import substring

from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatusData,
    TrafficLightStatusType,
)
from nuplan.planning.simulation.observation.idm.idm_agents_builder import (
    build_idm_agents_on_map_rails, get_starting_segment,
)
from nuplan.planning.simulation.observation.idm.idm_policy import IDMPolicy
from nuplan.planning.simulation.observation.idm.idm_agent_manager import IDMAgentManager
from nuplan.planning.simulation.observation.idm.idm_states import (
    IDMAgentState,
    IDMLeadAgentState,
)
from nuplan.planning.simulation.observation.idm.utils import (
    create_path_from_se2,
    path_to_linestring,
)
from nuplan.planning.simulation.observation.idm_agents import IDMAgents
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)

from odyssey.common.dataclasses import Trajectory
from odyssey.components.agents.policy.idm_route_intent import (
    build_source_route_intent, choose_source_branch,
)
from odyssey.components.agents.policy.base_policy import BasePolicy
from odyssey.components.agents.policy.intersection_manager import (
    EgoSafetyState,
    IntersectionManager,
    IntersectionManagerConfig,
    IntersectionWorldState,
    TargetedLeadIDMAgentManager,
    _path_to_go_linestring,
)
from odyssey.components.agents.policy.pdm_planner.utils.odyssey_to_pdm_utils import (
    OdysseyToNuPlanConverter,
)
from odyssey.utils.nuplan_map_utils import resolve_map_location
from odyssey.utils.cadence import synchronized_gt_warmup_steps
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

logger = logging.getLogger(__name__)

#: IDM parameters. nuPlan's simulation defaults, kept in one place so a sweep has one edit point.
DEFAULTS = dict(target_velocity=10.0, min_gap_to_lead_agent=1.0, headway_time=1.5,
                accel_max=1.0, decel_max=2.0, emergency_decel_max=4.905,
                minimum_path_length=20.0, radius=100.0)

# IDM-only, scene-scoped connector exclusions, {(map location, scene id): connector ids}: a map
# overlay for IDM route building, not a change to nuPlan map data or to the scorer/renderer's
# vector map. No benchmark scene uses one.
_IDM_EXCLUDED_CONNECTORS = {}


class _IDMStartingSegmentMap:
    """Builder-only view: exclude selected connectors and wrap heading differences.

    nuPlan's get_starting_segment subtracts headings without wrapping at +/-pi. Give its
    builder only the true closest candidate; every other map operation delegates unchanged.
    """

    def __init__(self, map_api, excluded_connector_ids=()):
        self._map_api = map_api
        self._excluded_connector_ids = frozenset(map(str, excluded_connector_ids))

    def __getattr__(self, name):
        return getattr(self._map_api, name)

    def get_all_map_objects(self, position, layer):
        segments = list(self._map_api.get_all_map_objects(position, layer))
        if layer not in {SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR}:
            return segments
        segments = [
            segment for segment in segments
            if str(segment.id) not in self._excluded_connector_ids
        ]
        if not segments:
            return []
        heading = float(position.heading)

        def wrapped_heading_error(segment):
            nearest = segment.baseline_path.get_nearest_pose_from_position(position)
            delta = float(nearest.heading) - heading
            return abs(float(np.arctan2(np.sin(delta), np.cos(delta))))

        return [min(segments, key=wrapped_heading_error)]


def _plan_route_excluding_connectors(agent, traffic_light_status, excluded_connector_ids):
    """Extend IDM rails using local GT intent; RED routes still stop at stop lines."""
    while (
        agent.get_progress_to_go()
        < agent._minimum_path_length + agent._policy.target_velocity * agent._policy.headway_time
    ):
        candidates = []
        stock_candidates = []
        for edge in agent.end_segment.outgoing_edges:
            if str(edge.id) in excluded_connector_ids:
                continue
            red = edge.id in traffic_light_status[TrafficLightStatusType.RED]
            green = edge.id in traffic_light_status[TrafficLightStatusType.GREEN]
            # A known RED branch may be committed as a route; the manager inserts its
            # stop line as a virtual lead. UNKNOWN signal states remain unavailable.
            if edge.has_traffic_lights() and not (red or green):
                continue
            candidates.append(edge)
            if green or (not red and not edge.has_traffic_lights()):
                stock_candidates.append(edge)
        stock = min(
            stock_candidates,
            key=lambda candidate: abs(
                candidate.baseline_path.get_curvature_at_arc_length(0.0)
            ),
        ) if stock_candidates else None
        edge = choose_source_branch(
            agent, candidates, stock, getattr(agent, "_odyssey_route_decisions", None)
        )
        if edge is None:
            break
        agent._route.append(edge)
        agent._path = create_path_from_se2(
            agent.get_path_to_go() + edge.baseline_path.discrete_path
        )
        agent._state.progress = 0


def _install_idm_connector_filter(agent, excluded_connector_ids):
    """Apply branch-intent routing on this IDM instance; leave nuPlan unmodified."""
    excluded = frozenset(map(str, excluded_connector_ids))
    if getattr(agent, "_odyssey_excluded_connector_ids", None) == excluded:
        return
    agent._odyssey_excluded_connector_ids = excluded

    def plan_route(this, traffic_light_status):
        return _plan_route_excluding_connectors(this, traffic_light_status, excluded)

    agent.plan_route = MethodType(plan_route, agent)


def _assert_no_excluded_idm_routes(agents, excluded_connector_ids):
    """Fail a rollout if an excluded parking connector nevertheless reaches a live IDM route."""
    if not excluded_connector_ids:
        return
    for token, agent in agents.items():
        forbidden = sorted(
            str(edge.id) for edge in agent.get_route()
            if str(edge.id) in excluded_connector_ids
        )
        if forbidden:
            raise RuntimeError(
                f"IDM agent {token} entered excluded connector(s): {','.join(forbidden)}"
            )


# nuPlan's stock ``idm_agents_observation.yaml`` list.  Vehicles which cannot be put on a map
# rail are intentionally not open-loop detections: the reference builder drops them rather than
# replaying a second, unsimulated vehicle population beside the IDM fleet.
OPEN_LOOP_DETECTION_TYPES = [
    "PEDESTRIAN", "BICYCLE", "BARRIER", "CZONE_SIGN", "TRAFFIC_CONE", "GENERIC_OBJECT",
]


class _IDMPolicyWithEmergencyBrake(IDMPolicy):
    """Stock nuPlan IDM equation with an adapter-local emergency braking clamp."""

    def __init__(self, *idm_params, emergency_decel_max: float):
        super().__init__(*idm_params)
        if emergency_decel_max < self.decel_max:
            raise ValueError(
                "emergency_decel_max must be greater than or equal to decel_max, got "
                f"{emergency_decel_max} < {self.decel_max}"
            )
        self._emergency_decel_max = emergency_decel_max

    def solve_forward_euler_idm_policy(
        self,
        agent: IDMAgentState,
        lead_agent: IDMLeadAgentState,
        sampling_time: float,
    ) -> IDMAgentState:
        """Use nuPlan's desired gap and acceleration request, changing only its hard clamp."""
        x_dot, velocity_dot = self.idm_model(
            [], agent.to_array(), lead_agent.to_array(), self.idm_params
        )
        return IDMAgentState(
            agent.progress + sampling_time * x_dot,
            agent.velocity
            + sampling_time
            * min(max(-self._emergency_decel_max, velocity_dot), self._accel_max),
        )


class _ScenarioAdapter:
    """The five members nuPlan's IDM reads off a scenario, backed by the converter.

    Duck-typed rather than an AbstractScenario subclass: that ABC has ~30 abstract methods and
    the IDM path touches exactly these five (idm_agents_builder.py:82-84,126 and
    idm_agents.py:129,153). Implementing the rest to satisfy the ABC would be dead code that
    still has to be read by whoever debugs this.
    """

    def __init__(self, converter: OdysseyToNuPlanConverter, map_api):
        self._c = converter
        self.map_api = map_api
        # IDMAgents asks for the same source frame in update_observation(),
        # get_observation(), and our late-admission guard.  Rebuilding hundreds of
        # Agent/OrientedBox objects for each request is pure adapter overhead and has no
        # counterpart in a database-backed nuPlan scenario.  Keep only adjacent frames:
        # long rollouts must not retain every dense observation.
        self._track_cache = OrderedDict()
        self._track_cache_size = 8
        self._traffic_light_transform = None
        self._initial_iteration = 0

    def set_initial_iteration(self, iteration: int) -> None:
        """Choose the single source snapshot used to construct the IDM population."""
        self._initial_iteration = int(iteration)

    def set_traffic_light_transform(self, transform) -> None:
        """Install an IDM-local signal transform without changing the ego converter."""
        self._traffic_light_transform = transform

    def _tracked_objects_at(self, iteration: int):
        iteration = int(iteration)
        cached = self._track_cache.get(iteration)
        if cached is not None:
            self._track_cache.move_to_end(iteration)
            return cached
        tracks = self._c.convert_to_detections_tracks_from_scene(iteration)
        self._track_cache[iteration] = tracks
        self._track_cache.move_to_end(iteration)
        while len(self._track_cache) > self._track_cache_size:
            self._track_cache.popitem(last=False)
        return tracks

    @property
    def initial_tracked_objects(self):
        return self._tracked_objects_at(self._initial_iteration)

    def get_ego_state_at_iteration(self, iteration: int):
        # nuPlan's builder always asks for iteration zero even when the simulator's
        # reactive phase starts later.  Runtime IDM updates do not call this method.
        if int(iteration) == 0:
            iteration = self._initial_iteration
        return self._c.convert_to_current_ego_state(iteration)

    def get_tracked_objects_at_iteration(self, iteration: int, *a, **kw):
        return self._tracked_objects_at(iteration)

    def get_traffic_light_status_at_iteration(self, iteration: int):
        data = self._c.convert_to_traffic_lights(iteration)
        if self._traffic_light_transform is None:
            return data
        return self._traffic_light_transform(int(iteration), data)


class _HistoryShim:
    """update_observation only reads `history.current_state[0]`."""

    def __init__(self, ego_state):
        self.current_state = (ego_state, None)


class _IDMAgentsWithVehicleFallback(IDMAgents):
    """Stock IDM observation plus selected source-pose vehicles kept open loop.

    nuPlan normally makes every routable vehicle reactive and therefore omits VEHICLE from
    its open-loop classes.  Long Odyssey clips also contain parked curb/unstructured
    vehicles.  Snapping those onto a lane rail turns scenery into a stopped lane blocker.
    Tokens placed in ``extra_open_loop_vehicle_tokens`` remain at their recorded pose instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_open_loop_vehicle_tokens = set()

    def _get_open_loop_track_objects(self, iteration):
        objects = list(super()._get_open_loop_track_objects(iteration))
        if not self.extra_open_loop_vehicle_tokens:
            return objects
        detections = self._scenario.get_tracked_objects_at_iteration(iteration)
        objects.extend(
            obj
            for obj in detections.tracked_objects.get_tracked_objects_of_type(
                TrackedObjectType.VEHICLE
            )
            if str(obj.track_token) in self.extra_open_loop_vehicle_tokens
        )
        return objects


class _PinnedVehiclesAsObstacles:
    """Builder view of one snapshot in which pinned vehicles are obstacles, not candidates.

    ``build_idm_agents_on_map_rails`` reads only these two queries.  Pinned (whole-episode
    stationary) vehicles are withheld from its VEHICLE loop and handed to its open-loop query
    instead, so the builder collision-checks every candidate against their true log box -- the
    same obstacle IDM will see at run time -- rather than against a lane-snapped copy.
    """

    def __init__(self, detections, pinned_tokens, excluded_tokens=()):
        self._objects = detections.tracked_objects
        self._pinned = pinned_tokens
        self._excluded = frozenset(excluded_tokens)
        self.tracked_objects = self

    def _pinned_vehicles(self, pinned):
        return [
            obj for obj in self._objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
            if str(obj.track_token) not in self._excluded
            and (str(obj.track_token) in self._pinned) == pinned
        ]

    def get_tracked_objects_of_type(self, tracked_object_type):
        if tracked_object_type == TrackedObjectType.VEHICLE:
            return self._pinned_vehicles(False)
        return [
            obj for obj in self._objects.get_tracked_objects_of_type(tracked_object_type)
            if str(getattr(obj, "track_token", "")) not in self._excluded
        ]

    def get_tracked_objects_of_types(self, tracked_object_types):
        return ([
            obj for obj in self._objects.get_tracked_objects_of_types(tracked_object_types)
            if str(getattr(obj, "track_token", "")) not in self._excluded
        ] + self._pinned_vehicles(True))


class IDMPropagateError(RuntimeError):
    """The reactive fleet could not be advanced; the rollout must fail, not freeze traffic."""


class NuPlanIDMBatch:
    """One IDMAgents for the whole scene, stepped once per sim step.

    Held on the engine rather than on any one agent: every background agent shares the single
    propagate() and the single occupancy map, which is the whole point of using nuPlan's IDM.
    """

    # nuplan_idm_static_pin_and_predrop.  Class defaults keep the switch off (byte-identical
    # legacy behaviour) for every construction path, including lightweight ``__new__`` tests.
    _static_pin_and_predrop = False
    _static_vehicle_tokens = frozenset()
    _source_path_predrop_max_m = 0.0
    _source_vehicle_path_lengths = {}
    _excluded_idm_connector_ids = frozenset()
    # nuplan_idm_fail_on_propagate_error / data_output_dir (see __init__).
    _fail_on_propagate_error = True
    _failure_output_dir = None

    def __init__(self, engine, config):
        self.engine = engine
        scenario_manager = engine.managers['scenario_manager']
        self.scene = scenario_manager.current_scene
        ego_agent = engine.managers['agent_manager'].ego_agent
        self._ego_agent = ego_agent
        self.converter = OdysseyToNuPlanConverter(self.scene, ego_agent, engine)
        map_root = engine.global_config['nuplan_map_root']
        self._map_location = resolve_map_location(self.scene, map_root)
        map_api = get_maps_api(map_root, "nuplan-maps-v1.0", self._map_location)
        self._excluded_idm_connector_ids = _IDM_EXCLUDED_CONNECTORS.get(
            (self._map_location, str(self.scene.get("id", ""))), frozenset()
        )
        if self._excluded_idm_connector_ids:
            logger.info(
                "[NUPLAN-IDM-ROUTE] scene=%s excluding IDM-only connectors %s",
                self.scene.get("id"), sorted(self._excluded_idm_connector_ids),
            )
        self._scenario = _ScenarioAdapter(
            self.converter,
            _IDMStartingSegmentMap(map_api, self._excluded_idm_connector_ids),
        )

        p = {k: float(config.get(f"nuplan_idm_{k}", v)) for k, v in DEFAULTS.items()}
        self._obs = _IDMAgentsWithVehicleFallback(
            target_velocity=p["target_velocity"],
            min_gap_to_lead_agent=p["min_gap_to_lead_agent"],
            headway_time=p["headway_time"],
            accel_max=p["accel_max"],
            decel_max=p["decel_max"],
            open_loop_detections_types=OPEN_LOOP_DETECTION_TYPES,
            scenario=self._scenario,
            minimum_path_length=p["minimum_path_length"],
            radius=p["radius"],
        )
        self._dt = float(engine.sim_dt)
        self._gt_warmup_steps = synchronized_gt_warmup_steps(config)
        self._initialization_mode = str(config.get(
            "nuplan_idm_initialization_mode", "gt_merge"
        ))
        if self._initialization_mode not in {
            "gt_merge", "centerline_snap_warmup"
        }:
            raise ValueError(
                "nuplan_idm_initialization_mode must be gt_merge or "
                f"centerline_snap_warmup, got {self._initialization_mode!r}"
            )
        self._idm_start_step = None
        self._idm_handoff_prebuild_seconds = 0.0
        self._handoff_snap_diagnostics_enabled = bool(config.get(
            "nuplan_idm_handoff_snap_diagnostics_enabled", False
        ))
        self._handoff_snap_stats = {}
        self._handoff_smooth_merge_enabled = bool(config.get(
            "nuplan_idm_handoff_smooth_merge_enabled", True
        ))
        # A logged vehicle which cannot be admitted to reactive IDM must not continue as an
        # open-loop replay object in a closed-loop rollout.  It cannot react to the diverged ego
        # or IDM traffic and can therefore drive through them.  Keep the old preservation mode
        # only as an explicit ablation; production follows nuPlan's normal drop/retry semantics.
        self._vehicle_gt_fallback_enabled = bool(config.get(
            "nuplan_idm_vehicle_gt_fallback_enabled", False
        ))
        # Parked cars (classified by agent_manager at reset) never enter IDM; they stay at their
        # log pose for the whole clip as static agents. IDM sees them as open-loop obstacles --
        # they take the existing open-loop vehicle path (extra_open_loop_vehicle_tokens), so they
        # enter propagate's occupancy map. Cars that would fail the handoff / late-admission gate
        # never appear at all (_predropped) instead of being GT-replayed or skipped. Warm-up is
        # pure log replay, so the handoff decision is already fixed at reset.
        self._static_pin_and_predrop = bool(config.get(
            "nuplan_idm_static_pin_and_predrop", False
        ))
        self._static_vehicle_tokens = frozenset(
            getattr(engine.managers['agent_manager'], '_static_pinned_vehicle_ids', ())
        ) if self._static_pin_and_predrop else frozenset()
        self._obs.extra_open_loop_vehicle_tokens.update(self._static_vehicle_tokens)
        #: token -> reason. Vehicles the IDM gate refused; never published, never retried.
        self._predropped = {}
        # Any IDM propagation failure ends the run as failed. Swallowing it would score every
        # car frozen at its last pose (often the GT warm-up pose). false is only an ablation
        # escape hatch.
        self._fail_on_propagate_error = bool(config.get(
            "nuplan_idm_fail_on_propagate_error", True
        ))
        self._failure_output_dir = config.get("data_output_dir")
        self._handoff_smooth_merge_max_offset_m = float(config.get(
            "nuplan_idm_handoff_smooth_merge_max_offset_m", 3.0
        ))
        self._handoff_smooth_merge_max_heading_deg = float(config.get(
            "nuplan_idm_handoff_smooth_merge_max_heading_deg", 45.0
        ))
        if (
            self._handoff_smooth_merge_max_offset_m <= 0
            or not 0 < self._handoff_smooth_merge_max_heading_deg <= 180
        ):
            raise ValueError("handoff smooth-merge thresholds must be positive")
        self._handoff_smooth_merge_tokens = set()
        self._handoff_gt_fallback_tokens = set()
        # The PKL velocity field may be zero while the GT pose rail moves. Recover
        # forward source speed at the actor's mapped row for every admitted IDM vehicle.
        self._initial_speed_sanity_max_mps = float(config.get(
            "nuplan_idm_initial_speed_sanity_max_mps", 40.0
        ))
        if self._initial_speed_sanity_max_mps <= 0:
            raise ValueError("nuplan_idm_initial_speed_sanity_max_mps must be positive")
        # An IDM token is admitted only once. Keep a deterministic NPZ audit per token.
        self._idm_initial_speed_records = {}
        self._idm_route_decisions = []
        solve_dt = float(config.get("nuplan_idm_solve_dt", self._dt) or self._dt)
        self._handoff_source_vehicle_tokens = set()
        self._handoff_built_idm_tokens = set()
        self._handoff_routable_not_built_tokens = set()
        self._handoff_dropped_routable_tokens = set()
        solve_ratio = solve_dt / self._dt
        self._solve_stride = int(round(solve_ratio))
        if self._solve_stride < 1 or not np.isclose(solve_ratio, self._solve_stride):
            raise ValueError(
                "nuplan_idm_solve_dt must be an integer multiple of the simulation "
                f"step ({self._dt}s), got {solve_dt}s"
            )
        self._solve_dt = self._solve_stride * self._dt
        self._origin = np.asarray(self.converter.initial_ego_center, dtype=np.float64).reshape(-1)[:2]
        self._step = -1
        self._poses = {}
        self._source_modes = {}
        # Multi-rate state.  nuPlan owns the states on solve boundaries; Odyssey only
        # interpolates their presentation on its finer outer ticks.
        self._interval_start_step = None
        self._interval_target_step = None
        self._interval_start_poses = {}
        self._interval_target_poses = {}
        self._interval_start_modes = {}
        self._interval_target_modes = {}
        self._idm_params = p
        #: Tokens which cannot be routed, or whose source geometry is permanently malformed.
        #: Transient collision refusals are deliberately not remembered and can be retried.
        self._unroutable = set()
        #: An IDM-owned token gets one simulated lifetime.  Once it has actually been exposed
        #: and then disappears (source end, radius filtering, or explicit removal), do not
        #: resurrect it later from a future log row with a discontinuous fresh IDM state.
        self._ever_admitted = set()
        self._retired = set()
        self._admitted_after_start = 0
        self._spawn_sector_waiting_tokens = set()
        self._spawn_crossing_deferrals = 0
        self._spawn_crossing_horizon_s = float(
            config.get("nuplan_idm_spawn_crossing_horizon_s", 1.0)
        )
        self._spawn_crossing_sample_dt = float(
            config.get("nuplan_idm_spawn_crossing_sample_dt", 0.1)
        )
        self._spawn_rail_conflict_lookahead_m = float(
            config.get("nuplan_idm_spawn_rail_conflict_lookahead_m", 100.0)
        )
        self._spawn_gate = str(config.get("nuplan_idm_spawn_gate", "conservative"))
        if self._spawn_gate not in {"conservative", "footprint_overlap_only"}:
            raise ValueError(
                "nuplan_idm_spawn_gate must be conservative or footprint_overlap_only, got "
                f"{self._spawn_gate!r}"
            )
        if (
            self._spawn_crossing_horizon_s <= 0
            or self._spawn_crossing_sample_dt <= 0
            or self._spawn_rail_conflict_lookahead_m <= 0
        ):
            raise ValueError("spawn crossing horizon and sample dt must be positive")
        self._late_spawn_open_loop_braking_gap_enabled = bool(config.get(
            "nuplan_idm_late_spawn_open_loop_braking_gap_enabled", False
        ))
        self._source_motion_spawn_braking_gap_enabled = bool(config.get(
            "nuplan_idm_source_motion_spawn_braking_gap_enabled", True
        ))
        self._ego_lane_spawn_clearance_enabled = bool(config.get(
            "nuplan_idm_ego_lane_spawn_clearance_enabled", True
        ))
        self._ego_lane_spawn_reaction_time_s = float(config.get(
            "nuplan_idm_ego_lane_spawn_reaction_time_s", 1.0
        ))
        self._ego_lane_spawn_stopping_distance_multiplier = float(config.get(
            "nuplan_idm_ego_lane_spawn_stopping_distance_multiplier", 3.0
        ))
        self._ego_lane_spawn_front_min_m = float(config.get(
            "nuplan_idm_ego_lane_spawn_front_min_m", 15.0
        ))
        self._ego_lane_spawn_rear_m = float(config.get(
            "nuplan_idm_ego_lane_spawn_rear_m", 20.0
        ))
        self._ego_lane_spawn_heading_tolerance_rad = np.deg2rad(float(config.get(
            "nuplan_idm_ego_lane_spawn_heading_tolerance_deg", 30.0
        )))
        self._ego_lane_spawn_lateral_margin_m = float(config.get(
            "nuplan_idm_ego_lane_spawn_lateral_margin_m", 0.5
        ))
        if (
            self._ego_lane_spawn_reaction_time_s < 0
            or self._ego_lane_spawn_stopping_distance_multiplier < 0
            or self._ego_lane_spawn_front_min_m < 0
            or self._ego_lane_spawn_rear_m < 0
            or not 0 <= self._ego_lane_spawn_heading_tolerance_rad <= np.pi
            or self._ego_lane_spawn_lateral_margin_m < 0
        ):
            raise ValueError("ego-lane spawn-clearance parameters must be non-negative")
        self._spawn_ego_lane_clearance_deferrals = 0
        # Retained in the rollout schema for backward-compatible analysis. A synchronized
        # handoff has no "initial spawn" gate, so this counter must remain zero.
        self._initial_ego_lane_clearance_deferrals = 0
        # Parked/off-rail source vehicles are physical detections, but not reactive traffic.
        # Keeping them at GT pose avoids nuPlan's builder teleporting curb vehicles to a lane
        # centre and manufacturing a queue at reset or late admission.
        self._parked_vehicle_fallback_enabled = bool(config.get(
            "nuplan_idm_parked_vehicle_fallback_enabled", True
        ))
        self._parked_vehicle_speed_threshold_mps = float(config.get(
            "nuplan_idm_parked_vehicle_speed_threshold_mps", 0.5
        ))
        self._parked_vehicle_snap_distance_m = float(config.get(
            "nuplan_idm_parked_vehicle_snap_distance_m", 2.0
        ))
        self._parked_vehicle_max_track_displacement_m = float(config.get(
            "nuplan_idm_parked_vehicle_max_track_displacement_m", 3.0
        ))
        self._parked_vehicle_min_valid_samples = int(config.get(
            "nuplan_idm_parked_vehicle_min_valid_samples", 3
        ))
        # Optional ablation: admit only source vehicles which demonstrate enough motion over
        # their complete valid CKPT-derived track.  Vehicles below the threshold remain exact
        # GT open-loop objects; zero disables the gate and preserves the production policy.
        self._reactive_min_source_displacement_m = float(config.get(
            "nuplan_idm_reactive_min_source_displacement_m", 0.0
        ))
        if (
            self._parked_vehicle_speed_threshold_mps < 0
            or self._parked_vehicle_snap_distance_m < 0
            or self._parked_vehicle_max_track_displacement_m < 0
            or self._parked_vehicle_min_valid_samples < 2
            or self._reactive_min_source_displacement_m < 0
        ):
            raise ValueError("parked-vehicle fallback thresholds must be non-negative")
        # The source velocity field is often zero even for a vehicle which moves elsewhere in
        # the clip.  Classifying from one frame therefore turns traffic-light queues into
        # "parked" open-loop objects.  Precompute a whole-valid-track motion envelope once and
        # require it to remain spatially local before applying the geometry-based fallback.
        self._source_vehicle_motion_profiles = self._build_source_vehicle_motion_profiles(
            self.scene
        )
        # Source motion below this limit has no reliable reactive route signal.  In the
        # static-pin/predrop policy such vehicles are removed before GT warm-up and scoring,
        # using cumulative distance over valid PKL poses.
        self._source_path_predrop_max_m = float(config.get(
            "nuplan_idm_source_path_predrop_max_m", 3.0
        ))
        if self._source_path_predrop_max_m < 0 or not np.isfinite(self._source_path_predrop_max_m):
            raise ValueError("nuplan_idm_source_path_predrop_max_m must be finite and non-negative")
        # Report the effective threshold in the NPZ; an off-mode rollout must not
        # advertise a 3 m predrop that never ran.
        if not self._static_pin_and_predrop:
            self._source_path_predrop_max_m = 0.0
        self._source_vehicle_path_lengths = (
            self._build_source_vehicle_path_lengths(self.scene)
            if self._source_path_predrop_max_m > 0 else {}
        )
        self._predrop_short_source_paths()
        self._source_motion_open_loop_tokens = set()
        self._source_motion_spawned_tokens = set()
        self._source_motion_spawn_deferrals = 0
        self._source_vehicle_final_valid_steps = (
            self._build_source_vehicle_final_valid_steps(self.scene)
        )
        self._unstable_short_track_filter_enabled = bool(config.get(
            "nuplan_idm_unstable_short_track_filter_enabled", True
        ))
        self._unstable_short_track_tokens = self._build_unstable_short_track_tokens(
            self.scene,
            max_valid_samples=int(config.get(
                "nuplan_idm_unstable_short_track_max_valid_samples", 2
            )),
            max_displacement_m=float(config.get(
                "nuplan_idm_unstable_short_track_max_displacement_m", 1.0
            )),
            heading_jump_deg=float(config.get(
                "nuplan_idm_unstable_short_track_heading_jump_deg", 90.0
            )),
        ) if self._unstable_short_track_filter_enabled else set()
        self._parked_vehicle_fallbacks = 0
        #: Routable vehicles which are visible in the source frame but cannot yet be inserted
        #: without overlapping the live closed-loop world. They must remain absent, not fall
        #: through to GT replay at the pose which just failed the collision check.
        self._deferred_tokens = set()
        #: Long Odyssey windows need late admission: unlike nuPlan's short scenarios, many
        #: actors have zero-filled leading frames and first appear well after iteration 0. It can
        #: still be disabled for an exact nuPlan-style fixed initial population.
        self._midspawn = bool(config.get("nuplan_idm_midspawn", True))
        self._intersection_manager = None
        self._intersection_stall_liveness_enabled = bool(config.get(
            "nuplan_idm_intersection_stall_liveness_enabled", False
        ))
        self._source_ended_active_stall_relief_enabled = bool(config.get(
            "nuplan_idm_source_ended_active_stall_relief_enabled", True
        ))
        self._source_ended_active_stall_retirements = 0
        if bool(config.get("nuplan_idm_intersection_manager_enabled", False)):
            self._intersection_manager = IntersectionManager(
                IntersectionManagerConfig(
                    sim_dt=self._dt,
                    decel_max=p["decel_max"],
                    reaction_margin_s=float(config.get(
                        "nuplan_idm_intersection_reaction_margin_s", 0.7)),
                    distance_margin_m=float(config.get(
                        "nuplan_idm_intersection_distance_margin_m", 3.0)),
                    stop_offset_m=float(config.get(
                        "nuplan_idm_intersection_stop_offset_m", 1.0)),
                    downstream_margin_m=float(config.get(
                        "nuplan_idm_intersection_downstream_margin_m", 2.0)),
                    ego_envelope_horizon_s=float(config.get(
                        "nuplan_idm_intersection_ego_horizon_s", 1.5)),
                    ego_envelope_margin_m=float(config.get(
                        "nuplan_idm_intersection_ego_margin_m", 0.75)),
                    stopped_rear_ego_release_distance_m=float(config.get(
                        "nuplan_idm_intersection_stopped_rear_ego_release_distance_m",
                        15.0)),
                    stopped_rear_ego_release_lateral_m=float(config.get(
                        "nuplan_idm_intersection_stopped_rear_ego_release_lateral_m",
                        3.0)),
                    allow_non_conflicting_movements=bool(config.get(
                        "nuplan_idm_intersection_allow_non_conflicting_movements", True)),
                    contention_only=bool(config.get(
                        "nuplan_idm_intersection_contention_only", True)),
                    protect_signal_conflicts=bool(config.get(
                        "nuplan_idm_intersection_protect_signal_conflicts", False)),
                    wait_bonus_per_tick=float(config.get(
                        "nuplan_idm_intersection_wait_bonus_per_tick", 0.01)),
                    stall_timeout_s=float(config.get(
                        "nuplan_idm_intersection_stall_timeout_s", 8.0)),
                    progress_epsilon_m=float(config.get(
                        "nuplan_idm_intersection_progress_epsilon_m", 0.5)),
                    retire_active_stalls=(
                        self._intersection_stall_liveness_enabled
                        or self._source_ended_active_stall_relief_enabled
                    ),
                    downstream_stopped_persistence_s=float(config.get(
                        "nuplan_idm_intersection_downstream_stopped_persistence_s",
                        2.0)),
                    reservation_commit_distance_m=float(config.get(
                        "nuplan_idm_intersection_reservation_commit_distance_m", 2.0)),
                    reservation_ttl_s=float(config.get(
                        "nuplan_idm_intersection_reservation_ttl_s", 2.0)),
                    log_interval_ticks=int(config.get(
                        "nuplan_idm_intersection_log_interval_ticks", 100)),
                )
            )
        # This is a signal-data recovery policy for stock nuPlan IDM route extension; it does
        # not require the optional IntersectionManager.  Keeping it coupled to that manager
        # silently disabled the configured UNKNOWN -> unprotected fallback in the default
        # (stock-IDM-only) setup, leaving agents parked behind a virtual red stop line.
        self._unknown_signal_fallback_enabled = bool(config.get(
            "nuplan_idm_unknown_signal_unprotected_fallback_enabled", True
        ))
        self._signal_connector_intersection_cache = {}
        self._unknown_signal_fallback_step = None
        self._unknown_signal_fallback_connector_ids = frozenset()
        self._unknown_signal_fallback_active_intersections = frozenset()
        self._vehicle_only_signal_bypass_connector_ids = frozenset()
        self._unknown_signal_fallback_events = 0
        self._unknown_signal_fallback_intersection_ticks = 0
        self._unknown_signal_fallback_connector_ticks = 0
        if self._unknown_signal_fallback_enabled:
            self._scenario.set_traffic_light_transform(
                self._apply_unknown_signal_unprotected_fallback
            )
        self._wait_cycle_monitor_enabled = bool(config.get(
            "nuplan_idm_wait_cycle_monitor_enabled", False
        )) or self._intersection_manager is not None
        self._wait_cycle_timeout_ticks = max(1, int(round(float(config.get(
            "nuplan_idm_wait_cycle_timeout_s", 2.0
        )) / self._dt)))
        self._ego_deadlock_timeout_ticks = max(1, int(round(float(config.get(
            "nuplan_idm_ego_deadlock_timeout_s", 1.0
        )) / self._dt)))
        self._wait_cycle_stopped_speed_mps = float(config.get(
            "nuplan_idm_wait_cycle_stopped_speed_mps", 0.5
        ))
        self._physical_lead_margin_m = float(config.get(
            "nuplan_idm_physical_lead_margin_m", 0.0
        ))
        if self._physical_lead_margin_m < 0:
            raise ValueError("nuplan_idm_physical_lead_margin_m must be non-negative")
        vehicle_only_map = str(config.get(
            "nuplan_idm_vehicle_only_lead_map", ""
        ) or "")
        vehicle_only_lane_ids = {
            str(lane_id)
            for lane_id in (
                config.get("nuplan_idm_vehicle_only_lead_lane_ids", []) or []
            )
        }
        self._vehicle_only_lead_lane_ids = (
            vehicle_only_lane_ids
            if bool(config.get("nuplan_idm_vehicle_only_lead_enabled", False))
            and self._map_location == vehicle_only_map
            else set()
        )
        # A source track ending is not a physical reason to delete its closed-loop car.
        # Most map rails can keep planning normally.  At a true graph boundary, however,
        # stock IDMAgent clamps to the final path point forever.  Give only source-ended
        # vehicles at such a boundary a short heading-continuous egress tail, after which
        # the ordinary ego-radius lifecycle can retire them out of view.
        self._source_end_route_extension_enabled = bool(config.get(
            "nuplan_idm_source_end_route_extension_enabled", False
        ))
        self._source_end_route_extension_length_m = float(config.get(
            "nuplan_idm_source_end_route_extension_length_m", 100.0
        ))
        self._source_end_route_extension_trigger_m = float(config.get(
            "nuplan_idm_source_end_route_extension_trigger_m", 20.0
        ))
        if self._source_end_route_extension_length_m <= 0:
            raise ValueError("source-end route extension length must be positive")
        if self._source_end_route_extension_trigger_m < 0:
            raise ValueError("source-end route extension trigger must be non-negative")
        self._source_end_extended_tokens = set()
        self._source_end_route_extensions = 0
        self._source_end_route_exhaustion_despawn_enabled = bool(config.get(
            "nuplan_idm_source_end_route_exhaustion_despawn_enabled", True
        ))
        self._source_end_route_exhaustion_margin_m = float(config.get(
            "nuplan_idm_source_end_route_exhaustion_margin_m", 0.5
        ))
        if self._source_end_route_exhaustion_margin_m < 0:
            raise ValueError("source-end route exhaustion margin must be non-negative")
        self._source_end_route_exhaustion_despawns = 0
        # A configured map lane can be a genuine graph dead-end. Its first car clamps at the
        # path end and may block traffic behind it. Keep this escape hatch fully data-driven:
        # map/lane scoping prevents it from affecting the rest of the benchmark, while the
        # optional follower guard supports either queue-only or final-agent retirement.
        relief_map = str(config.get("nuplan_idm_dead_end_queue_relief_map", "") or "")
        relief_lane_ids = {
            str(lane_id)
            for lane_id in (config.get("nuplan_idm_dead_end_queue_relief_lane_ids", []) or [])
        }
        self._dead_end_queue_relief_enabled = bool(config.get(
            "nuplan_idm_dead_end_queue_relief_enabled", False
        )) and self._map_location == relief_map and bool(relief_lane_ids)
        self._dead_end_queue_relief_lane_ids = relief_lane_ids
        self._dead_end_queue_relief_timeout_ticks = max(1, int(round(float(config.get(
            "nuplan_idm_dead_end_queue_relief_timeout_s", 4.0
        )) / self._dt)))
        self._dead_end_queue_relief_progress_epsilon_m = float(config.get(
            "nuplan_idm_dead_end_queue_relief_progress_epsilon_m", 0.5
        ))
        self._dead_end_queue_relief_stopped_speed_mps = float(config.get(
            "nuplan_idm_dead_end_queue_relief_stopped_speed_mps", 0.5
        ))
        self._dead_end_queue_relief_trigger_m = float(config.get(
            "nuplan_idm_dead_end_queue_relief_trigger_m", 5.0
        ))
        self._dead_end_queue_relief_follower_distance_m = float(config.get(
            "nuplan_idm_dead_end_queue_relief_follower_distance_m", 30.0
        ))
        self._dead_end_queue_relief_require_follower = bool(config.get(
            "nuplan_idm_dead_end_queue_relief_require_follower", True
        ))
        if (
            self._dead_end_queue_relief_progress_epsilon_m <= 0
            or self._dead_end_queue_relief_stopped_speed_mps < 0
            or self._dead_end_queue_relief_trigger_m <= 0
            or self._dead_end_queue_relief_follower_distance_m <= 0
        ):
            raise ValueError("dead-end queue-relief thresholds must be positive")
        self._dead_end_queue_progress = {}
        self._dead_end_queue_retirements = 0
        logger.info(
            "[NUPLAN-IDM] enabled: %s, solve_dt=%s (every %d sim step%s)",
            ", ".join(f"{k}={v}" for k, v in p.items()), self._solve_dt,
            self._solve_stride, "s" if self._solve_stride != 1 else "",
        )
        if self._intersection_manager is not None:
            logger.info(
                "[IDM-IM] enabled: NPC-only adaptive movement admission + downstream storage"
            )
        if self._unknown_signal_fallback_enabled:
            logger.info(
                "[IDM-IM] all-UNKNOWN/absent signal frames fall back to unsignalized rules"
            )
        if self._vehicle_only_lead_lane_ids:
            logger.info(
                "[NUPLAN-IDM] vehicle-only open-loop leads on map=%s lanes=%s",
                self._map_location,
                ",".join(sorted(self._vehicle_only_lead_lane_ids)),
            )

    def _advance(self, step: int):
        """Propagate every agent to `step`. Idempotent within a step."""
        if step == self._step:
            return
        snapped_warmup = (
            getattr(self, "_initialization_mode", "gt_merge")
            == "centerline_snap_warmup"
        )
        if step < self._gt_warmup_steps and not snapped_warmup:
            self._publish_gt_warmup(step)
            self._step = step
            return

        if self._idm_start_step is None:
            # The reference protocol constructs one fleet from the shared GT handoff snapshot.
            # The snap-warmup protocol instead constructs nuPlan's rail-snapped fleet at t=0 and
            # lets it settle against the still-GT Ego before scoring begins.
            self._idm_start_step = 0 if snapped_warmup else step
            self._scenario.set_initial_iteration(self._idm_start_step)
        if self._solve_stride == 1:
            # Materialize the initialization snapshot without advancing it. Subsequent calls
            # propagate one ordinary simulator interval at a time.  During snapped warm-up the
            # converter reads the current GT Ego, so IDM sees the same deterministic moving Ego
            # that will seed PDM/E2E at the handoff.
            prev = step if step == self._idm_start_step else max(step - 1, self._idm_start_step)
            self._solve_once(step, prev, step)
        else:
            self._advance_multirate(step)
        self._step = step

    def _spawn_eligible(self, track_token: str) -> bool:
        """Whether the simulator-side first-admission clock has opened for this token."""
        checker = getattr(self.engine.agent_manager, 'is_idm_spawn_eligible', None)
        return True if checker is None else bool(checker(str(track_token)))

    def _remove_ineligible_initial_agents(self, manager, step: int) -> None:
        """Undo nuPlan's t=0 admission for sectors the live ego has not released.

        The stock builder sees every vehicle in its source snapshot. Filtering only the
        Odyssey proxies would leave invisible IDM actors in the occupancy map, where they
        could brake or block eligible traffic. Remove them from both coupled dictionaries;
        they are deliberately not retired, so the normal late-admission path can retry them
        after their sector opens.
        """
        waiting = {
            str(token) for token in manager.agents
            if not self._spawn_eligible(str(token))
        }
        if not waiting:
            return
        for token in sorted(waiting):
            manager.agents.pop(token, None)
            if manager.agent_occupancy.contains(token):
                manager.agent_occupancy.remove([token])
        self._spawn_sector_waiting_tokens.update(waiting)
        logger.info(
            '[NUPLAN-IDM] held %d vehicle(s) outside unopened intersection sector '
            'at tick=%d', len(waiting), step,
        )

    def _publish_gt_warmup(self, step: int) -> None:
        """Expose the exact shared source snapshot without constructing or running IDM."""
        if self._obs._idm_agent_manager is None:
            self._prebuild_idm_handoff()
        tracks = self._scenario.get_tracked_objects_at_iteration(step)
        poses = {}
        for obj in getattr(tracks.tracked_objects, "tracked_objects", []) or []:
            token = str(getattr(obj, "track_token", "") or "")
            if not token or token in self._predropped_tokens():
                continue
            center = obj.center
            velocity = getattr(obj, "velocity", None)
            speed = float(np.hypot(velocity.x, velocity.y)) if velocity is not None else 0.0
            poses[token] = (
                float(center.x) - self._origin[0],
                float(center.y) - self._origin[1],
                float(center.heading),
                speed,
            )
        self._poses = poses
        self._source_modes = {token: "gt_warmup" for token in poses}

    def _prebuild_idm_handoff(self) -> None:
        """Build the deterministic handoff fleet during warm-up, then leave it frozen."""
        handoff_step = self._gt_warmup_steps
        tracks = self._scenario.get_tracked_objects_at_iteration(handoff_step)
        state = self._ego_agent.object_track
        state_step = min(handoff_step, len(state["position"]) - 1)
        center_local = np.asarray(state["position"][state_step], dtype=float)[:2]
        center_world = center_local + self._origin
        heading = float(state["heading"][state_step])
        ego_box = OrientedBox(
            center=StateSE2(center_world[0], center_world[1], heading),
            length=float(self._ego_agent.length),
            width=float(self._ego_agent.width),
            height=float(self._ego_agent.height),
        )

        excluded_tokens = self._predropped_tokens()
        builder_tracks = (
            _PinnedVehiclesAsObstacles(tracks, self._static_vehicle_tokens, excluded_tokens)
            if self._static_vehicle_tokens or excluded_tokens else tracks
        )

        class _AtHandoff:
            map_api = self._scenario.map_api
            initial_tracked_objects = builder_tracks

            @staticmethod
            def get_ego_state_at_iteration(_iteration):
                return SimpleNamespace(
                    agent=SimpleNamespace(token="ego", box=ego_box)
                )

        started = time.perf_counter()
        p = self._idm_params
        agents, occupancy = build_idm_agents_on_map_rails(
            p["target_velocity"],
            p["min_gap_to_lead_agent"],
            p["headway_time"],
            p["accel_max"],
            p["decel_max"],
            p["minimum_path_length"],
            _AtHandoff(),
            self._obs._open_loop_detections_types,
        )
        for token, agent in agents.items():
            _install_idm_connector_filter(agent, self._excluded_idm_connector_ids)
            self._attach_idm_route_intent(token, agent, handoff_step)
        source_vehicles = {
            str(track.track_token): track
            for track in builder_tracks.tracked_objects.get_tracked_objects_of_type(
                TrackedObjectType.VEHICLE
            )
            if track.track_token is not None
        }
        self._handoff_source_vehicle_tokens = set(source_vehicles)
        builder_agent_tokens = set(agents)
        # Filter the builder result directly. The nuPlan adapter can expose a vehicle in its
        # handoff snapshot one source index earlier than the raw ``valid`` mask, so intersecting
        # only with ``source_vehicles`` misses exactly those boundary tracks.
        for token in sorted(set(agents) & self._unstable_short_track_tokens):
            agents.pop(token, None)
            if occupancy.contains(token):
                occupancy.remove([token])
            self._obs.extra_open_loop_vehicle_tokens.add(token)
            logger.info(
                "[NUPLAN-IDM-HANDOFF] keeping temporally unstable short track %s "
                "in source mode",
                token,
            )
        for token in sorted(set(agents)):
            reason = self._source_motion_open_loop_reason(token)
            if reason is None:
                continue
            agents.pop(token, None)
            if occupancy.contains(token):
                occupancy.remove([token])
            self._keep_source_motion_vehicle_open_loop(token, reason, handoff_step)
        if self._handoff_smooth_merge_enabled:
            for token, track in source_vehicles.items():
                agent = agents.get(token)
                if agent is None:
                    if token not in builder_agent_tokens:
                        self._handoff_routable_not_built_tokens.add(token)
                    if self._static_pin_and_predrop:
                        self._predrop_vehicle(token, "builder_refused", handoff_step)
                        continue
                    # Preserve stock nuPlan semantics by default. Replaying a refused vehicle
                    # open-loop is unsafe after the world diverges from the log because it has
                    # no collision response. The opt-in branch exists only for controlled
                    # ablations of the former behavior.
                    if self._vehicle_gt_fallback_enabled:
                        self._keep_handoff_vehicle_at_gt(
                            token, "builder_refused_or_unroutable", handoff_step
                        )
                    continue

                fallback_reason = self._parked_vehicle_fallback_reason(track, agent)
                if fallback_reason is None:
                    fallback_reason = self._handoff_merge_fallback_reason(track, agent)
                if fallback_reason is not None:
                    agents.pop(token, None)
                    if occupancy.contains(token):
                        occupancy.remove([token])
                    self._handoff_dropped_routable_tokens.add(token)
                    if self._static_pin_and_predrop:
                        self._predrop_vehicle(token, fallback_reason, handoff_step)
                    elif self._vehicle_gt_fallback_enabled:
                        self._keep_handoff_vehicle_at_gt(
                            token, fallback_reason, handoff_step
                        )
                    else:
                        # This vehicle was already visible during GT warm-up. Reclassifying a
                        # failed handoff as a later "new" spawn would make it disappear and then
                        # teleport back onto a rail. Omit it for this rollout instead; a safe
                        # continuous merge is the only permitted warm-up-to-IDM transition.
                        self._retired.add(token)
                        logger.info(
                            "[NUPLAN-IDM-HANDOFF] omitting vehicle %s at tick=%d: %s",
                            token,
                            handoff_step,
                            fallback_reason,
                        )
                    continue

                merge_installed = self._install_handoff_merge_path(track, agent)
                if merge_installed:
                    occupancy.set(token, agent.polygon)
                    self._handoff_smooth_merge_tokens.add(token)
                else:
                    # The vehicle was visible during GT warm-up. Publishing an unsafe lane snap
                    # can place it on a crossing rail, while retrying it as a late spawn would
                    # create a disappear/reappear teleport. Omit it for this rollout.
                    agents.pop(token, None)
                    if occupancy.contains(token):
                        occupancy.remove([token])
                    self._handoff_dropped_routable_tokens.add(token)
                    self._retired.add(token)
                    logger.warning(
                        "[NUPLAN-IDM-HANDOFF] rejected non-forward merge for %s; "
                        "omitting unsafe handoff",
                        token,
                    )
                    if self._static_pin_and_predrop:
                        self._predrop_vehicle(token, "non_forward_merge", handoff_step)

        # Install velocity after successful smooth-path replacement: the path starts at the
        # GT heading and progress remains zero, so pose and longitudinal motion are continuous.
        # A sector-held vehicle is *not* admitted here; it is rebuilt later and must then read
        # its own late-admission source row rather than this synchronized handoff row.
        for token, agent in agents.items():
            if str(token) in source_vehicles and self._spawn_eligible(str(token)):
                self._seed_idm_initial_speed(str(token), agent, handoff_step)

        # Snapshot successful map-rail handoffs before the intersection-sector gate removes
        # temporarily ineligible vehicles from the live IDM manager.
        self._handoff_built_idm_tokens = set(agents) & set(source_vehicles)

        self._obs._idm_agent_manager = IDMAgentManager(
            agents, occupancy, self._scenario.map_api
        )
        self._remove_ineligible_initial_agents(
            self._obs._idm_agent_manager, handoff_step
        )
        self._idm_handoff_prebuild_seconds = time.perf_counter() - started
        logger.info(
            "[NUPLAN-IDM-HANDOFF] prebuilt %d IDM vehicle(s) from GT step=%d in %.3fs "
            "(smooth_merge=%d, gt_fallback=%d)",
            len(agents),
            handoff_step,
            self._idm_handoff_prebuild_seconds,
            len(self._handoff_smooth_merge_tokens),
            len(self._handoff_gt_fallback_tokens),
        )

    def _predrop_vehicle(self, token: str, reason: str, step: int) -> None:
        """Keep a vehicle which fails the IDM gate out of the whole rollout.

        At the handoff the decision is taken on the handoff-tick log pose during reset, so the
        vehicle is also withheld from GT warm-up: it never appears, rather than appearing and
        then vanishing (retired/pruned) or turning into a non-reactive GT replay.
        """
        token = str(token)
        self._retired.add(token)
        self._predropped[token] = str(reason)
        logger.info(
            "[NUPLAN-IDM-PREDROP] vehicle %s withheld from the rollout (gate tick=%d): %s",
            token, step, reason,
        )

    def _keep_handoff_vehicle_at_gt(self, token: str, reason: str, step: int) -> None:
        """Preserve a source vehicle which cannot make a faithful reactive handoff."""
        token = str(token)
        self._obs.extra_open_loop_vehicle_tokens.add(token)
        self._handoff_gt_fallback_tokens.add(token)
        logger.info(
            "[NUPLAN-IDM-HANDOFF] keeping vehicle %s at GT pose at tick=%d: %s",
            token,
            step,
            reason,
        )

    def _handoff_merge_fallback_reason(self, track, agent):
        """Reject a lane association too remote or opposed for a short physical merge."""
        source_pose = track.center
        rail_pose = agent.to_se2()
        offset_m = float(np.hypot(
            float(rail_pose.x) - float(source_pose.x),
            float(rail_pose.y) - float(source_pose.y),
        ))
        heading_deg = float(np.rad2deg(abs(
            (float(rail_pose.heading) - float(source_pose.heading) + np.pi)
            % (2.0 * np.pi)
            - np.pi
        )))
        agent._handoff_snap_distance_m = offset_m
        agent._handoff_snap_heading_deg = heading_deg
        if offset_m > self._handoff_smooth_merge_max_offset_m:
            return f"rail_offset_{offset_m:.2f}m"
        if heading_deg > self._handoff_smooth_merge_max_heading_deg:
            return f"rail_heading_{heading_deg:.1f}deg"
        return None

    @staticmethod
    def _install_handoff_merge_path(track, agent) -> bool:
        """Prepend a short, forward-monotone GT-pose-to-centreline merge.

        The operation is one-time and O(number of sampled merge points). Runtime IDM remains
        purely longitudinal on its cached path; no per-tick route search or lateral planner is
        introduced.

        Do not build this from ``get_route()[0]``.  That is only the first lane object and
        replacing ``agent._path`` with it both discards the rest of the route and can make a
        Hermite tangent curl backwards on a short first segment.  Instead, retain the full
        builder path and join the GT pose to a point ``merge_lookahead`` ahead on it.

        The GT pose is a boundary condition, not something re-derived from sampled points:
        the curve is a cubic Hermite whose start pose is the GT position *and heading* and
        whose end pose lies on the rail with the rail tangent.  A stationary vehicle therefore
        keeps exactly its GT heading.  (The former positional blend derived the heading from
        the first chord; on a 0.45 m remaining rail it rotated a parked car by 40 degrees.)

        The lookahead must be physically available.  When the builder route ends before it
        (a vehicle waiting at a stop line), successors are appended here using clear
        source branch intent or the stock lowest-curvature fallback. Signal compliance is
        unaffected: nuPlan's ``_get_relevant_stop_lines`` stops an agent before every RED
        connector on its route.
        The agent is mutated only when the merge is accepted.
        """
        # Work in the agent's *remaining* path coordinate system. ``agent.progress`` is
        # expressed in the original InterpolatedPath's progress domain; treating it as a
        # Shapely distance along ``get_sampled_path()`` can select a different branch on a
        # curved/doubled-back route.  That mismatch manufactured the observed 120--170 degree
        # flips shortly after handoff.
        get_path_to_go = getattr(agent, "get_path_to_go", None)
        if get_path_to_go is not None:
            sampled_path = list(get_path_to_go())
            rail_progress = 0.0
        else:
            # Lightweight test doubles do not implement get_path_to_go. Their synthetic
            # progress uses the same zero-based distance domain as the sampled path.
            sampled_path = list(agent._path.get_sampled_path())
            rail_progress = float(agent.progress)
        if len(sampled_path) < 2:
            return False
        source = track.center
        source_xy = np.asarray([source.x, source.y], dtype=np.float64)
        source_heading = float(source.heading)
        line = LineString([(float(state.x), float(state.y)) for state in sampled_path])
        rail_progress = float(np.clip(rail_progress, 0.0, float(line.length)))
        rail_point = line.interpolate(rail_progress)
        lateral_offset = float(np.hypot(
            source_xy[0] - rail_point.x, source_xy[1] - rail_point.y
        ))
        speed = NuPlanIDMBatch._track_speed(track)
        wanted_lookahead = max(8.0, 3.0 * lateral_offset, 1.5 * speed)

        # Extend the rail until the merge and one rail sample after it fit.
        appended_edges = []
        end_segment = getattr(agent, "end_segment", None) if get_path_to_go else None
        while (
            end_segment is not None
            and float(line.length) - rail_progress < wanted_lookahead + 1.0
        ):
            outgoing = [
                edge for edge in (getattr(end_segment, "outgoing_edges", ()) or ())
                if str(getattr(edge, "id", "")) not in getattr(agent, "_odyssey_excluded_connector_ids", ())
            ]
            if not outgoing:
                break
            stock = min(
                outgoing,
                key=lambda edge: abs(edge.baseline_path.get_curvature_at_arc_length(0.0)),
            )
            end_segment = choose_source_branch(
                agent, outgoing, stock, incoming_edge=end_segment
            )
            appended_edges.append(end_segment)
            sampled_path.extend(end_segment.baseline_path.discrete_path)
            line = LineString([(float(state.x), float(state.y)) for state in sampled_path])

        available = max(0.0, float(line.length) - rail_progress)
        merge_lookahead = min(available, wanted_lookahead)
        merge_progress = rail_progress + merge_lookahead

        def rail_heading(progress: float) -> float:
            lo = line.interpolate(max(0.0, progress - 0.25))
            hi = line.interpolate(min(float(line.length), progress + 0.25))
            return float(np.arctan2(hi.y - lo.y, hi.x - lo.x))

        merge_end = line.interpolate(merge_progress)
        end_xy = np.asarray([merge_end.x, merge_end.y], dtype=np.float64)
        end_heading = rail_heading(merge_progress)
        start_tangent = merge_lookahead * np.asarray(
            [np.cos(source_heading), np.sin(source_heading)]
        )
        end_tangent = merge_lookahead * np.asarray([np.cos(end_heading), np.sin(end_heading)])

        sample_count = max(3, int(np.ceil(max(merge_lookahead, 1.0) / 0.5)) + 1)
        coordinates, headings, rail_progresses = [], [], []
        for u in np.linspace(0.0, 1.0, sample_count):
            xy = (
                (2 * u**3 - 3 * u**2 + 1) * source_xy
                + (u**3 - 2 * u**2 + u) * start_tangent
                + (-2 * u**3 + 3 * u**2) * end_xy
                + (u**3 - u**2) * end_tangent
            )
            derivative = (
                (6 * u**2 - 6 * u) * source_xy
                + (3 * u**2 - 4 * u + 1) * start_tangent
                + (-6 * u**2 + 6 * u) * end_xy
                + (3 * u**2 - 2 * u) * end_tangent
            )
            coordinates.append(xy)
            headings.append(float(np.arctan2(derivative[1], derivative[0])))
            rail_progresses.append(rail_progress + u * merge_lookahead)

        for progress in np.arange(merge_progress + 1.0, float(line.length), 1.0):
            point = line.interpolate(float(progress))
            coordinates.append(np.asarray([point.x, point.y], dtype=np.float64))
            headings.append(rail_heading(float(progress)))
            rail_progresses.append(float(progress))
        line_end = line.interpolate(float(line.length))
        line_end_xy = np.asarray([line_end.x, line_end.y], dtype=np.float64)
        # When the remaining rail is shorter than the requested merge lookahead, the curve
        # already ends on the path endpoint.  Appending it again creates a zero-length
        # terminal segment, which the forward-path validator correctly rejects.
        if np.linalg.norm(line_end_xy - coordinates[-1]) > 1e-3:
            coordinates.append(line_end_xy)
            headings.append(rail_heading(float(line.length)))
            rail_progresses.append(float(line.length))

        # The curve must travel along the rail, not across or against it: every sample stays
        # within 45 degrees of the rail tangent at the matching progress (this also rejects a
        # source heading opposed to the rail), and no adjacent pair may form a hairpin.  A
        # rejected merge leaves the agent untouched; the caller omits the vehicle.
        path_headings = np.asarray(headings, dtype=np.float64)
        rail_headings = np.asarray([rail_heading(p) for p in rail_progresses])
        rail_errors = np.abs(np.arctan2(
            np.sin(path_headings - rail_headings), np.cos(path_headings - rail_headings)
        ))
        heading_steps = np.abs(np.arctan2(
            np.sin(np.diff(path_headings)), np.cos(np.diff(path_headings))
        ))
        segment_lengths = np.linalg.norm(np.diff(np.asarray(coordinates), axis=0), axis=1)
        if (
            np.any(rail_errors > np.deg2rad(45.0))
            or np.any(heading_steps > np.deg2rad(45.0))
            or np.any(segment_lengths <= 1e-3)
        ):
            return False

        path = [
            StateSE2(float(xy[0]), float(xy[1]), heading)
            for xy, heading in zip(coordinates, headings)
        ]
        tail = path[-1]
        path.append(StateSE2(
            float(tail.x) + 0.1 * np.cos(float(tail.heading)),
            float(tail.y) + 0.1 * np.sin(float(tail.heading)),
            float(tail.heading),
        ))
        for edge in appended_edges:
            agent._route.append(edge)
        agent._path = create_path_from_se2(path)
        agent._state.progress = 0.0
        agent._requires_state_update = True
        return True

    def _solve_once(self, step: int, prev: int, ego_step: int):
        """Run one unmodified nuPlan observation update over ``prev -> step``.

        ``ego_step`` is normally ``step`` for the legacy 10 Hz adapter.  In multi-rate
        mode it is the beginning of the solve interval, matching a causal simulator: the
        reactive fleet cannot inspect a future closed-loop ego pose.
        """
        try:
            self._deferred_tokens.clear()
            ego_state = self.converter.convert_to_current_ego_state(ego_step)
            # This global iteration keys the observation cache. In sector mode the converter
            # resolves each actor's own source row through BaseAgentManager; propagation and
            # IDM integration still use real simulation time.
            source_step = int(getattr(self.engine.agent_manager, 'idm_source_step', step))
            gt_tracks = self._scenario.get_tracked_objects_at_iteration(source_step)
            source_vehicle_tokens = {
                str(track.track_token)
                for track in gt_tracks.tracked_objects.get_tracked_objects_of_type(
                    TrackedObjectType.VEHICLE
                )
                if track.track_token is not None
            }
            mgr = self._obs._get_idm_agent_manager()
            for token, agent in mgr.agents.items():
                _install_idm_connector_filter(agent, self._excluded_idm_connector_ids)
                self._attach_idm_route_intent(token, agent, step)
            # centerline_snap_warmup / zero warm-up build through the stock builder, which
            # does not know about pinned vehicles. Keep them as open-loop obstacles only.
            for token in sorted(set(mgr.agents) & (
                    set(self._static_vehicle_tokens) | set(self._predropped_tokens()))):
                mgr.agents.pop(token, None)
                if mgr.agent_occupancy.contains(token):
                    mgr.agent_occupancy.remove([token])
            self._remove_ineligible_initial_agents(mgr, step)
            unstable_existing = set(mgr.agents) & self._unstable_short_track_tokens
            for token in sorted(unstable_existing):
                mgr.agents.pop(token, None)
                if mgr.agent_occupancy.contains(token):
                    mgr.agent_occupancy.remove([token])
                self._obs.extra_open_loop_vehicle_tokens.add(token)
                self._unroutable.add(token)
                logger.info(
                    "[NUPLAN-IDM] removed temporally unstable short track %s "
                    "from reactive fleet at tick=%d",
                    token,
                    step,
                )
            if (
                self._intersection_manager is not None
                or self._wait_cycle_monitor_enabled
                or self._vehicle_only_lead_lane_ids
            ):
                mgr = self._ensure_targeted_lead_manager(mgr)
                mgr.configure_wait_cycle_monitor(
                    self._wait_cycle_timeout_ticks,
                    self._wait_cycle_stopped_speed_mps,
                    self._ego_deadlock_timeout_ticks,
                )
                mgr.configure_physical_path_leads(self._physical_lead_margin_m)
                mgr.configure_scoped_vehicle_only_leads(
                    self._vehicle_only_lead_lane_ids
                )
            self._configure_emergency_braking(mgr.agents.values())

            # The t=0 builder has already snapped every routable vehicle to a rail. Undo that
            # for stationary curb/unstructured objects before either the intersection manager
            # or propagation can treat the artificial lane-centre pose as traffic.
            self._demote_initial_parked_vehicles(mgr, gt_tracks, step)
            self._reject_invalid_initial_centerline_snaps(mgr, gt_tracks, step)

            if (
                step == self._idm_start_step
                and self._handoff_snap_diagnostics_enabled
                and not self._handoff_snap_stats
            ):
                # Audit the state which is actually exposed after parked/off-rail vehicles
                # have been restored to GT. Measuring the raw builder output here falsely
                # reports a rail snap which the production policy immediately undoes.
                self._handoff_snap_stats = self._measure_handoff_snap(
                    gt_tracks, ego_state, mgr
                )

            # Stock IDMAgents inserts the configured open-loop detections while propagating and
            # removes them afterwards.  Late admission is our long-window extension and happens
            # just before that propagation, so expose the same objects to its collision check as
            # well.  Remove them before update_observation, which performs the official insertion
            # itself.
            admission_open_loop_tokens = self._insert_open_loop_for_admission(gt_tracks, mgr)
            newly_admitted = set()
            if self._midspawn:
                try:
                    newly_admitted = self._admit_new_agents(step, gt_tracks, ego_state)
                finally:
                    removable = [tok for tok in admission_open_loop_tokens
                                 if mgr.agent_occupancy.contains(tok)]
                    if removable:
                        mgr.agent_occupancy.remove(removable)
            elif admission_open_loop_tokens:
                mgr.agent_occupancy.remove(admission_open_loop_tokens)

            # Low-motion vehicles deliberately kept out of IDM are still physical GT replay
            # objects.  Do not publish one into an occupied pose: defer it by one simulation
            # tick and retry while its source track remains valid.  Once admitted, stock IDM
            # sees it as an open-loop obstacle and reacts normally.
            self._admit_source_motion_open_loop_vehicles(
                gt_tracks, mgr, ego_state, step
            )

            if self._intersection_manager is not None:
                try:
                    # Extend the same NPC route rails that stock propagation will consume.  This
                    # is not a second planner and it does not read any ego route/planner state.
                    traffic_light_status = self._traffic_light_status(
                        step, mgr.agents.values()
                    )
                    # Two precisely configured Las Vegas circular-road lanes have erroneous
                    # signalized frontiers in the map/log.  "Vehicle-only leads" on those lanes
                    # therefore also means no virtual signal stop: rewrite only their immediate
                    # outgoing connectors as green for route extension, then remove them from
                    # the manager's controlled set so normal unsignalized FIFO/corridor rules
                    # still serialize conflicting vehicles.
                    signal_bypass = {
                        str(edge.id)
                        for agent in mgr.agents.values()
                        if agent.is_active(step)
                        and agent.has_valid_path()
                        and mgr.uses_scoped_vehicle_only_leads(agent)
                        for edge in (
                            getattr(agent.end_segment, "outgoing_edges", ()) or ()
                        )
                        if edge.has_traffic_lights()
                    }
                    self._vehicle_only_signal_bypass_connector_ids = frozenset(
                        signal_bypass
                    )
                    if signal_bypass:
                        for signal_state in tuple(traffic_light_status):
                            traffic_light_status[signal_state] = [
                                connector_id
                                for connector_id in traffic_light_status[signal_state]
                                if connector_id not in signal_bypass
                            ]
                        traffic_light_status[TrafficLightStatusType.GREEN].extend(
                            sorted(signal_bypass)
                        )
                    for token, agent in mgr.agents.items():
                        if agent.is_active(step) and agent.has_valid_path():
                            agent.plan_route(traffic_light_status)
                            if (
                                not self._source_end_route_exhaustion_despawn_enabled
                                and str(token) not in source_vehicle_tokens
                            ):
                                self._extend_source_end_dead_end_path(str(token), agent)
                    mgr.mark_routes_preplanned(step)
                    ego_velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d
                    decisions = self._intersection_manager.update(
                        IntersectionWorldState(
                            tick=step,
                            occupancy=mgr.agent_occupancy,
                            controlled_lane_connector_ids=frozenset(
                                connector_id
                                for connector_ids in traffic_light_status.values()
                                for connector_id in connector_ids
                            ) - self._unknown_signal_fallback_connector_ids
                            - self._vehicle_only_signal_bypass_connector_ids,
                            green_lane_connector_ids=frozenset(
                                traffic_light_status[TrafficLightStatusType.GREEN]
                            ) - self._unknown_signal_fallback_connector_ids
                            - self._vehicle_only_signal_bypass_connector_ids,
                        ),
                        mgr.agents,
                        EgoSafetyState(
                            footprint=ego_state.car_footprint.geometry,
                            velocity_x=float(ego_velocity.x),
                            velocity_y=float(ego_velocity.y),
                        ),
                    )
                    eligible_stalled_tokens = None
                    if not self._intersection_stall_liveness_enabled:
                        # Source presence is checked at the current frame only. Thus a red/queue
                        # wait whose log track is still alive cannot be deleted by this relief.
                        eligible_stalled_tokens = set(mgr.agents) - source_vehicle_tokens
                    stalled_tokens = self._intersection_manager.consume_active_stall_retirements(
                        eligible_stalled_tokens
                    )
                    for token in stalled_tokens:
                        self.remove_agent(token)
                    if self._source_ended_active_stall_relief_enabled:
                        self._source_ended_active_stall_retirements += len(
                            stalled_tokens - source_vehicle_tokens
                        )
                    if stalled_tokens:
                        logger.warning(
                            "[IDM-IM] tick=%d retired %d active stalled NPC(s): %s",
                            step,
                            len(stalled_tokens),
                            ",".join(sorted(stalled_tokens)),
                        )
                    mgr.set_targeted_virtual_leads(
                        self._intersection_manager.targeted_virtual_leads(decisions)
                    )
                    mgr.configure_active_intersection_tokens(
                        self._intersection_manager.active_npc_ids()
                    )
                except Exception:  # noqa: BLE001 - policy failure must degrade to stock IDM
                    mgr.set_targeted_virtual_leads({})
                    logger.exception(
                        "[IDM-IM] update failed at step %s; running stock IDM for this tick",
                        step,
                    )
            # True map-dead-end relief is an independent, explicitly scoped lifecycle valve;
            # it must not require the optional intersection-admission controller to be on.
            # Keeping the call outside that block lets stock IDM use the same narrow rule.
            if self._dead_end_queue_relief_enabled:
                dead_end_tokens = self._collect_dead_end_queue_retirements(mgr, step)
                for token in dead_end_tokens:
                    self.remove_agent(token)
                if dead_end_tokens:
                    logger.warning(
                        "[NUPLAN-IDM] tick=%d retired %d queued true-dead-end lead(s): %s",
                        step,
                        len(dead_end_tokens),
                        ",".join(sorted(dead_end_tokens)),
                    )
            self._obs.update_observation(
                SimulationIteration(TimePoint(int(prev * self._dt * 1e6)), prev),
                SimulationIteration(TimePoint(int(step * self._dt * 1e6)), step),
                _HistoryShim(ego_state),
            )
            _assert_no_excluded_idm_routes(
                mgr.agents, self._excluded_idm_connector_ids
            )
            exhausted_tokens = self._collect_dead_end_route_exhaustions(mgr, step)
            for token in exhausted_tokens:
                self.remove_agent(token)
            if exhausted_tokens:
                logger.info(
                    "[NUPLAN-IDM] tick=%d despawned %d vehicle(s) at "
                    "true IDM route dead-end: %s",
                    step,
                    len(exhausted_tokens),
                    ",".join(sorted(exhausted_tokens)),
                )
            tracks = self._obs.get_observation()
            rejected_after_propagation = self._reject_overlapping_new_agents(
                newly_admitted, tracks, ego_state, mgr, step)
            self._admitted_after_start += len(newly_admitted - rejected_after_propagation)
            poses = {}
            source_modes = {}
            for obj in getattr(tracks, "tracked_objects", []) or []:
                tok = str(getattr(obj, "track_token", "") or "")
                if not tok or tok in rejected_after_propagation:
                    continue
                c = obj.center
                v = getattr(obj, "velocity", None)
                speed = float(np.hypot(v.x, v.y)) if v is not None else 0.0
                poses[tok] = (float(c.x) - self._origin[0], float(c.y) - self._origin[1],
                              float(c.heading), speed)
                source_modes[tok] = "idm" if tok in mgr.agents else "open_loop"

            # Stock nuPlan permanently forgets an agent after radius filtering.  Preserve that
            # one-way lifecycle for the long-window admission extension too: only agents which
            # were actually exposed as IDM traffic count as having appeared.  Builder refusals
            # and collision-deferred candidates therefore remain retryable.
            current_idm_tokens = {
                token for token, mode in source_modes.items() if mode == "idm"
            }
            self._retired.update(self._ever_admitted - current_idm_tokens)
            self._ever_admitted.update(current_idm_tokens)
            self._poses = poses
            self._source_modes = source_modes
        except Exception as exc:                                        # noqa: BLE001
            mgr = self._obs._get_idm_agent_manager()
            removable = [tok for tok in locals().get("admission_open_loop_tokens", [])
                         if mgr.agent_occupancy.contains(tok)]
            if removable:
                mgr.agent_occupancy.remove(removable)
            if not self._fail_on_propagate_error:
                # Ablation only: agents hold their last pose and the log says which step
                # lost them.
                logger.warning("[NUPLAN-IDM] propagate failed at step %s: %s", step, exc)
                return
            raise IDMPropagateError(self._record_propagate_failure(step, exc, mgr)) from exc

    def _idm_invariant_violations(self, mgr, step):
        """IDM agents failing exactly what ``propagate_agents`` asserts: the occupancy entry of
        the agent must intersect ``path_to_go.buffer(width / 2)`` ("Agent's baseline does not
        intersect the agent itself").  Reports the tested geometries, not the rail polyline: a
        car with a 0x0 source box has an EMPTY buffer, while a point-to-line distance on the
        polyline reads a misleading 0.0 m.
        """
        from odyssey.manager.agent_manager import _is_pinned_static_vehicle

        scene_tracks = (getattr(self, "scene", None) or {}).get("object_track", {})
        violations = []
        for token, agent in mgr.agents.items():
            record = {"token": str(token)}
            try:
                if not (agent.is_active(step) and agent.has_valid_path()):
                    continue
                path = path_to_linestring(agent.get_path_to_go())
                buffer = path.buffer(agent.width / 2, cap_style=CAP_STYLE.flat)
                if mgr.agent_occupancy.intersects(buffer).contains(token):
                    continue
                pose = agent.to_se2()
                occupied = (mgr.agent_occupancy.get(token)
                            if mgr.agent_occupancy.contains(token) else None)
                record.update(
                    reason=("zero_size_box" if agent.width <= 0 or buffer.is_empty
                            else "not_in_occupancy" if occupied is None
                            else "occupancy_off_path"),
                    pose_xy_heading=[float(pose.x), float(pose.y), float(pose.heading)],
                    box_length_width_m=[float(agent.length), float(agent.width)],
                    path_buffer_empty=bool(buffer.is_empty),
                    occupancy_area_m2=None if occupied is None else float(occupied.area),
                    occupancy_to_path_buffer_m=(
                        None if occupied is None or buffer.is_empty
                        else float(occupied.distance(buffer))),
                )
            except Exception as check_exc:                          # noqa: BLE001
                record["check_error"] = f"{type(check_exc).__name__}: {check_exc}"
            record["source_mode"] = getattr(self, "_source_modes", {}).get(str(token), "unknown")
            # The scenario-track predicate, independent of whether the switch is on.
            record["pinned_static"] = bool(
                str(token) in scene_tracks
                and _is_pinned_static_vehicle(scene_tracks[str(token)]))
            record["flag_static_pin_on"] = bool(self._static_pin_and_predrop)
            violations.append(record)
        if violations:
            # propagate_agents walks mgr.agents in order and stops at the first failure.
            violations[0]["asserted_on"] = True
        return violations

    def _record_propagate_failure(self, step, exc, mgr):
        """Log, archive (<data_output_dir>/idm_failure.json) and summarize an IDM failure."""
        try:
            agents = self._idm_invariant_violations(mgr, step)
        except Exception as check_exc:                              # noqa: BLE001
            agents = [{"check_error": f"{type(check_exc).__name__}: {check_exc}"}]
        message = (str(exc).strip().splitlines() or [type(exc).__name__])[0][:100]
        summary = f"IDM_PROPAGATE_FAILED step {step}: {message}"
        named = [a["token"] for a in agents if "token" in a]
        if named:
            summary += f" (agent {named[0][:8]}" + (f" +{len(named) - 1}" if len(named) > 1 else "") + ")"
        record = {
            "summary": summary,
            "step": int(step),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "agents": agents,
            "traceback": traceback.format_exc(),
        }
        logger.error("[NUPLAN-IDM] %s | %s: %s | agents=%s",
                     summary, type(exc).__name__, exc, agents)
        if self._failure_output_dir:
            try:
                os.makedirs(self._failure_output_dir, exist_ok=True)
                with open(os.path.join(self._failure_output_dir, "idm_failure.json"), "w") as fh:
                    json.dump(record, fh, indent=2)
            except OSError as write_exc:
                logger.warning("[NUPLAN-IDM] could not write idm_failure.json: %s", write_exc)
        return summary

    def _extend_source_end_dead_end_path(self, token: str, agent: object) -> bool:
        """Append one physical egress tail at a true map-graph boundary.

        A red signal can temporarily leave ``plan_route`` with no selectable successor,
        but the current segment still has outgoing edges.  That case must remain stopped.
        We extend only when the map itself has no successor and only once per vehicle.
        """
        if (
            not self._source_end_route_extension_enabled
            or token in self._source_end_extended_tokens
            or float(agent.get_progress_to_go()) > self._source_end_route_extension_trigger_m
            or tuple(getattr(agent.end_segment, "outgoing_edges", ()) or ())
        ):
            return False

        remaining = list(agent.get_path_to_go())
        if not remaining:
            return False
        tail = remaining[-1]
        heading = float(tail.heading)
        spacing_m = 5.0
        distances = np.arange(
            spacing_m,
            # nuPlan's create_path_from_se2 intentionally drops the final sample while
            # filtering repeated progress values. Supply one guard sample so the usable
            # interpolated path retains the full requested tail length.
            self._source_end_route_extension_length_m + 2.0 * spacing_m,
            spacing_m,
        )
        path = [StateSE2(float(state.x), float(state.y), float(state.heading)) for state in remaining]
        path.extend(
            StateSE2(
                float(tail.x) + float(distance) * np.cos(heading),
                float(tail.y) + float(distance) * np.sin(heading),
                heading,
            )
            for distance in distances
        )
        if len(path) < 2:
            return False
        agent._path = create_path_from_se2(path)
        agent._state.progress = 0.0
        agent._requires_state_update = True
        self._source_end_extended_tokens.add(token)
        self._source_end_route_extensions += 1
        logger.info(
            "[NUPLAN-IDM] source-ended agent %s received %.1f m map-edge egress tail",
            token,
            self._source_end_route_extension_length_m,
        )
        return True

    def _collect_dead_end_route_exhaustions(self, manager, step: int):
        """Despawn an IDM vehicle before its front bumper passes a true route end.

        Source-log validity does not extend a map rail. A RED or UNKNOWN signal only
        delays a formal successor; it is not a dead-end. Scene-excluded connectors,
        however, are not legal IDM successors and cannot protect a stalled vehicle.
        """
        if not self._source_end_route_exhaustion_despawn_enabled:
            return set()
        retirements = set()
        for token, agent in manager.agents.items():
            end_segment = getattr(agent, "end_segment", None)
            if (
                not agent.is_active(step)
                or not agent.has_valid_path()
                or end_segment is None
            ):
                continue
            excluded = getattr(agent, "_odyssey_excluded_connector_ids", ())
            usable_successors = (
                edge for edge in (getattr(end_segment, "outgoing_edges", ()) or ())
                if str(edge.id) not in excluded
            )
            if any(usable_successors):
                continue
            front_margin = max(
                0.0, float(getattr(agent, "length", 0.0)) / 2.0
            ) + self._source_end_route_exhaustion_margin_m
            if float(agent.get_progress_to_go()) <= front_margin:
                retirements.add(str(token))
        self._source_end_route_exhaustion_despawns += len(retirements)
        return retirements

    def _collect_dead_end_queue_retirements(self, manager, step: int):
        """Return stalled leads at configured true map dead-ends with a queued follower.

        ``get_progress_to_go`` is measured to each agent's route end.  Agents sharing the same
        terminal segment can therefore be ordered without guessing from Euclidean distance on
        a curved lane: the smaller value is farther ahead.  Requiring a real graph dead-end,
        a near-end lead, sustained physical stagnation, and a same-terminal follower prevents
        this valve from treating red lights or ordinary intersection waits as disposable cars.
        """
        if not self._dead_end_queue_relief_enabled:
            return set()

        candidates = []
        for token, agent in manager.agents.items():
            if not (agent.is_active(step) and agent.has_valid_path()):
                continue
            end_segment = agent.end_segment
            lane_id = str(getattr(end_segment, "id", ""))
            if (
                lane_id not in self._dead_end_queue_relief_lane_ids
                or tuple(getattr(end_segment, "outgoing_edges", ()) or ())
            ):
                continue
            progress_to_go = float(agent.get_progress_to_go())
            candidates.append((str(token), agent, lane_id, progress_to_go))

        eligible_tokens = {token for token, _agent, _lane_id, _progress in candidates}
        for token in tuple(self._dead_end_queue_progress):
            if token not in eligible_tokens:
                self._dead_end_queue_progress.pop(token, None)

        retirements = set()
        for token, agent, lane_id, progress_to_go in candidates:
            near_end = progress_to_go <= self._dead_end_queue_relief_trigger_m
            stopped = float(agent.velocity) <= self._dead_end_queue_relief_stopped_speed_mps
            follower_present = any(
                other_lane_id == lane_id
                and other_progress > progress_to_go
                and other_progress - progress_to_go
                <= self._dead_end_queue_relief_follower_distance_m
                for other_token, _other, other_lane_id, other_progress in candidates
                if other_token != token
            )
            if not (
                near_end
                and stopped
                and (
                    follower_present
                    or not getattr(
                        self, "_dead_end_queue_relief_require_follower", True
                    )
                )
            ):
                self._dead_end_queue_progress.pop(token, None)
                continue

            point = np.array([agent.to_se2().x, agent.to_se2().y], dtype=float)
            first_tick, anchor = self._dead_end_queue_progress.get(
                token, (step, point.copy())
            )
            if np.linalg.norm(point - anchor) >= self._dead_end_queue_relief_progress_epsilon_m:
                first_tick, anchor = step, point.copy()
            self._dead_end_queue_progress[token] = (first_tick, anchor)
            if step - first_tick >= self._dead_end_queue_relief_timeout_ticks:
                retirements.add(token)

        for token in retirements:
            self._dead_end_queue_progress.pop(token, None)
        self._dead_end_queue_retirements += len(retirements)
        return retirements

    @staticmethod
    def _interpolate_idm_poses(start, target, alpha):
        """Interpolate agents present at both nuPlan solve boundaries.

        A newly appearing target is intentionally hidden until its solve boundary.  An
        agent which disappears at the target remains at its last state during the interval
        and is removed on the boundary.  This avoids manufacturing a spawn/despawn pose.
        """
        poses = {}
        for token, pose0 in start.items():
            pose1 = target.get(token)
            if pose1 is None:
                if alpha < 1.0:
                    poses[token] = pose0
                continue
            x0, y0, h0, v0 = pose0
            x1, y1, h1, v1 = pose1
            dh = (h1 - h0 + np.pi) % (2 * np.pi) - np.pi
            poses[token] = (
                float(x0 + alpha * (x1 - x0)),
                float(y0 + alpha * (y1 - y0)),
                float(h0 + alpha * dh),
                float(v0 + alpha * (v1 - v0)),
            )
        if alpha >= 1.0:
            for token, pose in target.items():
                poses.setdefault(token, pose)
        return poses

    def _current_open_loop_poses(self, step):
        """Return stock configured open-loop objects at the fine Odyssey tick."""
        tracks = self._scenario.get_tracked_objects_at_iteration(step)
        poses = {}
        for obj in getattr(tracks.tracked_objects, "tracked_objects", []) or []:
            if getattr(obj, "tracked_object_type", None) not in self._obs._open_loop_detections_types:
                continue
            token = str(getattr(obj, "track_token", "") or "")
            if not token:
                continue
            center = obj.center
            velocity = getattr(obj, "velocity", None)
            speed = float(np.hypot(velocity.x, velocity.y)) if velocity is not None else 0.0
            poses[token] = (
                float(center.x) - self._origin[0], float(center.y) - self._origin[1],
                float(center.heading), speed,
            )
        return poses

    def _advance_multirate(self, step):
        """Solve nuPlan at its configured cadence and expose interpolated fine ticks."""
        if self._interval_target_step is None:
            # Materialize the exact initial state without advancing time, then compute the
            # first future solve boundary.  The manager remains at the future boundary while
            # the proxies traverse the interval; no further occupancy query occurs meanwhile.
            self._solve_once(step, step, step)
            self._interval_start_step = step
            self._interval_start_poses = {
                token: pose for token, pose in self._poses.items()
                if self._source_modes.get(token) == "idm"
            }
            self._interval_start_modes = {
                token: "idm" for token in self._interval_start_poses
            }
            target_step = step + self._solve_stride
            self._solve_once(target_step, step, step)
            self._interval_target_step = target_step
            self._interval_target_poses = {
                token: pose for token, pose in self._poses.items()
                if self._source_modes.get(token) == "idm"
            }
            self._interval_target_modes = {
                token: "idm" for token in self._interval_target_poses
            }

        while step >= self._interval_target_step:
            boundary = self._interval_target_step
            self._interval_start_step = boundary
            self._interval_start_poses = self._interval_target_poses
            self._interval_start_modes = self._interval_target_modes
            target_step = boundary + self._solve_stride
            self._solve_once(target_step, boundary, boundary)
            self._interval_target_step = target_step
            self._interval_target_poses = {
                token: pose for token, pose in self._poses.items()
                if self._source_modes.get(token) == "idm"
            }
            self._interval_target_modes = {
                token: "idm" for token in self._interval_target_poses
            }

        alpha = (step - self._interval_start_step) / self._solve_stride
        poses = self._interpolate_idm_poses(
            self._interval_start_poses, self._interval_target_poses, alpha)
        source_modes = {token: "idm" for token in poses}
        open_loop = self._current_open_loop_poses(step)
        poses.update(open_loop)
        source_modes.update({token: "open_loop" for token in open_loop})
        self._poses = poses
        self._source_modes = source_modes

    def _insert_open_loop_for_admission(self, gt_tracks, mgr):
        """Temporarily mirror nuPlan's configured open-loop occupancy for late admission."""
        inserted = []
        for obj in getattr(gt_tracks.tracked_objects, "tracked_objects", []) or []:
            tok = str(getattr(obj, "track_token", "") or "")
            if ((getattr(obj, "tracked_object_type", None)
                    not in self._obs._open_loop_detections_types
                    and tok not in self._static_vehicle_tokens)
                    or not tok or mgr.agent_occupancy.contains(tok)):
                continue
            geometry = obj.box.geometry
            if geometry.is_empty or not geometry.is_valid:
                continue
            mgr.agent_occupancy.insert(tok, geometry)
            inserted.append(tok)
        return inserted

    @staticmethod
    def _overlap_pairs(geometries):
        """Return token pairs whose physical boxes overlap by a nontrivial area."""
        items = sorted(geometries.items())
        pairs = set()
        for index, (left_token, left_geometry) in enumerate(items):
            if left_geometry.is_empty or not left_geometry.is_valid:
                continue
            for right_token, right_geometry in items[index + 1:]:
                if right_geometry.is_empty or not right_geometry.is_valid:
                    continue
                if left_geometry.intersection(right_geometry).area > 0.01:
                    pairs.add((left_token, right_token))
        return pairs

    def _measure_handoff_snap(self, gt_tracks, ego_state, manager):
        """Measure the one GT-to-rail transition without changing admission or motion."""
        objects = list(getattr(gt_tracks.tracked_objects, "tracked_objects", []) or [])
        candidate_raw_geometries = {"ego": ego_state.car_footprint.geometry}
        candidate_snap_geometries = {"ego": ego_state.car_footprint.geometry}
        actual_raw_geometries = {"ego": ego_state.car_footprint.geometry}
        actual_snap_geometries = {"ego": ego_state.car_footprint.geometry}
        candidate_metrics = {}
        outliers = []
        routable_tokens = set()
        vehicle_tokens = set()
        objects_by_token = {}

        for obj in objects:
            token = str(getattr(obj, "track_token", "") or "")
            if not token:
                continue
            objects_by_token[token] = obj
            candidate_raw_geometries[token] = obj.box.geometry
            candidate_snap_geometries[token] = obj.box.geometry
            if getattr(obj, "tracked_object_type", None) != TrackedObjectType.VEHICLE:
                continue
            vehicle_tokens.add(token)
            route, _progress = get_starting_segment(obj, self._scenario.map_api)
            if route is None:
                continue
            routable_tokens.add(token)
            state_on_path = route.baseline_path.get_nearest_pose_from_position(
                obj.center.point
            )
            snapped_box = OrientedBox.from_new_pose(
                obj.box,
                StateSE2(
                    state_on_path.x,
                    state_on_path.y,
                    state_on_path.heading,
                ),
            )
            candidate_snap_geometries[token] = snapped_box.geometry
            displacement = float(np.hypot(
                state_on_path.x - obj.center.x,
                state_on_path.y - obj.center.y,
            ))
            heading_error_deg = float(np.rad2deg(abs(
                (state_on_path.heading - obj.center.heading + np.pi)
                % (2.0 * np.pi)
                - np.pi
            )))
            candidate_metrics[token] = (displacement, heading_error_deg)
            outliers.append((displacement, heading_error_deg, token))

        built_tokens = vehicle_tokens.intersection(manager.agents)
        fallback_tokens = vehicle_tokens.intersection(
            self._obs.extra_open_loop_vehicle_tokens
        )
        exposed_tokens = built_tokens | fallback_tokens | {
            token
            for token, obj in objects_by_token.items()
            if getattr(obj, "tracked_object_type", None)
            in self._obs._open_loop_detections_types
        }
        for token in exposed_tokens:
            obj = objects_by_token.get(token)
            if obj is None:
                continue
            actual_raw_geometries[token] = obj.box.geometry
            if token in built_tokens:
                agent = manager.agents[token]
                actual_snap_geometries[token] = agent.polygon
            else:
                actual_snap_geometries[token] = obj.box.geometry

        candidate_raw_pairs = self._overlap_pairs(candidate_raw_geometries)
        candidate_snap_pairs = self._overlap_pairs(candidate_snap_geometries)
        actual_raw_pairs = self._overlap_pairs(actual_raw_geometries)
        actual_snap_pairs = self._overlap_pairs(actual_snap_geometries)
        displacements = [candidate_metrics[token][0] for token in built_tokens]
        heading_errors_deg = [candidate_metrics[token][1] for token in built_tokens]
        dropped_routable_tokens = routable_tokens - built_tokens - fallback_tokens

        def percentile(values, q):
            return float(np.percentile(values, q)) if values else 0.0

        stats = {
            "step": int(self._idm_start_step),
            "source_vehicles": len(vehicle_tokens),
            "routable_vehicles": len(routable_tokens),
            "built_idm_vehicles": len(built_tokens),
            "gt_fallback_vehicles": len(fallback_tokens),
            "dropped_routable_vehicles": len(dropped_routable_tokens),
            # Backward-compatible name used by the first audit. It now means genuinely
            # absent at handoff, not a vehicle deliberately kept at its GT pose.
            "routable_not_built": len(dropped_routable_tokens),
            "snap_distance_p50_m": percentile(displacements, 50),
            "snap_distance_p95_m": percentile(displacements, 95),
            "snap_distance_max_m": max(displacements, default=0.0),
            "snap_heading_p95_deg": percentile(heading_errors_deg, 95),
            "snap_heading_max_deg": max(heading_errors_deg, default=0.0),
            "snap_over_0_5m": sum(value > 0.5 for value in displacements),
            "snap_over_1m": sum(value > 1.0 for value in displacements),
            "snap_over_2m": sum(value > 2.0 for value in displacements),
            "raw_overlap_pairs": len(actual_raw_pairs),
            "snapped_overlap_pairs": len(actual_snap_pairs),
            "new_snap_overlap_pairs": len(actual_snap_pairs - actual_raw_pairs),
            "resolved_snap_overlap_pairs": len(actual_raw_pairs - actual_snap_pairs),
            "candidate_new_snap_overlap_pairs": len(
                candidate_snap_pairs - candidate_raw_pairs
            ),
        }
        logger.info(
            "[NUPLAN-IDM-HANDOFF] scene=%s stats=%s",
            self.scene.get("name", self.scene.get("id", "<scene>")),
            stats,
        )
        notable = sorted(outliers, reverse=True)[:5]
        if notable:
            logger.info(
                "[NUPLAN-IDM-HANDOFF] largest_snap=(distance_m,heading_deg,token) %s",
                notable,
            )
        new_pairs = sorted(actual_snap_pairs - actual_raw_pairs)
        if new_pairs:
            logger.warning(
                "[NUPLAN-IDM-HANDOFF] snap introduced %d overlap pair(s): %s",
                len(new_pairs),
                new_pairs[:10],
            )
        missing_tokens = sorted(dropped_routable_tokens)
        if missing_tokens:
            candidate_overlap_partners = {
                token: {
                    "raw": sorted({
                        right if left == token else left
                        for left, right in candidate_raw_pairs
                        if token in (left, right)
                    }),
                    "snapped": sorted({
                        right if left == token else left
                        for left, right in candidate_snap_pairs
                        if token in (left, right)
                    }),
                }
                for token in missing_tokens
            }
            logger.info(
                "[NUPLAN-IDM-HANDOFF] routable-but-not-built tokens=%s "
                "candidate_overlap_partners=%s",
                missing_tokens,
                candidate_overlap_partners,
            )
        return stats

    def _signal_intersection_id(self, lane_connector_id: str):
        """Resolve one signal connector to its physical intersection, with negative caching."""
        lane_connector_id = str(lane_connector_id)
        if lane_connector_id in self._signal_connector_intersection_cache:
            return self._signal_connector_intersection_cache[lane_connector_id]
        intersection_id = None
        try:
            connector = self._scenario.map_api.get_map_object(
                lane_connector_id, SemanticMapLayer.LANE_CONNECTOR
            )
            parent = connector.parent if connector is not None else None
            intersection = parent.intersection if parent is not None else None
            if callable(intersection):
                intersection = intersection()
            if intersection is not None:
                intersection_id = str(intersection.id)
        except (AttributeError, KeyError, NotImplementedError, TypeError, ValueError):
            intersection_id = None
        self._signal_connector_intersection_cache[lane_connector_id] = intersection_id
        return intersection_id

    def _record_signal_fallback(self, step, connector_ids):
        """Publish one tick's signal-fallback scope and update transition counters."""
        fallback_connectors = frozenset(str(connector_id) for connector_id in connector_ids)
        fallback_intersections = frozenset(
            intersection_id
            for connector_id in fallback_connectors
            if (intersection_id := self._signal_intersection_id(connector_id)) is not None
        )

        if self._unknown_signal_fallback_step != int(step):
            self._unknown_signal_fallback_events += len(
                fallback_intersections
                - self._unknown_signal_fallback_active_intersections
            )
            self._unknown_signal_fallback_intersection_ticks += len(
                fallback_intersections
            )
            self._unknown_signal_fallback_connector_ticks += len(fallback_connectors)
            self._unknown_signal_fallback_active_intersections = fallback_intersections
            self._unknown_signal_fallback_step = int(step)
            self._unknown_signal_fallback_connector_ids = fallback_connectors
            return

        # The scenario adapter is normally queried twice per solve.  A record-less query may
        # discover its active route-frontier connectors only in ``_traffic_light_status``;
        # merge that late discovery without double-counting an already published connector.
        new_connectors = fallback_connectors - self._unknown_signal_fallback_connector_ids
        new_intersections = (
            fallback_intersections
            - self._unknown_signal_fallback_active_intersections
        )
        self._unknown_signal_fallback_events += len(new_intersections)
        self._unknown_signal_fallback_intersection_ticks += len(new_intersections)
        self._unknown_signal_fallback_connector_ticks += len(new_connectors)
        self._unknown_signal_fallback_active_intersections |= fallback_intersections
        self._unknown_signal_fallback_connector_ids |= fallback_connectors

    def _apply_unknown_signal_unprotected_fallback(self, step, traffic_light_data):
        """Treat an all-UNKNOWN/absent signal frame as unsignalized for IDM.

        UNKNOWN connectors are rewritten to GREEN only for stock ``IDMAgent.plan_route``. The
        same connector IDs are removed from the manager's controlled set in ``_solve_once``, so
        passage admission is still governed by the unsignalized FIFO/corridor safety rules.
        """
        if not self._unknown_signal_fallback_enabled:
            return traffic_light_data

        records = list(traffic_light_data)
        # An empty frame has no connector IDs to rewrite.  Defer it to
        # ``_traffic_light_status``, which can recover the signalized connectors directly from
        # the active IDM route frontiers.  In particular, do not clear a scope already recovered
        # earlier in the same solve when stock IDMAgents queries the adapter for a second time.
        if not records:
            return records

        connector_intersections = {}
        intersection_statuses = defaultdict(list)
        for data in records:
            connector_id = str(data.lane_connector_id)
            intersection_id = self._signal_intersection_id(connector_id)
            if intersection_id is None:
                continue
            connector_intersections[connector_id] = intersection_id
            intersection_statuses[intersection_id].append(data.status)

        # Signal validity is local to a physical intersection.  A long rollout can reach an
        # intersection whose connectors are all UNKNOWN while an unrelated, distant junction
        # still has a clamped RED/GREEN record.  Requiring the entire scene to be UNKNOWN leaves
        # the former permanently closed even though its own controller is unavailable.
        fallback_intersections = frozenset(
            intersection_id
            for intersection_id, statuses in intersection_statuses.items()
            if statuses
            and all(status == TrafficLightStatusType.UNKNOWN for status in statuses)
        )
        fallback_connectors = frozenset(
            connector_id
            for connector_id, intersection_id in connector_intersections.items()
            if intersection_id in fallback_intersections
        )
        transformed = [
            TrafficLightStatusData(
                status=(
                    TrafficLightStatusType.GREEN
                    if str(data.lane_connector_id) in fallback_connectors
                    and data.status == TrafficLightStatusType.UNKNOWN
                    else data.status
                ),
                lane_connector_id=data.lane_connector_id,
                timestamp=data.timestamp,
            )
            for data in records
        ]

        self._record_signal_fallback(step, fallback_connectors)
        return transformed

    def _traffic_light_status(self, step, agents=()):
        """Build route-extension input, including a truly absent-signal fallback.

        nuPlan's stock route planner refuses every traffic-light connector that is absent from
        its GREEN list.  Some long clips exhaust their signal log and return no records at all;
        at that point the active signalized route frontiers must be admitted as unprotected
        movements and left to the intersection manager's FIFO/corridor conflict checks.
        """
        status = defaultdict(list)
        records = list(self._scenario.get_traffic_light_status_at_iteration(step))
        for data in records:
            status[data.status].append(str(data.lane_connector_id))
        if self._unknown_signal_fallback_enabled and not records:
            fallback_connectors = {
                str(edge.id)
                for agent in agents
                if agent.is_active(step) and agent.has_valid_path()
                for edge in (getattr(agent.end_segment, "outgoing_edges", ()) or ())
                if edge.has_traffic_lights()
            }
            status[TrafficLightStatusType.GREEN].extend(sorted(fallback_connectors))
            self._record_signal_fallback(step, fallback_connectors)
        # plan_route indexes both RED and GREEN even when the source frame has no records.
        status[TrafficLightStatusType.RED]
        status[TrafficLightStatusType.GREEN]
        return status

    def _ensure_targeted_lead_manager(self, manager):
        """Replace the stock manager without changing its agents or occupancy state."""
        if isinstance(manager, TargetedLeadIDMAgentManager):
            return manager
        manager = TargetedLeadIDMAgentManager(
            manager.agents, manager.agent_occupancy, manager._map_api
        )
        self._obs._idm_agent_manager = manager
        return manager

    @property
    def intersection_manager_stats(self):
        """Scene-local counters for rollout logging/evaluation."""
        # Metrics are queried during GT warm-up. Do not let an observational read construct
        # the lazy IDM manager against the wrong source frame before the handoff boundary.
        manager = self._obs._idm_agent_manager if self._idm_start_step is not None else None
        wait_stats = (
            manager.get_wait_cycle_stats()
            if isinstance(manager, TargetedLeadIDMAgentManager)
            else {}
        )
        if (self._intersection_manager is None and not wait_stats
                and not self._dead_end_queue_relief_enabled
                and not self._source_end_route_exhaustion_despawn_enabled
                and not self._vehicle_only_lead_lane_ids
                and not self._ego_lane_spawn_clearance_enabled
                and not self._spawn_sector_waiting_tokens
                and self._reactive_min_source_displacement_m <= 0
                and not self._idm_initial_speed_records):
            return None
        stats = self._intersection_manager.get_stats() if self._intersection_manager else {}
        stats.update(wait_stats)
        stats["spawn_crossing_deferrals"] = self._spawn_crossing_deferrals
        stats["spawn_ego_lane_clearance_deferrals"] = (
            self._spawn_ego_lane_clearance_deferrals
        )
        stats["spawn_sector_waiting_vehicles"] = len(
            self._spawn_sector_waiting_tokens
        )
        stats["initial_ego_lane_clearance_deferrals"] = (
            self._initial_ego_lane_clearance_deferrals
        )
        stats["handoff_prebuild_seconds"] = self._idm_handoff_prebuild_seconds
        stats["handoff_smooth_merge_vehicles"] = len(
            self._handoff_smooth_merge_tokens
        )
        stats["handoff_gt_fallback_vehicles"] = len(
            self._handoff_gt_fallback_tokens
        )
        stats["handoff_source_vehicles"] = len(
            self._handoff_source_vehicle_tokens
        )
        stats["handoff_built_idm_vehicles"] = len(self._handoff_built_idm_tokens)
        stats["handoff_routable_not_built"] = len(
            self._handoff_routable_not_built_tokens
        )
        stats["handoff_dropped_routable_vehicles"] = len(
            self._handoff_dropped_routable_tokens
        )
        route_decisions = [
            decision for decision in getattr(self, "_idm_route_decisions", ())
            if decision[0] in self._idm_initial_speed_records
        ]
        stats["route_intent_tokens"] = [decision[0] for decision in route_decisions]
        stats["route_intent_source_rows"] = [decision[1] for decision in route_decisions]
        stats["route_intent_from_edges"] = [decision[2] for decision in route_decisions]
        stats["route_intent_stock_edges"] = [decision[3] for decision in route_decisions]
        stats["route_intent_chosen_edges"] = [decision[4] for decision in route_decisions]
        stats["route_intent_match_m"] = [decision[5] for decision in route_decisions]
        stats["route_intent_margin_m"] = [decision[6] for decision in route_decisions]
        stats["route_intent_changed_branches"] = sum(
            decision[3] != decision[4] for decision in route_decisions
        )
        initial_speed_records = [
            self._idm_initial_speed_records[token]
            for token in sorted(self._idm_initial_speed_records)
        ]
        stats["initial_speed_tokens"] = [r["token"] for r in initial_speed_records]
        stats["initial_speed_source_rows"] = [r["source_row"] for r in initial_speed_records]
        stats["initial_speed_raw_mps"] = [r["raw_mps"] for r in initial_speed_records]
        stats["initial_speed_derived_mps"] = [r["derived_mps"] for r in initial_speed_records]
        stats["initial_speed_applied_mps"] = [r["applied_mps"] for r in initial_speed_records]
        stats["initial_speed_source_kind"] = [r["source_kind"] for r in initial_speed_records]
        stats["initial_speed_clamp_reason"] = [r["clamp_reason"] for r in initial_speed_records]
        applied_speeds = np.asarray(stats["initial_speed_applied_mps"], dtype=np.float64)
        stats["initial_speed_seeded_vehicles"] = len(initial_speed_records)
        stats["initial_speed_zero_fallbacks"] = sum(
            r["source_kind"] == "zero_fallback" for r in initial_speed_records
        )
        stats["initial_speed_applied_p50_mps"] = float(
            np.nanpercentile(applied_speeds, 50) if len(applied_speeds) else 0.0
        )
        stats["initial_speed_applied_p95_mps"] = float(
            np.nanpercentile(applied_speeds, 95) if len(applied_speeds) else 0.0
        )
        stats["initial_speed_applied_max_mps"] = float(
            np.nanmax(applied_speeds) if len(applied_speeds) else 0.0
        )
        for key, value in self._handoff_snap_stats.items():
            stats[f"handoff_{key}"] = value
        stats["unknown_signal_fallback_events"] = self._unknown_signal_fallback_events
        stats["unknown_signal_fallback_intersection_ticks"] = (
            self._unknown_signal_fallback_intersection_ticks
        )
        stats["unknown_signal_fallback_connector_ticks"] = (
            self._unknown_signal_fallback_connector_ticks
        )
        stats["source_end_route_extensions"] = self._source_end_route_extensions
        # Preserve the old NPZ key for existing analysis; it now includes vehicles
        # whose source track is still valid at a true IDM graph dead-end.
        stats["source_end_route_exhaustion_despawns"] = (
            self._source_end_route_exhaustion_despawns
        )
        stats["dead_end_route_despawns"] = self._source_end_route_exhaustion_despawns
        stats["dead_end_queue_retirements"] = self._dead_end_queue_retirements
        stats["parked_vehicle_fallbacks"] = self._parked_vehicle_fallbacks
        stats["source_motion_open_loop_vehicles"] = len(
            self._source_motion_open_loop_tokens
        )
        stats["source_motion_spawned_vehicles"] = len(
            self._source_motion_spawned_tokens
        )
        stats["source_motion_spawn_deferrals"] = (
            self._source_motion_spawn_deferrals
        )
        stats["source_ended_active_stall_retirements"] = (
            self._source_ended_active_stall_retirements
        )
        stats["wait_cycle_agent_seconds"] = round(
            float(stats.get("wait_cycle_agent_ticks", 0)) * self._dt, 3
        )
        stats["intersection_manager_enabled"] = self._intersection_manager is not None
        return stats

    def _configure_emergency_braking(self, agents):
        """Keep stock IDM anticipation while allowing a separate hard-braking ceiling."""
        emergency_decel = self._idm_params["emergency_decel_max"]
        for agent in agents:
            if not isinstance(agent._policy, _IDMPolicyWithEmergencyBrake):
                agent._policy = _IDMPolicyWithEmergencyBrake(
                    *agent._policy.idm_params,
                    emergency_decel_max=emergency_decel,
                )

    @staticmethod
    def _has_safe_stopping_gap(
        bumper_gap: float,
        follower_speed: float,
        leader_speed: float,
        emergency_decel: float,
        solve_dt: float,
        minimum_gap: float,
    ) -> bool:
        """Whether one solve tick plus differential 0.5 g braking fits in the gap."""
        closing_speed = max(0.0, follower_speed - leader_speed)
        differential_stop = max(
            0.0,
            (follower_speed ** 2 - leader_speed ** 2) / (2.0 * emergency_decel),
        )
        required_gap = minimum_gap + closing_speed * solve_dt + differential_stop
        return bumper_gap >= required_gap

    def _unsafe_longitudinal_neighbors(
        self, candidate, ego_state, mgr, open_loop_vehicles=()
    ):
        """Find same-direction actors for which late insertion is not braking-feasible.

        Stock nuPlan never inserts a new reactive vehicle midway through a scenario. Its
        projected-footprint check is useful geometric screening, but it is not a stopping-
        distance guarantee and can be shortened at the end of a candidate's current map rail.
        This guard therefore belongs specifically to our long-window admission extension.
        """
        candidate_pose = candidate.to_se2()
        candidate_xy = np.array([candidate_pose.x, candidate_pose.y], dtype=np.float64)
        forward = np.array(
            [np.cos(candidate_pose.heading), np.sin(candidate_pose.heading)], dtype=np.float64
        )
        lateral = np.array([-forward[1], forward[0]], dtype=np.float64)
        candidate_speed = float(candidate.velocity)

        neighbors = [
            (
                str(token),
                other.to_se2(),
                float(other.velocity),
                float(other.length),
                float(other.width),
            )
            for token, other in mgr.agents.items()
        ]
        neighbors.extend(
            (
                str(track.track_token),
                track.center,
                self._track_speed(track),
                float(track.box.length),
                float(track.box.width),
            )
            for track in open_loop_vehicles
            if getattr(track, "track_token", None) is not None
        )
        ego_velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d
        ego_parameters = ego_state.car_footprint.vehicle_parameters
        neighbors.append(
            (
                "ego",
                ego_state.center,
                float(np.hypot(ego_velocity.x, ego_velocity.y)),
                float(ego_parameters.length),
                float(ego_parameters.width),
            )
        )

        unsafe = []
        for token, pose, speed, length, width in neighbors:
            heading_error = abs(
                (float(pose.heading) - candidate_pose.heading + np.pi) % (2.0 * np.pi) - np.pi
            )
            if heading_error > np.deg2rad(30.0):
                continue
            delta = np.array([pose.x, pose.y], dtype=np.float64) - candidate_xy
            # Half-widths plus 0.5 m catches a merge without treating an adjacent lane as a
            # longitudinal pair. Geometric projected-footprint admission still handles crossing.
            if abs(float(delta @ lateral)) > (candidate.width + width) / 2.0 + 0.5:
                continue
            signed_distance = float(delta @ forward)
            if abs(signed_distance) < 1e-6:
                unsafe.append(token)
                continue
            bumper_gap = abs(signed_distance) - (candidate.length + length) / 2.0
            if signed_distance > 0.0:
                follower_speed, leader_speed = candidate_speed, speed
            else:
                follower_speed, leader_speed = speed, candidate_speed
            if not self._has_safe_stopping_gap(
                bumper_gap,
                follower_speed,
                leader_speed,
                self._idm_params["emergency_decel_max"],
                self._solve_dt,
                self._idm_params["min_gap_to_lead_agent"],
            ):
                unsafe.append(token)
        return unsafe

    def _ego_lane_spawn_clearance(self, candidate, ego_state):
        """Return a violated ego-centred same-lane clearance, if any.

        The check uses current physical state only.  Similar heading plus overlap of the two
        footprint-width corridors is the local same-lane test; this deliberately excludes an
        adjacent lane without depending on a future ego route or planner output.  Distances are
        bumper-to-bumper rather than centre-to-centre.
        """
        if not self._ego_lane_spawn_clearance_enabled:
            return None

        candidate_pose = candidate.to_se2()
        ego_pose = ego_state.center
        heading_error = abs(
            (float(candidate_pose.heading) - float(ego_pose.heading) + np.pi)
            % (2.0 * np.pi)
            - np.pi
        )
        if heading_error > self._ego_lane_spawn_heading_tolerance_rad:
            return None

        forward = np.array(
            [np.cos(float(ego_pose.heading)), np.sin(float(ego_pose.heading))],
            dtype=np.float64,
        )
        lateral = np.array([-forward[1], forward[0]], dtype=np.float64)
        delta = (
            np.array([candidate_pose.x, candidate_pose.y], dtype=np.float64)
            - np.array([ego_pose.x, ego_pose.y], dtype=np.float64)
        )
        lateral_distance = abs(float(delta @ lateral))
        ego_parameters = ego_state.car_footprint.vehicle_parameters
        same_lane_half_width = (
            (float(ego_parameters.width) + float(candidate.width)) / 2.0
            + self._ego_lane_spawn_lateral_margin_m
        )
        if lateral_distance > same_lane_half_width:
            return None

        signed_center_distance = float(delta @ forward)
        bumper_gap = (
            abs(signed_center_distance)
            - (float(ego_parameters.length) + float(candidate.length)) / 2.0
        )
        if signed_center_distance >= 0.0:
            ego_velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d
            ego_speed = float(np.hypot(ego_velocity.x, ego_velocity.y))
            stopping_distance = (
                ego_speed * self._ego_lane_spawn_reaction_time_s
                + ego_speed ** 2
                / (2.0 * self._idm_params["emergency_decel_max"])
            )
            required_gap = max(
                self._ego_lane_spawn_front_min_m,
                stopping_distance
                * self._ego_lane_spawn_stopping_distance_multiplier,
            )
            direction = "ahead"
        else:
            required_gap = self._ego_lane_spawn_rear_m
            direction = "behind"

        if bumper_gap >= required_gap:
            return None
        return direction, bumper_gap, required_gap

    @staticmethod
    def _track_speed(track):
        velocity = getattr(track, "velocity", None)
        return float(np.hypot(velocity.x, velocity.y)) if velocity is not None else 0.0

    @staticmethod
    def _valid_source_pose(positions, valid, row):
        """Whether ``row`` holds a finite, source-valid XY pose."""
        return (
            positions.ndim == 2
            and positions.shape[1] >= 2
            and 0 <= row < len(valid)
            and row < len(positions)
            and bool(valid[row])
            and bool(np.isfinite(positions[row, :2]).all())
        )

    def _attach_idm_route_intent(self, token, agent, simulation_step):
        """Attach the actor's fixed source path once, using its mapped source row."""
        if hasattr(agent, "_odyssey_route_intent"):
            return
        # Lightweight test batches may not carry a complete scene/converter.
        if not hasattr(self, "scene") or not hasattr(self, "converter"):
            return
        token = str(token)
        row = int(self.converter.object_source_row(token, int(simulation_step)))
        state = self.scene.get("object_track", {}).get(token, {}).get("state", {})
        agent._odyssey_route_intent = build_source_route_intent(state, self._origin, row)
        agent._odyssey_route_token = token
        agent._odyssey_route_source_row = row
        agent._odyssey_route_decisions = getattr(self, "_idm_route_decisions", None)

    def _recover_idm_initial_speed(self, token, agent, simulation_step):
        """Recover source forward motion at this actor's real source row.

        A finite, non-zero raw velocity has priority.  If it is zero or absent, use the
        previous valid pose; only the first valid pose may use the next valid pose.  Motion is
        projected on the installed path's start heading (the GT heading after smooth merge).
        """
        row = int(self.converter.object_source_row(str(token), int(simulation_step)))
        record = {
            "token": str(token), "source_row": row, "raw_mps": np.nan,
            "derived_mps": np.nan, "applied_mps": 0.0,
            "source_kind": "zero_fallback", "clamp_reason": "invalid_source",
        }
        source = self.scene.get("object_track", {}).get(str(token), {})
        state = source.get("state", {})
        positions = np.asarray(state.get("position", []), dtype=np.float64)
        valid = np.asarray(state.get("valid", []), dtype=bool).reshape(-1)
        if not self._valid_source_pose(positions, valid, row):
            return record

        pose = agent.to_se2()
        forward = np.asarray([np.cos(float(pose.heading)), np.sin(float(pose.heading))])
        velocity = np.asarray(state.get("velocity", []), dtype=np.float64)
        raw_vector = None
        if velocity.ndim == 2 and velocity.shape[1] >= 2 and 0 <= row < len(velocity):
            candidate = velocity[row, :2]
            if np.isfinite(candidate).all():
                record["raw_mps"] = float(np.dot(candidate, forward))
                if float(np.linalg.norm(candidate)) > 1e-6:
                    raw_vector = candidate

        def accept(vector, kind, field):
            magnitude = float(np.linalg.norm(vector))
            longitudinal = float(np.dot(vector, forward))
            record[field] = longitudinal
            record["source_kind"] = kind
            if magnitude > self._initial_speed_sanity_max_mps:
                record["clamp_reason"] = "sanity_limit"
                return None
            if longitudinal <= 0:
                record["clamp_reason"] = "nonforward_motion"
                return None
            record["clamp_reason"] = "none"
            return longitudinal

        recovered = None
        if raw_vector is not None:
            recovered = accept(raw_vector, "raw_velocity", "raw_mps")
        else:
            previous = np.flatnonzero(valid[:row])
            if len(previous):
                other_row = int(previous[-1])
                vector = (positions[row, :2] - positions[other_row, :2]) / (
                    (row - other_row) * self._dt
                )
                recovered = accept(vector, "backward_pose_difference", "derived_mps")
            else:
                following = np.flatnonzero(valid[row + 1:])
                if len(following):
                    other_row = int(row + 1 + following[0])
                    vector = (positions[other_row, :2] - positions[row, :2]) / (
                        (other_row - row) * self._dt
                    )
                    recovered = accept(vector, "forward_pose_difference", "derived_mps")

        applied = 0.0 if recovered is None else float(recovered)
        record["applied_mps"] = applied
        agent._state.velocity = applied
        agent._requires_state_update = True
        return record

    def _seed_idm_initial_speed(self, token, agent, simulation_step):
        """Apply and retain exactly one audit record for an actually-admitted IDM agent."""
        token = str(token)
        if token not in self._idm_initial_speed_records:
            self._idm_initial_speed_records[token] = self._recover_idm_initial_speed(
                token, agent, simulation_step
            )
        return self._idm_initial_speed_records[token]

    @staticmethod
    def _build_source_vehicle_motion_profiles(scene):
        """Summarize each vehicle over every valid source sample in the scene clip."""
        profiles = {}
        for token, source_track in scene.get("object_track", {}).items():
            if not source_track.get("metadata", {}).get("simulation_enabled", True):
                continue
            if str(source_track.get("type", "")).upper() not in {"VEHICLE", "VEH"}:
                continue
            state = source_track.get("state", {})
            positions = np.asarray(state.get("position", []), dtype=np.float64)
            valid = np.asarray(state.get("valid", []), dtype=bool).reshape(-1)
            if (
                positions.ndim != 2
                or positions.shape[1] < 2
                or len(valid) != len(positions)
            ):
                continue
            points = positions[valid, :2]
            points = points[np.isfinite(points).all(axis=1)]
            if not len(points):
                continue
            # Maximum distance from the first valid observation catches a vehicle which moves
            # and later returns close to its starting point, unlike endpoint displacement.
            max_displacement = float(np.linalg.norm(points - points[0], axis=1).max())
            profiles[str(token)] = (int(len(points)), max_displacement)
        return profiles

    @staticmethod
    def _build_source_vehicle_path_lengths(scene):
        """Cumulative XY distance across valid poses of each IDM-candidate vehicle."""
        lengths = {}
        for token, track in scene.get("object_track", {}).items():
            metadata = track.get("metadata", {})
            if (not metadata.get("simulation_enabled", True)
                    or metadata.get("control_mode") in {"static", "replay"}
                    or str(track.get("type", "")).upper() not in {"VEHICLE", "VEH"}):
                continue
            state = track.get("state", {})
            positions = np.asarray(state.get("position", []), dtype=np.float64)
            valid = np.asarray(state.get("valid", []), dtype=bool).reshape(-1)
            if (positions.ndim != 2 or positions.shape[1] < 2
                    or len(positions) != len(valid) or not valid.any()):
                continue
            points = positions[valid, :2]
            if not np.isfinite(points).all():
                continue
            lengths[str(token)] = float(
                np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
            )
        return lengths

    def _predrop_short_source_paths(self):
        """Reject low-motion IDM candidates before they enter any scored snapshot."""
        if not self._static_pin_and_predrop or self._source_path_predrop_max_m <= 0:
            return
        reason = f"source_path_le_{self._source_path_predrop_max_m:g}m"
        for token, distance in sorted(self._source_vehicle_path_lengths.items()):
            if token not in self._static_vehicle_tokens and distance <= self._source_path_predrop_max_m:
                self._predrop_vehicle(token, reason, 0)

    @staticmethod
    def _build_source_vehicle_final_valid_steps(scene):
        """Return the true final source row for each simulation-enabled vehicle.

        Exact-row absence may be a temporary sensor gap. Route-exhaustion despawn is a terminal
        lifecycle action, so it must wait until the progress clock passes the track's final valid
        row instead of treating an intermittent invalid sample as source completion.
        """
        final_steps = {}
        for token, source_track in scene.get("object_track", {}).items():
            if not source_track.get("metadata", {}).get("simulation_enabled", True):
                continue
            if str(source_track.get("type", "")).upper() not in {"VEHICLE", "VEH"}:
                continue
            valid = np.asarray(
                source_track.get("state", {}).get("valid", []), dtype=bool
            ).reshape(-1)
            indices = np.flatnonzero(valid)
            if len(indices):
                final_steps[str(token)] = int(indices[-1])
        return final_steps

    @staticmethod
    def _build_unstable_short_track_tokens(
        scene, max_valid_samples=2, max_displacement_m=1.0, heading_jump_deg=90.0
    ):
        """Find too-short stationary tracks whose consecutive headings contradict each other."""
        unstable = set()
        max_valid_samples = max(2, int(max_valid_samples))
        heading_jump_rad = np.deg2rad(float(heading_jump_deg))
        # ScenarioManager inserts interpolated subframes before policy construction.  Diagnose
        # source quality on the original samples; otherwise one corrupt two-frame detection
        # becomes six apparently smooth observations and evades this gate.
        cadence = scene.get("cadence") if hasattr(scene, "get") else None
        source_stride = max(1, int(getattr(cadence, "upsample_n", 1)))
        for token, source_track in scene.get("object_track", {}).items():
            if not source_track.get("metadata", {}).get("simulation_enabled", True):
                continue
            if str(source_track.get("type", "")).upper() not in {"VEHICLE", "VEH"}:
                continue
            state = source_track.get("state", {})
            positions = np.asarray(state.get("position", []), dtype=np.float64)
            headings = np.asarray(state.get("heading", []), dtype=np.float64).reshape(-1)
            valid = np.asarray(state.get("valid", []), dtype=bool).reshape(-1)
            if (
                positions.ndim != 2
                or positions.shape[1] < 2
                or len(valid) != len(positions)
                or len(headings) != len(positions)
            ):
                continue
            source_samples = np.arange(len(valid)) % source_stride == 0
            indices = np.flatnonzero(valid & np.isfinite(headings) & source_samples)
            if len(indices) < 2 or len(indices) > max_valid_samples:
                continue
            points = positions[indices, :2]
            if (
                not np.isfinite(points).all()
                or float(np.linalg.norm(points - points[0], axis=1).max())
                > float(max_displacement_m)
            ):
                continue
            consecutive = np.diff(indices) == source_stride
            if not np.any(consecutive):
                continue
            delta = np.diff(headings[indices])
            heading_jumps = np.abs(np.arctan2(np.sin(delta), np.cos(delta)))
            if np.any(heading_jumps[consecutive] >= heading_jump_rad):
                unstable.add(str(token))
        return unstable

    def _parked_vehicle_fallback_reason(self, track, agent):
        """Return why a stationary-over-its-full-track vehicle stays at logged pose."""
        if (
            not self._parked_vehicle_fallback_enabled
            or self._track_speed(track) > self._parked_vehicle_speed_threshold_mps
        ):
            return None
        token = str(getattr(track, "track_token", ""))
        valid_samples, max_displacement = self._source_vehicle_motion_profiles.get(
            token, (0, np.inf)
        )
        if (
            valid_samples < self._parked_vehicle_min_valid_samples
            or max_displacement > self._parked_vehicle_max_track_displacement_m
        ):
            return None
        source_xy = np.asarray(track.center.point.array, dtype=np.float64)[:2]
        snapped_pose = agent.to_se2()
        snapped_xy = np.asarray([snapped_pose.x, snapped_pose.y], dtype=np.float64)
        snap_distance = float(np.linalg.norm(source_xy - snapped_xy))
        if snap_distance > self._parked_vehicle_snap_distance_m:
            return "off_rail_snap"
        return None

    def _source_motion_open_loop_reason(self, token):
        """Return an audit reason when a source track is below the reactive-motion gate."""
        threshold = self._reactive_min_source_displacement_m
        if threshold <= 0:
            return None
        valid_samples, max_displacement = self._source_vehicle_motion_profiles.get(
            str(token), (0, 0.0)
        )
        if valid_samples > 0 and max_displacement >= threshold:
            return None
        return (
            f"source_motion_{max_displacement:.2f}m_below_"
            f"{threshold:.2f}m"
        )

    def _keep_source_motion_vehicle_open_loop(self, token, reason, step):
        token = str(token)
        if token in self._source_motion_open_loop_tokens:
            return
        self._source_motion_open_loop_tokens.add(token)
        logger.info(
            "[NUPLAN-IDM] queued low-motion vehicle %s for collision-safe GT "
            "open-loop spawn at tick=%d: %s",
            token,
            step,
            reason,
        )

    def _unsafe_source_motion_spawn_neighbors(self, candidate, manager):
        """Find reactive vehicles that cannot brake safely around a GT-replay spawn.

        A free footprint is not enough for a low-motion vehicle which first becomes visible
        only a few metres in front of a moving IDM vehicle.  The reactive vehicle needs the
        same one-tick-plus-emergency-braking clearance used for late IDM admission.  This
        remains a current-state check: it uses neither the candidate's future source poses nor
        a nuPlan log trajectory.
        """
        candidate_pose = candidate.center
        candidate_xy = np.asarray(
            [candidate_pose.x, candidate_pose.y], dtype=np.float64
        )
        forward = np.asarray(
            [np.cos(candidate_pose.heading), np.sin(candidate_pose.heading)],
            dtype=np.float64,
        )
        lateral = np.asarray([-forward[1], forward[0]], dtype=np.float64)
        candidate_speed = self._track_speed(candidate)
        candidate_length = float(candidate.box.length)
        candidate_width = float(candidate.box.width)

        unsafe = []
        for token, other in manager.agents.items():
            other_pose = other.to_se2()
            heading_error = abs(
                (float(other_pose.heading) - float(candidate_pose.heading) + np.pi)
                % (2.0 * np.pi)
                - np.pi
            )
            if heading_error > np.deg2rad(30.0):
                continue
            delta = (
                np.asarray([other_pose.x, other_pose.y], dtype=np.float64)
                - candidate_xy
            )
            if abs(float(delta @ lateral)) > (
                candidate_width + float(other.width)
            ) / 2.0 + 0.5:
                continue
            signed_distance = float(delta @ forward)
            if abs(signed_distance) < 1e-6:
                unsafe.append(str(token))
                continue
            bumper_gap = (
                abs(signed_distance)
                - (candidate_length + float(other.length)) / 2.0
            )
            if signed_distance > 0.0:
                follower_speed = candidate_speed
                leader_speed = float(other.velocity)
            else:
                follower_speed = float(other.velocity)
                leader_speed = candidate_speed
            if not self._has_safe_stopping_gap(
                bumper_gap,
                follower_speed,
                leader_speed,
                self._idm_params["emergency_decel_max"],
                self._solve_dt,
                self._idm_params["min_gap_to_lead_agent"],
            ):
                unsafe.append(str(token))
        return unsafe

    def _admit_source_motion_open_loop_vehicles(
        self, gt_tracks, manager, ego_state, step
    ):
        """Publish queued GT vehicles only when their current physical pose is free.

        This is an admission gate, not a trajectory oracle: it reads only the current source
        row and current simulated boxes. A refused object remains absent and is retried on the
        next 0.1 s solve. If its valid source interval ends first, it simply never appears.
        """
        pending = (
            self._source_motion_open_loop_tokens
            - self._source_motion_spawned_tokens
        )
        if not pending:
            return set()

        objects = list(
            getattr(gt_tracks.tracked_objects, "tracked_objects", []) or []
        )
        by_token = {
            str(obj.track_token): obj
            for obj in objects
            if getattr(obj, "track_token", None) is not None
        }
        blockers = [("ego", ego_state.car_footprint.geometry)]
        blockers.extend(
            (str(token), agent.polygon)
            for token, agent in manager.agents.items()
        )
        # Pedestrians, bicycles and static objects are always source-replayed. Previously
        # accepted low-motion vehicles are source-replayed too, but are not persistent in
        # IDMAgentManager's occupancy outside propagate_agents(), so add both explicitly.
        blockers.extend(
            (str(obj.track_token), obj.box.geometry)
            for obj in objects
            if (
                getattr(obj, "track_token", None) is not None
                and (
                    getattr(obj, "tracked_object_type", None)
                    in self._obs._open_loop_detections_types
                    or str(obj.track_token) in self._source_motion_spawned_tokens
                    or str(obj.track_token) in self._static_vehicle_tokens
                )
            )
        )

        admitted = set()
        for token in sorted(pending):
            obj = by_token.get(token)
            if obj is None:
                continue
            geometry = obj.box.geometry
            if geometry.is_empty or not geometry.is_valid:
                self._source_motion_spawn_deferrals += 1
                continue
            overlaps = [
                other for other, other_geometry in blockers
                if other != token
                and geometry.intersection(other_geometry).area > 0.01
            ]
            if overlaps:
                self._source_motion_spawn_deferrals += 1
                logger.debug(
                    "[NUPLAN-IDM] deferring low-motion GT vehicle %s at step %s: "
                    "current footprint overlap with %s",
                    token,
                    step,
                    overlaps[:5],
                )
                continue
            unsafe_neighbors = (
                self._unsafe_source_motion_spawn_neighbors(obj, manager)
                if getattr(
                    self, "_source_motion_spawn_braking_gap_enabled", True
                )
                else []
            )
            if unsafe_neighbors:
                self._source_motion_spawn_deferrals += 1
                logger.debug(
                    "[NUPLAN-IDM] deferring low-motion GT vehicle %s at step %s: "
                    "unsafe braking gap to %s",
                    token,
                    step,
                    unsafe_neighbors[:5],
                )
                continue
            self._obs.extra_open_loop_vehicle_tokens.add(token)
            self._source_motion_spawned_tokens.add(token)
            admitted.add(token)
            blockers.append((token, geometry))
            logger.info(
                "[NUPLAN-IDM] admitted low-motion vehicle %s as GT open-loop at "
                "tick=%d after collision-safe spawn check",
                token,
                step,
            )
        return admitted

    def _keep_vehicle_open_loop(self, token, reason, step):
        token = str(token)
        if token in self._obs.extra_open_loop_vehicle_tokens:
            return
        self._obs.extra_open_loop_vehicle_tokens.add(token)
        self._parked_vehicle_fallbacks += 1
        logger.info(
            "[NUPLAN-IDM] keeping stationary vehicle %s open-loop at tick=%d: %s",
            token,
            step,
            reason,
        )

    def _demote_initial_parked_vehicles(self, manager, gt_tracks, step):
        """Remove parked/off-rail vehicles from the prebuilt t=0 IDM population."""
        if (
            step != self._idm_start_step
            or (
                not self._parked_vehicle_fallback_enabled
                and self._reactive_min_source_displacement_m <= 0
            )
        ):
            return set()
        source_by_token = {
            str(track.track_token): track
            for track in gt_tracks.tracked_objects.get_tracked_objects_of_type(
                TrackedObjectType.VEHICLE
            )
            if track.track_token is not None
        }
        demoted = {}
        for token, agent in list(manager.agents.items()):
            track = source_by_token.get(str(token))
            reason = self._source_motion_open_loop_reason(token)
            if reason is None and track is not None:
                reason = self._parked_vehicle_fallback_reason(track, agent)
            if reason is not None:
                demoted[str(token)] = reason
        for token, reason in demoted.items():
            manager.agents.pop(token, None)
            if manager.agent_occupancy.contains(token):
                manager.agent_occupancy.remove([token])
            if reason.startswith("source_motion_"):
                self._keep_source_motion_vehicle_open_loop(token, reason, step)
            else:
                self._keep_vehicle_open_loop(token, reason, step)
        return set(demoted)

    def _reject_invalid_initial_centerline_snaps(self, manager, gt_tracks, step):
        """Apply the normal 3 m / 45 degree association gate to snap-warmup vehicles.

        nuPlan's stock builder chooses the closest-heading rail but imposes no maximum lateral
        or angular error.  That is acceptable for its short benchmark scenes, but a badly
        associated actor can spin for a long time in our extended rollout.  Collision-refused
        candidates are still retried by the ordinary mid-spawn path; only a geometrically
        implausible rail association is permanently rejected here.
        """
        if (
            step != self._idm_start_step
            or self._initialization_mode != "centerline_snap_warmup"
        ):
            return set()
        source_by_token = {
            str(track.track_token): track
            for track in gt_tracks.tracked_objects.get_tracked_objects_of_type(
                TrackedObjectType.VEHICLE
            )
            if track.track_token is not None
        }
        rejected = {}
        for token, agent in list(manager.agents.items()):
            track = source_by_token.get(str(token))
            if track is None:
                continue
            reason = self._handoff_merge_fallback_reason(track, agent)
            if reason is not None:
                rejected[str(token)] = reason
        for token, reason in rejected.items():
            manager.agents.pop(token, None)
            if manager.agent_occupancy.contains(token):
                manager.agent_occupancy.remove([token])
            self._unroutable.add(token)
            logger.info(
                "[NUPLAN-IDM-SNAP-WARMUP] rejecting vehicle %s at tick=%d: %s",
                token, step, reason,
            )
        return set(rejected)

    def _unsafe_crossing_spawn_neighbors(self, candidate, mgr, prediction_cache=None):
        """Find current-state crossing trajectories that overlap at the same near-term time.

        This is deliberately a late-spawn guard, not a second planner.  It translates each
        actor's current physical box with its current scalar speed and heading, samples both
        actors at identical times, and never reads a source/GT future trajectory.
        """
        candidate_pose = candidate.to_se2()
        candidate_heading = float(candidate_pose.heading)
        candidate_velocity = np.array(
            [np.cos(candidate_heading), np.sin(candidate_heading)], dtype=np.float64
        ) * float(candidate.velocity)
        sample_count = max(
            1,
            int(np.ceil(
                self._spawn_crossing_horizon_s / self._spawn_crossing_sample_dt
            )),
        )
        sample_times = [
            min(
                self._spawn_crossing_horizon_s,
                sample * self._spawn_crossing_sample_dt,
            )
            for sample in range(sample_count + 1)
        ]
        candidate_boxes = [
            translate(
                candidate.polygon,
                xoff=float(candidate_velocity[0] * time_s),
                yoff=float(candidate_velocity[1] * time_s),
            )
            for time_s in sample_times
        ]
        # Several deferred vehicles can retry on the same simulation tick.  The live IDM fleet
        # does not propagate while this admission loop is running, so its constant-velocity
        # boxes are identical for every candidate.  Cache them for this one admission pass;
        # newly admitted agents are absent from the cache and are populated lazily below.
        prediction_cache = {} if prediction_cache is None else prediction_cache
        unsafe = []
        for token, other in mgr.agents.items():
            cached = prediction_cache.get(token)
            if cached is None:
                other_pose = other.to_se2()
                other_heading = float(other_pose.heading)
                other_velocity = np.array(
                    [np.cos(other_heading), np.sin(other_heading)], dtype=np.float64,
                ) * float(other.velocity)
                other_boxes = tuple(
                    translate(
                        other.polygon,
                        xoff=float(other_velocity[0] * time_s),
                        yoff=float(other_velocity[1] * time_s),
                    )
                    for time_s in sample_times
                )
                cached = (other_heading, other_boxes)
                prediction_cache[token] = cached
            other_heading, other_boxes = cached
            heading_error = abs(
                (other_heading - candidate_heading + np.pi)
                % (2.0 * np.pi)
                - np.pi
            )
            # Same-flow insertions already have the stronger braking-distance check.
            if heading_error <= np.deg2rad(30.0):
                continue
            for candidate_box, other_box in zip(candidate_boxes, other_boxes):
                if candidate_box.intersection(other_box).area > 0.01:
                    unsafe.append(str(token))
                    break
        return unsafe

    def _unsafe_converging_rail_spawn_neighbors(self, candidate, mgr, rail_cache=None):
        """Find live adjacent flows whose near rails merge with a late candidate.

        Constant-velocity boxes miss curved merges: two vehicles can be far apart when a
        candidate first appears and still be routed onto the same receiving lane.  This check
        runs only during late admission and compares at most the next 100 m of rail.  A vehicle
        already in the same lane is deliberately left to the braking-gap and stock IDM checks.
        """
        candidate_line = _path_to_go_linestring(candidate)
        if candidate_line is None or candidate_line.is_empty:
            return []
        candidate_pose = candidate.to_se2()
        candidate_heading = float(candidate_pose.heading)
        candidate_xy = np.array(
            [candidate_pose.x, candidate_pose.y], dtype=np.float64
        )
        candidate_lateral = np.array(
            [-np.sin(candidate_heading), np.cos(candidate_heading)], dtype=np.float64
        )
        candidate_rail = substring(
            candidate_line,
            0.0,
            min(float(candidate_line.length), self._spawn_rail_conflict_lookahead_m),
        ).buffer(float(candidate.width) / 2.0 + 0.25, cap_style=CAP_STYLE.flat)
        if candidate_rail.is_empty:
            return []

        # Buffering the next 100 m of a curved rail is substantially more expensive than the
        # intersection test itself.  All existing agents are stationary with respect to this
        # admission pass, so reuse each buffered rail across candidates on this tick.  A newly
        # admitted candidate is cached lazily when the next candidate examines it.
        rail_cache = {} if rail_cache is None else rail_cache
        unsafe = []
        for token, other in mgr.agents.items():
            cached = rail_cache.get(token)
            if cached is None:
                other_pose = other.to_se2()
                other_heading = float(other_pose.heading)
                other_xy = np.array(
                    [other_pose.x, other_pose.y], dtype=np.float64
                )
                other_line = _path_to_go_linestring(other)
                other_rail = None
                if other_line is not None and not other_line.is_empty:
                    other_rail = substring(
                        other_line,
                        0.0,
                        min(
                            float(other_line.length),
                            self._spawn_rail_conflict_lookahead_m,
                        ),
                    ).buffer(
                        float(other.width) / 2.0 + 0.25,
                        cap_style=CAP_STYLE.flat,
                    )
                cached = (other_heading, other_xy, other_rail)
                rail_cache[token] = cached
            other_heading, other_xy, other_rail = cached
            heading_error = abs(
                (other_heading - candidate_heading + np.pi)
                % (2.0 * np.pi)
                - np.pi
            )
            delta = other_xy - candidate_xy
            lateral_separation = abs(float(delta @ candidate_lateral))
            same_lane_now = (
                heading_error <= np.deg2rad(30.0)
                and lateral_separation
                <= (float(candidate.width) + float(other.width)) / 2.0 + 0.5
            )
            if same_lane_now:
                continue
            if other_rail is None or other_rail.is_empty:
                continue
            if candidate_rail.intersects(other_rail):
                unsafe.append(str(token))
        return unsafe

    def _admit_new_agents(self, step, gt_tracks, ego_state):
        """Let vehicles that appear AFTER t=0 become IDM agents.

        build_idm_agents_on_map_rails only ever reads scenario.initial_tracked_objects, i.e. the
        t=0 frame. That is fine for nuPlan's 20 s scenarios; over an 80 s window cars enter
        throughout and would never be simulated. IDMAgent already takes a start_iteration, so a
        late admission is what it is built for.

        Construction goes through nuPlan's OWN builder rather than assembling IDMAgent here.
        propagate_agents asserts that each agent's path buffer contains its own box
        (idm_agent_manager.py:79), so the exact snapped geometry returned by the builder is
        validated before either of the manager's two coupled dictionaries is changed.

        The builder walks every vehicle, so it is called only when an unknown token is both
        visible and inside the IDM radius. The set/radius checks below are cheap; the builder is
        not. Filtering here also prevents a far-away actor from being admitted and immediately
        deleted by IDMAgentManager._filter_agents_out_of_range on every step.
        """
        mgr = self._obs._get_idm_agent_manager()
        known = set(mgr.agents)
        vehicles = gt_tracks.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
        ego_xy = np.array([ego_state.center.x, ego_state.center.y], dtype=np.float64)
        radius_sq = self._idm_params["radius"] ** 2
        vehicles_by_token = {
            a.track_token: a for a in vehicles
            if a.track_token is not None
            and np.sum((a.center.point.array - ego_xy) ** 2) <= radius_sq
        }
        open_loop_vehicles = []
        if self._late_spawn_open_loop_braking_gap_enabled:
            open_loop_vehicle_tokens = (
                set(self._obs.extra_open_loop_vehicle_tokens)
                | set(self._source_motion_spawned_tokens)
            )
            open_loop_vehicles = [
                vehicle
                for token, vehicle in vehicles_by_token.items()
                if token in open_loop_vehicle_tokens
            ]
        tokens_now = set(vehicles_by_token)
        new_tokens = (
            tokens_now
            - known
            - self._unroutable
            - self._retired
            - self._obs.extra_open_loop_vehicle_tokens
            - self._source_motion_open_loop_tokens
        )
        sector_waiting = {
            str(token) for token in new_tokens
            if not self._spawn_eligible(str(token))
        }
        if sector_waiting:
            self._spawn_sector_waiting_tokens.update(sector_waiting)
            new_tokens -= sector_waiting
        unstable_tokens = new_tokens & self._unstable_short_track_tokens
        if unstable_tokens:
            self._obs.extra_open_loop_vehicle_tokens.update(unstable_tokens)
            new_tokens -= unstable_tokens
            logger.info(
                "[NUPLAN-IDM] keeping temporally unstable short track(s) in source mode "
                "at tick=%d: %s",
                step,
                ",".join(sorted(unstable_tokens)),
            )
        low_motion_tokens = {
            token for token in new_tokens
            if self._source_motion_open_loop_reason(token) is not None
        }
        for token in sorted(low_motion_tokens):
            self._keep_source_motion_vehicle_open_loop(
                token,
                self._source_motion_open_loop_reason(token),
                step,
            )
        new_tokens -= low_motion_tokens
        if not new_tokens:
            return set()

        # A view of the scene whose "initial" frame is THIS step, so the builder does its normal
        # t=0 work against the current poses. Pass only the genuinely new candidates: giving it
        # the whole frame rebuilds every visible vehicle whenever one collision-deferred token is
        # retried (often every 0.1 s). Existing IDM actors are checked against their more accurate
        # live occupancy below, and candidates in this batch still collision-check one another.
        candidate_tracks = DetectionsTracks(TrackedObjects(
            [vehicles_by_token[tok] for tok in sorted(new_tokens)]
        ))

        class _AtStep:
            map_api = self._scenario.map_api
            initial_tracked_objects = candidate_tracks

            @staticmethod
            def get_ego_state_at_iteration(_i):
                return ego_state

        p = self._idm_params
        built, _occ = build_idm_agents_on_map_rails(
            p["target_velocity"], p["min_gap_to_lead_agent"], p["headway_time"],
            p["accel_max"], p["decel_max"], p["minimum_path_length"], _AtStep(), [])
        for token, agent in built.items():
            _install_idm_connector_filter(agent, self._excluded_idm_connector_ids)
            self._attach_idm_route_intent(token, agent, step)

        admitted = set()
        crossing_prediction_cache = {}
        rail_cache = {}
        for tok in sorted(new_tokens):
            a = built.get(tok)
            if a is None:
                # The builder declines both genuinely unroutable agents and agents which happen
                # to collide at this frame. Cache only the former; the latter may safely enter on
                # a later frame. The extra query occurs only on a refusal.
                route, _ = get_starting_segment(vehicles_by_token[tok], self._scenario.map_api)
                if route is None:
                    self._unroutable.add(tok)
                    if self._static_pin_and_predrop:
                        self._predrop_vehicle(tok, "unroutable", step)
                else:
                    self._deferred_tokens.add(tok)
                    logger.debug(
                        "[NUPLAN-IDM] deferring late agent %s at step %s: builder collision",
                        tok, step)
                continue

            self._configure_emergency_braking([a])

            fallback_reason = self._parked_vehicle_fallback_reason(
                vehicles_by_token[tok], a
            )
            if fallback_reason is None:
                fallback_reason = self._handoff_merge_fallback_reason(
                    vehicles_by_token[tok], a
                )
            if fallback_reason is not None:
                if self._static_pin_and_predrop:
                    # Same rule as the handoff: a failed gate means the vehicle never appears,
                    # neither as GT replay nor as a later retry.
                    self._predrop_vehicle(tok, fallback_reason, step)
                elif self._vehicle_gt_fallback_enabled:
                    self._keep_vehicle_open_loop(tok, fallback_reason, step)
                else:
                    self._deferred_tokens.add(tok)
                    logger.debug(
                        "[NUPLAN-IDM] deferring late agent %s at step %s: %s",
                        tok, step, fallback_reason,
                    )
                continue

            # This token was not visible during the shared GT warm-up, so there is no displayed
            # pose to preserve across a transition. Keep the builder's lane-centre pose and
            # heading (a hard snap before first publication). A synthetic source-to-rail merge
            # can create a backward tangent on curved/doubled-back routes and make a newly
            # spawned vehicle spin. The 3 m / 45 degree association gate above plus the overlap,
            # braking-distance, crossing, and ego-clearance gates below decide whether the hard
            # snap is safe; a refused token remains deferred and is retried later.

            # The builder does not reject zero-sized/invalid boxes. A zero width makes the flat
            # path buffer empty, which is precisely the assertion seen for late-spawn actors.
            # Keep this guard even though the converter now reads dimensions at `step`: malformed
            # source data must not poison the whole batch.
            if a.width <= 0.0 or a.length <= 0.0 or not _occ.contains(tok):
                logger.warning(
                    "[NUPLAN-IDM] refusing malformed late agent %s at step %s "
                    "(length=%s, width=%s)", tok, step, a.length, a.width)
                self._unroutable.add(tok)
                continue

            # The builder's lane-centre box is the geometry that will actually be published and
            # must pass every admission gate.
            geometry = a.polygon
            path_line = _path_to_go_linestring(a)
            path_buffer = (
                path_line.buffer(a.width / 2, cap_style=CAP_STYLE.flat)
                if path_line is not None else None
            )
            if (geometry.is_empty or not geometry.is_valid or path_buffer is None
                    or path_buffer.is_empty
                    or not geometry.intersects(path_buffer)):
                logger.warning(
                    "[NUPLAN-IDM] refusing late agent %s at step %s: snapped box does "
                    "not intersect its path", tok, step)
                self._unroutable.add(tok)
                continue

            # Seed before the projected-footprint gate so that it evaluates the speed that
            # would actually be published.  Record only after all admission gates pass.
            initial_speed_record = self._recover_idm_initial_speed(tok, a, step)

            # The standalone builder checks current GT boxes. A late vehicle also needs enough
            # room for its initial speed: a merely non-overlapping box can advance into another
            # actor before IDM's bounded deceleration can stop it. nuPlan represents that exact
            # headway requirement with IDMAgent.projected_footprint, the same geometry its manager
            # stores after every propagation. Existing IDM actors may already have diverged from
            # GT, so check this projected footprint against their live projected occupancy. A
            # transient refusal is retried and never permanently blacklists the token.
            projected_geometry = a.projected_footprint
            if self._spawn_gate == "footprint_overlap_only":
                # Root/exact-log lifecycle baseline: delay only a physical overlap at this tick.
                # Existing IDM occupancy stores projected footprints, so inspect the live boxes
                # directly; otherwise this supposedly exact gate would silently inherit the
                # conservative look-ahead policy.  No logged future pose is read here.
                current_geometry = a.polygon
                blockers = [("ego", ego_state.car_footprint.geometry)]
                blockers.extend((str(other), agent.polygon)
                                for other, agent in mgr.agents.items())
                overlaps = [other for other, blocker in blockers
                            if current_geometry.intersection(blocker).area > 0.01]
                if overlaps:
                    self._deferred_tokens.add(tok)
                    logger.debug(
                        "[NUPLAN-IDM] deferring late agent %s at step %s: "
                        "current footprint overlap with %s", tok, step, overlaps[:5])
                    continue
            else:
                mgr.agent_occupancy.set("ego", ego_state.car_footprint.geometry)
                if (projected_geometry.is_empty or not projected_geometry.is_valid
                        or not mgr.agent_occupancy.intersects(projected_geometry).is_empty()):
                    self._deferred_tokens.add(tok)
                    logger.debug(
                        "[NUPLAN-IDM] deferring late agent %s at step %s: projected occupancy collision",
                        tok, step)
                    continue

                ego_clearance = self._ego_lane_spawn_clearance(a, ego_state)
                if ego_clearance is not None:
                    direction, bumper_gap, required_gap = ego_clearance
                    self._deferred_tokens.add(tok)
                    self._spawn_ego_lane_clearance_deferrals += 1
                    logger.debug(
                        "[NUPLAN-IDM] deferring late agent %s at step %s: ego-lane %s "
                        "bumper gap %.2fm < %.2fm",
                        tok, step, direction, bumper_gap, required_gap,
                    )
                    continue

                unsafe_neighbors = self._unsafe_longitudinal_neighbors(
                    a,
                    ego_state,
                    mgr,
                    (
                        open_loop_vehicles
                        if self._late_spawn_open_loop_braking_gap_enabled
                        else ()
                    ),
                )
                if unsafe_neighbors:
                    self._deferred_tokens.add(tok)
                    logger.debug(
                        "[NUPLAN-IDM] deferring late agent %s at step %s: unsafe braking gap to %s",
                        tok, step, unsafe_neighbors[:5])
                    continue

                unsafe_crossing_neighbors = self._unsafe_crossing_spawn_neighbors(
                    a, mgr, crossing_prediction_cache
                )
                if unsafe_crossing_neighbors:
                    self._deferred_tokens.add(tok)
                    self._spawn_crossing_deferrals += 1
                    logger.debug(
                        "[NUPLAN-IDM] deferring late agent %s at step %s: "
                        "current-state crossing envelope with %s",
                        tok, step, unsafe_crossing_neighbors[:5])
                    continue

                unsafe_rail_neighbors = self._unsafe_converging_rail_spawn_neighbors(
                    a, mgr, rail_cache
                )
                if unsafe_rail_neighbors:
                    self._deferred_tokens.add(tok)
                    self._spawn_crossing_deferrals += 1
                    logger.debug(
                        "[NUPLAN-IDM] deferring late agent %s at step %s: "
                        "converging live rail with %s",
                        tok, step, unsafe_rail_neighbors[:5])
                    continue

            a._start_iteration = step
            mgr.agent_occupancy.insert(tok, projected_geometry)
            mgr.agents[tok] = a
            self._idm_initial_speed_records[str(tok)] = initial_speed_record
            self._deferred_tokens.discard(tok)
            admitted.add(tok)
        return admitted

    def _reject_overlapping_new_agents(self, admitted, tracks, ego_state, mgr, step):
        """Hide a late agent if its first post-propagation box overlaps the live world.

        Admission checks nuPlan's projected footprint before propagation.  On the same step,
        ``plan_route`` may extend/select a merging segment and propagation can then produce a box
        which was not represented by that pre-propagation geometry.  Stock nuPlan has no late
        population, so this is a guard only for our long-window extension.  Existing agents are
        never removed by it.
        """
        if not admitted:
            return set()
        objects = {
            str(getattr(obj, "track_token", "") or ""): obj
            for obj in getattr(tracks.tracked_objects, "tracked_objects", []) or []
        }
        blockers = [("ego", ego_state.car_footprint.geometry)]
        blockers.extend(
            (tok, obj.box.geometry) for tok, obj in objects.items()
            if tok and tok not in admitted
        )
        rejected = set()
        for tok in sorted(admitted):
            obj = objects.get(tok)
            if obj is None:
                continue
            geometry = obj.box.geometry
            colliders = [other for other, other_geometry in blockers
                         if geometry.intersection(other_geometry).area > 1e-6]
            if colliders:
                mgr.agents.pop(tok, None)
                if mgr.agent_occupancy.contains(tok):
                    mgr.agent_occupancy.remove([tok])
                self._idm_initial_speed_records.pop(str(tok), None)
                self._deferred_tokens.add(tok)
                rejected.add(tok)
                logger.debug(
                    "[NUPLAN-IDM] deferring late agent %s at step %s: "
                    "post-propagation overlap with %s", tok, step, colliders[:5])
            else:
                blockers.append((tok, geometry))
        return rejected

    def prepare_step(self, step: int) -> None:
        """Advance the shared fleet before Odyssey materializes or observes its proxies."""
        self._advance(step)

    def can_materialize(self, track_token: str) -> bool:
        """Whether this source track has a collision-safe pose for the current batch step."""
        # A pinned vehicle is a static agent held at its log pose by Odyssey itself; the
        # multi-rate presentation path does not publish extra open-loop vehicle poses.
        return (str(track_token) in self._poses
                or str(track_token) in self._static_vehicle_tokens)

    def source_mode_for(self, track_token: str) -> str:
        """Return whether the current pose was propagated by IDM or replayed from GT."""
        if str(track_token) in self._static_vehicle_tokens:
            return "static"
        return self._source_modes.get(str(track_token), "unknown")

    def _predropped_tokens(self):
        return getattr(self, "_predropped", {})

    @property
    def predropped_vehicles(self):
        """token -> reason for vehicles withheld by nuplan_idm_static_pin_and_predrop."""
        return dict(self._predropped_tokens())

    def is_gt_replay_vehicle(self, track_token: str) -> bool:
        """Whether this vehicle follows source poses instead of reactive IDM dynamics.

        Odyssey's source-end retention is intentionally an IDM-only policy.  A vehicle
        which was refused/demoted from IDM and published through nuPlan's open-loop fallback
        must disappear with its recorded validity interval; retaining it by ego distance would
        freeze its last GT pose into a synthetic obstacle.
        """
        token = str(track_token)
        if token in self._static_vehicle_tokens:
            # Pinned vehicles are whole-episode static agents, not sector-shifted replay.
            return False
        return (
            token in self._source_motion_open_loop_tokens
            or token in self._obs.extra_open_loop_vehicle_tokens
        )

    def requires_source_validity(self, track_token: str) -> bool:
        """Whether this deliberately non-reactive proxy may exist only in valid GT frames.

        Normal IDM vehicles outlive a short source track by design.  The exception is a
        temporally corrupt track which was rejected from IDM admission: retaining its final
        open-loop pose would turn a two-frame detection into a permanent obstacle.
        """
        return str(track_token) in self._unstable_short_track_tokens

    def expire_source_limited_track(self, track_token: str) -> None:
        """Permanently retire a corrupt source-only proxy after its final valid frame."""
        token = str(track_token)
        if token not in self._unstable_short_track_tokens:
            return
        self.remove_agent(token)
        self._retired.add(token)
        self._unroutable.add(token)
        self._obs.extra_open_loop_vehicle_tokens.discard(token)

    def pose_for(self, step: int, track_token: str):
        self._advance(step)
        return self._poses.get(str(track_token))

    def remove_agent(self, track_token: str) -> None:
        """Remove a Odyssey track from both coupled nuPlan IDM containers.

        Odyssey owns the source-valid lifecycle.  When a long-window track ends, keeping
        only its IDMAgent leaves an obstacle which no longer exists in the source observation;
        keeping only its BaseAgent leaves a frozen blue box.  nuPlan's manager likewise removes
        the agent and its occupancy entry as one operation when it falls outside its radius.
        """
        token = str(track_token)
        mgr = self._obs._idm_agent_manager
        if self._idm_start_step is None or mgr is None:
            # Source lifecycle can remove a proxy during GT warm-up. It was never admitted
            # to IDM, so removing its published GT pose is the complete operation.
            self._poses.pop(token, None)
            self._source_modes.pop(token, None)
            return
        # A proxy can be pruned before its candidate has ever passed admission.  Such a token is
        # still allowed to retry; only retire a vehicle which truly belonged to the IDM fleet.
        if token in self._ever_admitted or token in mgr.agents:
            self._ever_admitted.add(token)
            self._retired.add(token)
        mgr.agents.pop(token, None)
        if mgr.agent_occupancy.contains(token):
            mgr.agent_occupancy.remove([token])
        self._poses.pop(token, None)
        self._source_modes.pop(token, None)

    def summary(self):
        mgr = self._obs._idm_agent_manager
        if self._idm_start_step is None or mgr is None:
            return f"IDM not started (GT warm-up through step {self._gt_warmup_steps - 1})"
        return (f"IDM agents {len(mgr.agents)} "
                f"(+{self._admitted_after_start} admitted after t=0), "
                f"sector-waited {len(self._spawn_sector_waiting_tokens)}, "
                f"retired {len(self._retired)}, unroutable {len(self._unroutable)}, "
                f"solve_dt {self._solve_dt:g}s")


def _batch(engine, config) -> NuPlanIDMBatch:
    b = getattr(engine, "_nuplan_idm_batch", None)
    if b is None or b.scene is not engine.managers['scenario_manager'].current_scene:
        b = NuPlanIDMBatch(engine, config)
        engine._nuplan_idm_batch = b
    return b


class NuPlanIDMPolicy(BasePolicy):
    """A background agent's share of the batch result."""

    def __init__(self, agent, config=None, random_seed=None):
        super().__init__(agent=agent, config=config, random_seed=random_seed)
        self.dt = float(self.engine.sim_dt)

    def act(self) -> Optional[Trajectory]:
        b = _batch(self.engine, self.engine.global_config)
        step = int(self.engine.episode_step)
        token = str(getattr(self.agent, "id", "") or getattr(self.agent, "name", ""))
        pose = b.pose_for(step, token)
        if pose is None:
            # Not an IDM agent this step (out of radius, or it left the scene). Hold still --
            # the manager's own despawn decides whether it stays at all.
            here = np.asarray(self.agent.current_position, dtype=np.float64)[:2]
            h = float(self.agent.current_heading)
            wp = np.stack([here, here])
            return Trajectory(waypoints=wp, velocities=np.zeros((2, 2)),
                              headings=np.array([h, h]), angular_velocities=np.zeros(2),
                              wp_dt=self.dt)
        x, y, heading, speed = pose
        here = np.asarray(self.agent.current_position, dtype=np.float64)[:2]
        wp = np.stack([here, np.array([x, y])])
        h0 = float(self.agent.current_heading)
        # wp_dt is declared, not inferred: the consumer raises without it, which is what keeps a
        # producer's cadence from being guessed at (see Trajectory.wp_dt).
        # velocities is a per-waypoint VECTOR, not a speed: base_agent.current_speed reads
        # velocity[0]/velocity[1]. Give it the speed along each waypoint's own heading.
        vel = np.stack([speed * np.array([np.cos(h0), np.sin(h0)]),
                        speed * np.array([np.cos(heading), np.sin(heading)])])
        return Trajectory(waypoints=wp, velocities=vel,
                          headings=np.array([h0, heading]),
                          angular_velocities=np.array([(heading - h0) / self.dt] * 2),
                          wp_dt=self.dt)

    def remove_from_batch(self) -> None:
        """Mirror destruction of the Odyssey proxy into the shared IDMAgents batch."""
        batch = getattr(self.engine, "_nuplan_idm_batch", None)
        if batch is not None:
            token = str(getattr(self.agent, "id", "") or getattr(self.agent, "name", ""))
            batch.remove_agent(token)

    @property
    def is_current_step_valid(self):
        return True
