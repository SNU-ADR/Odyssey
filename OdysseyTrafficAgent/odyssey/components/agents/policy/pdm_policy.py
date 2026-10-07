# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
import os
import logging
import numpy as np
from typing import Any, Dict, Optional, Tuple

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.common.actor_state.state_representation import TimePoint, Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from odyssey.components.agents.policy.base_policy import BasePolicy
from odyssey.utils.nuplan_map_utils import (
    resolve_map_location,
    route_roadblock_ids,
)
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from odyssey.components.agents.policy.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from odyssey.components.agents.policy.pdm_planner.reference_config import (
    LATERAL_OFFSETS,
    MAP_RADIUS,
    PROPOSAL_NUM_POSES,
    SAMPLE_INTERVAL,
    TRAJECTORY_NUM_POSES,
    build_reference_idm_policy,
)
from collections import Counter, defaultdict
from shapely.geometry import Point, Polygon

from odyssey.common.dataclasses import Trajectory
from odyssey.components.agents.policy.pdm_planner.utils.odyssey_to_pdm_utils import (
    OdysseyToNuPlanConverter,
    denormalize_from_ego_center,
)
from odyssey.scenario.scenarios.scenario_description import ScenarioDescription as SD
from odyssey.components.agents.policy.pdm_planner.utils.pdm_enums import StateIDMIndex
from odyssey.utils.cadence import synchronized_gt_warmup_steps


def _resolve_pdm_lateral_offsets(config):
    """Return the PDM proposal offsets while preserving the upstream fallback.

    The centerline proposal is always added by ``PDMClosedPlanner``; this list contains only
    the extra parallel paths. ``None``/``[]`` deliberately request centerline-only planning.
    """
    configured = config.get("pdm_lateral_offsets", LATERAL_OFFSETS)
    if configured is None:
        return []
    offsets = [float(value) for value in configured]
    if not np.isfinite(offsets).all():
        raise ValueError("pdm_lateral_offsets must contain only finite distances")
    return offsets


def _sim_dt(scene):
    """This scene's outer step length (s), resolved once by ScenarioManager via resolve_cadence.

    Do not rebuild it as scene["sample_rate"] * 0.05: the formula is right (sample_rate counts
    0.05 s periods), but deriving the same number in several places eventually diverges.
    Fails if missing rather than guessing.
    """
    c = scene.get("cadence") if hasattr(scene, "get") else None
    if c is None:
        raise RuntimeError("scene['cadence'] is missing. sim_dt is not re-derived here.")
    return float(c.sim_dt)


class PDMPolicy(BasePolicy):
    """
    PDM Policy class.
    """ 

    def __init__(self, agent, config=None, random_seed=None):
        """
        Constructor for PDMPolicy

        Signature matches every other policy (build_policy calls `Policy(agent)`).
        :param agent: BaseAgent this policy drives
        :param config: Dict, the configuration for the policy
        :param random_seed: int, seed for reproducibility
        """
        super(PDMPolicy, self).__init__(agent=agent, config=config, random_seed=random_seed)
        # Reference PDM-Closed: 4 s proposals and an 8 s output trajectory, both at 10 Hz.
        # These are planner semantics, not the 0.5 s grid on which MetricManager grades a
        # rollout.  Conflating those clocks would give 20 s proposals at 2 Hz.
        self._proposal_sampling = TrajectorySampling(
            num_poses=PROPOSAL_NUM_POSES, interval_length=SAMPLE_INTERVAL)
        self._future_sampling = TrajectorySampling(
            num_poses=TRAJECTORY_NUM_POSES, interval_length=SAMPLE_INTERVAL)
        self._map_radius = MAP_RADIUS
        self._speed_scale = float(self.agent.config["pdm_speed_scale"])
        self._lateral_offsets = _resolve_pdm_lateral_offsets(self.agent.config)
        self._lane_keeping_weight = float(
            self.agent.config.get("pdm_lane_keeping_weight", 0.0)
        )
        if not np.isfinite(self._lane_keeping_weight) or self._lane_keeping_weight < 0.0:
            raise ValueError(
                "pdm_lane_keeping_weight must be finite and non-negative"
            )
        # PDM is a stand-in used to exercise the reactive world, while the production planner path
        # replays the ego's logged motion until enough history exists for the external planner.
        # Keep that handoff contract here as well: IDM still propagates during these ticks, but
        # PDM does not take control before the real planner would.
        self._gt_warmup_steps = synchronized_gt_warmup_steps(self.agent.config)
        # The route is a property of the SCENE, not of the step: it is read off the logged ego
        # track, which does not change while the rollout runs. Computed on first use rather
        # than here because the converter and the agent's object_track are not ready yet.
        self._route_roadblocks_ids = None
        #: (map_api id, route ids) the PDM planner was last initialize()d for.
        self._pdm_initialized_for = None
        self._stop_diagnostics_enabled = bool(
            self.agent.config.get("pdm_stop_diagnostics_enabled", True)
        )
        self._stop_diagnostics_interval_steps = max(
            1, int(self.agent.config.get("pdm_stop_diagnostics_interval_steps", 10))
        )
        self.scenario_manager = self.engine.managers['scenario_manager']
        self._pdm_closed = PDMClosedPlanner(
            trajectory_sampling=self._future_sampling,
            proposal_sampling=self._proposal_sampling,
            idm_policies=build_reference_idm_policy(speed_scale=self._speed_scale),
            lateral_offsets=self._lateral_offsets,
            map_radius=self._map_radius,
            lane_keeping_weight=self._lane_keeping_weight,
        ) # register planner

        self.current_scene = self.scenario_manager.current_scene
        sim_dt = _sim_dt(self.current_scene)
        plan_dt = float(self.agent.config.get("pdm_plan_dt", sim_dt) or sim_dt)
        plan_ratio = plan_dt / sim_dt
        self._plan_stride = int(round(plan_ratio))
        if self._plan_stride < 1 or not np.isclose(plan_ratio, self._plan_stride):
            raise ValueError(
                "pdm_plan_dt must be an integer multiple of the simulation step: "
                f"plan_dt={plan_dt}, sim_dt={sim_dt}"
            )
        self._plan_dt = self._plan_stride * sim_dt
        # The policy rebuilds its local drivable-area map around the CURRENT ego every step, so
        # the reference 50 m radius remains sufficient even for an 80 s Odyssey rollout.
        # MetricManager may use a wider map to grade the complete saved trajectory at once; that
        # is an evaluator adapter and must not change the driving policy.
        self.converter = OdysseyToNuPlanConverter(self.current_scene, self.agent, self.engine)
        ego_source_state = self.current_scene["object_track"][
            self.current_scene["sdc_id"]
        ]["state"]
        ego_source_position = np.asarray(ego_source_state.get("position", []), dtype=float)
        ego_source_valid = np.asarray(ego_source_state.get("valid", []), dtype=bool)
        ego_source_heading = np.asarray(ego_source_state.get("heading", []), dtype=float)
        self._gt_reference_centerline = None
        if (
            ego_source_position.ndim == 2
            and ego_source_position.shape[1] >= 2
            and len(ego_source_valid) == len(ego_source_position)
            and np.any(ego_source_valid)
        ):
            self._gt_destination_world = (
                ego_source_position[ego_source_valid, :2][-1]
                + np.asarray(self.converter.initial_ego_center, dtype=float)[:2]
            )
            if (
                bool(self.agent.config.get("pdm_gt_reference_centerline_enabled", True))
                and len(ego_source_heading) == len(ego_source_position)
            ):
                valid = ego_source_valid & np.isfinite(ego_source_heading)
                valid &= np.isfinite(ego_source_position[:, :2]).all(axis=1)
                xy = (
                    ego_source_position[valid, :2]
                    + np.asarray(self.converter.initial_ego_center, dtype=float)[:2]
                )
                headings = ego_source_heading[valid]
                if len(xy) >= 2:
                    # PDMPath's progress interpolator requires strictly increasing samples.
                    # Logged red-light/queue stops legitimately repeat poses, so retain their
                    # geometry once rather than encoding dwell time into the route rail.
                    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-3]
                    reference = np.column_stack((xy[keep], headings[keep]))
                    if len(reference) >= 2:
                        self._gt_reference_centerline = reference
        else:
            self._gt_destination_world = None
        nuplan_map_root = self.engine.global_config['nuplan_map_root']
        map_location = resolve_map_location(self.scenario_manager.current_scene, nuplan_map_root)
        self.map_api = get_maps_api(nuplan_map_root, "nuplan-maps-v1.0", map_location)

        self.ego_states_list = []
        self.observations_list = []
        self._logger = logging.getLogger(__name__)
        self._logger.info(
            "[PDM] plan_dt=%s (every %d sim step%s), ego_speed_scale=%.3f, "
            "lateral_offsets=%s, lane_keeping_weight=%.3f",
            self._plan_dt,
            self._plan_stride,
            "s" if self._plan_stride != 1 else "",
            self._speed_scale,
            self._lateral_offsets,
            self._lane_keeping_weight,
        )

    def act(self):
        """
        Run the PDM policy.

        """
        self.current_step = self.engine.episode_step 
        if self.current_scene is None:
            self._logger.error("No scene available for PDM policy.")
            return None
        else:
            current_ego_state = self.converter.convert_to_current_ego_state(self.current_step)
            self.ego_states_list.append(current_ego_state)

        if self.current_step < self._gt_warmup_steps:
            self.observations_list.append(
                self.converter.convert_to_detections_tracks_from_agent_input(
                    self.current_step
                )
            )
            return self._logged_warmup_trajectory(self.current_step)

        # Keep the 10 Hz observation/history clock even when planning at a lower rate.
        # The controller consumes one more waypoint from the existing time-parameterized
        # trajectory on skipped ticks, so the 10 Hz plant remains unchanged.
        current_trajectory = getattr(self.agent, "trajectory", None)
        should_plan = (
            current_trajectory is None
            or self.current_step % self._plan_stride == 0
        )
        if not should_plan:
            self.observations_list.append(
                self.converter.convert_to_detections_tracks_from_agent_input(
                    self.current_step
                )
            )
            return current_trajectory
        
        planner_input, planner_initialization = self._get_planner_inputs(self.current_scene)

        # initialize() is nuPlan's ONCE-PER-SCENARIO setup, not a per-step call. It reloads the
        # route dicts (two map-object lookups per roadblock, then every lane of each) and then
        # runs a full gc.collect(). Doing that every step, in a process holding the map and a few
        # hundred agent tracks, costs seconds per step, against the tens of ms PDM-Closed is
        # supposed to take.
        #
        # Nothing in it depends on the step -- the route is a scene constant (see
        # get_route_roadblocks_ids) and so is the map api. Re-run it only if either actually
        # changes, which also keeps a policy reused across scenarios correct.
        _key = (id(planner_initialization["map_api"]),
                tuple(planner_initialization["route_roadblock_dict_ids"]))
        if _key != self._pdm_initialized_for:
            self._pdm_closed.initialize(planner_initialization)
            self._pdm_initialized_for = _key

        # as_trajectory=True: we DRIVE this plan, so we need the InterpolatedTrajectory the
        # converter below expects, not the raw proposal array dense_reward_manager wants.
        planned_trajectory = self._pdm_closed.compute_planner_trajectory(
            planner_input, as_trajectory=True)
        if not getattr(self, "_route_plan_logged", False):
            def _safe_len(value):
                try:
                    return len(value)
                except TypeError:
                    return 0

            def _safe_float(value):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return float("nan")

            self._logger.info(
                "[PDM-ROUTE] roadblocks=%d path_found=%s start_lane=%s target_rb=%s "
                "route_lanes=%d centerline=%.1fm",
                _safe_len(getattr(self._pdm_closed, "_route_roadblock_dict", {})),
                getattr(self._pdm_closed, "_last_centerline_path_found", None),
                getattr(self._pdm_closed, "_last_centerline_start_lane_id", "?"),
                getattr(self._pdm_closed, "_last_centerline_target_roadblock_id", "?"),
                _safe_len(getattr(self._pdm_closed, "_last_centerline_lane_ids", ())),
                _safe_float(getattr(getattr(self._pdm_closed, "_centerline", None),
                                    "length", float("nan"))),
            )
            self._route_plan_logged = True
        self._log_stop_diagnostics(current_ego_state)
        
        trajectory = self.converter.convert_to_trajectory(
            planned_trajectory, wp_dt=self._future_sampling.interval_length)
        
        return trajectory

    def _log_stop_diagnostics(self, ego_state) -> None:
        """Explain a low-speed PDM decision without altering proposal selection."""
        if (
            not getattr(self, "_stop_diagnostics_enabled", False)
            or self.current_step
            % getattr(self, "_stop_diagnostics_interval_steps", 10)
        ):
            return
        velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d
        ego_speed = float(np.hypot(velocity.x, velocity.y))
        if ego_speed > 0.75:
            return
        best_idx = getattr(self._pdm_closed, "_last_best_idx", None)
        generator = self._pdm_closed._generator
        if best_idx is None or generator._state_idm_array is None:
            return
        planned_speeds = generator._state_idm_array[
            best_idx, :, StateIDMIndex.VELOCITY
        ]
        lead_tokens = [
            str(token)
            for token in generator._leading_agent_tokens[best_idx, 1:]
            if str(token)
        ]
        lead_counts = Counter(lead_tokens)
        lead_summary = []
        for token, count in lead_counts.most_common(3):
            if token == "__path_end__":
                object_type = "free/path_end"
            elif "red_light" in token:
                object_type = "red_light"
            else:
                obj = self._pdm_closed._observation.unique_objects.get(token)
                object_type = str(getattr(obj, "tracked_object_type", "unknown"))
            lead_summary.append(f"{token}:{object_type}x{count}")

        proposal = self._pdm_closed._proposal_manager[best_idx]
        current_progress = proposal.linestring.project(
            Point(*ego_state.rear_axle.point.array)
        )
        remaining_path_m = max(0.0, proposal.linestring.length - current_progress)
        scores = getattr(self._pdm_closed, "_last_proposal_scores", ())
        best_score = float(scores[best_idx]) if len(scores) else float("nan")
        path_found = getattr(self._pdm_closed, "_last_centerline_path_found", None)
        start_lane = getattr(self._pdm_closed, "_last_centerline_start_lane_id", "?")
        target_roadblock = getattr(
            self._pdm_closed, "_last_centerline_target_roadblock_id", "?"
        )
        route_lane_count = len(getattr(
            self._pdm_closed, "_last_centerline_lane_ids", ()
        ))
        if self._gt_destination_world is None:
            gt_goal_distance_m = float("nan")
        else:
            gt_goal_distance_m = float(np.linalg.norm(
                np.asarray(ego_state.center.point.array, dtype=float)[:2]
                - self._gt_destination_world
            ))
        self._logger.info(
            "[PDM-STOP-DIAG] tick=%d ego_v=%.3f best=%d score=%.4f "
            "planned_v[min/end]=%.3f/%.3f remaining_path=%.2fm gt_goal=%.2fm "
            "path_found=%s start_lane=%s target_rb=%s route_lanes=%d leads=%s",
            self.current_step,
            ego_speed,
            best_idx,
            best_score,
            float(np.min(planned_speeds)),
            float(planned_speeds[-1]),
            remaining_path_m,
            gt_goal_distance_m,
            path_found,
            start_lane,
            target_roadblock,
            route_lane_count,
            ",".join(lead_summary) if lead_summary else "none",
        )

    def _logged_warmup_trajectory(self, step: int) -> Trajectory:
        """Replay the same ego GT prefix used before an external E2E handoff."""
        track = self.agent.object_track
        stop = step + 9
        return Trajectory(
            waypoints=track[SD.POSITION][step:stop, :2],
            velocities=track["velocity"][step:stop],
            headings=track[SD.HEADING][step:stop],
            angular_velocities=track["angular_velocity"][step:stop],
            wp_dt=_sim_dt(self.current_scene),
        )
    
    def _get_planner_inputs(self, scene) -> Tuple[PlannerInput, PlannerInitialization]:
        """
        Creates planner input arguments from scenario object.
        :param scenario: scenario object of Odyssey
        :return: tuple of planner input and initialization objects
        """
        observation = self.converter.convert_to_detections_tracks_from_agent_input(self.current_step)
        self.observations_list.append(observation)
        base_timestamp = scene.get("base_timestamp", 0.0)
        time_stamp = base_timestamp + self.current_step * _sim_dt(scene) * 1e6
        
        route_roadblocks_ids = self.get_route_roadblocks_ids()
        
        # Initialize Planner
        planner_initialization = {
            "route_roadblock_dict_ids": route_roadblocks_ids,
            "map_api": self.map_api,
            # This route was sampled from every valid GT ego pose.  PDM's generic polygon-overlap
            # loop cutter may therefore not discard its endpoint; overlapping connectors are
            # resolved by the ordered map matcher above.
            "route_is_gt_matched": True,
            # Odyssey windows are 5-10x longer than the nuPlan PDM design horizon and can
            # contain lane changes or map seams that have no lane-graph edge.  Supply only the
            # logged route geometry; PDM still chooses speed, obstacle response, lateral
            # proposal, and control in closed loop.
            "reference_centerline": self._gt_reference_centerline,
        }   

        if self.current_step <= 5:
            buffer_size = self.current_step + 1
        else:
            buffer_size = 5

        history = SimulationHistoryBuffer.initialize_from_list(
            buffer_size=buffer_size,
            ego_states=self.ego_states_list, 
            observations=self.observations_list
        )

        traffic_light_data = self.converter.convert_to_traffic_lights(self.current_step)
        planner_input = PlannerInput(
            iteration = SimulationIteration(index=self.current_step, time_point=TimePoint(time_stamp)),
            history = history,
            traffic_light_data = traffic_light_data,
        )

        return planner_input, planner_initialization
    
    def get_route_roadblocks_ids(self):
        """Roadblocks the ego's logged route runs through, in order, as nuPlan map ids.

        Queried against the nuPlan map api in the GLOBAL frame, NOT read off
        navigation.checkpoint_lanes. Those lanes carry real nuPlan roadblock ids, so
        _load_route_dicts() would resolve every one, but using them mixes the simulator's two
        coordinate frames:

          * navigation builds its checkpoints by calling get_closest_lane_index() with
            object_track positions, which are SCENE-LOCAL (they start at (0, 0)).
          * the EgoState this planner reasons about comes from OdysseyToNuPlanConverter, which
            applies `old_origin_in_current_coordinate` and is therefore GLOBAL UTM.

        _get_starting_lane() would then intersect a global ego pose against a route picked out
        in scene-local space, find no intersecting lane, fall through to its "closest lane"
        fallback and build the centerline from an essentially arbitrary lane -- a planner that
        barely moves, with nothing raised.

        Sampling the ego's own logged path (rather than trusting a lane graph built in the other
        frame) also keeps the route ordered and gap-free, which is what PDM's Dijkstra centerline
        search expects.

        CACHED for the life of the policy. It depends only on the logged ego track, the
        converter's initial_ego_center and the map api -- all three constant for a scene.
        _get_planner_inputs() calls it every step, and it walks EVERY logged pose with one or two
        map-layer lookups each, so recomputing it per step would be pure waste.
        """
        if self._route_roadblocks_ids is not None:
            return self._route_roadblocks_ids

        positions = np.asarray(self.agent.object_track[SD.POSITION], dtype=np.float64)
        route_roadblocks_ids = route_roadblock_ids(
            positions, self.converter.initial_ego_center, self.map_api, self._logger)

        if not route_roadblocks_ids:
            self._logger.warning(
                "PDM route is empty: the planner will fall back to its nearest-lane "
                "heuristic and is likely to crawl.")
        else:
            # Once per scene, so it is cheap and it says what route the run was actually
            # scored against -- the first thing to check when a drive reads as off-route.
            self._logger.info(
                "[PDM] route resolved once from %d logged ego poses -> %d roadblocks",
                len(positions), len(route_roadblocks_ids))
        self._route_roadblocks_ids = route_roadblocks_ids
        return route_roadblocks_ids

    @property
    def is_current_step_valid(self):
        return True
