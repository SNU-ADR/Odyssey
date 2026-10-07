"""NPC-only admission control for IDM traffic at junctions.

The manager is deliberately outside the ego planning/observation stack.  It sees only
the route rails already owned by each IDM agent and a narrow current-state ego safety
snapshot.  A hold decision is enforced by inserting a temporary ``stop_line_*``
polygon into nuPlan's occupancy map, which makes the stock IDM lead-agent query treat
it as a stationary obstacle.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field, replace
from math import cos, hypot, sin
from typing import Dict, FrozenSet, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from shapely import buffer as shapely_buffer
from shapely import line_locate_point, points
from shapely.affinity import translate
from shapely.geometry import LineString, Point
from shapely.geometry.base import CAP_STYLE
from shapely.ops import nearest_points, substring, unary_union

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.geometry.transform import rotate_angle
from nuplan.planning.metrics.utils.expert_comparisons import principal_value
from nuplan.planning.simulation.observation.idm.idm_agent_manager import IDMAgentManager
from nuplan.planning.simulation.observation.idm.idm_states import IDMLeadAgentState
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring

logger = logging.getLogger(__name__)

HOLD_REASONS = (
    "active_npc",
    "downstream_blocked",
    "ego_safety_envelope",
    "fifo_wait",
)


def _path_to_go_linestring(agent: object) -> Optional[LineString]:
    """Build an agent's remaining path once per route/progress state.

    Intersection admission and IDM propagation inspect the same pre-propagation path.
    Real IDMAgents expose ``_path`` and ``progress``; lightweight test doubles fall
    back to uncached construction so mutable fake state cannot become stale.
    """
    route_path = getattr(agent, "_path", None)
    progress = getattr(agent, "progress", None)
    cacheable = route_path is not None and progress is not None
    cache_key = (id(route_path), float(progress)) if cacheable else None
    if cacheable and getattr(agent, "_odyssey_path_line_cache_key", None) == cache_key:
        return getattr(agent, "_odyssey_path_line_cache", None)

    direct_linestring = getattr(agent, "get_path_to_go_linestring", None)
    if direct_linestring is not None:
        line = direct_linestring()
    else:
        path_states = agent.get_path_to_go()
        line = path_to_linestring(path_states) if len(path_states) >= 2 else None
    if cacheable:
        agent._odyssey_path_line_cache_key = cache_key
        agent._odyssey_path_line_cache = line
    return line


@dataclass(frozen=True)
class EgoSafetyState:
    """The complete ego-facing API: current footprint and current velocity only."""

    footprint: object
    velocity_x: float
    velocity_y: float


@dataclass(frozen=True)
class IntersectionWorldState:
    """Simulator-side inputs that do not expose planner or ego-route state."""

    tick: int
    occupancy: object
    # Existing NPC traffic-light input.  Presence in this set means that the physical
    # intersection is already controlled by stock nuPlan IDM and must not be serialized here.
    controlled_lane_connector_ids: FrozenSet[str] = frozenset()
    green_lane_connector_ids: FrozenSet[str] = frozenset()


@dataclass
class IntersectionState:
    active_npc_ids: Set[str] = field(default_factory=set)
    granted_npc_ids: Set[str] = field(default_factory=set)
    arrival_tick: Dict[str, int] = field(default_factory=dict)
    last_progress_tick: int = 0

    @property
    def active_npc_id(self) -> Optional[str]:
        """Backward-compatible V1 view; V2 may contain several non-conflicting holders."""
        return min(self.active_npc_ids) if self.active_npc_ids else None

    @property
    def granted_npc_id(self) -> Optional[str]:
        """Backward-compatible V1 view; V2 may contain several non-conflicting grants."""
        return min(self.granted_npc_ids) if self.granted_npc_ids else None


@dataclass(frozen=True)
class AdmissionDecision:
    npc_id: str
    intersection_id: str
    enter: bool
    reason: Optional[str] = None
    virtual_lead_geometry: Optional[object] = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class IntersectionManagerConfig:
    sim_dt: float
    decel_max: float
    reaction_margin_s: float = 0.7
    distance_margin_m: float = 3.0
    stop_offset_m: float = 1.0
    virtual_lead_depth_m: float = 0.2
    virtual_lead_lateral_margin_m: float = 0.75
    downstream_margin_m: float = 2.0
    downstream_start_margin_m: float = 0.5
    downstream_lateral_margin_m: float = 0.5
    ego_envelope_horizon_s: float = 1.5
    ego_envelope_margin_m: float = 0.75
    stopped_rear_ego_release_distance_m: float = 15.0
    stopped_rear_ego_release_lateral_m: float = 3.0
    movement_corridor_margin_m: float = 0.5
    allow_non_conflicting_movements: bool = True
    contention_only: bool = False
    protect_signal_conflicts: bool = False
    wait_bonus_per_tick: float = 0.01
    stall_timeout_s: float = 8.0
    progress_epsilon_m: float = 0.5
    stopped_speed_mps: float = 0.2
    retire_active_stalls: bool = False
    downstream_stopped_persistence_s: float = 2.0
    reservation_commit_distance_m: float = 2.0
    reservation_ttl_s: float = 2.0
    log_interval_ticks: int = 100

    def __post_init__(self) -> None:
        if self.sim_dt <= 0 or self.decel_max <= 0:
            raise ValueError("sim_dt and decel_max must be positive")
        if self.stall_timeout_s <= 0 or self.progress_epsilon_m <= 0:
            raise ValueError("stall_timeout_s and progress_epsilon_m must be positive")
        if self.downstream_stopped_persistence_s <= 0:
            raise ValueError("downstream_stopped_persistence_s must be positive")
        if self.reservation_commit_distance_m < 0 or self.reservation_ttl_s <= 0:
            raise ValueError(
                "reservation_commit_distance_m must be non-negative and reservation_ttl_s positive"
            )
        if (
            self.stopped_rear_ego_release_distance_m < 0
            or self.stopped_rear_ego_release_lateral_m < 0
        ):
            raise ValueError("stopped rear-ego release bounds must be non-negative")


@dataclass(frozen=True)
class _Passage:
    npc_id: str
    intersection_id: str
    junction_geometry: object
    movement_corridor: object
    movement_vector: Tuple[float, float]
    downstream_corridor: Optional[object]
    downstream_vector: Optional[Tuple[float, float]]
    virtual_lead_geometry: object
    distance_to_entry_m: float
    distance_to_exit_m: float
    lane_connector_ids: Tuple[str, ...] = ()


class TargetedLeadIDMAgentManager(IDMAgentManager):
    """nuPlan IDM manager with per-NPC, rather than global, virtual stop lines.

    Stock nuPlan's occupancy map is shared by the fleet.  Leaving every admission gate in that
    map for the full propagation pass can stop the permit holder when its path crosses another
    NPC's gate.  This small propagation specialization inserts a gate only while evaluating its
    intended held NPC, then removes it before the next NPC is processed.
    """

    _VIRTUAL_PREFIX = "stop_line_intersection_manager_"

    def __init__(self, agents, agent_occupancy, map_api):
        super().__init__(agents, agent_occupancy, map_api)
        self._targeted_virtual_leads: Dict[str, object] = {}
        self._routes_preplanned_iteration = -1
        self._wait_cycle_timeout_ticks = 0
        self._ego_deadlock_timeout_ticks = 0
        self._wait_cycle_stopped_speed_mps = 0.5
        self._physical_lead_margin_m = 0.0
        self._rear_lead_rejections = 0
        self._adjacent_lead_rejections = 0
        self._vehicle_only_lead_lane_ids: Set[str] = set()
        self._scoped_non_vehicle_lead_suppressions = 0
        self._wait_cycle_first_tick: Dict[Tuple[str, ...], int] = {}
        self._reported_wait_cycles = set()
        self._wait_cycle_events = 0
        self._wait_cycle_agent_ticks = 0
        self._wait_cycle_release_leads: Dict[str, str] = {}
        self._wait_cycle_release_until: Dict[str, Tuple[str, int]] = {}
        self._wait_cycle_breaker_ticks = 0
        self._active_intersection_tokens: Set[str] = set()
        self._last_selected_physical_leads: Dict[str, str] = {}
        self._ego_deadlock_first_tick: Dict[str, int] = {}
        self._ego_deadlock_release_until: Dict[str, int] = {}
        self._ego_deadlock_breaker_events = 0
        self._ego_deadlock_breaker_ticks = 0

    def set_targeted_virtual_leads(self, leads: Mapping[str, object]) -> None:
        self._targeted_virtual_leads = {str(token): geometry for token, geometry in leads.items()}

    def mark_routes_preplanned(self, iteration: int) -> None:
        """Record that the admission pass already planned every active route this tick."""
        self._routes_preplanned_iteration = int(iteration)

    def configure_wait_cycle_monitor(
        self,
        timeout_ticks: int,
        stopped_speed_mps: float = 0.5,
        ego_deadlock_timeout_ticks: Optional[int] = None,
    ) -> None:
        """Enable read-only NPC wait-for-cycle accounting; zero ticks disables it."""
        self._wait_cycle_timeout_ticks = max(0, int(timeout_ticks))
        self._ego_deadlock_timeout_ticks = max(
            0,
            int(timeout_ticks if ego_deadlock_timeout_ticks is None else ego_deadlock_timeout_ticks),
        )
        self._wait_cycle_stopped_speed_mps = float(stopped_speed_mps)

    def configure_physical_path_leads(self, margin_m: float = 0.0) -> None:
        """Set extra bumper clearance for path-ranked physical lead selection."""
        if margin_m < 0:
            raise ValueError("physical path lead margin must be non-negative")
        self._physical_lead_margin_m = float(margin_m)

    def configure_scoped_vehicle_only_leads(self, lane_ids: Sequence[str]) -> None:
        """Ignore open-loop non-vehicles only while an IDM body occupies a listed lane."""
        self._vehicle_only_lead_lane_ids = {str(lane_id) for lane_id in lane_ids}

    def configure_active_intersection_tokens(self, tokens: Sequence[str]) -> None:
        """Provide current committed junction holders for ego/NPC deadlock recovery."""
        self._active_intersection_tokens = {str(token) for token in tokens}

    def _uses_vehicle_only_leads(self, agent: object) -> bool:
        if not self._vehicle_only_lead_lane_ids:
            return False
        body = agent.polygon
        for segment in agent.get_route():
            if str(getattr(segment, "id", "")) not in self._vehicle_only_lead_lane_ids:
                continue
            lane_polygon = getattr(segment, "polygon", None)
            if lane_polygon is not None and body.intersects(lane_polygon):
                return True
        return False

    def uses_scoped_vehicle_only_leads(self, agent: object) -> bool:
        """Public read-only query used while preparing this agent's route frontier."""
        return self._uses_vehicle_only_leads(agent)

    @staticmethod
    def _cycles_in_functional_graph(edges: Mapping[str, str]) -> set:
        cycles = set()
        finished = set()
        for start in edges:
            if start in finished:
                continue
            order = []
            indices = {}
            node = start
            while node in edges and node not in finished:
                if node in indices:
                    cycle = tuple(sorted(order[indices[node]:]))
                    if len(cycle) >= 2:
                        cycles.add(cycle)
                    break
                indices[node] = len(order)
                order.append(node)
                node = edges[node]
            finished.update(order)
        return cycles

    def _update_wait_cycles(self, iteration: int, observed_leads: Mapping[str, str]) -> None:
        if self._wait_cycle_timeout_ticks <= 0:
            self._wait_cycle_release_leads = {}
            self._wait_cycle_release_until = {}
            return
        self._wait_cycle_release_until = {
            token: (lead, until)
            for token, (lead, until) in self._wait_cycle_release_until.items()
            if int(iteration) < int(until)
            and token in self.agents
            and lead in self.agents
        }
        stopped = {
            str(token)
            for token, agent in self.agents.items()
            if float(agent.velocity) <= self._wait_cycle_stopped_speed_mps
        }
        edges = {
            str(follower): str(leader)
            for follower, leader in observed_leads.items()
            if str(follower) in stopped and str(leader) in stopped
        }
        current = self._cycles_in_functional_graph(edges)
        release_leads: Dict[str, str] = {
            token: lead
            for token, (lead, _until) in self._wait_cycle_release_until.items()
        }
        for cycle in list(self._wait_cycle_first_tick):
            if cycle not in current:
                self._wait_cycle_first_tick.pop(cycle, None)
                self._reported_wait_cycles.discard(cycle)
        for cycle in current:
            first_tick = self._wait_cycle_first_tick.setdefault(cycle, int(iteration))
            if int(iteration) - first_tick + 1 < self._wait_cycle_timeout_ticks:
                continue
            self._wait_cycle_agent_ticks += len(cycle)
            # A persistent mutual physical-lead cycle cannot be solved by FIFO grants alone:
            # both longitudinal controllers see the other body as their lead.  Let exactly one
            # vehicle ignore exactly its cyclic lead while it clears the conflict. Prefer the vehicle
            # furthest along its route (smallest distance-to-go); every other member continues
            # to see it and stays stopped.  This breaks the cycle without deleting either car or
            # globally weakening collision handling.
            def _remaining(token):
                try:
                    return float(self.agents[token].get_progress_to_go())
                except (AttributeError, TypeError, ValueError):
                    return float("inf")

            # Ignoring the wrong member of a crossing cycle is not a liveness fix: it lets that
            # vehicle accelerate through the stationary lead's body.  Release only a member
            # whose short forward motion increases physical separation from its cyclic lead.
            # This picks the vehicle already pointing out of the conflict.  If neither member
            # can move away, keep both physical leads; a collision is worse than a stalemate.
            safe_members = [
                token
                for token in cycle
                if self._cycle_release_moves_away(token, edges[token])
            ]
            if not safe_members:
                continue
            winner = min(safe_members, key=lambda token: (_remaining(token), token))
            release_leads[winner] = edges[winner]
            # One tick advances a stopped vehicle only centimetres and the same cycle forms
            # again.  Two timeout windows (4 s in production) are enough to clear a normal
            # vehicle length under bounded IDM acceleration, while every non-cyclic obstacle
            # remains visible to the winner.
            self._wait_cycle_release_until[winner] = (
                edges[winner], int(iteration) + 2 * self._wait_cycle_timeout_ticks
            )
            if cycle not in self._reported_wait_cycles:
                self._reported_wait_cycles.add(cycle)
                self._wait_cycle_events += 1
                logger.warning(
                    "[IDM-WAIT-CYCLE] tick=%d npcs=%s release=%s ignored_lead=%s",
                    iteration,
                    ",".join(cycle),
                    winner,
                    edges[winner],
                )
        self._wait_cycle_release_leads = release_leads
        self._wait_cycle_breaker_ticks += len(release_leads)

    def _cycle_release_moves_away(self, follower_token: str, leader_token: str) -> bool:
        """Return true only when a small forward step separates a mutual-lead pair."""
        follower = self.agents.get(str(follower_token))
        leader = self.agents.get(str(leader_token))
        follower_polygon = getattr(follower, "polygon", None)
        leader_polygon = getattr(leader, "polygon", None)
        to_se2 = getattr(follower, "to_se2", None)
        if follower_polygon is None or leader_polygon is None or to_se2 is None:
            # Preserve behavior for lightweight external managers which do not expose geometry;
            # production nuPlan IDM agents always take the checked branch above.
            return True
        if follower_polygon.intersection(leader_polygon).area > 1e-4:
            return False
        pose = to_se2()
        old_distance = float(follower_polygon.distance(leader_polygon))
        probe_distance_m = 0.5
        shifted = translate(
            follower_polygon,
            xoff=probe_distance_m * cos(float(pose.heading)),
            yoff=probe_distance_m * sin(float(pose.heading)),
        )
        return bool(
            shifted.intersection(leader_polygon).area <= 1e-4
            and float(shifted.distance(leader_polygon)) > old_distance + 0.05
        )

    def get_wait_cycle_stats(self) -> dict:
        return {
            "wait_cycle_events": self._wait_cycle_events,
            "wait_cycle_agent_ticks": self._wait_cycle_agent_ticks,
            "wait_cycle_breaker_ticks": self._wait_cycle_breaker_ticks,
            "rear_lead_rejections": self._rear_lead_rejections,
            "adjacent_lead_rejections": self._adjacent_lead_rejections,
            "ego_deadlock_breaker_events": self._ego_deadlock_breaker_events,
            "ego_deadlock_breaker_ticks": self._ego_deadlock_breaker_ticks,
            "scoped_non_vehicle_lead_suppressions": (
                self._scoped_non_vehicle_lead_suppressions
            ),
        }

    def _selected_physical_lead_rerank_reason(
        self,
        agent_token: str,
        agent: object,
        path_line: LineString,
        lead_id: str,
    ) -> Optional[str]:
        """Return why a stock same-flow lead needs physical path re-ranking.

        Besides a projected rear vehicle, a long vehicle on a curve can touch an adjacent
        lane's projected footprint with one body corner.  A same-direction actor whose centre
        is outside the combined lane-width envelope is not a longitudinal lead.  Crossing
        traffic is deliberately excluded from this test and remains a valid blocker.
        """
        lead_id = str(lead_id)
        if lead_id == str(agent_token) or lead_id not in self.agents:
            return None
        lead = self.agents[lead_id]
        if agent.polygon.intersects(lead.polygon):
            return None
        relative_heading = principal_value(
            float(lead.to_se2().heading) - float(agent.to_se2().heading)
        )
        # Crossing traffic remains a legitimate physical blocker.  The false-lead case is a
        # same-flow rear car whose long projected footprint curls into the follower's corridor.
        if abs(relative_heading) > np.pi / 3.0:
            return None
        if path_line is None or path_line.is_empty or path_line.length < 1e-6:
            return None
        start = path_line.interpolate(0.0)
        tangent_end = path_line.interpolate(min(1.0, path_line.length))
        tangent = np.array(
            [tangent_end.x - start.x, tangent_end.y - start.y], dtype=np.float64
        )
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-6:
            return None
        centroid = lead.polygon.centroid
        forward_offset = (
            (float(centroid.x) - float(start.x)) * tangent[0]
            + (float(centroid.y) - float(start.y)) * tangent[1]
        ) / norm
        if forward_offset < -0.25:
            return "rear"
        lateral_distance = float(path_line.distance(centroid))
        same_lane_envelope = (
            (float(agent.width) + float(lead.width)) / 2.0 + 0.25
        )
        if lateral_distance > same_lane_envelope:
            return "adjacent"
        return None

    def _nearest_physical_lead_on_path(
        self,
        agent_token: str,
        agent: object,
        path_line: LineString,
        intersecting_ids: Sequence[str],
        path_corridor: Optional[object] = None,
    ) -> Optional[Tuple[str, float]]:
        """Choose the nearest physical obstacle ahead along this agent's rail.

        Stock nuPlan asks for Euclidean distance between *projected* footprints.
        On a curve, a following vehicle's long projected polygon can be closer than the
        actual vehicle ahead, so the query can select a rear vehicle as the IDM lead.
        Keep projected polygons for broad-phase anticipation, then rank current physical
        boxes by progress along the follower's own remaining path.
        """
        if path_line is None or path_line.is_empty or path_line.length < 1e-6:
            return None
        start = path_line.interpolate(0.0)
        tangent_end = path_line.interpolate(min(1.0, path_line.length))
        tangent = np.array(
            [tangent_end.x - start.x, tangent_end.y - start.y], dtype=np.float64
        )
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-6:
            return None
        tangent /= norm
        corridor = path_corridor
        if corridor is None:
            corridor = path_line.buffer(
                float(agent.width) / 2.0, cap_style=CAP_STYLE.flat
            )
        own_half_length = float(agent.length) / 2.0
        candidates = []
        for raw_id in intersecting_ids:
            lead_id = str(raw_id)
            if lead_id == str(agent_token):
                continue
            if lead_id in self.agents:
                lead_agent = self.agents[lead_id]
                geometry = lead_agent.polygon
            elif self.agent_occupancy.contains(lead_id):
                lead_agent = None
                geometry = self.agent_occupancy.get(lead_id)
            else:
                continue
            if geometry is None or geometry.is_empty or not geometry.intersects(corridor):
                continue
            centroid = geometry.centroid
            if lead_agent is not None and not agent.polygon.intersects(geometry):
                relative_heading = principal_value(
                    float(lead_agent.to_se2().heading)
                    - float(agent.to_se2().heading)
                )
                if abs(relative_heading) <= np.pi / 3.0:
                    lateral_distance = float(path_line.distance(centroid))
                    same_lane_envelope = (
                        (float(agent.width) + float(lead_agent.width)) / 2.0
                        + 0.25
                    )
                    if lateral_distance > same_lane_envelope:
                        continue
            forward_offset = (
                (float(centroid.x) - float(start.x)) * tangent[0]
                + (float(centroid.y) - float(start.y)) * tangent[1]
            )
            overlap_measures = IntersectionManager._geometry_measures(
                path_line, path_line.intersection(geometry)
            )
            if overlap_measures:
                path_progress = min(overlap_measures)
            else:
                point_on_path, _point_on_obstacle = nearest_points(path_line, geometry)
                path_progress = float(path_line.project(point_on_path))
            # A broad-phase projected footprint can reach forward from a vehicle whose
            # physical body is behind us. It is not a lead unless it already overlaps.
            if path_progress <= 1e-6 and forward_offset <= 0.0:
                continue
            bumper_gap = max(
                0.0,
                path_progress - own_half_length - self._physical_lead_margin_m,
            )
            candidates.append((bumper_gap, lead_id))
        if not candidates:
            return None
        gap, lead_id = min(candidates, key=lambda item: (item[0], item[1]))
        return lead_id, gap

    def propagate_agents(
        self,
        ego_state,
        tspan,
        iteration,
        traffic_light_status,
        open_loop_detections,
        radius,
    ) -> None:
        """Stock nuPlan v1.2 propagation with one scoped virtual lead addition."""
        self.agent_occupancy.set("ego", ego_state.car_footprint.geometry)
        track_ids = []
        open_loop_types = {}
        observed_leads: Dict[str, str] = {}
        selected_physical_leads: Dict[str, str] = {}
        for track in open_loop_detections:
            # Cones are permanent static detections, while this background controller has no
            # lateral path planner with which to pass one.  Putting even a lane-edge cone in
            # the longitudinal occupancy query can therefore stop an IDM vehicle forever.
            # Keep the cone in the open-loop world/render output, but do not make it an IDM lead.
            if getattr(track, "tracked_object_type", None) == TrackedObjectType.TRAFFIC_CONE:
                continue
            token = str(track.track_token)
            track_ids.append(token)
            open_loop_types[token] = getattr(track, "tracked_object_type", None)
            self.agent_occupancy.insert(token, track.box.geometry)

        try:
            self._filter_agents_out_of_range(ego_state, radius)
            active_agents = [
                (agent_token, agent)
                for agent_token, agent in self.agents.items()
                if agent.is_active(iteration) and agent.has_valid_path()
            ]
            if self._routes_preplanned_iteration != int(iteration):
                for _agent_token, agent in active_agents:
                    agent.plan_route(traffic_light_status)

            path_lines = [_path_to_go_linestring(agent) for _, agent in active_agents]
            path_widths = np.asarray(
                [agent.width / 2.0 for _, agent in active_agents], dtype=np.float64
            )
            path_corridors = shapely_buffer(
                np.asarray(path_lines, dtype=object),
                path_widths,
                cap_style="flat",
            )

            for (agent_token, agent), agent_path, agent_corridor in zip(
                active_agents, path_lines, path_corridors
            ):
                ignored_open_loop = {}
                ignored_cycle_lead = None
                ignored_ego = None
                scoped_vehicle_only = self._uses_vehicle_only_leads(agent)
                if scoped_vehicle_only:
                    for token, tracked_type in open_loop_types.items():
                        if (
                            tracked_type != TrackedObjectType.VEHICLE
                            and self.agent_occupancy.contains(token)
                        ):
                            geometry = self.agent_occupancy.get(token)
                            if geometry.intersects(agent_corridor):
                                ignored_open_loop[token] = geometry
                    if ignored_open_loop:
                        self.agent_occupancy.remove(list(ignored_open_loop))
                        self._scoped_non_vehicle_lead_suppressions += len(
                            ignored_open_loop
                        )
                cycle_lead_id = self._wait_cycle_release_leads.get(str(agent_token))
                if (
                    cycle_lead_id is not None
                    and cycle_lead_id in self.agents
                    and self.agent_occupancy.contains(cycle_lead_id)
                ):
                    ignored_cycle_lead = (
                        cycle_lead_id, self.agent_occupancy.get(cycle_lead_id)
                    )
                    self.agent_occupancy.remove([cycle_lead_id])
                ego_velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d
                ego_speed = float(np.hypot(ego_velocity.x, ego_velocity.y))
                token = str(agent_token)
                release_until = self._ego_deadlock_release_until.get(token, -1)
                release_active = int(iteration) < int(release_until)
                waiting_on_ego = (
                    token in self._active_intersection_tokens
                    and self._last_selected_physical_leads.get(token) == "ego"
                    and float(agent.velocity) <= self._wait_cycle_stopped_speed_mps
                    and ego_speed <= self._wait_cycle_stopped_speed_mps
                    and not agent.polygon.intersects(ego_state.car_footprint.geometry)
                )
                if waiting_on_ego and not release_active:
                    first_tick = self._ego_deadlock_first_tick.setdefault(
                        token, int(iteration)
                    )
                    if (
                        int(iteration) - first_tick + 1
                        >= self._ego_deadlock_timeout_ticks
                    ):
                        release_active = True
                        self._ego_deadlock_release_until[token] = (
                            int(iteration) + 2 * self._wait_cycle_timeout_ticks
                        )
                        self._ego_deadlock_breaker_events += 1
                        logger.warning(
                            "[IDM-EGO-DEADLOCK] tick=%d npc=%s release=ego",
                            iteration,
                            token,
                        )
                elif not release_active:
                    self._ego_deadlock_first_tick.pop(token, None)
                    self._ego_deadlock_release_until.pop(token, None)
                if (
                    release_active
                    and token in self._active_intersection_tokens
                    and self.agent_occupancy.contains("ego")
                ):
                    ignored_ego = self.agent_occupancy.get("ego")
                    self.agent_occupancy.remove(["ego"])
                    self._ego_deadlock_breaker_ticks += 1
                # The configured Las Vegas circular-road scope is intentionally
                # vehicle-only: its highly curved rail intersects nearby signal
                # stop-line polygons which do not govern this movement.  Keeping
                # those virtual stop lines would negate the non-vehicle filter and
                # leave a bus stopped forever even after FIFO admission.  Physical
                # vehicles (and ego) remain in the occupancy map.
                stop_lines = (
                    [] if scoped_vehicle_only
                    else self._get_relevant_stop_lines(agent, traffic_light_status)
                )
                temporary_tokens = self._insert_stop_lines_into_occupancy_map(stop_lines)

                virtual_geometry = self._targeted_virtual_leads.get(str(agent_token))
                # The audited Las Vegas circular lanes are longitudinal car-following rails,
                # not independent at-grade junction approaches.  Treating every short map
                # connector as a fresh intersection makes the manager place virtual FIFO /
                # downstream blockers around the circle and can permanently lock a platoon.
                # In this narrowly configured scope, keep only physical vehicles (and ego) as
                # leads; actual bodies remain in occupancy and are ranked by route progress.
                if virtual_geometry is not None and not scoped_vehicle_only:
                    virtual_token = f"{self._VIRTUAL_PREFIX}{agent_token}"
                    self.agent_occupancy.set(virtual_token, virtual_geometry)
                    temporary_tokens.append(virtual_token)

                try:
                    intersecting_agents = self.agent_occupancy.intersects(
                        agent_corridor
                    )
                    assert intersecting_agents.contains(agent_token), (
                        "Agent's baseline does not intersect the agent itself"
                    )

                    if intersecting_agents.size > 1:
                        nearest_id, _nearest_polygon, relative_distance = (
                            intersecting_agents.get_nearest_entry_to(agent_token)
                        )
                        # Preserve stock nuPlan IDM spacing in the normal case.  Re-rank by
                        # physical progress only for the curve pathology where the selected
                        # projected footprint belongs to a same-flow vehicle behind us.
                        rerank_reason = self._selected_physical_lead_rerank_reason(
                            str(agent_token), agent, agent_path, str(nearest_id)
                        )
                        if scoped_vehicle_only:
                            # Always use Frenet-like progress on the remaining rail in the
                            # circular-road scope. Euclidean nearest-entry ordering is unstable
                            # on tight curves and can mistake a rear/adjacent vehicle for a lead.
                            rerank_reason = rerank_reason or "scoped_frenet"
                        if rerank_reason is not None:
                            corrected = self._nearest_physical_lead_on_path(
                                str(agent_token),
                                agent,
                                agent_path,
                                intersecting_agents.get_all_ids(),
                                agent_corridor,
                            )
                            if rerank_reason == "rear":
                                self._rear_lead_rejections += 1
                            elif rerank_reason == "adjacent":
                                self._adjacent_lead_rejections += 1
                            if corrected is None:
                                nearest_id = None
                            else:
                                nearest_id, relative_distance = corrected
                    else:
                        nearest_id = None

                    if nearest_id is not None:
                        selected_physical_leads[str(agent_token)] = str(nearest_id)
                        agent_heading = agent.to_se2().heading
                        if "ego" in nearest_id:
                            ego_velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d
                            longitudinal_velocity = np.hypot(ego_velocity.x, ego_velocity.y)
                            relative_heading = ego_state.rear_axle.heading - agent_heading
                        elif "stop_line" in nearest_id:
                            longitudinal_velocity = 0.0
                            relative_heading = 0.0
                        elif nearest_id in self.agents:
                            nearest_agent = self.agents[nearest_id]
                            longitudinal_velocity = nearest_agent.velocity
                            relative_heading = nearest_agent.to_se2().heading - agent_heading
                        else:
                            longitudinal_velocity = 0.0
                            relative_heading = 0.0
                        relative_heading = principal_value(relative_heading)
                        projected_velocity = rotate_angle(
                            StateSE2(longitudinal_velocity, 0, 0), relative_heading
                        ).x
                        length_rear = 0
                        if nearest_id in self.agents:
                            observed_leads[str(agent_token)] = str(nearest_id)
                    else:
                        projected_velocity = 0.0
                        relative_distance = agent.get_progress_to_go()
                        length_rear = agent.length / 2

                    agent.propagate(
                        IDMLeadAgentState(
                            progress=relative_distance,
                            velocity=projected_velocity,
                            length_rear=length_rear,
                        ),
                        tspan,
                    )
                    self.agent_occupancy.set(agent_token, agent.projected_footprint)
                finally:
                    removable = [
                        token for token in temporary_tokens if self.agent_occupancy.contains(token)
                    ]
                    if removable:
                        self.agent_occupancy.remove(removable)
                    for token, geometry in ignored_open_loop.items():
                        self.agent_occupancy.insert(token, geometry)
                    if ignored_cycle_lead is not None:
                        token, geometry = ignored_cycle_lead
                        if not self.agent_occupancy.contains(token):
                            self.agent_occupancy.insert(token, geometry)
                    if ignored_ego is not None and not self.agent_occupancy.contains("ego"):
                        self.agent_occupancy.insert("ego", ignored_ego)
            self._update_wait_cycles(iteration, observed_leads)
            self._last_selected_physical_leads = selected_physical_leads
        finally:
            removable_tracks = [
                token for token in track_ids if self.agent_occupancy.contains(token)
            ]
            if removable_tracks:
                self.agent_occupancy.remove(removable_tracks)
            self._targeted_virtual_leads = {}


class IntersectionManager:
    """NPC-only junction admission with buffered movement-conflict checks."""

    _VIRTUAL_PREFIX = "stop_line_intersection_manager_"

    def __init__(self, config: IntersectionManagerConfig):
        self.config = config
        self.states: Dict[str, IntersectionState] = {}
        self._intersection_geometry: Dict[str, object] = {}
        self._route_segment_intersection: Dict[str, Optional[str]] = {}
        self._passage_cache: Dict[str, Tuple[int, float, _Passage]] = {}
        self._controlled_intersections = set()
        self._holder_geometry: Dict[Tuple[str, str], object] = {}
        self._holder_movement_corridor: Dict[Tuple[str, str], object] = {}
        self._holder_movement_vector: Dict[Tuple[str, str], Tuple[float, float]] = {}
        # A permit reserves its route-aligned receiving slot as well as the junction token.
        # This is global across intersections, preventing two nearby junctions from both
        # committing an NPC into the same finite downstream space.
        self._downstream_reservations: Dict[Tuple[str, str], object] = {}
        self._downstream_reservation_vectors: Dict[
            Tuple[str, str], Tuple[float, float]
        ] = {}
        self._downstream_reservation_created_tick: Dict[Tuple[str, str], int] = {}
        self._reservation_revoked_holders: Set[Tuple[str, str]] = set()
        self._reservation_conflicts = 0
        self._reservation_ttl_revocations = 0
        self._peak_reservations = 0
        self._progress_checkpoint: Dict[Tuple[str, str], Point] = {}
        self._holder_last_progress_tick: Dict[Tuple[str, str], int] = {}
        self._last_holder_pose: Dict[Tuple[str, str], Point] = {}
        self._entered_holders = set()
        self._reported_stalls = set()
        self._recovered_active_holders: Set[Tuple[str, str]] = set()
        self._preexisting_active_stalls = 0
        # An active holder has already entered the physical junction, so revoking its
        # permit alone cannot restore liveness: stock IDM would leave the same stopped
        # body in occupancy.  The owning batch consumes these opt-in requests and removes
        # the vehicle atomically from both the IDM fleet and occupancy map.
        self._active_stall_retirement_requests: Set[Tuple[str, str]] = set()
        self._active_stall_retirements = 0
        self._last_decisions: Dict[Tuple[str, str], Tuple[bool, Optional[str]]] = {}
        self._hold_reason_ticks: Counter = Counter()
        self._stall_events = 0
        self._intersection_stopped_agent_ticks = 0
        self._candidate_ticks = 0
        self._contention_candidate_ticks = 0
        self._safety_candidate_ticks = 0
        self._unmanaged_candidate_ticks = 0
        # A same-flow vehicle in the receiving lane is normally handled by stock IDM.
        # It becomes a don't-block-the-box obstacle only after it has remained physically
        # stopped in this candidate's downstream corridor for the configured hysteresis.
        self._downstream_stopped_first_tick: Dict[Tuple[str, str, str], int] = {}
        self._downstream_observations_seen: Set[Tuple[str, str, str]] = set()
        self._downstream_moving_ignored_ticks = 0
        self._downstream_transient_ignored_ticks = 0
        self._downstream_persistent_blocker_ticks = 0
        self._downstream_cross_direction_blocker_ticks = 0
        self._downstream_insufficient_path_ticks = 0
        self._downstream_unclassified_blocker_ticks = 0
        self._rear_ego_hold_suppressions = 0
        self._updates = 0

    def precompute_or_cache_intersection_geometry(
        self,
        route: Sequence[object],
        controlled_lane_connector_ids: FrozenSet[str] = frozenset(),
    ) -> Dict[str, object]:
        """Get physical junctions only through map objects already on an NPC's route rail."""
        result = {}
        for segment in route:
            segment_id = str(getattr(segment, "id", id(segment)))
            if segment_id in self._route_segment_intersection:
                cached_intersection_id = self._route_segment_intersection[segment_id]
                if cached_intersection_id is not None:
                    result[cached_intersection_id] = self._intersection_geometry[cached_intersection_id]
                    if segment_id in controlled_lane_connector_ids:
                        self._controlled_intersections.add(cached_intersection_id)
                continue
            try:
                parent = segment.parent
                intersection = parent.intersection
                if callable(intersection):
                    intersection = intersection()
            except (AttributeError, KeyError, NotImplementedError, TypeError):
                self._route_segment_intersection[segment_id] = None
                continue
            if intersection is None:
                self._route_segment_intersection[segment_id] = None
                continue
            intersection_id = str(intersection.id)
            geometry = self._intersection_geometry.get(intersection_id)
            if geometry is None:
                geometry = intersection.polygon
                if geometry is None or geometry.is_empty or not geometry.is_valid:
                    self._route_segment_intersection[segment_id] = None
                    continue
                self._intersection_geometry[intersection_id] = geometry
            self._route_segment_intersection[segment_id] = intersection_id
            if segment_id in controlled_lane_connector_ids:
                self._controlled_intersections.add(intersection_id)
            result[intersection_id] = geometry
        return result

    def active_npc_ids(self) -> Set[str]:
        """Return current committed holders without exposing mutable state."""
        return {
            str(token)
            for state in self.states.values()
            for token in state.active_npc_ids
        }

    @staticmethod
    def _geometry_measures(line: LineString, geometry: object) -> Sequence[float]:
        coordinates = []

        def visit(item: object) -> None:
            if item is None or item.is_empty:
                return
            if hasattr(item, "geoms"):
                for child in item.geoms:
                    visit(child)
                return
            item_coordinates = getattr(item, "coords", None)
            if item_coordinates is None or len(item_coordinates) == 0:
                return
            coordinates.append(item_coordinates[0])
            if len(item_coordinates) > 1:
                coordinates.append(item_coordinates[-1])

        visit(geometry)
        if not coordinates:
            return []
        return [
            float(measure)
            for measure in line_locate_point(line, points(coordinates))
        ]

    def _make_virtual_lead(self, line: LineString, distance: float, width: float) -> object:
        center = line.interpolate(max(0.0, min(distance, line.length)))
        before = line.interpolate(max(0.0, distance - 0.25))
        after = line.interpolate(min(line.length, distance + 0.25))
        dx, dy = after.x - before.x, after.y - before.y
        norm = hypot(dx, dy)
        if norm < 1e-6:
            dx, dy, norm = 1.0, 0.0, 1.0
        nx, ny = -dy / norm, dx / norm
        half_width = width / 2.0 + self.config.virtual_lead_lateral_margin_m
        crossbar = LineString(
            [
                (center.x - nx * half_width, center.y - ny * half_width),
                (center.x + nx * half_width, center.y + ny * half_width),
            ]
        )
        return crossbar.buffer(
            self.config.virtual_lead_depth_m / 2.0,
            cap_style=CAP_STYLE.flat,
        )

    def _next_passage(self, npc_id: str, agent: object) -> Optional[_Passage]:
        route = agent.get_route()
        junctions = self.precompute_or_cache_intersection_geometry(route)
        if not junctions:
            return None
        route_path = getattr(agent, "_path", None)
        progress = getattr(agent, "progress", None)
        cached = self._passage_cache.get(npc_id)
        if route_path is not None and progress is not None and cached is not None:
            cached_path_id, cached_progress, cached_passage = cached
            progress_delta = float(progress) - cached_progress
            if (
                cached_path_id == id(route_path)
                and progress_delta >= 0.0
                and cached_passage.distance_to_entry_m > progress_delta
                and not agent.polygon.intersects(cached_passage.junction_geometry)
            ):
                return replace(
                    cached_passage,
                    distance_to_entry_m=cached_passage.distance_to_entry_m - progress_delta,
                    distance_to_exit_m=cached_passage.distance_to_exit_m - progress_delta,
                )
        line = _path_to_go_linestring(agent)
        if line is None or line.is_empty or line.length < 1e-3:
            return None

        passages = []
        for intersection_id, junction in junctions.items():
            within_junction = line.intersection(junction)
            measures = self._geometry_measures(line, within_junction)
            if not measures:
                continue
            entry = min(measures)
            exit_distance = max(measures)
            if junction.covers(Point(line.coords[0])):
                entry = 0.0
            if exit_distance - entry < 0.1:
                continue

            movement_line = substring(line, entry, exit_distance)
            movement_coordinates = list(movement_line.coords)
            movement_dx = movement_coordinates[-1][0] - movement_coordinates[0][0]
            movement_dy = movement_coordinates[-1][1] - movement_coordinates[0][1]
            movement_norm = max(hypot(movement_dx, movement_dy), 1e-6)
            movement_vector = (
                movement_dx / movement_norm,
                movement_dy / movement_norm,
            )
            movement_corridor = movement_line.buffer(
                agent.width / 2.0 + self.config.movement_corridor_margin_m,
                cap_style=CAP_STYLE.flat,
            ).intersection(junction)

            required_storage = agent.length + self.config.downstream_margin_m
            downstream_start = exit_distance + self.config.downstream_start_margin_m
            downstream_end = downstream_start + required_storage
            downstream_corridor = None
            downstream_vector = None
            if downstream_end <= line.length:
                downstream_line = substring(line, downstream_start, downstream_end)
                downstream_coordinates = list(downstream_line.coords)
                downstream_dx = downstream_coordinates[-1][0] - downstream_coordinates[0][0]
                downstream_dy = downstream_coordinates[-1][1] - downstream_coordinates[0][1]
                downstream_norm = max(hypot(downstream_dx, downstream_dy), 1e-6)
                downstream_vector = (
                    downstream_dx / downstream_norm,
                    downstream_dy / downstream_norm,
                )
                downstream_corridor = downstream_line.buffer(
                    agent.width / 2.0 + self.config.downstream_lateral_margin_m,
                    cap_style=CAP_STYLE.flat,
                )

            stop_distance = max(0.0, entry - self.config.stop_offset_m)
            passages.append(
                _Passage(
                    npc_id=npc_id,
                    intersection_id=intersection_id,
                    junction_geometry=junction,
                    movement_corridor=movement_corridor,
                    movement_vector=movement_vector,
                    downstream_corridor=downstream_corridor,
                    downstream_vector=downstream_vector,
                    virtual_lead_geometry=self._make_virtual_lead(
                        line, stop_distance, float(agent.width)
                    ),
                    distance_to_entry_m=max(0.0, entry - agent.length / 2.0),
                    distance_to_exit_m=exit_distance,
                    lane_connector_ids=tuple(
                        str(getattr(segment, "id", id(segment)))
                        for segment in route
                        if self._route_segment_intersection.get(
                            str(getattr(segment, "id", id(segment)))
                        ) == intersection_id
                    ),
                )
            )
        result = min(passages, key=lambda item: item.distance_to_entry_m) if passages else None
        if (
            result is not None
            and route_path is not None
            and progress is not None
            and result.distance_to_entry_m > 0.0
            and not agent.polygon.intersects(result.junction_geometry)
        ):
            self._passage_cache[npc_id] = (id(route_path), float(progress), result)
        else:
            self._passage_cache.pop(npc_id, None)
        return result

    def collect_candidates(
        self,
        npc_agents: Mapping[str, object],
        passages: Optional[Mapping[str, _Passage]] = None,
    ) -> Dict[str, _Passage]:
        candidates = {}
        for raw_npc_id, agent in npc_agents.items():
            npc_id = str(raw_npc_id)
            if not agent.has_valid_path():
                continue
            passage = passages.get(npc_id) if passages is not None else self._next_passage(npc_id, agent)
            if passage is None or agent.polygon.intersects(passage.junction_geometry):
                continue
            braking_gate = (
                float(agent.velocity) ** 2 / (2.0 * self.config.decel_max)
                + float(agent.velocity) * self.config.reaction_margin_s
                + self.config.distance_margin_m
            )
            if passage.distance_to_entry_m <= braking_gate:
                candidates[npc_id] = passage
        return candidates

    def _downstream_block_kind(
        self,
        passage: _Passage,
        occupancy: object,
        npc_id: str,
        npc_agents: Optional[Mapping[str, object]] = None,
        tick: int = 0,
    ) -> Optional[str]:
        """Require a full vehicle length plus margin after the junction exit.

        nuPlan's IDM occupancy stores ``projected_footprint`` polygons extended by
        velocity * headway.  That is appropriate for longitudinal car-following, but it
        is not an observation of currently occupied downstream storage.  Use each known
        NPC's current physical polygon here and retain occupancy geometry only for ego
        and open-loop objects.
        """
        if passage.downstream_corridor is None or passage.downstream_corridor.is_empty:
            # A truncated route rail is absence of evidence, not evidence that the
            # receiving lane is occupied. Stock IDM is the safe no-op fallback and the
            # diagnostic counter makes this geometry coverage gap measurable.
            self._downstream_insufficient_path_ticks += 1
            return None
        current_agents = {
            str(token): agent for token, agent in (npc_agents or {}).items()
        }
        block_kind = None
        persistence_ticks = max(
            1,
            int(round(
                self.config.downstream_stopped_persistence_s / self.config.sim_dt
            )),
        )
        for token in occupancy.get_all_ids():
            token = str(token)
            if token == npc_id or token.startswith(self._VIRTUAL_PREFIX):
                continue
            agent = current_agents.get(token)
            geometry = agent.polygon if agent is not None else occupancy.get(token)
            if geometry.is_empty or not geometry.intersects(passage.downstream_corridor):
                continue

            # Ego and open-loop objects have no NPC route/velocity contract here. Keep
            # treating their current physical occupancy as an immediate conservative block.
            if agent is None:
                self._downstream_unclassified_blocker_ticks += 1
                block_kind = block_kind or "unclassified_physical"
                continue

            heading = float(agent.to_se2().heading)
            heading_vector = (cos(heading), sin(heading))
            receiving_vector = passage.downstream_vector or passage.movement_vector
            if not self._same_direction(receiving_vector, heading_vector):
                self._downstream_cross_direction_blocker_ticks += 1
                block_kind = block_kind or "cross_direction_physical"
                continue

            observation_key = (passage.intersection_id, npc_id, token)
            if float(agent.velocity) > self.config.stopped_speed_mps:
                # A moving same-flow lead is ordinary car-following, not unavailable
                # storage. Reset immediately so a later stop must satisfy hysteresis anew.
                self._downstream_stopped_first_tick.pop(observation_key, None)
                self._downstream_moving_ignored_ticks += 1
                continue

            self._downstream_observations_seen.add(observation_key)
            first_tick = self._downstream_stopped_first_tick.setdefault(
                observation_key, int(tick)
            )
            if int(tick) - first_tick + 1 < persistence_ticks:
                self._downstream_transient_ignored_ticks += 1
                continue
            self._downstream_persistent_blocker_ticks += 1
            block_kind = block_kind or "persistent_same_flow"

        own_key = (passage.intersection_id, npc_id)
        for key, geometry in self._downstream_reservations.items():
            if key == own_key or not geometry.intersects(passage.downstream_corridor):
                continue
            vector = self._downstream_reservation_vectors.get(key)
            if vector is not None and self._same_direction(
                passage.downstream_vector or passage.movement_vector, vector
            ):
                # Stock IDM safely meters a same-flow receiving lane. Reserving it here
                # recreated the V3.2 queue amplification that contention_only avoided.
                continue
            self._reservation_conflicts += 1
            block_kind = block_kind or "conflicting_reservation"
        return block_kind

    def has_downstream_storage(
        self,
        passage: _Passage,
        occupancy: object,
        npc_id: str,
        npc_agents: Optional[Mapping[str, object]] = None,
        tick: int = 0,
    ) -> bool:
        return self._downstream_block_kind(
            passage, occupancy, npc_id, npc_agents, tick
        ) is None

    def _ego_envelope(self, ego_state: EgoSafetyState) -> object:
        dx = ego_state.velocity_x * self.config.ego_envelope_horizon_s
        dy = ego_state.velocity_y * self.config.ego_envelope_horizon_s
        translated = translate(ego_state.footprint, xoff=dx, yoff=dy)
        return unary_union([ego_state.footprint, translated]).convex_hull.buffer(
            self.config.ego_envelope_margin_m
        )

    def ego_blocks_entry(
        self,
        passage: _Passage,
        ego_state: EgoSafetyState,
        ego_envelope: Optional[object] = None,
    ) -> bool:
        envelope = (
            ego_envelope if ego_envelope is not None else self._ego_envelope(ego_state)
        )
        return envelope.intersects(passage.movement_corridor)

    def ego_approaches_candidate_from_rear(
        self,
        passage: _Passage,
        agent: object,
        ego_state: EgoSafetyState,
        ego_envelope: Optional[object] = None,
    ) -> bool:
        """Detect unsafe new virtual stops using current ego state only.

        The manager never commands ego. If ego is already closing from behind and its
        constant-current-velocity envelope reaches the candidate, injecting a new stop
        lead can create the rear-end collision observed in the V3.2 pilot. Same-direction
        stock IDM remains in control in that narrow case.
        """
        speed = hypot(float(ego_state.velocity_x), float(ego_state.velocity_y))
        ego_center = ego_state.footprint.centroid
        candidate_center = agent.polygon.centroid
        relative_x = ego_center.x - candidate_center.x
        relative_y = ego_center.y - candidate_center.y
        longitudinal = (
            relative_x * passage.movement_vector[0]
            + relative_y * passage.movement_vector[1]
        )
        if longitudinal >= 0.0:
            return False
        if speed <= self.config.stopped_speed_mps:
            lateral = abs(
                relative_x * passage.movement_vector[1]
                - relative_y * passage.movement_vector[0]
            )
            return bool(
                -longitudinal <= self.config.stopped_rear_ego_release_distance_m
                and lateral <= self.config.stopped_rear_ego_release_lateral_m
            )
        ego_vector = (
            float(ego_state.velocity_x) / speed,
            float(ego_state.velocity_y) / speed,
        )
        if not self._same_direction(passage.movement_vector, ego_vector):
            return False
        envelope = (
            ego_envelope if ego_envelope is not None else self._ego_envelope(ego_state)
        )
        return envelope.intersects(agent.polygon)

    def select_grant(
        self,
        tick: int,
        state: IntersectionState,
        candidates: Sequence[str],
    ) -> Optional[str]:
        if not candidates:
            return None

        def priority(npc_id: str) -> Tuple[float, int, str]:
            arrival = state.arrival_tick[npc_id]
            waited = tick - arrival
            score = -arrival + self.config.wait_bonus_per_tick * waited
            return (-score, arrival, npc_id)

        return min(candidates, key=priority)

    @staticmethod
    def _same_direction(
        first: Tuple[float, float], second: Tuple[float, float]
    ) -> bool:
        # A 30-degree tolerance leaves ordinary lane following/parallel flow on stock
        # IDM while turns, merges from distinct approaches, and crossings stay managed.
        return first[0] * second[0] + first[1] * second[1] >= 0.866

    def _movements_conflict(
        self,
        first: object,
        second: object,
        first_vector: Optional[Tuple[float, float]] = None,
        second_vector: Optional[Tuple[float, float]] = None,
    ) -> bool:
        """V1 serializes all movements; V2 admits disjoint buffered corridors in parallel."""
        if not self.config.allow_non_conflicting_movements:
            return True
        if not first.intersects(second):
            return False
        if (
            self.config.contention_only
            and first_vector is not None
            and second_vector is not None
            and self._same_physical_flow(
                first, second, first_vector, second_vector
            )
        ):
            return False
        return True

    def _same_physical_flow(
        self,
        first: object,
        second: object,
        first_vector: Optional[Tuple[float, float]],
        second_vector: Optional[Tuple[float, float]],
    ) -> bool:
        """Distinguish a duplicated same lane from sibling-ID merge geometry."""
        if (
            first_vector is None
            or second_vector is None
            or not self._same_direction(first_vector, second_vector)
        ):
            return False
        minimum_area = min(float(first.area), float(second.area))
        if minimum_area <= 1e-6:
            return False
        # Duplicated/segmented map-intersection IDs can describe the same through lane.
        # Their corridors substantially coincide. A shallow-angle merge only clips a small
        # fraction and must remain serialized even when its tangent differs by under 30 deg.
        return float(first.intersection(second).area) / minimum_area >= 0.5

    def _downstream_corridors_conflict(self, first: _Passage, second: _Passage) -> bool:
        return bool(
            first.downstream_corridor is not None
            and second.downstream_corridor is not None
            and first.downstream_corridor.intersects(second.downstream_corridor)
            and not (
                self.config.contention_only
                and self._same_physical_flow(
                    first.downstream_corridor,
                    second.downstream_corridor,
                    first.downstream_vector or first.movement_vector,
                    second.downstream_vector or second.movement_vector,
                )
            )
        )

    def _contended_candidates(
        self,
        intersection_id: str,
        npc_ids: Sequence[str],
        candidates: Mapping[str, _Passage],
        state: IntersectionState,
    ) -> Set[str]:
        """Identify entries where the simulator-side shield can change an outcome.

        A lone, non-conflicting entry stays on stock IDM. Once another movement competes
        for the same conflict area or receiving slot, both become managed. An NPC already
        inside the junction is treated as contention for an overlapping new movement.
        """
        if not self.config.contention_only:
            return set(npc_ids)

        contended = set(npc_ids) & set(state.granted_npc_ids)
        active_movements = [
            (
                holder,
                self._holder_movement_corridor[(intersection_id, holder)],
                self._holder_movement_vector.get((intersection_id, holder)),
                False,
            )
            for holder in state.active_npc_ids
            if (intersection_id, holder) in self._holder_movement_corridor
        ]
        active_movements.extend(self._foreign_held_movements(intersection_id))
        reservations = [
            (geometry, self._downstream_reservation_vectors.get(key))
            for key, geometry in self._downstream_reservations.items()
        ]
        for npc_id in npc_ids:
            passage = candidates[npc_id]
            if any(
                self._movements_conflict(
                    passage.movement_corridor,
                    corridor,
                    passage.movement_vector,
                    vector,
                )
                and not (
                    foreign
                    and self._same_physical_flow(
                        passage.movement_corridor,
                        corridor,
                        passage.movement_vector,
                        vector,
                    )
                )
                for holder, corridor, vector, foreign in active_movements
                if holder != npc_id
            ) or any(
                passage.downstream_corridor is not None
                and passage.downstream_corridor.intersects(reservation)
                and (
                    vector is None
                    or not self._same_direction(
                        passage.downstream_vector or passage.movement_vector,
                        vector,
                    )
                )
                for reservation, vector in reservations
            ):
                contended.add(npc_id)

        for index, first_id in enumerate(npc_ids):
            first = candidates[first_id]
            for second_id in npc_ids[index + 1:]:
                second = candidates[second_id]
                if self._movements_conflict(
                    first.movement_corridor,
                    second.movement_corridor,
                    first.movement_vector,
                    second.movement_vector,
                ) or self._downstream_corridors_conflict(first, second):
                    contended.update((first_id, second_id))
        return contended

    def _foreign_held_movements(self, intersection_id: str) -> list:
        """Return live corridors owned by overlapping sibling map intersections.

        nuPlan occasionally assigns adjacent/overlapping connector polygons to different
        parent-intersection IDs.  Admission state remains keyed by the map IDs, but safety
        cannot stop at that bookkeeping boundary: two physical movement corridors which
        intersect must see one another.  Non-overlapping intersections are a no-op because
        ``_movements_conflict`` performs the geometric test at the call site.
        """
        movements = []
        for (other_intersection, holder), corridor in self._holder_movement_corridor.items():
            if other_intersection == intersection_id:
                continue
            state = self.states.get(other_intersection)
            if state is None or holder not in (
                set(state.active_npc_ids) | set(state.granted_npc_ids)
            ):
                continue
            movements.append((
                holder,
                corridor,
                self._holder_movement_vector.get((other_intersection, holder)),
                True,
            ))
        return movements

    def _install_grant(
        self,
        tick: int,
        intersection_id: str,
        npc_id: str,
        passage: _Passage,
        agent: object,
        state: IntersectionState,
    ) -> None:
        key = (intersection_id, npc_id)
        state.granted_npc_ids.add(npc_id)
        state.last_progress_tick = tick
        self._holder_geometry[key] = passage.junction_geometry
        self._holder_movement_corridor[key] = passage.movement_corridor
        self._holder_movement_vector[key] = passage.movement_vector
        if passage.distance_to_entry_m <= self.config.reservation_commit_distance_m:
            self._publish_downstream_reservation(key, passage, tick)
        pose = agent.to_se2()
        point = Point(pose.x, pose.y)
        self._progress_checkpoint[key] = point
        self._holder_last_progress_tick[key] = tick
        self._last_holder_pose[key] = point
        logger.info(
            "[IDM-IM] tick=%d intersection=%s npc=%s permit=granted",
            tick,
            intersection_id,
            npc_id,
        )

    def _mark_progress(self, key: Tuple[str, str], point: Point, state: IntersectionState, tick: int) -> None:
        checkpoint = self._progress_checkpoint.get(key)
        if checkpoint is None or checkpoint.distance(point) >= self.config.progress_epsilon_m:
            self._progress_checkpoint[key] = point
            self._holder_last_progress_tick[key] = tick
            state.last_progress_tick = tick
            self._reported_stalls.discard(key)
            self._active_stall_retirement_requests.discard(key)
        self._last_holder_pose[key] = point

    def _record_stall(self, key: Tuple[str, str], tick: int, kind: str) -> None:
        if key in self._reported_stalls:
            return
        self._reported_stalls.add(key)
        self._stall_events += 1
        if kind == "active" and key in self._recovered_active_holders:
            self._preexisting_active_stalls += 1
        if kind == "active" and self.config.retire_active_stalls:
            self._active_stall_retirement_requests.add(key)
        logger.warning(
            "[IDM-IM] tick=%d intersection=%s npc=%s stall=%s",
            tick,
            key[0],
            key[1],
            kind,
        )

    def consume_active_stall_retirements(
        self, eligible_npc_ids: Optional[Set[str]] = None
    ) -> Set[str]:
        """Return eligible stalled holders, retaining ineligible requests for later.

        ``None`` preserves the broad opt-in liveness behavior.  Supplying a set lets the
        caller impose a physical lifecycle condition (currently: source track has ended)
        without teaching this geometry-only manager about dataset observations.
        """
        requests = {
            key
            for key in self._active_stall_retirement_requests
            if eligible_npc_ids is None or key[1] in eligible_npc_ids
        }
        self._active_stall_retirement_requests.difference_update(requests)
        self._active_stall_retirements += len(requests)
        for intersection_id, npc_id in requests:
            self._clear_holder(intersection_id, npc_id)
            state = self.states.get(intersection_id)
            if state is not None:
                state.arrival_tick.pop(npc_id, None)
        return {npc_id for _intersection_id, npc_id in requests}

    def _clear_holder(self, intersection_id: str, npc_id: str) -> None:
        key = (intersection_id, npc_id)
        state = self.states.get(intersection_id)
        if state is not None:
            state.active_npc_ids.discard(npc_id)
            state.granted_npc_ids.discard(npc_id)
        self._holder_geometry.pop(key, None)
        self._holder_movement_corridor.pop(key, None)
        self._holder_movement_vector.pop(key, None)
        self._downstream_reservations.pop(key, None)
        self._downstream_reservation_vectors.pop(key, None)
        self._downstream_reservation_created_tick.pop(key, None)
        self._reservation_revoked_holders.discard(key)
        self._progress_checkpoint.pop(key, None)
        self._holder_last_progress_tick.pop(key, None)
        self._last_holder_pose.pop(key, None)
        self._entered_holders.discard(key)
        self._reported_stalls.discard(key)
        self._recovered_active_holders.discard(key)
        self._active_stall_retirement_requests.discard(key)

    def _publish_downstream_reservation(
        self,
        key: Tuple[str, str],
        passage: _Passage,
        tick: int,
    ) -> None:
        """Publish a short-lived receiving-slot claim only after entry is committed."""
        if (
            key in self._reservation_revoked_holders
            or key in self._downstream_reservations
            or passage.downstream_corridor is None
        ):
            return
        self._downstream_reservations[key] = passage.downstream_corridor
        self._downstream_reservation_vectors[key] = (
            passage.downstream_vector or passage.movement_vector
        )
        self._downstream_reservation_created_tick[key] = int(tick)
        self._peak_reservations = max(
            self._peak_reservations, len(self._downstream_reservations)
        )

    def _expire_downstream_reservations(self, tick: int) -> None:
        """Revoke only the reservation after its TTL; the holder permit remains intact."""
        ttl_ticks = max(1, int(round(self.config.reservation_ttl_s / self.config.sim_dt)))
        expired = [
            key
            for key, created_tick in self._downstream_reservation_created_tick.items()
            if int(tick) - created_tick >= ttl_ticks
        ]
        for key in expired:
            self._downstream_reservations.pop(key, None)
            self._downstream_reservation_vectors.pop(key, None)
            self._downstream_reservation_created_tick.pop(key, None)
            self._reservation_revoked_holders.add(key)
            self._reservation_ttl_revocations += 1

    def update_and_release(
        self,
        tick: int,
        npc_agents: Mapping[str, object],
        passages: Mapping[str, Optional[_Passage]],
        recoverable_npc_ids: Optional[Set[str]] = None,
    ) -> None:
        stall_ticks = max(1, int(round(self.config.stall_timeout_s / self.config.sim_dt)))
        agents = {str(token): agent for token, agent in npc_agents.items()}

        for intersection_id, state in list(self.states.items()):
            for active in list(state.active_npc_ids):
                agent = agents.get(active)
                key = (intersection_id, active)
                junction = self._holder_geometry.get(key)
                if agent is None or junction is None:
                    self._clear_holder(intersection_id, active)
                else:
                    point = Point(agent.to_se2().x, agent.to_se2().y)
                    inside = agent.polygon.intersects(junction)
                    if inside:
                        self._entered_holders.add(key)
                        self._mark_progress(key, point, state, tick)
                        if float(agent.velocity) <= self.config.stopped_speed_mps:
                            self._intersection_stopped_agent_ticks += 1
                        if tick - self._holder_last_progress_tick.get(key, tick) >= stall_ticks:
                            self._record_stall(key, tick, "active")
                    elif key in self._entered_holders:
                        logger.info(
                            "[IDM-IM] tick=%d intersection=%s npc=%s permit=released",
                            tick,
                            intersection_id,
                            active,
                        )
                        self._clear_holder(intersection_id, active)
                        state.arrival_tick.pop(active, None)

            for granted in list(state.granted_npc_ids):
                agent = agents.get(granted)
                key = (intersection_id, granted)
                junction = self._holder_geometry.get(key)
                if agent is None or junction is None:
                    self._clear_holder(intersection_id, granted)
                else:
                    pose = agent.to_se2()
                    point = Point(pose.x, pose.y)
                    if agent.polygon.intersects(junction):
                        state.active_npc_ids.add(granted)
                        state.granted_npc_ids.discard(granted)
                        self._entered_holders.add(key)
                        passage = passages.get(granted)
                        if passage is not None:
                            self._publish_downstream_reservation(key, passage, tick)
                        self._mark_progress(key, point, state, tick)
                        logger.info(
                            "[IDM-IM] tick=%d intersection=%s npc=%s permit=active",
                            tick,
                            intersection_id,
                            granted,
                        )
                    else:
                        previous = self._last_holder_pose.get(key)
                        passage = passages.get(granted)
                        if (
                            passage is not None
                            and passage.distance_to_entry_m
                            <= self.config.reservation_commit_distance_m
                        ):
                            self._publish_downstream_reservation(key, passage, tick)
                        swept_junction = (
                            previous is not None
                            and LineString(
                                [(previous.x, previous.y), (point.x, point.y)]
                            ).intersects(junction)
                        )
                        if swept_junction and (
                            passage is None or passage.intersection_id != intersection_id
                        ):
                            self._clear_holder(intersection_id, granted)
                            state.arrival_tick.pop(granted, None)
                        else:
                            self._mark_progress(key, point, state, tick)
                            if tick - self._holder_last_progress_tick.get(key, tick) >= stall_ticks:
                                self._record_stall(key, tick, "granted_before_entry")
                                self._clear_holder(intersection_id, granted)
                                # Recompute FIFO instead of immediately re-granting the same stalled NPC.
                                state.arrival_tick[granted] = tick

        # Recover conservatively when the manager starts while an NPC is already in a junction.
        inside_by_intersection: Dict[str, list] = {}
        for npc_id, agent in agents.items():
            if recoverable_npc_ids is not None and npc_id not in recoverable_npc_ids:
                continue
            # ``update`` caches a result (including None) for every active agent. Keep
            # the fallback for direct callers/tests that supply only a partial map.
            passage = (
                passages[npc_id]
                if npc_id in passages
                else self._next_passage(npc_id, agent)
            )
            if passage is None or not agent.polygon.intersects(passage.junction_geometry):
                continue
            inside_by_intersection.setdefault(passage.intersection_id, []).append((npc_id, passage))

        for intersection_id, occupants in inside_by_intersection.items():
            state = self.states.setdefault(intersection_id, IntersectionState(last_progress_tick=tick))
            for npc_id, passage in occupants:
                if npc_id in state.active_npc_ids:
                    continue
                state.granted_npc_ids.discard(npc_id)
                state.active_npc_ids.add(npc_id)
                key = (intersection_id, npc_id)
                self._holder_geometry[key] = passage.junction_geometry
                self._holder_movement_corridor[key] = passage.movement_corridor
                self._holder_movement_vector[key] = passage.movement_vector
                self._recovered_active_holders.add(key)
                self._publish_downstream_reservation(key, passage, tick)
                self._entered_holders.add(key)
                pose = agents[npc_id].to_se2()
                self._mark_progress(key, Point(pose.x, pose.y), state, tick)

    def update(
        self,
        world_state: IntersectionWorldState,
        npc_agents: Mapping[str, object],
        ego_state: EgoSafetyState,
    ) -> Dict[str, AdmissionDecision]:
        """Return enter/hold decisions before the IDM propagation of this tick."""
        if not isinstance(ego_state, EgoSafetyState):
            raise TypeError(
                "IntersectionManager accepts only EgoSafetyState; ego route, future trajectory, "
                "HLC, reservation, priority, and planner output are outside this API"
            )
        tick = int(world_state.tick)
        agents = {str(token): agent for token, agent in npc_agents.items()}

        # Register signal-controlled physical intersections from the route rails before
        # computing any passage.  Stock IDM already owns traffic-light compliance there;
        # serializing those intersections destroys throughput and can create network queues.
        # This set is tick-local: an all-UNKNOWN signal group can be downgraded to the
        # unsignalized manager rules after previously reporting a valid signal.
        self._controlled_intersections.clear()
        for agent in agents.values():
            if agent.has_valid_path():
                self.precompute_or_cache_intersection_geometry(
                    agent.get_route(), world_state.controlled_lane_connector_ids
                )
        if not self.config.protect_signal_conflicts:
            for intersection_id in self._controlled_intersections:
                state = self.states.get(intersection_id)
                if state is not None:
                    for holder in set(state.active_npc_ids) | set(state.granted_npc_ids):
                        self._clear_holder(intersection_id, holder)
                    self.states.pop(intersection_id, None)

        active_agents = {
            token: agent for token, agent in agents.items() if agent.is_active(tick)
        }
        self._downstream_observations_seen = set()
        # Cache both raw passages and negative lookups. Recovery historically sees a
        # signal-controlled passage too (candidate admission below does not), so retain
        # that subtle behavior while avoiding the duplicate geometry calculation.
        passage_by_agent: Dict[str, Optional[_Passage]] = {}
        for token, agent in active_agents.items():
            passage_by_agent[token] = self._next_passage(token, agent)
        all_passages = {
            token: passage
            for token, passage in passage_by_agent.items()
            if passage is not None
            and (
                self.config.protect_signal_conflicts
                or passage.intersection_id not in self._controlled_intersections
            )
        }
        self._expire_downstream_reservations(tick)
        # A red/unknown signalized approach can geometrically overlap nuPlan's broad physical
        # intersection polygon while it is merely waiting at its stock stop line.  Recovering
        # such an NPC as an already-entered holder lets it block a genuinely green movement.
        # Existing holders are still tracked above (including across a phase change); this gate
        # affects only state reconstructed without a prior manager grant.
        controlled = world_state.controlled_lane_connector_ids
        green = world_state.green_lane_connector_ids
        recoverable_npc_ids = set()
        for token, passage in passage_by_agent.items():
            if passage is None:
                continue
            passage_connectors = set(passage.lane_connector_ids)
            signalized = bool(passage_connectors & controlled)
            if not signalized or (
                self.config.protect_signal_conflicts
                and bool(passage_connectors & green)
            ):
                recoverable_npc_ids.add(token)
        self.update_and_release(
            tick,
            active_agents,
            passage_by_agent,
            recoverable_npc_ids=recoverable_npc_ids,
        )
        candidates = self.collect_candidates(active_agents, all_passages)
        if self.config.protect_signal_conflicts:
            controlled = world_state.controlled_lane_connector_ids
            green = world_state.green_lane_connector_ids
            candidates = {
                token: passage
                for token, passage in candidates.items()
                # Stock IDM owns red/unknown signal compliance. The supplemental
                # manager arbitrates only a route whose relevant controlled connector
                # is green, preventing conflicting simultaneous greens from colliding.
                if not (set(passage.lane_connector_ids) & controlled)
                or bool(set(passage.lane_connector_ids) & green)
            }
        ego_envelope = self._ego_envelope(ego_state)

        grouped: Dict[str, list] = {}
        for npc_id, passage in candidates.items():
            grouped.setdefault(passage.intersection_id, []).append(npc_id)
            self.states.setdefault(
                passage.intersection_id, IntersectionState(last_progress_tick=tick)
            )

        decisions: Dict[str, AdmissionDecision] = {}
        for intersection_id, all_npc_ids in grouped.items():
            state = self.states[intersection_id]
            contended = self._contended_candidates(
                intersection_id, all_npc_ids, candidates, state
            )
            # V3.2: ``contention_only`` is a throughput optimization, not an escape
            # hatch from the two simulator-side safety rules.  V3.1 evaluated ego and
            # downstream occupancy only *after* a candidate had been classified as
            # contended.  Consequently a lone arrival (and same-direction arrivals)
            # could enter an occupied receiving slot without ever reaching those
            # checks.  Pre-arm only candidates which are blocked by current geometry;
            # an empty intersection is still a strict stock-IDM no-op.
            safety_blocks: Dict[str, str] = {}
            for npc_id in all_npc_ids:
                passage = candidates[npc_id]
                rear_ego_approach = self.ego_approaches_candidate_from_rear(
                    passage,
                    active_agents[npc_id],
                    ego_state,
                    ego_envelope,
                )
                downstream_block_kind = self._downstream_block_kind(
                    passage,
                    world_state.occupancy,
                    npc_id,
                    active_agents,
                    tick,
                )
                if (
                    self.ego_blocks_entry(passage, ego_state, ego_envelope)
                    and not rear_ego_approach
                ):
                    safety_blocks[npc_id] = "ego_safety_envelope"
                elif (
                    downstream_block_kind is not None
                    and not rear_ego_approach
                ):
                    safety_blocks[npc_id] = "downstream_blocked"
                elif rear_ego_approach and downstream_block_kind is not None:
                    self._rear_ego_hold_suppressions += 1
            managed = set(contended) | set(safety_blocks)
            self._candidate_ticks += len(all_npc_ids)
            self._contention_candidate_ticks += len(contended)
            self._safety_candidate_ticks += len(safety_blocks)
            self._unmanaged_candidate_ticks += len(set(all_npc_ids) - managed)
            for npc_id in all_npc_ids:
                if npc_id not in managed:
                    decisions[npc_id] = AdmissionDecision(
                        npc_id, intersection_id, True
                    )
                    state.arrival_tick.pop(npc_id, None)
            npc_ids = [npc_id for npc_id in all_npc_ids if npc_id in managed]
            for npc_id in npc_ids:
                state.arrival_tick.setdefault(npc_id, tick)
            if not npc_ids:
                continue
            blocked: Dict[str, str] = {}
            admitted = set()
            active_corridors = [
                (
                    holder,
                    self._holder_movement_corridor[(intersection_id, holder)],
                    self._holder_movement_vector.get((intersection_id, holder)),
                    False,
                )
                for holder in state.active_npc_ids
                if (intersection_id, holder) in self._holder_movement_corridor
            ]
            active_corridors.extend(self._foreign_held_movements(intersection_id))
            granted_corridors = []

            # A grant remains provisional until entry. Re-check current ego occupancy,
            # receiving space, and any active/recovered junction movements every tick.
            existing_grants = sorted(
                set(npc_ids) & set(state.granted_npc_ids),
                key=lambda token: (state.arrival_tick[token], token),
            )
            for npc_id in existing_grants:
                passage = candidates[npc_id]
                reason = safety_blocks.get(npc_id)
                if reason is None and any(
                    self._movements_conflict(
                        passage.movement_corridor,
                        corridor,
                        passage.movement_vector,
                        vector,
                    )
                    and not (
                        foreign
                        and self._same_physical_flow(
                            passage.movement_corridor,
                            corridor,
                            passage.movement_vector,
                            vector,
                        )
                    )
                    for holder, corridor, vector, foreign in active_corridors
                    if holder != npc_id
                ):
                    reason = "active_npc"
                elif any(
                    self._movements_conflict(
                        passage.movement_corridor,
                        corridor.movement_corridor,
                        passage.movement_vector,
                        corridor.movement_vector,
                    )
                    for corridor in granted_corridors
                ):
                    reason = "fifo_wait"

                if reason is not None:
                    self._clear_holder(intersection_id, npc_id)
                    blocked[npc_id] = reason
                    continue
                admitted.add(npc_id)
                granted_corridors.append(passage)

            # Iterate in FIFO order. Each accepted movement immediately reserves its
            # downstream slot, so later candidates cannot over-commit finite storage.
            pending = set(npc_ids) - admitted - set(blocked)
            while pending:
                npc_id = self.select_grant(tick, state, tuple(pending))
                assert npc_id is not None
                pending.remove(npc_id)
                passage = candidates[npc_id]

                if npc_id in safety_blocks:
                    blocked[npc_id] = safety_blocks[npc_id]
                elif any(
                    self._movements_conflict(
                        passage.movement_corridor,
                        corridor,
                        passage.movement_vector,
                        vector,
                    )
                    and not (
                        foreign
                        and self._same_physical_flow(
                            passage.movement_corridor,
                            corridor,
                            passage.movement_vector,
                            vector,
                        )
                    )
                        for holder, corridor, vector, foreign in active_corridors
                        if holder != npc_id
                ):
                    blocked[npc_id] = "active_npc"
                elif any(
                    self._movements_conflict(
                        passage.movement_corridor,
                        corridor.movement_corridor,
                        passage.movement_vector,
                        corridor.movement_vector,
                    )
                    for corridor in granted_corridors
                ):
                    blocked[npc_id] = "fifo_wait"
                else:
                    self._install_grant(
                        tick,
                        intersection_id,
                        npc_id,
                        passage,
                        active_agents[npc_id],
                        state,
                    )
                    admitted.add(npc_id)
                    granted_corridors.append(passage)

            for npc_id in npc_ids:
                passage = candidates[npc_id]
                enter = npc_id in admitted
                decisions[npc_id] = AdmissionDecision(
                    npc_id,
                    intersection_id,
                    enter,
                    None if enter else blocked[npc_id],
                    None if enter else passage.virtual_lead_geometry,
                )

        self._record_decisions(tick, decisions)
        candidate_pairs = {
            (decision.intersection_id, decision.npc_id) for decision in decisions.values()
        }
        for intersection_id, state in self.states.items():
            holders = set(state.active_npc_ids) | set(state.granted_npc_ids)
            for npc_id in list(state.arrival_tick):
                if npc_id not in holders and (intersection_id, npc_id) not in candidate_pairs:
                    state.arrival_tick.pop(npc_id, None)
        # Any moving blocker or corridor exit resets persistence immediately. Keeping
        # only observations seen during this update also prevents stale hysteresis when
        # a candidate temporarily leaves the braking gate.
        self._downstream_stopped_first_tick = {
            key: first_tick
            for key, first_tick in self._downstream_stopped_first_tick.items()
            if key in self._downstream_observations_seen
        }
        self._updates += 1
        if self.config.log_interval_ticks > 0 and self._updates % self.config.log_interval_ticks == 0:
            logger.info("[IDM-IM] tick=%d summary=%s", tick, self.get_stats())
        return decisions

    def _record_decisions(self, tick: int, decisions: Mapping[str, AdmissionDecision]) -> None:
        current_keys = set()
        for decision in decisions.values():
            key = (decision.npc_id, decision.intersection_id)
            current_keys.add(key)
            value = (decision.enter, decision.reason)
            if not decision.enter:
                if decision.reason not in HOLD_REASONS:
                    raise AssertionError(f"Unknown hold reason: {decision.reason}")
                self._hold_reason_ticks[decision.reason] += 1
            if self._last_decisions.get(key) != value:
                logger.info(
                    "[IDM-IM] tick=%d intersection=%s npc=%s decision=%s reason=%s",
                    tick,
                    decision.intersection_id,
                    decision.npc_id,
                    "enter" if decision.enter else "hold",
                    decision.reason or "granted",
                )
        self._last_decisions = {
            key: value for key, value in self._last_decisions.items() if key in current_keys
        }
        self._last_decisions.update(
            {
                (decision.npc_id, decision.intersection_id): (decision.enter, decision.reason)
                for decision in decisions.values()
            }
        )

    @staticmethod
    def targeted_virtual_leads(
        decisions: Mapping[str, AdmissionDecision]
    ) -> Dict[str, object]:
        """Return gates keyed by the only NPC allowed to observe each one."""
        return {
            decision.npc_id: decision.virtual_lead_geometry
            for decision in decisions.values()
            if not decision.enter and decision.virtual_lead_geometry is not None
        }

    def get_stats(self) -> dict:
        return {
            "updates": self._updates,
            "hold_reason_ticks": {reason: int(self._hold_reason_ticks[reason]) for reason in HOLD_REASONS},
            "stall_events": self._stall_events,
            "preexisting_active_stall_events": self._preexisting_active_stalls,
            "active_stall_retirements": self._active_stall_retirements,
            "downstream_reservation_conflicts": self._reservation_conflicts,
            "downstream_reservation_ttl_revocations": self._reservation_ttl_revocations,
            "peak_downstream_reservations": self._peak_reservations,
            "intersection_stopped_seconds": round(
                self._intersection_stopped_agent_ticks * self.config.sim_dt, 3
            ),
            "candidate_ticks": self._candidate_ticks,
            "contention_candidate_ticks": self._contention_candidate_ticks,
            "safety_candidate_ticks": self._safety_candidate_ticks,
            "unmanaged_candidate_ticks": self._unmanaged_candidate_ticks,
            "downstream_moving_same_flow_ignored_ticks": (
                self._downstream_moving_ignored_ticks
            ),
            "downstream_transient_stop_ignored_ticks": (
                self._downstream_transient_ignored_ticks
            ),
            "downstream_persistent_blocker_ticks": (
                self._downstream_persistent_blocker_ticks
            ),
            "downstream_cross_direction_blocker_ticks": (
                self._downstream_cross_direction_blocker_ticks
            ),
            "downstream_insufficient_path_ticks": (
                self._downstream_insufficient_path_ticks
            ),
            "downstream_unclassified_blocker_ticks": (
                self._downstream_unclassified_blocker_ticks
            ),
            "rear_ego_hold_suppressions": self._rear_ego_hold_suppressions,
        }
