# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
import copy
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.utils.collision_utils import CollisionType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely import Point, creation, distance

from odyssey.components.agents.policy.pdm_planner.observation.pdm_observation import (
    PDMObservation,
)
from odyssey.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)
from odyssey.components.agents.policy.pdm_planner.scoring.pdm_comfort_metrics import (
    ego_is_comfortable,
)
from odyssey.components.agents.policy.pdm_planner.scoring.pdm_scorer_utils import (
    get_collision_type,
)
from odyssey.components.agents.policy.pdm_planner.utils.pdm_array_representation import (
    coords_array_to_polygon_array,
    state_array_to_coords_array,
)
from odyssey.components.agents.policy.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
    EgoAreaIndex,
    MultiMetricIndex,
    StateIndex,
    WeightedMetricIndex,
)
from odyssey.components.agents.policy.pdm_planner.utils.pdm_path import PDMPath

# constants
# TODO: Add to config
WEIGHTED_METRICS_WEIGHTS = np.zeros(len(WeightedMetricIndex), dtype=np.float64)
WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.PROGRESS] = 5.0
WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.TTC] = 5.0
WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.COMFORTABLE] = 2.0
WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.LANE_KEEPING] = 0.0
WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.DRIVING_DIRECTION] = 0.0

# TODO: Add to config
DRIVING_DIRECTION_COMPLIANCE_THRESHOLD = 2.0  # [m] (driving direction)

# RouteDS per-collision factors (P_col; Bench2Drive's PENALTY_VALUE_DICT). They are kept out of
# the PDM score: NO_COLLISION is a multiplicative gate there. Only agent classes are charged --
# the reconstruction bakes no static obstacles (cones, barriers, generic objects), so the model
# cannot see them and a contact with one is not charged or counted.
COLLISION_PENALTY = {
    TrackedObjectType.PEDESTRIAN: 0.5,
    TrackedObjectType.BICYCLE: 0.6,
    TrackedObjectType.VEHICLE: 0.6,
}
# Bench2Drive CollisionTest's re-arming thresholds: two contacts with the same object count as
# one until the ego has moved this far away, or this much time has passed. See
# _calculate_no_at_fault_collision for why a broken contact is also required here but not
# upstream (CARLA ticks at 20 Hz, this scores at 0.5 s).
COLLISION_RADIUS_M = 5.0       # B2D CollisionTest.COLLISION_RADIUS
COLLISION_MAX_ID_TIME_S = 5.0  # B2D CollisionTest.MAX_ID_TIME
# A contact counts as BROKEN only once the polygons are at least this far apart. `intersects`
# is a knife edge: an ego sliding along a parked car can report millimetre gaps on single ticks
# in the middle of one scrape, which would charge one sideswipe twice. The margin makes the break a physical separation rather than a floating-point outcome.
COLLISION_BREAK_GAP_M = 0.3
DRIVING_DIRECTION_VIOLATION_THRESHOLD = 6.0  # [m] (driving direction)
STOPPED_SPEED_THRESHOLD = 5e-03  # [m/s] (ttc)
PROGRESS_DISTANCE_THRESHOLD = 0.1  # [m] (progress)
LANE_KEEPING_DEVATION_LIMIT = 0.5  # [m] (lane keeping) (hydraMDP++)
LANE_KEEPING_HORIZON_WINDOW = 2.0  # [s] (lane keeping) (hydraMDP++)


class PDMScorer:
    """Class to score proposals in PDM pipeline. Re-implements nuPlan's closed-loop metrics."""

    def __init__(
        self,
        proposal_sampling: TrajectorySampling,
        lane_keeping_weight: float = 0.0,
    ):
        """
        Constructor of PDMScorer
        :param proposal_sampling: Sampling parameters for proposals
        """
        self._proposal_sampling = proposal_sampling
        lane_keeping_weight = float(lane_keeping_weight)
        if not np.isfinite(lane_keeping_weight) or lane_keeping_weight < 0.0:
            raise ValueError("lane_keeping_weight must be finite and non-negative")
        # Keep the public/reference score constants immutable. The driving policy may use a
        # small selector-only lane preference, while MetricManager constructs the scorer with
        # the default 0.0 and therefore reports the unchanged reference PDM score.
        self._weighted_metric_weights = WEIGHTED_METRICS_WEIGHTS.copy()
        self._weighted_metric_weights[
            WeightedMetricIndex.LANE_KEEPING
        ] = lane_keeping_weight

        # lazy loaded
        self._initial_ego_state: Optional[EgoState] = None
        self._observation: Optional[PDMObservation] = None
        self._centerline: Optional[PDMPath] = None
        self._route_lane_dict: Optional[Dict[str, LaneGraphEdgeMapObject]] = None
        self._drivable_area_map: Optional[PDMOccupancyMap] = None
        self._map_api: Optional[AbstractMap] = None

        self._num_proposals: Optional[int] = None
        self._states: Optional[npt.NDArray[np.float64]] = None
        self._ego_coords: Optional[npt.NDArray[np.float64]] = None
        self._ego_polygons: Optional[npt.NDArray[np.object_]] = None

        self._ego_areas: Optional[npt.NDArray[np.bool_]] = None

        self._multi_metrics: Optional[npt.NDArray[np.float64]] = None
        self._weighted_metrics: Optional[npt.NDArray[np.float64]] = None
        self._progress_raw: Optional[npt.NDArray[np.float64]] = None

        self._collision_time_idcs: Optional[npt.NDArray[np.float64]] = None
        self._ttc_time_idcs: Optional[npt.NDArray[np.float64]] = None

    def P_col(self, proposal_idx: int) -> float:
        """
        RouteDS collision term: the product of COLLISION_PENALTY over the charged at-fault
        collision events (Bench2Drive's event splitting). An object is re-armed after the ego
        has driven away and the contact has broken, so two separate hits on the same car are
        two events; a sustained overlap is never re-charged, however far the ego travels while
        it lasts -- see _calculate_no_at_fault_collision.
        :return: penalty factor in (0, 1]; 1.0 when there was no at-fault collision
        """
        types = getattr(self, "_charged_collision_types", None)
        if not types:
            return 1.0
        penalty = 1.0
        for object_type in types[proposal_idx]:
            penalty *= COLLISION_PENALTY[object_type]
        return penalty

    def collision_count(self, proposal_idx: int) -> int:
        """Number of separately charged at-fault collision events (the events behind P_col)."""
        types = getattr(self, "_charged_collision_types", None)
        return len(types[proposal_idx]) if types else 0

    def offroad_stats(self, proposal_idx: int) -> Dict[str, float]:
        """
        Bench2Drive-style off-road accounting behind P_off, in meters: the off-road distance,
        the distance travelled, their ratio, and the union split into its two disjoint parts
        (non-drivable / outside the route lanes) which only say which failure it was.
        """
        if getattr(self, "_offroad_ratio", None) is None:
            return {"offroad_distance": 0.0, "total_distance": 0.0, "offroad_ratio": 0.0,
                    "offroad_distance_nondrivable": 0.0, "offroad_distance_offroute": 0.0}
        return {
            "offroad_distance": float(self._offroad_distance[proposal_idx]),
            "total_distance": float(self._total_distance[proposal_idx]),
            "offroad_ratio": float(self._offroad_ratio[proposal_idx]),
            "offroad_distance_nondrivable": float(self._offroad_distance_nondrivable[proposal_idx]),
            "offroad_distance_offroute": float(self._offroad_distance_offroute[proposal_idx]),
        }

    def P_off(self, proposal_idx: int) -> float:
        """
        Bench2Drive's OUTSIDE_ROUTE_LANES_INFRACTION penalty, whose per-unit value is 0:
        score_penalty *= 1 - (1 - 0) * percentage / 100, i.e. exactly the fraction of the
        driven distance that stayed on the road.
        :return: penalty factor in [0, 1]; 1.0 when the ego never left the drivable area
        """
        if getattr(self, "_offroad_ratio", None) is None:
            return 1.0
        return float(1.0 - self._offroad_ratio[proposal_idx])

    def time_to_at_fault_collision(self, proposal_idx: int) -> float:
        """
        Returns time to at-fault collision for given proposal
        :param proposal_idx: index for proposal
        :return: time to infraction
        """
        return (
            self._collision_time_idcs[proposal_idx]
            * self._proposal_sampling.interval_length
        )

    def time_to_ttc_infraction(self, proposal_idx: int) -> float:
        """
        Returns time to ttc infraction for given proposal
        :param proposal_idx: index for proposal
        :return: time to infraction
        """
        return (
            self._ttc_time_idcs[proposal_idx] * self._proposal_sampling.interval_length
        )

    def score_proposals(
        self,
        states: npt.NDArray[np.float64],
        initial_ego_state: EgoState,
        observation: PDMObservation,
        centerline: PDMPath,
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
        drivable_area_map: PDMOccupancyMap,
        map_api: AbstractMap,
        batch = False
    ) -> npt.NDArray[np.float64]:
        """
        Scores proposal similar to nuPlan's closed-loop metrics
        :param states: array representation of simulated proposals
        :param initial_ego_state: ego-vehicle state at current iteration
        :param observation: PDM's observation class
        :param centerline: path of the centerline
        :param route_lane_dict: dictionary containing on-route lanes
        :param drivable_area_map: Occupancy map of drivable are polygons
        :param map_api: map object
        :param batch: whether to score in batch mode
        :return: array containing score of each proposal
        """

        # initialize & lazy load class values
        self._reset(
            states,
            initial_ego_state,
            observation,
            centerline,
            route_lane_dict,
            drivable_area_map,
            map_api,
        )

        # fill value ego-area array (used across multiple metrics)
        self._calculate_ego_area()

        # 1. multiplicative metrics
        self._calculate_no_at_fault_collision()
        self._calculate_drivable_area_compliance()

        # 2. weighted metrics
        self._calculate_progress()
        self._calculate_ttc_optimized()
        self._calculate_is_comfortable()
        self._calculate_lane_keeping()
        self._calculate_driving_direction_compliance()

        return self._aggregate_scores(batch)

    def _aggregate_scores(self, batch=False) -> npt.NDArray[np.float64]:
        """
        Aggregates metrics with multiplicative and weighted average.
        :return: array containing score of each proposal
        """

        # accumulate multiplicative metrics
        multiplicate_metric_scores = self._multi_metrics.prod(axis=0)

        # normalize progress values first
        if not batch:
            max_raw_progress = np.max(self._progress_raw)
            if max_raw_progress > PROGRESS_DISTANCE_THRESHOLD:
                normalized_progress = self._progress_raw / max_raw_progress
            else:
                normalized_progress = np.ones(len(self._progress_raw), dtype=np.float64)
        else:
            # [NOTE]: expert reference [0], we make direct batched comparison:
            ref_raw_progress = np.ones(self._num_proposals, ) * self._progress_raw[0]
            comp_raw_progress = np.stack([ref_raw_progress, self._progress_raw], axis=1)
            max_raw_progress = np.max(comp_raw_progress, axis=1)
            
            dist_large_mask = (max_raw_progress > PROGRESS_DISTANCE_THRESHOLD).astype(np.float64)
            normalized_progress = dist_large_mask * np.nan_to_num(self._progress_raw / max_raw_progress, nan=1e-5, posinf=1e-5, neginf=1e-5) + (1 - dist_large_mask)

        # apply multiplicative metrics after normalization
        normalized_progress = normalized_progress * multiplicate_metric_scores

        self._weighted_metrics[WeightedMetricIndex.PROGRESS] = normalized_progress

        # accumulate weighted metrics
        weighted_metric_scores = (
            self._weighted_metrics * self._weighted_metric_weights[..., None]
        ).sum(axis=0)
        weighted_metric_scores /= self._weighted_metric_weights.sum()

        # calculate final scores
        final_scores = multiplicate_metric_scores * weighted_metric_scores

        return final_scores

    def _reset(
        self,
        states: npt.NDArray[np.float64],
        initial_ego_state: EgoState,
        observation: PDMObservation,
        centerline: PDMPath,
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
        drivable_area_map: PDMOccupancyMap,
        map_api: AbstractMap,
    ) -> None:
        """
        Resets metric values and lazy loads input classes.
        :param states: array representation of simulated proposals
        :param initial_ego_state: ego-vehicle state at current iteration
        :param observation: PDM's observation class
        :param centerline: path of the centerline
        :param route_lane_dict: dictionary containing on-route lanes
        :param drivable_area_map: Occupancy map of drivable are polygons
        :param map_api: map object
        """
        assert states.ndim == 3
        assert states.shape[1] == self._proposal_sampling.num_poses + 1
        assert states.shape[2] == StateIndex.size()

        self._initial_ego_state = initial_ego_state
        self._observation = observation
        self._centerline = centerline
        self._route_lane_dict = route_lane_dict
        self._drivable_area_map = drivable_area_map
        self._map_api = map_api

        self._num_proposals = states.shape[0]

        # save ego state values
        self._states = states

        # calculate coordinates of ego corners and center
        self._ego_coords = state_array_to_coords_array(
            states, initial_ego_state.car_footprint.vehicle_parameters
        )

        # initialize all ego polygons from corners
        self._ego_polygons = coords_array_to_polygon_array(self._ego_coords)

        # zero initialize all remaining arrays.
        self._ego_areas = np.zeros(
            (
                self._num_proposals,
                self._proposal_sampling.num_poses + 1,
                len(EgoAreaIndex),
            ),
            dtype=np.bool_,
        )
        self._multi_metrics = np.ones(
            (len(MultiMetricIndex), self._num_proposals), dtype=np.float64
        )
        self._weighted_metrics = np.zeros(
            (len(WeightedMetricIndex), self._num_proposals), dtype=np.float64
        )
        self._progress_raw = np.zeros(self._num_proposals, dtype=np.float64)

        # (time_idx, token) of every at-fault contact, per proposal (dense_series reads it).
        self._at_fault_contacts = [[] for _ in range(self._num_proposals)]
        self._last_contact = [{} for _ in range(self._num_proposals)]
        # token -> whether the current uninterrupted overlap was at-fault ON THE FRAME IT
        # BEGAN. That verdict holds until the polygons stop overlapping; a later, separate
        # contact with the same actor is classified afresh. See _calculate_no_at_fault_collision.
        self._contact_origin_at_fault = [{} for _ in range(self._num_proposals)]
        # Collision event de-duplication (Bench2Drive's rule): an object is re-armed once the
        # ego has moved COLLISION_RADIUS_M away or COLLISION_MAX_ID_TIME_S has passed, so
        # hitting the same car twice at opposite ends of a street is two events, not one.
        #
        # The distance rule alone cannot be ported as-is: CARLA ticks at 20 Hz while this scores
        # at 0.5 s, and the ego covers well over 5 m between frames, so "moved 5 m" is true on
        # essentially every frame and a single sustained contact would re-arm every frame. So a
        # re-arm also requires the contact to have actually broken (a frame in between with no
        # overlap).
        # token -> (last charged center xy, last charged time_idx, contact seen last frame)
        self._collision_state = [dict() for _ in range(self._num_proposals)]
        # Object type of every charged event, per proposal, in time order (P_col multiplies it).
        self._charged_collision_types = [[] for _ in range(self._num_proposals)]

        # P_off's off-road distance accounting (meters); filled by _calculate_offroad_distance().
        self._offroad_distance = np.zeros(self._num_proposals, dtype=np.float64)
        self._total_distance = np.zeros(self._num_proposals, dtype=np.float64)
        self._offroad_ratio = np.zeros(self._num_proposals, dtype=np.float64)
        self._offroad_distance_nondrivable = np.zeros(self._num_proposals, dtype=np.float64)
        self._offroad_distance_offroute = np.zeros(self._num_proposals, dtype=np.float64)

        # initialize infraction arrays with infinity (meaning no infraction occurs)
        self._collision_time_idcs = np.zeros(self._num_proposals, dtype=np.float64)
        self._ttc_time_idcs = np.zeros(self._num_proposals, dtype=np.float64)
        self._collision_time_idcs.fill(np.inf)
        self._ttc_time_idcs.fill(np.inf)

    def _calculate_ego_area(self) -> None:
        """
        Determines the area of proposals over time.
        Areas are (1) in multiple lanes, (2) non-drivable area, or (3) oncoming traffic
        """

        n_proposals, n_horizon, n_points, _ = self._ego_coords.shape
        coordinates = self._ego_coords.reshape(n_proposals * n_horizon * n_points, 2)

        in_polygons = self._drivable_area_map.points_in_polygons(coordinates)
        in_polygons = in_polygons.reshape(
            len(self._drivable_area_map), n_proposals, n_horizon, n_points
        ).transpose(
            1, 2, 0, 3
        )  # shape: n_proposals, n_horizon, n_polygons, n_points

        drivable_area_on_route_idcs: List[int] = [
            idx
            for idx, token in enumerate(self._drivable_area_map.tokens)
            if token in self._route_lane_dict.keys()
        ]  # index mask for on-route lanes

        corners_in_polygon = in_polygons[..., :-1]  # ignore center coordinate
        center_in_polygon = in_polygons[..., -1]  # only center

        # in_multiple_lanes: if
        # - more than one drivable polygon contains at least one corner
        # - no polygon contains all corners
        batch_multiple_lanes_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_multiple_lanes_mask = (corners_in_polygon.sum(axis=-1) > 0).sum(
            axis=-1
        ) > 1

        batch_not_single_lanes_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_not_single_lanes_mask = np.all(
            corners_in_polygon.sum(axis=-1) != 4, axis=-1
        )

        multiple_lanes_mask = np.logical_and(
            batch_multiple_lanes_mask, batch_not_single_lanes_mask
        )
        self._ego_areas[multiple_lanes_mask, EgoAreaIndex.MULTIPLE_LANES] = True

        # in_nondrivable_area: if at least one corner is not within any drivable polygon
        batch_nondrivable_area_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_nondrivable_area_mask = (corners_in_polygon.sum(axis=-2) > 0).sum(
            axis=-1
        ) < 4
        self._batch_nondrivable_area_mask = batch_nondrivable_area_mask
        self._ego_areas[
            batch_nondrivable_area_mask, EgoAreaIndex.NON_DRIVABLE_AREA
        ] = True

        # in_oncoming_traffic: if center not in any drivable polygon that is on-route
        batch_oncoming_traffic_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_oncoming_traffic_mask = (
            center_in_polygon[..., drivable_area_on_route_idcs].sum(axis=-1) == 0
        )
        self._ego_areas[
            batch_oncoming_traffic_mask, EgoAreaIndex.ONCOMING_TRAFFIC
        ] = True

    def _calculate_no_at_fault_collision(self) -> None:
        """
        Re-implementation of nuPlan's at-fault collision metric.
        """
        no_at_fault_collision_scores = np.ones(self._num_proposals, dtype=np.float64)

        proposal_collided_track_ids = {
            proposal_idx: copy.deepcopy(self._observation.collided_track_ids)
            for proposal_idx in range(self._num_proposals)
        }

        for time_idx in range(self._proposal_sampling.num_poses + 1):
            ego_polygons = self._ego_polygons[:, time_idx]
            intersecting = self._observation[time_idx].query(
                ego_polygons, predicate="intersects"
            )

            if len(intersecting) == 0:
                continue

            for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                token = self._observation[time_idx].tokens[geometry_idx]
                if (self._observation.red_light_token in token) or (
                    token in proposal_collided_track_ids[proposal_idx]
                ) or token == 'ego':
                    continue

                ego_in_multiple_lanes_or_nondrivable_area = (
                    self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                    or self._ego_areas[
                        proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA
                    ]
                )

                tracked_object = self._observation.object_at(time_idx, token)
                previous_touch = self._last_contact[proposal_idx].get(token)
                self._last_contact[proposal_idx][token] = time_idx

                # classify collision
                collision_type: CollisionType = get_collision_type(
                    self._states[proposal_idx, time_idx],
                    self._ego_polygons[proposal_idx, time_idx],
                    tracked_object,
                    self._observation[time_idx][token],
                )
                collisions_at_stopped_track_or_active_front: bool = collision_type in [
                    CollisionType.ACTIVE_FRONT_COLLISION,
                    CollisionType.STOPPED_TRACK_COLLISION,
                ]
                collision_at_lateral: bool = (
                    collision_type == CollisionType.ACTIVE_LATERAL_COLLISION
                )
                # An overlap is judged by the frame it BEGAN on, and that verdict holds for as
                # long as the polygons keep overlapping.
                #
                # Why the first frame is the only honest one: there is no physics here. Actors
                # are log replay -- they drive their recorded path whatever the ego does, straight
                # through it, never avoiding it. Two bodies cannot interpenetrate in the world being modelled, so every
                # frame after contact begins shows a geometry that would not exist.
                # Reclassifying on those frames scores the ego against a fiction.
                #
                # It is also not a second event. The contact never broke; _last_contact is
                # updated above, so an actual separation still re-arms the latch and a genuine
                # new contact is classified afresh.
                #
                # The latch covers every collision type, not only ACTIVE_REAR_COLLISION: an actor
                # overtaking a near-stopped ego in the next lane begins as ACTIVE_LATERAL
                # (non-fault) and would otherwise be reclassified ACTIVE_FRONT the moment its
                # polygon sweeps across the ego front-bumper segment. (is_agent_behind measures
                # from the ego REAR AXLE with a 150 deg tolerance, so a vehicle passing alongside
                # is never "behind".)
                at_fault_now: bool = collisions_at_stopped_track_or_active_front or (
                    ego_in_multiple_lanes_or_nondrivable_area and collision_at_lateral
                )
                if previous_touch is None or time_idx > previous_touch + 1:
                    self._contact_origin_at_fault[proposal_idx][token] = at_fault_now
                # 1. at fault collision -- the verdict from the frame contact began on.
                if self._contact_origin_at_fault[proposal_idx][token]:
                    self._at_fault_contacts[proposal_idx].append((time_idx, token))
                    is_agent = tracked_object.tracked_object_type in AGENT_TYPES
                    no_at_fault_collision_score = 0.0 if is_agent else 0.5
                    # P_col's event accounting (Bench2Drive-style re-arming, see _reset()).
                    # Agent classes only: a static obstacle is not in the reconstruction, so
                    # the PDM gate sees its 0.5 but RouteDS neither charges nor counts it.
                    # state = (xy where this object was last CHARGED, time_idx of the last
                    # frame it was TOUCHING). The two differ during a sustained contact, and
                    # both are needed: distance/time are measured from the charge, while
                    # "did contact break" is measured from the last touch.
                    if is_agent:
                        obj_type = tracked_object.tracked_object_type
                        ego_xy = self._ego_coords[proposal_idx, time_idx, BBCoordsIndex.CENTER]
                        st = self._collision_state[proposal_idx].get(token)
                        if st is None:
                            charged_xy, charged_idx = ego_xy, time_idx
                            self._charged_collision_types[proposal_idx].append(obj_type)
                        else:
                            charged_xy, charged_idx, _ = st
                            broke = self._contact_broke(
                                proposal_idx, token, previous_touch, time_idx)
                            moved = float(np.linalg.norm(ego_xy - charged_xy))
                            elapsed = ((time_idx - charged_idx)
                                       * self._proposal_sampling.interval_length)
                            # BOTH halves of B2D's rule require the ego to have gone somewhere.
                            # On its own, the time half would re-arm a contact lasting more than
                            # MAX_ID_TIME on any break no matter how little the ego had moved,
                            # so a jam with a jittering polygon would charge one stationary car
                            # every 5 s. It is kept, because a slow ego that
                            # genuinely pulls away and comes back IS a second event, but it
                            # carries the same separation floor -- the ego must be clear of the
                            # actor, not merely not-overlapping it.
                            if broke and (moved > COLLISION_RADIUS_M
                                          or (elapsed > COLLISION_MAX_ID_TIME_S
                                              and moved > COLLISION_BREAK_GAP_M)):
                                self._charged_collision_types[proposal_idx].append(obj_type)
                                charged_xy, charged_idx = ego_xy, time_idx
                        self._collision_state[proposal_idx][token] = (
                            charged_xy, charged_idx, time_idx)
                    no_at_fault_collision_scores[proposal_idx] = np.minimum(
                        no_at_fault_collision_scores[proposal_idx],
                        no_at_fault_collision_score,
                    )
                    self._collision_time_idcs[proposal_idx] = min(
                        time_idx, self._collision_time_idcs[proposal_idx]
                    )

                # Other non-fault contacts (e.g. lateral) do not immunize the actor;
                # their dynamics and our responsibility can change mid-episode.

        self._multi_metrics[
            MultiMetricIndex.NO_COLLISION
        ] = no_at_fault_collision_scores

    def _contact_broke(self, proposal_idx, token, previous_touch, time_idx) -> bool:
        """Did the contact with ``token`` genuinely end between the two touches?

        Non-overlap for a tick is not separation. `intersects` flips on a boundary that a
        0.1 mm jitter crosses, so the raw "there was a tick without overlap" test turned one
        continuous scrape into several events (see COLLISION_BREAK_GAP_M). A break requires
        the polygons to have reached COLLISION_BREAK_GAP_M apart at some tick in between.

        The actor may also leave the scene during the gap. An absent actor is a real
        separation, not a missing measurement, so it counts as broken.
        """
        if previous_touch is None:
            return True
        if time_idx <= previous_touch + 1:
            return False
        for gap_idx in range(previous_touch + 1, time_idx):
            try:
                actor_geometry = self._observation[gap_idx][token]
            except KeyError:
                return True
            gap = self._ego_polygons[proposal_idx, gap_idx].distance(actor_geometry)
            if gap >= COLLISION_BREAK_GAP_M:
                return True
        return False

    def _calculate_progress(self) -> None:
        """
        Re-implementation of nuPlan's progress metric (non-normalized).
        Calculates progress along the centerline.
        """

        # calculate raw progress in meter
        progress_in_meter = np.zeros(self._num_proposals, dtype=np.float64)
        for proposal_idx in range(self._num_proposals):
            start_point = Point(
                *self._ego_coords[proposal_idx, 0, BBCoordsIndex.CENTER]
            )
            end_point = Point(*self._ego_coords[proposal_idx, -1, BBCoordsIndex.CENTER])
            start_progress = self._centerline.project_near(
                start_point,
                heading=float(self._states[proposal_idx, 0, StateIndex.HEADING]),
            )
            # A PDM proposal spans four seconds and cannot legitimately advance hundreds of
            # metres.  Anchor its end to the ordered branch selected at the start so a P-turn
            # crossing is scored as normal forward travel, without deleting the loop itself.
            end_progress = self._centerline.project_near(
                end_point,
                reference_progress=start_progress,
                heading=float(self._states[proposal_idx, -1, StateIndex.HEADING]),
                backward_tolerance=10.0,
                forward_tolerance=100.0,
            )
            progress_in_meter[proposal_idx] = end_progress - start_progress

        self._progress_raw = np.clip(progress_in_meter, a_min=0, a_max=None)

    def _calculate_is_comfortable(self) -> None:
        """
        Re-implementation of nuPlan's comfortability metric.
        """
        time_point_s: npt.NDArray[np.float64] = (
            np.arange(0, self._proposal_sampling.num_poses + 1).astype(np.float64)
            * self._proposal_sampling.interval_length
        )
        is_comfortable = ego_is_comfortable(self._states, time_point_s)
        self._weighted_metrics[WeightedMetricIndex.COMFORTABLE] = np.all(
            is_comfortable, axis=-1
        )

    def _calculate_drivable_area_compliance(self) -> None:
        """
        Re-implementation of nuPlan's drivable area compliance metric
        """
        drivable_area_compliance_scores = np.ones(self._num_proposals, dtype=np.float64)
        off_road_mask = self._ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA].any(
            axis=-1
        )
        drivable_area_compliance_scores[off_road_mask] = 0.0
        self._multi_metrics[
            MultiMetricIndex.DRIVABLE_AREA
        ] = drivable_area_compliance_scores

    def _ego_in_intersection(self) -> npt.NDArray[np.bool_]:
        """(n_proposals, n_horizon) -- is the ego CENTRE inside an INTERSECTION polygon?

        The centre, matching EgoAreaIndex.ONCOMING_TRAFFIC's own test, so the carve-out lines up
        with the mask it cancels. INTERSECTION polygons are already in the drivable-area map
        (PDMDrivableMap.from_simulation queries that layer), so this needs no extra map lookup --
        it reuses the same vectorised points_in_polygons the ego-area masks are built from.

        Returns all-False when the map carries no intersection polygons, which leaves the off-road
        union exactly as it was.
        """
        centers = self._ego_coords[:, :, BBCoordsIndex.CENTER]
        n_proposals, n_horizon = centers.shape[:2]
        indices = self._drivable_area_map.get_indices_of_map_type(
            [SemanticMapLayer.INTERSECTION]
        )
        if not indices:
            return np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        inside = self._drivable_area_map.points_in_polygons(centers.reshape(-1, 2))
        inside = inside.reshape(
            len(self._drivable_area_map), n_proposals, n_horizon
        ).transpose(1, 2, 0)
        return inside[:, :, indices].any(axis=-1)

    def _calculate_offroad_distance(self) -> None:
        """
        Bench2Drive-style off-road accounting, measured in METERS rather than frames.

        OutsideRouteLanesTest accumulates, per step, the distance the ego travelled and
        adds that same distance to a "wrong" accumulator whenever the step was off-road;
        the reported infraction is wrong_distance / total_distance. This mirrors that:
        each consecutive pair of scored poses contributes its segment length, and the
        segment counts as off-road when either endpoint is flagged non-drivable.

        Attributing a segment on "either endpoint" (rather than only the start) matches
        the CARLA behaviour of flagging the step in which the violation is detected, and
        keeps a single-frame excursion from being silently dropped.

        WRONG-WAY DRIVING IS FOLDED IN HERE, not scored as a separate multiplicative term.
        Bench2Drive counts driving against traffic inside this same OUTSIDE_ROUTE_LANES
        infraction, so the distance spent in oncoming traffic joins the off-road distance
        (union, so a segment that is both is charged once, never twice). nuPlan's own
        driving_direction_compliance is a 3-level gate (1.0 / 0.5 / 0.0) built for the PDM
        score; multiplying that 0.0 into RouteDS made ANY sustained wrong-way
        drive collapse DS to exactly 0, discarding route completion and every other
        penalty -- the same "all failures look alike" flattening DS exists to avoid.

        Feeds RouteDS only -- the PDM score's DRIVABLE_AREA and DRIVING_DIRECTION
        metrics are both untouched.
        """
        # Center of the ego box per proposal/frame -> segment lengths between poses.
        centers = self._ego_coords[:, :, BBCoordsIndex.CENTER]  # (n_proposals, n_horizon, 2)
        deltas = np.diff(centers, axis=1)
        segment_lengths = np.linalg.norm(deltas, axis=-1)  # (n_proposals, n_horizon - 1)

        # Union of "not on a drivable polygon" and "on the wrong side of the road": both are
        # route the ego traversed illegally, which is exactly what Bench2Drive accumulates.
        #
        # EXCEPT INSIDE A JUNCTION, where the ONCOMING half means nothing. That mask is not a
        # direction test -- it is set whenever the ego CENTRE is in no on-route polygon
        # (_calculate_ego_area) -- and a junction is full of overlapping lane connectors of which
        # only the one the GT drive took is on-route. So an ego crossing an intersection perfectly
        # legally is flagged for as long as it is inside. nuPlan's reference implementation zeroes
        # oncoming progress there for exactly this reason (navsim pdm_scorer, "remove
        # intersection"); we fold wrong-way into this distance instead of scoring it as its own
        # gate, so the carve-out has to come with it.
        #
        # In our experiments about two thirds of charged off-route distance fell inside
        # intersections. Without this the metric partly scores "did you pass through a junction"
        # rather than "did you leave your route".
        #
        # NON_DRIVABLE is deliberately NOT carved out: leaving the road surface is a violation
        # wherever it happens, and the reference does not exempt drivable-area compliance either.
        in_intersection = self._ego_in_intersection()
        self._offroute_mask = self._ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC] & ~in_intersection
        off_road = np.logical_or(
            self._ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA],
            np.logical_and(
                self._ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC],
                np.logical_not(in_intersection),
            ),
        )
        segment_off_road = np.logical_or(off_road[:, :-1], off_road[:, 1:])

        total_distance = segment_lengths.sum(axis=-1)
        offroad_distance = (segment_lengths * segment_off_road).sum(axis=-1)

        # A stationary ego has no distance to attribute; report 0 rather than 0/0.
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(total_distance > 0.0, offroad_distance / total_distance, 0.0)

        self._offroad_distance = offroad_distance
        self._total_distance = total_distance
        self._offroad_ratio = np.clip(ratio, 0.0, 1.0)

        # Diagnostics -- the union above is what the penalty uses and is NOT changed here.
        # The union alone cannot say whether 8 m of offroad was the ego leaving the road or
        # sitting outside its route lanes, and those are different failures. Splitting the
        # distance (never the penalty) makes that visible.
        #
        # The split is disjoint by construction: NON_DRIVABLE wins, and the offroute part is
        # only what is left over. So nondrivable + offroute == union exactly, which is what
        # lets a later 2b split them into separate coefficients without double-charging any
        # segment. Both masks are true on the same frame far more often than not.
        # The junction carve-out above applies here too -- the two parts must still sum to the
        # union the penalty charges, so the offroute part uses the same carved mask.
        nondrivable = self._ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA]
        offroute_only = np.logical_and(
            np.logical_and(
                self._ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC],
                np.logical_not(in_intersection),
            ),
            np.logical_not(nondrivable),
        )
        seg_nondrivable = np.logical_or(nondrivable[:, :-1], nondrivable[:, 1:])
        # A segment already charged as non-drivable must not be counted again here, or the
        # two parts would sum to more than the union whenever a segment straddles both.
        seg_offroute = np.logical_and(
            np.logical_or(offroute_only[:, :-1], offroute_only[:, 1:]),
            np.logical_not(seg_nondrivable),
        )
        self._offroad_distance_nondrivable = (segment_lengths * seg_nondrivable).sum(axis=-1)
        self._offroad_distance_offroute = (segment_lengths * seg_offroute).sum(axis=-1)

    def _calculate_driving_direction_compliance(self) -> None:
        """
        Re-implementation of nuPlan's driving direction compliance metric
        """
        center_coordinates = self._ego_coords[:, :, BBCoordsIndex.CENTER]
        cum_progress = np.zeros(
            (self._num_proposals, self._proposal_sampling.num_poses + 1),
            dtype=np.float64,
        )
        cum_progress[:, 1:] = (
            (center_coordinates[:, 1:] - center_coordinates[:, :-1]) ** 2.0
        ).sum(axis=-1) ** 0.5

        # mask out progress along the driving direction
        oncoming_traffic_masks = self._ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC]
        cum_progress[~oncoming_traffic_masks] = 0.0

        driving_direction_compliance_scores = np.ones(
            self._num_proposals, dtype=np.float64
        )

        for proposal_idx in range(self._num_proposals):
            oncoming_traffic_progress, oncoming_traffic_mask = (
                cum_progress[proposal_idx],
                oncoming_traffic_masks[proposal_idx],
            )

            # split progress whenever ego changes traffic direction
            oncoming_progress_splits = np.split(
                oncoming_traffic_progress,
                np.where(np.diff(oncoming_traffic_mask))[0] + 1,
            )

            # sum up progress of splitted intervals
            # Note: splits along the driving direction will have a sum of zero.
            max_oncoming_traffic_progress = max(
                oncoming_progress.sum()
                for oncoming_progress in oncoming_progress_splits
            )

            if max_oncoming_traffic_progress < DRIVING_DIRECTION_COMPLIANCE_THRESHOLD:
                driving_direction_compliance_scores[proposal_idx] = 1.0
            elif max_oncoming_traffic_progress < DRIVING_DIRECTION_VIOLATION_THRESHOLD:
                driving_direction_compliance_scores[proposal_idx] = 0.5
            else:
                driving_direction_compliance_scores[proposal_idx] = 0.0

        self._weighted_metrics[
            WeightedMetricIndex.DRIVING_DIRECTION
        ] = driving_direction_compliance_scores

    def _calculate_lane_keeping(self) -> None:
        """
        Revised implementation of hydraMDP++'s lane keeping metric.
        The trajectory is considered failing lane-keeping only if it deviates beyond
        the lateral threshold continuously for at least certain seconds.

        """
        # Initialize lane-keeping scores to 1.0
        lane_keeping_scores = np.ones(self._num_proposals, dtype=np.float64)
        lateral_deviation_limit = LANE_KEEPING_DEVATION_LIMIT

        interval_length = self._proposal_sampling.interval_length
        continuous_steps_required = int(
            np.ceil(LANE_KEEPING_HORIZON_WINDOW / interval_length)
        )

        centerline = self._centerline.linestring

        ego_positions = creation.points(
            self._ego_coords[:, :, BBCoordsIndex.CENTER]
        )
        exceeds_limit = distance(ego_positions, centerline) > lateral_deviation_limit
        num_steps = exceeds_limit.shape[1]
        if continuous_steps_required <= num_steps:
            windows = np.lib.stride_tricks.sliding_window_view(
                exceeds_limit,
                window_shape=continuous_steps_required,
                axis=1,
            )
            lane_keeping_scores[np.any(np.all(windows, axis=-1), axis=1)] = 0.0

        self._weighted_metrics[WeightedMetricIndex.LANE_KEEPING] = lane_keeping_scores

    # ==================== Optimized TTC ====================

    @staticmethod
    def _batch_relative_angles(
        ego_xy: npt.NDArray[np.float64],
        ego_heading: npt.NDArray[np.float64],
        track_xy: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Vectorized get_agent_relative_angle for N pairs.
        :param ego_xy: (N, 2)
        :param ego_heading: (N,)
        :param track_xy: (N, 2)
        :return: relative angles (N,) in radians [0, pi]
        """
        agent_vec = track_xy - ego_xy  # (N, 2)
        norms = np.linalg.norm(agent_vec, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # avoid division by zero
        agent_vec_normed = agent_vec / norms
        ego_vec = np.stack([np.cos(ego_heading), np.sin(ego_heading)], axis=1)  # (N, 2)
        dot = np.clip((ego_vec * agent_vec_normed).sum(axis=1), -1.0, 1.0)
        return np.arccos(dot)

    def _calculate_ttc_optimized(self):
        """
        Optimized TTC with:
        1. Lazy polygon creation (only for active proposals)
        2. Early termination (skip proposals with score=0)
        3. Vectorized is_agent_ahead / is_agent_behind

        Has been verified, outputs are same with before.
        """
        AHEAD_THRESHOLD = np.deg2rad(30)
        BEHIND_THRESHOLD = np.deg2rad(150)

        ttc_scores = np.ones(self._num_proposals, dtype=np.float64)
        temp_collided_track_ids = {
            proposal_idx: copy.deepcopy(self._observation.collided_track_ids)
            for proposal_idx in range(self._num_proposals)
        }

        future_time_idcs = np.arange(0, int(1 / self._proposal_sampling.interval_length) + 1)
        n_future_steps = len(future_time_idcs)

        # Precompute exterior coordinates (replace CENTER with FRONT_LEFT)
        coords_exterior = self._ego_coords.copy()
        coords_exterior[:, :, BBCoordsIndex.CENTER, :] = coords_exterior[
            :, :, BBCoordsIndex.FRONT_LEFT, :
        ]

        speeds = np.hypot(
            self._states[..., StateIndex.VELOCITY_X],
            self._states[..., StateIndex.VELOCITY_Y],
        )
        dxy_per_s = np.stack(
            [
                np.cos(self._states[..., StateIndex.HEADING]) * speeds,
                np.sin(self._states[..., StateIndex.HEADING]) * speeds,
            ],
            axis=-1,
        )

        num_ttc_poses = self._proposal_sampling.num_poses - int(1 / self._proposal_sampling.interval_length)

        # Precompute per-step time deltas
        delta_ts = future_time_idcs.astype(np.float64) * self._proposal_sampling.interval_length

        # Precompute boolean masks
        stopped_mask = speeds < STOPPED_SPEED_THRESHOLD
        ego_area_flags = (
            self._ego_areas[:, :, EgoAreaIndex.MULTIPLE_LANES]
            | self._ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA]
        )

        # Track active proposals
        active_mask = np.ones(self._num_proposals, dtype=np.bool_)

        # Cache for is_in_layer results: (proposal_idx, time_idx) -> bool
        intersection_cache = {}

        for time_idx in range(num_ttc_poses + 1):
            if not active_mask.any():
                break

            active_indices = np.where(active_mask)[0]

            for step_idx in range(n_future_steps):
                current_time_idx = time_idx + future_time_idcs[step_idx]
                delta_t = delta_ts[step_idx]

                # Lazy polygon creation: only for active proposals at this step
                active_coords = coords_exterior[active_indices, time_idx].copy()
                active_coords += dxy_per_s[active_indices, time_idx, None, :] * delta_t
                active_polygons = creation.polygons(active_coords)

                intersecting = self._observation[current_time_idx].query(
                    active_polygons, predicate="intersects"
                )

                if len(intersecting[0]) == 0:
                    continue

                # Map local indices back to original proposal indices
                local_idxs = intersecting[0]
                geometry_idxs = intersecting[1]
                proposal_idxs = active_indices[local_idxs]

                # Gather tokens for all pairs
                obs_at_t = self._observation[current_time_idx]
                tokens = [obs_at_t.tokens[gi] for gi in geometry_idxs]

                # ---- Batch filter known exclusions ----
                keep = np.ones(len(tokens), dtype=np.bool_)
                for i, (pidx, token) in enumerate(zip(proposal_idxs, tokens)):
                    if (
                        (self._observation.red_light_token in token)
                        or (token in temp_collided_track_ids[pidx])
                        or stopped_mask[pidx, time_idx]
                        or token == 'ego'
                        or not active_mask[pidx]
                    ):
                        keep[i] = False

                if not keep.any():
                    continue

                kept_proposal_idxs = proposal_idxs[keep]
                kept_tokens = [t for t, k in zip(tokens, keep) if k]
                kept_geometry_idxs = geometry_idxs[keep]

                # ---- Batch gather ego and track geometry ----
                ego_xy = self._states[kept_proposal_idxs, time_idx, :2]  # STATE_SE2 x, y
                ego_heading = self._states[kept_proposal_idxs, time_idx, StateIndex.HEADING]

                # Gather track centroids and headings
                track_xy = np.empty((len(kept_tokens), 2), dtype=np.float64)
                track_headings = np.empty(len(kept_tokens), dtype=np.float64)
                for i, (token, gi) in enumerate(zip(kept_tokens, kept_geometry_idxs)):
                    centroid = obs_at_t._geometries[obs_at_t._token_to_idx[token]].centroid
                    track_xy[i, 0] = centroid.x
                    track_xy[i, 1] = centroid.y
                    track_headings[i] = self._observation.unique_objects[token].box.center.heading

                # ---- Vectorized is_agent_ahead / is_agent_behind ----
                rel_angles = self._batch_relative_angles(ego_xy, ego_heading, track_xy)
                is_ahead = rel_angles < AHEAD_THRESHOLD
                is_behind = rel_angles > BEHIND_THRESHOLD

                # ---- Process each pair with precomputed geometry ----
                for i in range(len(kept_proposal_idxs)):
                    pidx = kept_proposal_idxs[i]
                    token = kept_tokens[i]

                    if not active_mask[pidx]:
                        continue

                    if is_ahead[i]:
                        at_fault = True
                    elif not is_behind[i]:
                        # Need to check ego_area_flags or is_in_layer
                        if ego_area_flags[pidx, time_idx]:
                            at_fault = True
                        else:
                            cache_key = (pidx, time_idx)
                            if cache_key not in intersection_cache:
                                ego_rear_axle = StateSE2(
                                    *self._states[pidx, time_idx, StateIndex.STATE_SE2]
                                )
                                intersection_cache[cache_key] = self._map_api.is_in_layer(
                                    ego_rear_axle, layer=SemanticMapLayer.INTERSECTION
                                )
                            at_fault = intersection_cache[cache_key]
                    else:
                        at_fault = False

                    if at_fault:
                        ttc_scores[pidx] = 0.0
                        self._ttc_time_idcs[pidx] = float(time_idx)
                        active_mask[pidx] = False
                    else:
                        temp_collided_track_ids[pidx].append(token)

        self._weighted_metrics[WeightedMetricIndex.TTC] = ttc_scores
