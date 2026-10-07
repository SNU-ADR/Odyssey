# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
from abc import ABC
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import (
    LaneGraphEdgeMapObject,
    RoadBlockGraphEdgeMapObject,
)
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from shapely.geometry import Point

from odyssey.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)
from odyssey.components.agents.policy.pdm_planner.utils.graph_search.dijkstra import (
    Dijkstra,
)
from odyssey.components.agents.policy.pdm_planner.utils.pdm_geometry_utils import (
    normalize_angle,
)
from odyssey.components.agents.policy.pdm_planner.utils.pdm_path import PDMPath
from odyssey.components.agents.policy.pdm_planner.utils.route_utils import (
    route_roadblock_correction,
)


class AbstractPDMPlanner(AbstractPlanner, ABC):
    """
    Interface for planners incorporating PDM-* variants.
    """

    def __init__(
        self,
        map_radius: float,
    ):
        """
        Constructor of AbstractPDMPlanner.
        :param map_radius: radius around ego to consider
        """

        self._map_radius: int = map_radius  # [m]
        self._iteration: int = 0

        # lazy loaded
        self._map_api: Optional[AbstractMap] = None
        self._route_roadblock_dict: Optional[Dict] = None
        self._route_lane_dict: Optional[Dict[str, LaneGraphEdgeMapObject]] = None

        self._centerline: Optional[PDMPath] = None
        self._drivable_area_map: Optional[PDMOccupancyMap] = None
        self._route_is_gt_matched: bool = False
        self._reference_centerline_discrete_path: Optional[List[StateSE2]] = None

    def _load_route_dicts(self, route_roadblock_ids: List[str]) -> None:
        """
        Loads roadblock and lane dictionaries of the target route from the map-api.
        :param route_roadblock_ids: ID's of on-route roadblocks
        """
        # remove repeated ids while remaining order in list
        route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))

        self._route_roadblock_dict = {}
        self._route_lane_dict = {}

        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(
                id_, SemanticMapLayer.ROADBLOCK_CONNECTOR
            )

            self._route_roadblock_dict[block.id] = block

            for lane in block.interior_edges:
                self._route_lane_dict[lane.id] = lane

    def _route_roadblock_correction(self, ego_state: EgoState) -> None:
        """
        Corrects the roadblock route and reloads lane-graph dictionaries.
        :param ego_state: state of the ego vehicle.
        """
        route_roadblock_ids = route_roadblock_correction(
            ego_state,
            self._map_api,
            self._route_roadblock_dict,
            cut_route_loops=not self._route_is_gt_matched,
        )
        self._load_route_dicts(route_roadblock_ids)

    def _get_discrete_centerline(
        self, current_lane: LaneGraphEdgeMapObject, search_depth: int = 30
    ) -> List[StateSE2]:
        """
        Applies a Dijkstra search on the lane-graph to retrieve discrete centerline.
        :param current_lane: lane object of starting lane.
        :param search_depth: depth of search (for runtime), defaults to 30
        :return: list of discrete states on centerline (x,y,θ)
        """

        reference_centerline = getattr(
            self, "_reference_centerline_discrete_path", None
        )
        if reference_centerline:
            roadblocks = list(self._route_roadblock_dict.values())
            self._last_centerline_path_found = True
            self._last_centerline_start_lane_id = str(current_lane.id)
            self._last_centerline_target_roadblock_id = (
                str(roadblocks[-1].id) if roadblocks else "gt_endpoint"
            )
            self._last_centerline_lane_ids = tuple(
                str(lane.id) for lane in self._route_lane_dict.values()
            )
            return list(reference_centerline)

        roadblocks = list(self._route_roadblock_dict.values())
        roadblock_ids = list(self._route_roadblock_dict.keys())

        # Reference PDM plans short nuPlan scenarios and caps its one-time centerline at 30
        # roadblocks.  Odyssey windows are much longer, while PDMClosedPlanner creates its
        # proposal paths only on iteration zero.  A GT-matched route longer than that therefore
        # ended permanently at roadblock 30.  Expand only this one-time graph search to the full
        # trusted route; per-step proposal propagation is unchanged.
        if self._route_is_gt_matched:
            search_depth = max(search_depth, len(roadblocks))

        # find current roadblock index
        start_idx = np.argmax(
            np.array(roadblock_ids) == current_lane.get_roadblock_id()
        )
        roadblock_window = roadblocks[start_idx : start_idx + search_depth]

        graph_search = Dijkstra(current_lane, list(self._route_lane_dict.keys()))
        route_plan, path_found = graph_search.search(roadblock_window[-1])

        # A long GT window can require a turn hundreds of metres later.  nuPlan's lane graph has
        # no lane-change edges, so the physically occupied lane may terminate before that turn
        # even though an adjacent lane in the same roadblock reaches the route endpoint.  The
        # short reference scenarios rarely expose this.  For a GT-matched long route, select the
        # closest endpoint-reachable sibling once at initialization instead of accepting
        # Dijkstra's deepest dead-end fallback.  Runtime planning and longitudinal control remain
        # unchanged.
        selected_start_lane = current_lane
        if self._route_is_gt_matched and not path_found:
            current_roadblock = self._route_roadblock_dict.get(
                current_lane.get_roadblock_id()
            )
            reachable_siblings = []
            for sibling in getattr(current_roadblock, "interior_edges", ()):
                if sibling.id == current_lane.id:
                    continue
                sibling_plan, sibling_found = Dijkstra(
                    sibling, list(self._route_lane_dict.keys())
                ).search(roadblock_window[-1])
                if not sibling_found:
                    continue
                try:
                    lateral_distance = float(
                        current_lane.baseline_path.linestring.distance(
                            sibling.baseline_path.linestring
                        )
                    )
                except (AttributeError, TypeError, ValueError):
                    lateral_distance = 0.0
                reachable_siblings.append(
                    (lateral_distance, str(sibling.id), sibling, sibling_plan)
                )
            if reachable_siblings:
                _, _, selected_start_lane, route_plan = min(reachable_siblings)
                path_found = True

        # Diagnostic state consumed by the Odyssey wrapper.  A failed lane-graph search is
        # otherwise indistinguishable from a genuine route endpoint once PDM's IDM generator
        # stops on the short fallback path returned by Dijkstra.
        self._last_centerline_path_found = bool(path_found)
        self._last_centerline_start_lane_id = str(selected_start_lane.id)
        self._last_centerline_target_roadblock_id = str(roadblock_window[-1].id)
        self._last_centerline_lane_ids = tuple(str(lane.id) for lane in route_plan)

        centerline_discrete_path: List[StateSE2] = []
        for lane in route_plan:
            centerline_discrete_path.extend(lane.baseline_path.discrete_path)

        return centerline_discrete_path

    def _get_starting_lane(self, ego_state: EgoState) -> LaneGraphEdgeMapObject:
        """
        Returns the most suitable starting lane, in ego's vicinity.
        :param ego_state: state of ego-vehicle
        :return: lane object (on-route)
        """
        starting_lane: LaneGraphEdgeMapObject = None
        on_route_lanes, heading_error = self._get_intersecting_lanes(ego_state)

        if on_route_lanes:
            # 1. Option: find lanes from lane occupancy-map
            # select lane with lowest heading error
            starting_lane = on_route_lanes[np.argmin(np.abs(heading_error))]
            return starting_lane

        else:
            # 2. Option: find any intersecting or close lane on-route
            closest_distance = np.inf
            for edge in self._route_lane_dict.values():
                if edge.contains_point(ego_state.center):
                    starting_lane = edge
                    break

                distance = edge.polygon.distance(ego_state.car_footprint.geometry)
                if distance < closest_distance:
                    starting_lane = edge
                    closest_distance = distance

        return starting_lane

    def _get_intersecting_lanes(
        self, ego_state: EgoState
    ) -> Tuple[List[LaneGraphEdgeMapObject], List[float]]:
        """
        Returns on-route lanes and heading errors where ego-vehicle intersects.
        :param ego_state: state of ego-vehicle
        :return: tuple of lists with lane objects and heading errors [rad].
        """
        assert (
            self._drivable_area_map
        ), "AbstractPDMPlanner: Drivable area map must be initialized first!"

        ego_position_array: npt.NDArray[np.float64] = ego_state.rear_axle.array
        ego_rear_axle_point: Point = Point(*ego_position_array)
        ego_heading: float = ego_state.rear_axle.heading

        intersecting_lanes = self._drivable_area_map.intersects(ego_rear_axle_point)

        on_route_lanes, on_route_heading_errors = [], []
        for lane_id in intersecting_lanes:
            if lane_id in self._route_lane_dict.keys():
                # collect baseline path as array
                lane_object = self._route_lane_dict[lane_id]
                lane_discrete_path: List[
                    StateSE2
                ] = lane_object.baseline_path.discrete_path
                lane_state_se2_array = np.array(
                    [state.array for state in lane_discrete_path], dtype=np.float64
                )
                # calculate nearest state on baseline
                lane_distances = (
                    ego_position_array[None, ...] - lane_state_se2_array
                ) ** 2
                lane_distances = lane_distances.sum(axis=-1) ** 0.5

                # calculate heading error
                heading_error = (
                    lane_discrete_path[np.argmin(lane_distances)].heading - ego_heading
                )
                heading_error = np.abs(normalize_angle(heading_error))

                # add lane to candidates
                on_route_lanes.append(lane_object)
                on_route_heading_errors.append(heading_error)

        return on_route_lanes, on_route_heading_errors
