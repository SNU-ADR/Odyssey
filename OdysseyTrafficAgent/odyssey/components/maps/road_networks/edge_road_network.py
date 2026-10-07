# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
import logging
from collections import namedtuple
from typing import List

import numpy as np

from odyssey.components.maps.road_networks.base_road_network import BaseRoadNetwork
from odyssey.components.maps.road_networks.base_road_network import LaneIndex
from odyssey.scenario.scenarios.scenario_description import ScenarioDescription as SD

lane_info = namedtuple("edge_lane", ["lane", "entry_lanes", "exit_lanes", "left_lanes", "right_lanes"])


class EdgeRoadNetwork(BaseRoadNetwork):
    """
    Compared to NodeRoadNetwork representing the relation of lanes in a node-based graph, EdgeRoadNetwork stores the
    relationship in edge-based graph, which is more common in real map representation
    """
    def __init__(self):
        super(EdgeRoadNetwork, self).__init__()
        self.graph = {}
        self._lane_localization_cache = None

    def _invalidate_lane_localization_cache(self) -> None:
        self._lane_localization_cache = None

    def _build_lane_localization_cache(self):
        """Pack every center-line segment for one exact vectorized distance query."""
        lane_indexes = list(self.graph)
        lanes = [self.graph[index].lane for index in lane_indexes]
        if not lanes:
            return None

        starts = []
        ends = []
        directions = []
        lateral_directions = []
        prefix_lengths = []
        lane_slots = []
        offsets = []
        try:
            for lane_slot, lane in enumerate(lanes):
                offsets.append(len(starts))
                prefix = 0.0
                for segment, start, end in zip(
                    lane.lane_segments,
                    lane.lane_start_points,
                    lane.lane_end_points,
                ):
                    starts.append(start)
                    ends.append(end)
                    directions.append(segment["direction"])
                    lateral_directions.append(segment["lateral_direction"])
                    prefix_lengths.append(prefix)
                    lane_slots.append(lane_slot)
                    prefix += float(segment["length"])
        except (AttributeError, KeyError, TypeError):
            # A custom lane implementation can still use the scalar API.
            return None

        if not starts:
            return None
        return {
            "lane_indexes": lane_indexes,
            "lanes": lanes,
            "lane_lengths": np.asarray([lane.length for lane in lanes]),
            "starts": np.asarray(starts),
            "ends": np.asarray(ends),
            "directions": np.asarray(directions),
            "lateral_directions": np.asarray(lateral_directions),
            "prefix_lengths": np.asarray(prefix_lengths),
            "lane_slots": np.asarray(lane_slots, dtype=np.intp),
            "offsets": np.asarray(offsets, dtype=np.intp),
        }

    def _vectorized_lane_distances(self, position):
        cache = getattr(self, "_lane_localization_cache", None)
        if cache is None:
            cache = self._build_lane_localization_cache()
            self._lane_localization_cache = cache
        if cache is None:
            return None

        position = np.asarray(position)
        starts = cache["starts"]
        ends = cache["ends"]
        directions = cache["directions"]

        # This is BaseCenterLine.min_lineseg_dist over every map segment at once.
        s = np.multiply(starts - position, directions).sum(axis=1)
        t = np.multiply(position - ends, directions).sum(axis=1)
        h = np.maximum.reduce([s, t, np.zeros(len(s))])
        delta = position - starts
        cross = delta[:, 0] * directions[:, 1] - delta[:, 1] * directions[:, 0]
        segment_distances = np.hypot(h, cross)

        lane_slots = cache["lane_slots"]
        closest_distance = np.minimum.reduceat(segment_distances, cache["offsets"])
        closest_mask = segment_distances == closest_distance[lane_slots]
        closest_candidates = np.flatnonzero(closest_mask)
        _, first_candidate = np.unique(
            lane_slots[closest_candidates], return_index=True
        )
        segments = closest_candidates[first_candidate]

        local_delta = position - starts[segments]
        longitudinal = cache["prefix_lengths"][segments] + np.multiply(
            local_delta, directions[segments]
        ).sum(axis=1)
        lateral = np.multiply(
            local_delta, cache["lateral_directions"][segments]
        ).sum(axis=1)
        return (
            np.abs(lateral)
            + np.maximum(longitudinal - cache["lane_lengths"], 0.0)
            + np.maximum(-longitudinal, 0.0)
        )

    # add & get & update & remove lanes in the edge_road_network.
    def add_lane(self, lane) -> None:
        assert lane.index is not None, "Lane index can not be None"
        self.graph[lane.index] = lane_info(
            lane=lane,
            entry_lanes=lane.entry_lanes or [],
            exit_lanes=lane.exit_lanes or [],
            left_lanes=lane.left_lanes or [],
            right_lanes=lane.right_lanes or []
        )
        self._invalidate_lane_localization_cache()

    def get_lane(self, index: LaneIndex):
        return self.graph[index].lane

    def __isub__(self, other):
        for id, lane_info in other.graph.items():
            self.graph.pop(id)
        self._invalidate_lane_localization_cache()
        return self

    def add(self, other, no_intersect=True):
        for id, lane_info in other.graph.items():
            if no_intersect:
                assert id not in self.graph.keys(), "Intersect: {} exists in two network".format(id)
            self.graph[id] = other.graph[id]
        self._invalidate_lane_localization_cache()
        return self

    # find the closest lane to position.
    def get_closest_lane_index(self, position, return_all=False):
        distances = self._vectorized_lane_distances(position)
        if distances is None:
            index_distance_mapping = []
            for lane_index, lane in self.graph.items():
                index_distance_mapping.append((lane.lane.distance(position), lane_index, lane.lane))
        else:
            cache = self._lane_localization_cache
            index_distance_mapping = list(
                zip(distances, cache["lane_indexes"], cache["lanes"])
            )

        # sort the lane_index according to distance.
        index_distance_mapping = sorted(index_distance_mapping, key=lambda d: d[0])

        if return_all:
            return index_distance_mapping
        else:
            return index_distance_mapping[0]

    # find the shortest path from lane <start> to lane <end>.
    def shortest_path(self, start: str, goal: str):
        return next(self.bfs_paths(start, goal), [])

    def bfs_paths(self, start: str, goal: str) -> List[List[str]]:
        """
        BFS on all exit_lanes to find
         all routes from start to goal.

        :param start: starting edges
        :param goal: goal edge
        :return: list of paths from start to goal.
        """
        lanes = self.graph[start].left_lanes + self.graph[start].right_lanes + [start]

        queue = [(lane, [lane]) for lane in lanes]
        while queue:
            (lane, path) = queue.pop(0)
            if lane not in self.graph:
                yield []
            if len(self.graph[lane].exit_lanes) == 0:
                continue
            for _next in set(self.graph[lane].exit_lanes):
                if _next in path:
                    # circle
                    continue
                if _next == goal:
                    yield path + [_next]
                elif _next in self.graph:
                    queue.append((_next, path + [_next]))

    def get_peer_lanes_from_index(self, lane_index):
        """ Get all parallel lanes of the lane <lane_index>. """
        info = self.graph[lane_index]
        ret = [info.lane]
        return ret + (
            self.get_left_peer_lanes_from_index(lane_index) +
            self.get_right_peer_lanes_from_index(lane_index)
        )

    def get_left_peer_lanes_from_index(self, lane_index):
        """ Get all left parallel lanes of the lane <lane_index>. """
        ret = []
        info = self.graph[lane_index]
        for left_n in info.left_lanes:
            ret.append(self.graph[left_n].lane)
        return ret

    def get_right_peer_lanes_from_index(self, lane_index):
        """ Get all right parallel lanes of the lane <lane_index>. """
        ret = []
        info = self.graph[lane_index]
        for right_n in info.right_lanes:
            ret.append(self.graph[right_n].lane)
        return ret

    def destroy(self):
        """
        Destroy all lanes in this road network
        Returns: None

        """
        super(EdgeRoadNetwork, self).destroy()
        if self.graph is not None:
            for k, v in self.graph.items():
                v.lane.destroy()
                self.graph[k] = None
            self.graph = None

    def __del__(self):
        logging.debug("{} is released".format(self.__class__.__name__))

    def get_map_features(self, interval=2):

        ret = {}
        for id, lane_info in self.graph.items():
            assert id == lane_info.lane.index
            ret[id] = {
                SD.POLYLINE: lane_info.lane.get_polyline(interval),
                SD.POLYGON: lane_info.lane.polygon,
                SD.TYPE: lane_info.lane.type,
                SD.ENTRY: lane_info.entry_lanes,
                SD.EXIT: lane_info.exit_lanes,
                SD.LEFT_NEIGHBORS: lane_info.left_lanes,
                SD.RIGHT_NEIGHBORS: lane_info.right_lanes,
                "speed_limit_kmh": lane_info.lane.speed_limit
            }
        return ret

    def get_all_lanes(self):
        """
        This function will return all lanes in the road network
        :return: list of lanes
        """
        ret = []
        for id, lane_info in self.graph.items():
            ret.append(lane_info.lane)
        return ret
