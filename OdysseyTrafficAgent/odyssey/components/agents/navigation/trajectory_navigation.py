# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
"""
Used for other agent cars, following a given trajectory.
"""

from collections import deque
import logging

import numpy as np

from odyssey.engine.engine_utils import get_engine

from odyssey.scenario.scenarios.parse_scenario_state import parse_full_trajectory
from odyssey.scenario.scenarios.scenario_description import ScenarioDescription as SD
from odyssey.components.maps.lanes.center_lane import CenterLane

from odyssey.utils import math_utils


class TrajectoryNavigation:
    WAY_POINTS_INTERVAL = 2  # m, interval between waypoints.

    # Next waypoints for controller.
    NUM_WAY_POINTS = 10  # 10 is the same as MetaDrive.
    CHECK_POINT_INFO_DIM = 2  # 2 for x && y coordinates in the global coordinates.

    #  maximum navigation point distance,
    #   used to clip value, should be greater than WAY_POINTS_INTERVAL * MAX_NUM_WAY_POINT
    MAXIMUM_NAVI_POINT_DIST = 30  # m

    # maximum difference between the controlled trajectory
    #  and target route.
    MAX_LATERAL_DIST = 3  # m

    # route related properties.
    ROUTE_WIDTH = 2.0

    # Length to keep walking straight after the trajectory ends. The model's reference line is
    # 120 m, so leave some margin.
    ROUTE_EXTEND = 150.0             # m
    # Upper bound on lanes in the global route (guards against infinite loops).
    MAX_ROUTE_LANES = 200
    # Number and radius of start-lane candidates. A chain is built from each and the one that
    # best fits the trajectory is chosen.
    START_CANDIDATES = 4
    START_SEARCH_RADIUS = 6.0        # m
    # Fraction at the end of the polyline compared with the trajectory at a fork. The entry part
    # is shared by all candidates, so it is left out.
    GUIDE_TAIL_FRAC = 0.25
    # If that distance exceeds this, the trajectory is taken not to reach this fork; the route
    # stops following it.
    GUIDE_MATCH_TOL = 4.0            # m
    # Second check when the test above says "trajectory ended" (_walk_along's second chance).
    GUIDE_AHEAD_POINTS = 10          # how many not-yet-covered trajectory points to look at
    GUIDE_COVERED_TOL = 3.0          # m  a trajectory point within this counts as covered
    GUIDE_AHEAD_TOL = 8.0            # m  keep following a candidate within this of those points

    def __init__(self, agent):
        self.agent = agent

        self.full_traj, self.valid_indicator = parse_full_trajectory(self.agent.object_track)
        self.reference_route = CenterLane(self.full_traj, width=self.ROUTE_WIDTH)

        # agent trajectory information.
        self.last_and_current_long = deque([0.0, 0.0], maxlen=2)
        self.last_and_current_lat = deque([0.0, 0.0], maxlen=2)
        self.last_and_current_heading = deque([0.0, 0.0], maxlen=2)

        # navigation information.
        self._navi_info = np.zeros((self.get_navigation_info_dim(),), dtype=np.float32)

        # Route state. _global_route_lanes is built once in reset() and is immutable afterwards.
        self._global_route_lanes = []
        self._checkpoints_lane_indexes = []

    def reset(self):
        self.current_lane = self.reference_route

        self._global_route_lanes = self._build_global_route()
        self._checkpoints_lane_indexes = list(self._global_route_lanes)

        self.set_route()
        self.update_localization()

    # ----------------------------------------------------------- global route -- #
    def _build_global_route(self):
        """Build, once, the lanes full_traj passes through as a chain connected in the lane graph.

        ``checkpoint_lanes`` is consumed outside this class as "the ego's route"
        (DataManager._attach_route_info -> frame['route_roadblock_ids'], pdm_policy,
        metric_manager). Those consumers assume nuPlan's route convention -- a chain alternating
        ROADBLOCK and ROADBLOCK_CONNECTOR that is connected in the graph.

        Collecting the nearest lane for each trajectory point does not give such a chain: inside
        an intersection the connectors for every approach and direction physically overlap, so the
        nearest lane keeps jumping to neighbouring connectors, and navsim's
        route_roadblock_correction then "repairs" the unconnected list with a detour that can drop
        the very connector the ego had to pass.

        Here only the start lane is chosen by proximity; after that only exit_lanes are followed,
        so the chain cannot break (lane_a -> lane_b implies roadblock(a) -> roadblock(b)). A
        decision is needed only at forks, and there only the last GUIDE_TAIL_FRAC of each
        candidate polyline is compared with the trajectory -- candidates share the intersection
        entry, so the minimum distance over the whole polyline cannot tell them apart.
        """
        traj = np.asarray(self.full_traj, dtype=np.float64)
        if len(traj) < 2:
            return []

        # Picking the single nearest start lane would let that one lane decide the whole route.
        # Only exit_lanes are followed afterwards, so a wrong start cannot recover. Hence build a
        # chain from each of several nearest candidates and keep the one that best fits the
        # trajectory. A chain from a wrong start leaves the trajectory quickly, so they separate
        # clearly.
        net = self.map.road_network
        best_chain, best_score = [], None
        for dist, start, _ in net.get_closest_lane_index(traj[0], return_all=True)[:self.START_CANDIDATES]:
            if dist > self.START_SEARCH_RADIUS:
                break
            if start not in net.graph:
                continue
            chain = self._walk_along(start, traj)
            score = self._chain_fit(chain, traj)
            if score is not None and (best_score is None or score < best_score):
                best_chain, best_score = chain, score

        if not best_chain:
            return []

        logging.info("[navigation] global route: %d lane(s), trajectory fit %.2f m "
                     "(%d trajectory point(s))", len(best_chain), best_score, len(traj))
        return best_chain

    def _walk_along(self, start, traj):
        """Follow the trajectory from start along exit_lanes only. Graph edges only, so it cannot break.

        The fork test (_tail_distance) is kept as is, but when it says "the trajectory has ended"
        a second question is asked -- "is there still uncovered trajectory, and does a candidate
        cover it?" (_remaining_traj + _ahead_distance). _tail_distance measures a candidate's last
        25% against the *whole* trajectory, which is only 20 s (40 points), so even the correct
        branch looks far at its tail when the lane extends beyond the trajectory end. The walk
        then strays onto the straightest branch while trajectory remains, that road dead-ends a
        few tens of metres later, and the route is stuck off the planned path. Evaluated over all
        21,390 navsim scenes: scenes leaving the turn outside the route 544 -> 447, scenes with
        >= 95% coverage 20,070 -> 20,127 (99 recovered, 2 worse).

        Replacing the fork test with this criterion outright was also tried and was worse (19,860
        at >= 95%). So the original test stays wherever it works and this only runs just before
        giving up.

        When the trajectory ends (nothing left to cover and no candidate matches), the walk does
        not stop but continues ROUTE_EXTEND further along the straightest candidate. The model
        needs that much reference line anyway, and the part beyond the trajectory is outside the
        evaluated range, so that choice does not affect the score. Stopping would shorten the
        route near the end of a scenario and make the length the model receives inconsistent.
        """
        net = self.map.road_network
        chain = [start]
        extending, extended = False, 0.0
        while len(chain) < self.MAX_ROUTE_LANES:
            candidates = [c for c in net.graph[chain[-1]].exit_lanes
                          if c in net.graph and c not in chain]
            if not candidates:
                break                     # dead end
            if len(candidates) == 1:
                nxt = candidates[0]
            elif extending:
                nxt = self._straightest(chain[-1], candidates)
            else:
                nxt, best = None, None
                for c in candidates:
                    d = self._tail_distance(c, traj)
                    if d is not None and (best is None or d < best):
                        nxt, best = c, d
                if nxt is None or best > self.GUIDE_MATCH_TOL:
                    # The previous version gave up here immediately; ask once more.
                    ahead = self._remaining_traj(chain, traj)
                    nxt2, best2 = None, None
                    if ahead.shape[0]:
                        for c in candidates:
                            d = self._ahead_distance(c, ahead)
                            if d is not None and (best2 is None or d < best2):
                                nxt2, best2 = c, d
                    if nxt2 is not None and best2 <= self.GUIDE_AHEAD_TOL:
                        nxt = nxt2
                    else:
                        extending = True  # trajectory ends here -> extend going straight
                        nxt = self._straightest(chain[-1], candidates)
            if nxt is None:
                break
            chain.append(nxt)
            if extending:
                info = net.graph.get(nxt)
                extended += info.lane.length if info is not None else 0.0
                if extended >= self.ROUTE_EXTEND:
                    break
        return chain

    def _straightest(self, current, candidates):
        """Candidate for a fork the trajectory does not reach: the one turning least from the
        current lane's direction.

        Compares only the absolute difference between each candidate's end heading and the current
        lane's end heading. Without the sign there is no 180-degree wrap problem.
        """
        net = self.map.road_network
        try:
            base = net.graph[current].lane
            base_heading = base.heading_theta_at(base.length)
        except Exception:
            return candidates[0]
        best, best_turn = None, None
        for c in candidates:
            try:
                lane = net.graph[c].lane
                turn = abs(math_utils.wrap_to_pi(lane.heading_theta_at(lane.length) - base_heading))
            except Exception:
                continue
            if best_turn is None or turn < best_turn:
                best, best_turn = c, turn
        return best if best is not None else candidates[0]

    def _remaining_traj(self, chain, traj):
        """The next GUIDE_AHEAD_POINTS trajectory points not yet covered by chain.

        Measures "how far along" as a trajectory index and returns only the next few points.
        Testing forks against this segment alone keeps a candidate lane's tail from skewing the
        test when the lane is longer than the trajectory.
        """
        net = self.map.road_network
        pts = []
        for index in chain:            # chain is a list of lane indices, not lane objects
            try:
                pts.append(np.asarray(net.get_lane(index).get_polyline(1.0),
                                      dtype=np.float64)[:, :2])
            except Exception:
                continue
        if not pts:
            return traj[:self.GUIDE_AHEAD_POINTS]
        pts = np.concatenate(pts, axis=0)
        d = np.linalg.norm(traj[:, None, :] - pts[None, :, :], axis=-1).min(axis=1)
        covered = np.nonzero(d < self.GUIDE_COVERED_TOL)[0]
        start = int(covered.max()) + 1 if covered.size else 0
        return traj[start:start + self.GUIDE_AHEAD_POINTS]

    def _ahead_distance(self, lane_index, ahead):
        """How far the candidate lane is from the not-yet-covered trajectory points ahead [m]."""
        try:
            p = np.asarray(self.map.road_network.get_lane(lane_index).get_polyline(1.0),
                           dtype=np.float64)[:, :2]
        except Exception:
            return None
        if p.shape[0] == 0 or ahead.shape[0] == 0:
            return None
        return float(np.linalg.norm(ahead[:, None, :] - p[None, :, :], axis=-1).min(axis=1).mean())

    def _tail_distance(self, lane_index, traj):
        """Mean distance [m] between the last GUIDE_TAIL_FRAC of the lane polyline and the trajectory.

        The minimum distance over the whole polyline does not work: intersection candidates
        share the entry, so it is near 0 for all of them and cannot tell them apart.
        """
        try:
            poly = np.asarray(self.map.road_network.graph[lane_index]
                              .lane.get_polyline(self.WAY_POINTS_INTERVAL))[:, :2]
        except Exception:
            return None
        tail = poly[int(len(poly) * (1.0 - self.GUIDE_TAIL_FRAC)):]
        if len(tail) == 0:
            tail = poly
        return float(np.mean(np.min(
            np.linalg.norm(tail[:, None, :] - traj[None, :, :], axis=-1), axis=1)))

    def _chain_fit(self, chain, traj):
        """How well the chain covers the trajectory: mean distance from each point to the chain [m]."""
        pts = []
        for lane_index in chain:
            try:
                pts.append(np.asarray(self.map.road_network.graph[lane_index]
                                      .lane.get_polyline(self.WAY_POINTS_INTERVAL))[:, :2])
            except Exception:
                continue
        if not pts:
            return None
        poly = np.vstack(pts)
        return float(np.mean(np.min(
            np.linalg.norm(traj[:, None, :] - poly[None, :, :], axis=-1), axis=1)))

    def set_route(self):
        """
        Find the shortest path from current_lane_index to destination_lane_index.
        """
        self._checkpoints = self.discretize_reference_trajectory()

    def discretize_reference_trajectory(self):
        ret = []
        length = self.reference_route.length
        num = int(length / self.WAY_POINTS_INTERVAL)
        for i in range(num):
            ret.append(self.reference_route.position(i * self.WAY_POINTS_INTERVAL, 0))
        ret.append(self.reference_route.end)
        return ret

    def update_localization(self):
        """
        The method for updating route / checkpoints information
            according to the associated ego_vehicle motion.
        """
        assert self.reference_route is not None

        # Update ckpt index
        agent_position = self.agent.current_position
        long, lat = self.reference_route.local_coordinates(agent_position)
        route_heading = self.reference_route.heading_theta_at(long)
        self.last_and_current_long.append(long)
        self.last_and_current_lat.append(lat)
        self.last_and_current_heading.append(route_heading)

        # The route is the global chain built in reset(), unchanged. Do not trim it here -- the
        # consumer (navsim abstract_pdm_planner) finds the ego's roadblock in the list with
        #     start_idx = argmax(roadblock_ids == current_lane.get_roadblock_id())
        #     roadblock_window = roadblocks[start_idx : start_idx + 30]
        # uses only the part ahead, and samples the centerline 120 m from the ego's projection.
        # Trimming here would only make that centerline shorter.

        # find the next target goal points,
        #  and <NUM_WAY_POINTS> of target goal points.
        next_idx = max(int(long / self.WAY_POINTS_INTERVAL) + 1, 0)
        next_idx = min(next_idx, len(self.checkpoints) - 1)
        end_idx = min(next_idx + self.NUM_WAY_POINTS, len(self.checkpoints))
        ckpts = self.checkpoints[next_idx:end_idx]
        diff = self.NUM_WAY_POINTS - len(ckpts)
        assert diff >= 0, "Number of Navigation points error!"
        if diff > 0:  # not enough waypoints.
            ckpts += [self.checkpoints[-1] for _ in range(diff)]

        # update the navi information.
        # The navi_information includes:
        #  (NUM_WAY_POINTS * 2)
        #  the heading and rhs difference between current position to the next NUM_WAY_POINTS target points.
        #  (2)
        #  The lateral difference and angle difference between ego_car and associated lane.
        self._navi_info.fill(0.0)
        for index, ckpt in enumerate(ckpts):
            start = index * self.CHECK_POINT_INFO_DIM
            end = (index + 1) * self.CHECK_POINT_INFO_DIM
            self._navi_info[start:end] = self._get_info_for_checkpoint(ckpt)

        # finally add relative information of current position / heading
        # to the target route.
        self._navi_info[end] = math_utils.clip(
            (lat / self.MAX_LATERAL_DIST + 1) / 2, 0.0, 1.0)
        self._navi_info[end + 1] = math_utils.clip(
            (math_utils.wrap_to_pi(route_heading - self.agent.current_heading) / np.pi + 1) / 2, 0.0, 1.0
        )

        self._route_completion = long / self.reference_route.length

    def _get_info_for_checkpoint(self, ckpt):
        """ Get navigation information of agent from the target checkpoints.

        Args:
            ckpt: A numpy array with shape of [2], indicating the
                global position of the checkpoint.
        """

        navi_information = []
        # Project the checkpoint position into the target vehicle's coordination, where
        # +x is the heading and +y is the right hand side.
        dir_vec = ckpt - self.agent.current_position  # get the vector from center of vehicle to checkpoint
        dir_norm = math_utils.norm(dir_vec[0], dir_vec[1])
        # if the checkpoint is too far then crop the direction vector
        if dir_norm > self.MAXIMUM_NAVI_POINT_DIST:
            dir_vec = dir_vec / dir_norm * self.MAXIMUM_NAVI_POINT_DIST

        ckpt_in_heading, ckpt_in_rhs = self.agent.convert_to_local_coordinates(dir_vec)  # project to vehicle's coordination

        # Dim 1: the relative position of the checkpoint in the target vehicle's heading direction.
        navi_information.append(
            math_utils.clip(
                (ckpt_in_heading / self.MAXIMUM_NAVI_POINT_DIST + 1) / 2, 0.0, 1.0)
        )

        # Dim 2: the relative position of the checkpoint in the target vehicle's right hand side direction.
        navi_information.append(
            math_utils.clip(
                (ckpt_in_rhs / self.MAXIMUM_NAVI_POINT_DIST + 1) / 2, 0.0, 1.0)
        )

        return navi_information

    def get_navigation_info_dim(self):
        # The additional 2 is relative heading distance and relative latitude distance.
        return self.NUM_WAY_POINTS * self.CHECK_POINT_INFO_DIM + 2

    def destroy(self):
        self._checkpoints = None
        self._route_completion = 0

    @property
    def engine(self):
        return get_engine()

    @property
    def map(self):
        return self.engine.current_map

    @property
    def checkpoints(self):
        return self._checkpoints
    
    @property
    def checkpoints_lane_indexes(self):
        return self._checkpoints_lane_indexes

    @property
    def checkpoint_lanes(self):
        return [self.map.road_network.get_lane(ckpt) for ckpt in self.checkpoints_lane_indexes]

    @property
    def current_ref_lanes(self):
        return [self.reference_route]

    @property
    def navi_info(self):
        return self._navi_info

    def get_current_lateral_range(self) -> float:
        return self.current_lane.width * 2

    def get_current_lane_width(self) -> float:
        return self.current_lane.width

    def get_current_lane_num(self) -> float:
        return 1
