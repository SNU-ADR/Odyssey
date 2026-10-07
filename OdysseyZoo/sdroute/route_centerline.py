"""Build the routed lane centerline (PLUTO reference line) from a NAVSIM scene, in the ego frame.

Mirrors nuPlan/PDM centerline construction (the same helpers that fill MetricCache.centerline):
_load_route_dicts -> drivable-area map -> _get_starting_lane -> _get_discrete_centerline (Dijkstra).
The map only exists on the Scene, so this runs in the target builder (train + validation) and the
result is cached per token. Returns an ego-local polyline [x, y, dx, dy, heading] (P,5) + valid_mask
(P,), i.e. PLUTO's reference_line tensor for a single line.

`use_sd_route=True` re-matches the same HD centerline onto an SD/OSM graph (`graph`/`matcher`) before the
resample below runs; HD map access is still required to build the initial centerline (default off).

`use_gt_walk=True` replaces the route source: instead of the dataset roadblock_ids + BFS correction,
the chain is walked along the GT ego trajectory (`route_gt_walk`). Everything after that -- starting
lane, Dijkstra centerline, SD re-matching, resample -- is unchanged (default off).

route_target.py, the only caller, turns both on.
"""
import warnings

import numpy as np
import torch
from shapely.geometry import Point

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap

# The PDM helpers come from the HOST repo's navsim (whichever models/<repo> is running).
from navsim.planning.simulation.planner.pdm_planner.abstract_pdm_planner import AbstractPDMPlanner
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath

from . import matcher as sd_matcher
from .graph import get_graph as get_sd_graph
from .route_gt_walk import build_route_roadblock_ids


def _match_onto_sd_graph(discrete_path, map_location: str):
    """Re-match the HD routed centerline onto the SD (OSM) graph for `map_location`.

    Returns a new List[StateSE2], or None when the HMM finds no match. Never falls back to
    `discrete_path`: a cache silently mixing HD and SD routes is indistinguishable from a pure SD one.
    """
    graph = get_sd_graph(map_location)
    xy = np.array([[s.x, s.y] for s in discrete_path], dtype=float)
    matched = sd_matcher.match(graph, xy)
    if matched is None or len(matched["geom"]) < 2:
        return None
    geom = matched["geom"]
    heading = sd_matcher.headings(geom)
    return [StateSE2(float(x), float(y), float(h)) for (x, y), h in zip(geom, heading)]


class _CenterlinePlanner(AbstractPDMPlanner):
    """Concrete shell exposing AbstractPDMPlanner's route/centerline helpers (the base class is ABC)."""

    requires_scenario = False

    def initialize(self, initialization) -> None:
        pass

    def name(self) -> str:
        return "centerline"

    def observation_type(self):
        return None

    def compute_planner_trajectory(self, current_input):
        raise NotImplementedError


def _to_ego(xy: np.ndarray, ox: float, oy: float, oh: float) -> np.ndarray:
    """Global xy -> ego-local (x forward, y left)."""
    c, s = np.cos(oh), np.sin(oh)
    dx, dy = xy[:, 0] - ox, xy[:, 1] - oy
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)


def build_route_centerline_target(
    map_api: AbstractMap,
    roadblock_ids,
    ego_state: EgoState,
    ego_pose,
    num_points: int,
    horizon: float,
    map_radius: float = 50.0,
    use_sd_route: bool = False,
    map_location: str = "",
    use_gt_walk: bool = False,
    gt_xy_global=None,
):
    """Return (route_centerline (P,5) float32, valid_mask (P,) bool). Zeros + all-False on no-route/failure."""
    P = int(num_points)
    rc = torch.zeros(P, 5, dtype=torch.float32)
    mask = torch.zeros(P, dtype=torch.bool)
    if not roadblock_ids:
        return rc, mask

    # Build the routed centerline, mirroring PDMClosedPlanner (correction -> drivable map -> centerline)
    # so the result matches MetricCache.centerline. Expected no-route cases return zeros silently;
    # unexpected errors warn (never a silent no-op) so mass failure during caching is visible.
    try:
        planner = _CenterlinePlanner(map_radius)
        planner._map_api = map_api
        if use_gt_walk:
            # The walked chain is connected by construction, so the BFS correction must NOT run on
            # it -- it only ever repairs gaps, and would reinterpret overlapping connectors as loops.
            walked = build_route_roadblock_ids(map_api, ego_pose, gt_xy_global, list(roadblock_ids))
            if walked is None:
                return rc, mask
            planner._load_route_dicts(walked)
        else:
            planner._load_route_dicts(list(roadblock_ids))
            planner._route_roadblock_correction(ego_state)  # BFS repair of raw ids (as metric_cache does)
        planner._drivable_area_map = PDMDrivableMap.from_simulation(map_api, ego_state, map_radius)
        current_lane = planner._get_starting_lane(ego_state)
        if current_lane is None:
            return rc, mask
        discrete_path = planner._get_discrete_centerline(current_lane)
        if len(discrete_path) < 2:
            return rc, mask
        if use_sd_route:
            discrete_path = _match_onto_sd_graph(discrete_path, map_location)
            if discrete_path is None:
                return rc, mask
        path = PDMPath(discrete_path)
    except Exception as e:
        warnings.warn(f"route_centerline build failed ({type(e).__name__}: {e}); returning zeros")
        return rc, mask

    length = float(path.length)
    s0 = float(path.project(Point(ego_pose.x, ego_pose.y)))   # ego progress along the route
    if length - s0 < 1.0:
        return rc, mask

    # PLUTO parity: fixed ~1 m spacing from ego; tail beyond the route end is zero-padded (mask False).
    spacing = float(horizon) / P
    dist = s0 + np.arange(P) * spacing
    real = dist <= length                                     # points that lie on the actual route
    states = np.asarray(path.interpolate(np.clip(dist, s0, length), as_array=True))  # (P,3) global [x,y,heading]
    pos = _to_ego(states[:, :2], ego_pose.x, ego_pose.y, ego_pose.heading)         # (P,2) ego frame
    heading = states[:, 2] - ego_pose.heading                                      # (P,)
    vec = np.zeros_like(pos)
    vec[:-1] = np.diff(pos, axis=0)                                                 # forward vector; last = 0

    # valid iff this point and its successor are real (successor gives the vector) -> PLUTO's n_valid-1
    valid = real.copy()
    valid[:-1] &= real[1:]
    valid[-1] = False
    pos[~valid] = 0.0; vec[~valid] = 0.0; heading[~valid] = 0.0   # zero-pad like PLUTO's zero-init arrays
    if not valid.any():
        return rc, mask

    rc[:, 0:2] = torch.from_numpy(pos).float()
    rc[:, 2:4] = torch.from_numpy(vec).float()
    rc[:, 4] = torch.from_numpy(heading).float()
    mask[:] = torch.from_numpy(valid)
    return rc, mask
