"""Build the route roadblock chain by walking the lane graph along the GT ego trajectory.

Port of the Odyssey simulator's closed-loop route builder (`TrajectoryNavigation._build_global_route`)
onto the nuPlan map API. The dataset's own `roadblock_ids` can contain consecutive pairs that are
not connected in the lane graph; `route_roadblock_correction` then reads the gaps as "something is
missing" and invents a BFS detour, which can drop the roadblock ego actually drives through. Here
the chain is grown edge-by-edge from `outgoing_edges`, so a disconnected chain cannot be produced.

Deviation from the reference: once the GT trajectory no longer reaches any fork candidate, the
reference walks on with `_straightest`; this port follows the dataset `roadblock_ids` order instead
(`_next_on_dataset_route`) and only falls back to `_straightest` when no candidate is on that route.
NAVSIM scenes carry 5 s of future ego poses, so the trajectory guides roughly the first 25-75 m of
the 120 m reference line and the dataset route fills the rest.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle

# Constants from the reference; values kept as-is.
ROUTE_EXTEND = 150.0        # [m] walked past the end of the GT trajectory; > the 120 m reference line
MAX_ROUTE_LANES = 200       # loop guard
START_CANDIDATES = 4        # walking only follows edges, so a wrong start never recovers -> try several
START_SEARCH_RADIUS = 6.0   # [m]
GUIDE_TAIL_FRAC = 0.25      # fork candidates share the intersection entry; only their tails separate
GUIDE_MATCH_TOL = 4.0       # [m] no candidate within this -> the trajectory ended here

# Not from the reference. An oncoming lane sits well inside START_SEARCH_RADIUS and overlaps the GT
# path, so the position-only _chain_fit can prefer it -- the reference had a 200-point trajectory to
# break the tie, a NAVSIM scene has 5 s (and none at all when ego is stopped). pi/2 is the definition
# of "points backwards", not a tuned value; it matches _get_intersecting_lanes picking on heading error.
MAX_START_HEADING_ERROR = np.pi / 2


def _polyline(lane: LaneGraphEdgeMapObject, cache: Dict[str, np.ndarray]) -> np.ndarray:
    """(N,2) baseline path of `lane` in global coordinates."""
    poly = cache.get(lane.id)
    if poly is None:
        poly = np.array([[s.x, s.y] for s in lane.baseline_path.discrete_path], dtype=np.float64)
        cache[lane.id] = poly
    return poly


def _mean_min_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean over `a` of the distance to the nearest point of `b` [m]."""
    return float(np.mean(np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1), axis=1)))


def _tail_distance(lane, traj: np.ndarray, cache) -> Optional[float]:
    """Distance from the lane's tail to the trajectory [m].

    The full polyline cannot be used: at an intersection the candidates share the entry, so their
    whole-polyline minimum distance ties at ~0 and does not separate a left turn from a straight.
    """
    poly = _polyline(lane, cache)
    if len(poly) == 0:
        return None
    tail = poly[int(len(poly) * (1.0 - GUIDE_TAIL_FRAC)):]
    if len(tail) == 0:
        tail = poly
    return _mean_min_distance(tail, traj)


def _heading_error(lane, ego_pose, cache) -> Optional[float]:
    """|lane heading at the point nearest ego - ego heading| [rad], as _get_intersecting_lanes measures it."""
    path = lane.baseline_path.discrete_path
    poly = _polyline(lane, cache)
    if not path or len(poly) == 0:
        return None
    nearest = int(np.argmin(np.linalg.norm(poly - np.array([ego_pose.x, ego_pose.y]), axis=1)))
    return abs(normalize_angle(path[nearest].heading - ego_pose.heading))


def _chain_fit(chain: Sequence, traj: np.ndarray, cache) -> Optional[float]:
    """How well the chain covers the trajectory: mean distance from each GT point to the chain [m]."""
    polys = [p for p in (_polyline(lane, cache) for lane in chain) if len(p)]
    if not polys:
        return None
    return _mean_min_distance(traj, np.vstack(polys))


def _straightest(current, candidates):
    """Candidate that bends the current heading least (absolute value only -> no 180-degree wrap)."""
    base = current.baseline_path.discrete_path[-1].heading
    best, best_turn = None, None
    for c in candidates:
        path = c.baseline_path.discrete_path
        if not path:
            continue
        turn = abs(normalize_angle(path[-1].heading - base))
        if best_turn is None or turn < best_turn:
            best, best_turn = c, turn
    return best if best is not None else candidates[0]


def _next_on_dataset_route(current, candidates, order: Dict[str, int]):
    """Candidate whose roadblock lies on the dataset route, preferring the one that comes next.

    Returns None when no candidate is on that route, leaving the choice to `_straightest`.
    """
    cur = order.get(current.get_roadblock_id(), -1)
    on_route = [(order[rb], c) for c, rb in ((c, c.get_roadblock_id()) for c in candidates) if rb in order]
    if not on_route:
        return None
    ahead = [pair for pair in on_route if pair[0] > cur]
    return min(ahead or on_route, key=lambda pair: pair[0])[1]


def _walk_along(start, traj: np.ndarray, order: Dict[str, int], cache) -> List:
    """Follow `outgoing_edges` from `start`, steered by the trajectory while it still reaches ahead."""
    chain, seen = [start], {start.id}
    extending, extended = False, 0.0
    while len(chain) < MAX_ROUTE_LANES:
        candidates = [c for c in chain[-1].outgoing_edges if c.id not in seen]
        if not candidates:
            break
        if len(candidates) == 1:
            nxt = candidates[0]
        elif extending:
            nxt = _next_on_dataset_route(chain[-1], candidates, order) or _straightest(chain[-1], candidates)
        else:
            nxt, best = None, None
            for c in candidates:
                d = _tail_distance(c, traj, cache)
                if d is not None and (best is None or d < best):
                    nxt, best = c, d
            if best is None or best > GUIDE_MATCH_TOL:
                extending = True
                nxt = _next_on_dataset_route(chain[-1], candidates, order) or _straightest(chain[-1], candidates)
        if nxt is None:
            break
        chain.append(nxt)
        seen.add(nxt.id)
        if extending:
            extended += float(nxt.baseline_path.length)
            if extended >= ROUTE_EXTEND:
                break
    return chain


def build_route_roadblock_ids(
    map_api: AbstractMap,
    ego_pose,
    gt_xy_global,
    dataset_roadblock_ids: Sequence[str],
) -> Optional[List[str]]:
    """Return a graph-connected roadblock id chain, or None when no route could be built.

    :param ego_pose: StateSE2 of the current frame, GLOBAL frame
    :param gt_xy_global: (N,2) GLOBAL ego positions from the current frame onwards
    """
    if gt_xy_global is None:
        return None
    traj = np.asarray(gt_xy_global, dtype=np.float64)
    if traj.ndim != 2 or len(traj) < 2:
        return None

    layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    proximal = map_api.get_proximal_map_objects(Point2D(ego_pose.x, ego_pose.y), START_SEARCH_RADIUS, layers)
    lanes = [lane for layer in layers for lane in proximal.get(layer, [])]
    if not lanes:
        return None

    cache: Dict[str, np.ndarray] = {}
    ego = np.array([[ego_pose.x, ego_pose.y]], dtype=np.float64)
    scored = []
    for lane in lanes:
        err = _heading_error(lane, ego_pose, cache)
        if err is not None and err <= MAX_START_HEADING_ERROR:
            scored.append((_mean_min_distance(ego, _polyline(lane, cache)), lane))
    if not scored:
        return None
    scored.sort(key=lambda pair: pair[0])

    order = {rb: i for i, rb in enumerate(dict.fromkeys(dataset_roadblock_ids))}
    best_chain, best_score = None, None
    for _, start in scored[:START_CANDIDATES]:
        chain = _walk_along(start, traj, order, cache)
        score = _chain_fit(chain, traj, cache)
        if score is not None and (best_score is None or score < best_score):
            best_chain, best_score = chain, score
    if best_chain is None:
        return None

    return list(dict.fromkeys(lane.get_roadblock_id() for lane in best_chain))
