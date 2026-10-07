"""Local logged branch intent for IDM agents; dynamics remain nuPlan's IDM.

A route is matched only at a map fork, against 5-25 m of the actor's own future
source positions. Ambiguous or unavailable intent leaves stock curvature ranking.
"""
import numpy as np
from shapely.geometry import LineString, Point


def build_source_route_intent(state, origin, source_row):
    positions = np.asarray(state.get("position", []), dtype=np.float64)
    valid = np.asarray(state.get("valid", []), dtype=bool).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] < 2 or source_row < 0:
        return None
    count = min(len(positions), len(valid))
    if source_row >= count or not valid[source_row]:
        return None
    rows = np.flatnonzero(valid[source_row:count]) + source_row
    if len(rows) < 2:
        return None
    xy = positions[rows, :2] + np.asarray(origin, dtype=np.float64)
    if not np.isfinite(xy).all():
        return None
    lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    # Do not interpret missing intervals or teleports as branch intent.
    discontinuity = np.flatnonzero((np.diff(rows) > 5) | (lengths > 8.0))
    if len(discontinuity):
        end = int(discontinuity[0])
        xy, lengths = xy[:end + 1], lengths[:end]
    if not len(lengths) or float(np.sum(lengths)) < 12.0:
        return None
    return xy, np.r_[0.0, np.cumsum(lengths)]


def _edge_paths(edge, excluded):
    prefix = [(float(p.x), float(p.y)) for p in edge.baseline_path.discrete_path]
    if len(prefix) < 2:
        return []
    successors = [
        successor for successor in (getattr(edge, "outgoing_edges", ()) or ())
        if str(getattr(successor, "id", "")) not in excluded
    ]
    paths = []
    for successor in successors or [None]:
        coords = prefix + (
            [(float(p.x), float(p.y)) for p in successor.baseline_path.discrete_path]
            if successor is not None else []
        )
        paths.append(LineString(coords))
    return paths


def choose_source_branch(agent, candidates, stock, decisions=None, incoming_edge=None):
    """Return a confident GT-aligned edge or the stock signal-aware choice."""
    intent = getattr(agent, "_odyssey_route_intent", None)
    if intent is None or not candidates:
        return stock
    xy, progress = intent
    incoming_edge = incoming_edge or agent.end_segment
    incoming = incoming_edge.baseline_path.discrete_path
    if not incoming:
        return stock
    fork = np.asarray([incoming[-1].x, incoming[-1].y], dtype=np.float64)
    distances = np.linalg.norm(xy - fork, axis=1)
    anchor = int(np.argmin(distances))
    if distances[anchor] > 6.0:
        return stock
    ahead = progress - progress[anchor]
    sample_ids = np.flatnonzero((ahead >= 5.0) & (ahead <= 25.0))
    if not len(sample_ids) or float(ahead[sample_ids[-1]]) < 12.0:
        return stock
    if len(sample_ids) > 12:
        sample_ids = sample_ids[np.linspace(0, len(sample_ids) - 1, 12, dtype=int)]
    samples = [Point(*point) for point in xy[sample_ids]]
    ranked = []
    for edge in candidates:
        paths = _edge_paths(edge, getattr(agent, "_odyssey_excluded_connector_ids", ()))
        if paths:
            score = min(
                float(np.mean([path.distance(point) for point in samples]))
                for path in paths
            )
            ranked.append((score, edge))
    if not ranked:
        return stock
    ranked.sort(key=lambda pair: pair[0])
    best_score, best = ranked[0]
    margin = ranked[1][0] - best_score if len(ranked) > 1 else np.inf
    # Map/log disagreement and near-identical branches are not route evidence.
    if best_score > 4.5 or margin < 1.25:
        return stock
    if decisions is not None:
        decisions.append((
            str(getattr(agent, "_odyssey_route_token", "")),
            int(getattr(agent, "_odyssey_route_source_row", -1)),
            str(incoming_edge.id), str(stock.id) if stock is not None else "",
            str(best.id), float(best_score), float(margin) if np.isfinite(margin) else np.nan,
        ))
    return best
