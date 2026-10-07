"""Shared SDF matcher, independent of the driving planner's navsim installation."""
import importlib.util
from functools import lru_cache
import os
from pathlib import Path

import numpy as np

VERSION = 10
# v10: runs that end by arrival are matched only up to where they entered the
# arrival radius -- see sdroute_score.arrival_zone_keep.
# Branch decisions within this distance past the fork are deferred. A human log can end
# a little over 8 m past a fork while the two branches are still a few metres apart, so
# a shorter distance would score it off-route.
LOOK_M = 8.5
BACK_TOL = 5.0
METHOD = 'hmm_prefix_1m'


@lru_cache(maxsize=None)
def _module(name):
    # The SD-route builder has a single copy, in OdysseyZoo (required).
    zoo = os.environ.get('ODYSSEY_ZOO_ROOT')
    zoo = Path(zoo) if zoo else Path(__file__).resolve().parents[2] / 'OdysseyZoo'
    path = zoo / 'sdroute' / (name + '.py')
    if not path.is_file():
        raise FileNotFoundError(f'{path}: OdysseyZoo SD-route builder not found; set ODYSSEY_ZOO_ROOT')
    # These two self-contained modules only import numpy/shapely/pyproj. Loading
    # by path avoids replacing a planner's already-imported navsim package.
    spec = importlib.util.spec_from_file_location('_odyssey_sdf_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def matcher():
    m = _module('matcher')
    m.class_cost = lambda g, e: 0.0
    original = m.graph_dist

    def graph_dist(g, a, b, budget=m.DIJ_BUDGET):
        (ei, si), (ej, sj) = a, b
        if ei == ej and sj < si - 1e-6 and si - sj <= BACK_TOL:
            return 0.0
        return original(g, a, b, budget)

    m.graph_dist = graph_dist
    return m


@lru_cache(maxsize=16)
def graph(map_location, include_manual=False):
    base = _module('graph').get_graph(map_location)
    if not include_manual:
        return base
    from odyssey_bridge.sd_route import manual_edges, replaced_sd_edges
    from odyssey_bridge.sd_route_follow_graph import ManualSDGraph
    return ManualSDGraph(base, manual_edges(map_location), replaced_sd_edges(map_location))


def endpoint_scored_edges(g, route, seq, P, margin=LOOK_M):
    """Original SDF endpoint-branch deferral, shared without changing policy."""
    R = set(route)
    on = [i for i, e in enumerate(seq) if e in R]
    ignored = {'start': [], 'end': []}
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if (not on or margin <= 0 or arc[-1] <= 2 * margin
            or any(e < 0 or e >= len(g) for e in route)):
        return seq, ignored
    first, last = on[0], on[-1]
    for side, block in (('start', seq[:first]), ('end', seq[last + 1:])):
        if not block or any(int(g.twin[e]) >= 0 and int(g.twin[e]) in R for e in block):
            continue
        start = side == 'start'
        node = int(g.u[seq[first]] if start else g.v[seq[last]])
        if start:
            connected = int(g.v[block[-1]]) == node
            alternatives = any(int(g.v[e]) == node for e in R)
            distance = sum(float(g.length[e]) for e in block) - g.project(block[0], P[0])
        else:
            connected = int(g.u[block[0]]) == node
            alternatives = any(int(g.u[e]) == node for e in R)
            distance = sum(float(g.length[e]) for e in block[:-1]) + g.project(block[-1], P[-1])
        if not connected or not alternatives or distance > margin:
            continue
        if any(int(g.v[a]) != int(g.u[b]) for a, b in zip(block, block[1:])):
            continue
        point = g.geom[block[-1]][-1] if start else g.geom[block[0]][0]
        endpoint = P[0] if start else P[-1]
        if np.linalg.norm(endpoint - point) > margin:
            continue
        j = int(np.argmin(np.linalg.norm(P - point, axis=1)))
        observed = arc[j] if start else arc[-1] - arc[j]
        if observed <= margin:
            ignored[side] = block
    lo = first if ignored['start'] else 0
    hi = last + 1 if ignored['end'] else len(seq)
    return seq[lo:hi], ignored
