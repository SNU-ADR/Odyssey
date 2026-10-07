"""Scoring graph with stable sidecar IDs separate from dense matcher indices."""
import copy

import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, Point

from odyssey_benchmark.sdroute_sdf import _module

SDGraph = _module('graph').SDGraph


class ManualSDGraph(SDGraph):
    def __init__(self, base, manual, replaces):
        # Keep the shared planner graph and its shortest-path cache untouched.
        self.__dict__ = copy.copy(base.__dict__)
        self.edge_ids = list(range(len(base))) + list(manual)
        self.edge_index = {eid: i for i, eid in enumerate(self.edge_ids)}
        if any(e >= 0 for e in manual):
            raise ValueError('Manual edge IDs must be negative')
        if any(e < 0 or e >= len(base) for e in replaces):
            raise ValueError('Replacement edge does not exist in base graph')
        self.xy = dict(base.xy)
        self.geom = list(base.geom)
        self.u, self.v = list(base.u), list(base.v)
        self.way, self.hw, self.name = list(base.way), list(base.hw), list(base.name)
        self.twin = list(base.twin)
        for eid, edge in manual.items():
            xy = np.asarray(edge['xy'], dtype=float)
            if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
                raise ValueError(f'Invalid manual edge geometry: {eid}')
            if np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() <= 0:
                raise ValueError(f'Zero-length manual edge: {eid}')
            for node, p in ((edge['u'], xy[0]), (edge['v'], xy[-1])):
                if node in self.xy and not np.allclose(self.xy[node], p, atol=.01, rtol=0):
                    raise ValueError(f'Inconsistent manual endpoint: edge {eid}, node {node}')
                self.xy[node] = p
            self.geom.append(xy)
            self.u.append(edge['u']); self.v.append(edge['v'])
            self.way.append(0); self.hw.append('road'); self.name.append(edge['name'])
            self.twin.append(-1)
        self.u, self.v = np.asarray(self.u), np.asarray(self.v)
        self.way, self.twin = np.asarray(self.way), np.asarray(self.twin)
        self.length = np.array([np.linalg.norm(np.diff(g, axis=0), axis=1).sum() for g in self.geom])
        self._acc = [np.r_[0., np.cumsum(np.linalg.norm(np.diff(g, axis=0), axis=1))] for g in self.geom]
        self.lines = [LineString(g) for g in self.geom]
        self.active = [i for i in range(len(self.geom)) if i not in replaces]
        self.tree = STRtree([self.lines[i] for i in self.active])
        self.out = {}
        for i in self.active:
            self.out.setdefault(int(self.u[i]), []).append(i)
        self._dij = {}
        # Removed parent edges are not valid reverse matches for their retained twins.
        for i, twin in enumerate(self.twin):
            if twin in replaces:
                self.twin[i] = -1

    def candidates(self, p, radius):
        hits = self.tree.query(Point(float(p[0]), float(p[1])).buffer(radius))
        return [self.active[int(k)] for k in np.atleast_1d(hits)]
