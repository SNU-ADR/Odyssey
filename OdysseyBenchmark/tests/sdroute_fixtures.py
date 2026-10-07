"""Small directed graphs exercising the real production HMM without map assets."""
import numpy as np
from odyssey.manager.sdroute_progress import RouteGeometry


class TestGraph:
    __test__ = False

    def __init__(self, branching=False, offset=(0., 0.)):
        paths = ([[(0, 0), (40, 0)], [(40, 0), (70, 0)],
                  [(70, 0), (100, 0)], [(40, 0), (40, 20), (70, 20), (70, 0)]]
                 if branching else [[(0, 0), (100, 0)]])
        self.geom = [np.asarray(p, float) + offset for p in paths]
        self.routes = [RouteGeometry(p) for p in self.geom]
        self.length = [r.total_length for r in self.routes]
        self.u = [0, 1, 2, 1] if branching else [0]
        self.v = [1, 2, 3, 2] if branching else [1]
        self.twin = [-1] * len(paths)
        self.hw = ['road'] * len(paths)
        self.out = {}
        for i, u in enumerate(self.u):
            self.out.setdefault(u, []).append(i)
        self._dij = {}

    def __len__(self): return len(self.geom)
    def project(self, e, p): return self.routes[e].project(p).s
    def distance(self, e, p): return abs(self.routes[e].project(p).d)
    def heading_at(self, e, s): return self.routes[e].at_s(s).heading
    def candidates(self, p, radius):
        return [e for e in range(len(self)) if self.distance(e, p) <= radius]
    def slice(self, e, a, b):
        r = self.routes[e]
        return np.vstack([r.at_s(a).xy, r.xy[(r.s > a) & (r.s < b)], r.at_s(b).xy])
