"""Directed SD-map (OpenStreetMap) routing graph, built from a cached Overpass snapshot.

Construction
  1. nodes -> UTM xy (metres, the frame the cached ego2global_translation lives in)
  2. a way is cut at every node shared by >= 2 ways  ->  EDGE  (junction-to-junction segment)
  3. each edge is emitted as one or two DIRECTED edges depending on oneway

Directionality comes from the OSM convention that a way's node order IS its forward direction,
so `oneway=yes` means node-order only and `oneway=-1` means reverse-only. `junction=roundabout`
implies oneway even when untagged, as do motorway/motorway_link/trunk_link in practice.
"""
import functools
import json
import os

import numpy as np
import pyproj
from shapely import STRtree
from shapely.geometry import LineString, Point

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# UTM zone per NAVSIM/nuPlan map location -- the frame ego2global_translation lives in.
CITY_EPSG = {
    "us-ma-boston": 32619,
    "us-pa-pittsburgh-hazelwood": 32617,
    "us-nv-las-vegas-strip": 32611,
    "sg-one-north": 32648,
}

ONEWAY_YES = {"yes", "true", "1"}
ONEWAY_REV = {"-1", "reverse"}
IMPLIED_ONEWAY_HW = {"motorway", "motorway_link", "trunk_link"}


def _oneway(tags):
    """Return +1 forward-only, -1 backward-only, 0 bidirectional."""
    ow = str(tags.get("oneway", "")).lower()
    if ow in ONEWAY_YES:
        return 1
    if ow in ONEWAY_REV:
        return -1
    if ow in ("no", "false", "0"):
        return 0
    if tags.get("junction") in ("roundabout", "circular"):
        return 1
    if tags.get("highway") in IMPLIED_ONEWAY_HW:
        return 1
    return 0


class SDGraph:
    """Directed edge list + node adjacency + an STRtree over the directed geometries.

    Attributes (index-aligned, one entry per DIRECTED edge):
      geom[i]  (n,2) float64 polyline, already ordered along travel direction
      u[i]/v[i] start / end node id
      way[i]   OSM way id            twin[i]  the reverse directed edge, or -1 if oneway
      hw[i]    highway tag           name[i]  name tag ('' if absent)
      length[i]
    """

    def __init__(self, map_location):
        if map_location not in CITY_EPSG:
            raise KeyError(f"no SD graph for map_location={map_location!r}; have {list(CITY_EPSG)}")
        self.city = map_location
        els = json.load(open(os.path.join(ASSET_DIR, f"osm_raw_{map_location}.json")))["elements"]
        tr = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{CITY_EPSG[map_location]}", always_xy=True)
        nodes = {e["id"]: e for e in els if e["type"] == "node"}
        nid = np.fromiter(nodes.keys(), dtype=np.int64, count=len(nodes))
        lon = np.array([nodes[int(i)]["lon"] for i in nid])
        lat = np.array([nodes[int(i)]["lat"] for i in nid])
        x, y = tr.transform(lon, lat)
        self.xy = {int(i): np.array([px, py]) for i, px, py in zip(nid, x, y)}

        ways = [e for e in els if e["type"] == "way" and len(e.get("nodes", [])) >= 2]
        ways = [w for w in ways if all(n in self.xy for n in w["nodes"])]

        use = {}
        for w in ways:
            for n in w["nodes"]:
                use[n] = use.get(n, 0) + 1

        self.geom, self.u, self.v, self.way, self.hw, self.name, self.twin = [], [], [], [], [], [], []
        for w in ways:
            ns, tags = w["nodes"], w.get("tags", {})
            ow = _oneway(tags)
            cuts = [0] + [k for k in range(1, len(ns) - 1) if use[ns[k]] >= 2] + [len(ns) - 1]
            for a, b in zip(cuts[:-1], cuts[1:]):
                sub = ns[a : b + 1]
                if len(sub) < 2:
                    continue
                pts = np.array([self.xy[n] for n in sub])
                if np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() < 0.5:
                    continue
                fwd, bwd = ow >= 0, ow <= 0
                i0 = len(self.geom)
                if fwd:
                    self._add(pts, sub[0], sub[-1], w["id"], tags)
                if bwd:
                    self._add(pts[::-1], sub[-1], sub[0], w["id"], tags)
                if fwd and bwd:
                    self.twin[i0], self.twin[i0 + 1] = i0 + 1, i0

        self.u = np.array(self.u, dtype=np.int64)
        self.v = np.array(self.v, dtype=np.int64)
        self.way = np.array(self.way, dtype=np.int64)
        self.twin = np.array(self.twin, dtype=np.int64)
        self.length = np.array([np.linalg.norm(np.diff(g, axis=0), axis=1).sum() for g in self.geom])

        self.out = {}
        for i, uu in enumerate(self.u):
            self.out.setdefault(int(uu), []).append(i)

        self.lines = [LineString(g) for g in self.geom]
        self.tree = STRtree(self.lines)
        self._acc = [np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(g, axis=0), axis=1))]) for g in self.geom]
        self._dij = {}

    def _add(self, pts, u, v, wid, tags):
        self.geom.append(pts)
        self.u.append(u)
        self.v.append(v)
        self.way.append(wid)
        self.hw.append(tags.get("highway", ""))
        self.name.append(tags.get("name", ""))
        self.twin.append(-1)

    def __len__(self):
        return len(self.geom)

    # ---- geometry helpers -------------------------------------------------

    def project(self, i, p):
        """Arclength of the perpendicular projection of point p onto directed edge i."""
        return float(self.lines[i].project(Point(float(p[0]), float(p[1]))))

    def distance(self, i, p):
        return float(self.lines[i].distance(Point(float(p[0]), float(p[1]))))

    def point_at(self, i, s):
        a = self._acc[i]
        s = float(np.clip(s, 0.0, a[-1]))
        k = int(np.clip(np.searchsorted(a, s) - 1, 0, len(a) - 2))
        seg = a[k + 1] - a[k]
        t = 0.0 if seg <= 0 else (s - a[k]) / seg
        return self.geom[i][k] + t * (self.geom[i][k + 1] - self.geom[i][k])

    def tangent_at(self, i, s):
        """Unit travel direction of edge i at arclength s (always along the directed sense)."""
        L = self.length[i]
        s0 = float(np.clip(s - 0.5, 0.0, max(L - 1e-3, 0.0)))
        a, b = self.point_at(i, s0), self.point_at(i, min(s0 + 1.0, L))
        d = b - a
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else np.array([1.0, 0.0])

    def heading_at(self, i, s):
        t = self.tangent_at(i, s)
        return float(np.arctan2(t[1], t[0]))

    def slice(self, i, s0, s1):
        """Sub-polyline of edge i between arclengths s0 and s1 (keeps the original vertices)."""
        L = self.length[i]
        s0, s1 = float(np.clip(s0, 0, L)), float(np.clip(s1, 0, L))
        if s1 - s0 < 1e-6:
            return np.empty((0, 2))
        a = self._acc[i]
        keep = self.geom[i][(a > s0 + 1e-6) & (a < s1 - 1e-6)]
        return np.vstack([self.point_at(i, s0), keep, self.point_at(i, s1)])

    def candidates(self, p, radius):
        """Directed edge indices whose geometry passes within `radius` of point p."""
        return [int(k) for k in np.atleast_1d(self.tree.query(Point(float(p[0]), float(p[1])).buffer(radius)))]


@functools.lru_cache(maxsize=len(CITY_EPSG))
def get_graph(map_location):
    """Process-local cache: one SDGraph build per city per worker process."""
    return SDGraph(map_location)
