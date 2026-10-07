"""Direction-aware HMM map matching of a query polyline onto a directed SD (OSM) graph.

Newson-Krumm in the usual form, with one addition: the emission term carries a heading-agreement
penalty against the DIRECTED edge tangent. Without it, a bidirectional road's two directed twins
are indistinguishable (identical geometry) and, worse, a divided road's opposite carriageway is
often the geometrically nearer way -- both failure modes produce a route that points backwards.

  emission(k, e)   = 0.5 (d / SIGMA)^2 + W_DIR * (1 - cos(dpsi)) + class_cost(e)
  transition(a, b) = |graph_dist(a -> b) - euclid(a, b)| / BETA

`graph_dist` is a bounded Dijkstra on the directed edge graph, so it is 0 only for legal
manoeuvres; a U-turn on a oneway pair costs the whole way around the block, which is exactly
what kills the opposite-carriageway hypothesis at the sequence level.
"""
import heapq

import numpy as np

SIGMA = 8.0        # m, SD road centre vs. driven lane centre spread
W_DIR = 8.0        # weight on heading disagreement

# Log-prior on road class. Parking aisles and alleys are tagged highway=service and very often
# run parallel to, and nearer than, the street actually being driven -- with a pure distance
# emission the match hops onto them for a few samples and the route acquires a 10 m bulge.
# Dropping service outright is not an option (it costs 1-6% of ego-to-road coverage), so it is
# demoted instead: 1.5 is worth about 15 m of lateral distance.
CLASS_COST = {
    "motorway": 0.0, "trunk": 0.0, "primary": 0.0, "secondary": 0.0, "tertiary": 0.0,
    "motorway_link": 0.2, "trunk_link": 0.2, "primary_link": 0.2,
    "secondary_link": 0.2, "tertiary_link": 0.2,
    "unclassified": 0.3, "residential": 0.3, "busway": 0.8, "road": 0.5,
    "living_street": 1.2, "service": 1.5,
}

BETA = 6.0         # m, route-vs-straight slack
RADIUS = 35.0      # m, candidate search radius
TOPK = 10          # candidates kept per sample
STEP = 5.0         # m, query resampling
DIJ_BUDGET = 300.0  # m


def class_cost(g, e):
    return CLASS_COST.get(g.hw[e], 0.5)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def emission(g, e, p, psi):
    """Cost of explaining query point p (heading psi) with directed edge e. Returns (cost, s, d, dpsi)."""
    s = g.project(e, p)
    d = g.distance(e, p)
    dpsi = wrap(g.heading_at(e, s) - psi)
    return 0.5 * (d / SIGMA) ** 2 + W_DIR * (1.0 - np.cos(dpsi)) + class_cost(g, e), s, d, float(dpsi)


def resample(arr, step=STEP):
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-6])
    arr = arr[keep]
    if len(arr) < 2:
        return arr
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(arr, axis=0), axis=1))])
    u = np.arange(0.0, s[-1] + 1e-9, step)
    return np.column_stack([np.interp(u, s, arr[:, 0]), np.interp(u, s, arr[:, 1])])


def headings(pts):
    d = np.gradient(pts, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def dijkstra(g, src_node, budget=DIJ_BUDGET):
    """Bounded forward Dijkstra over nodes. Returns {node: (dist, parent_edge)}."""
    key = (src_node, budget)
    if key in g._dij:
        return g._dij[key]
    if len(g._dij) > 4000:
        g._dij.clear()
    dist = {src_node: (0.0, -1)}
    pq = [(0.0, src_node)]
    while pq:
        d, n = heapq.heappop(pq)
        if d > dist[n][0] + 1e-9 or d > budget:
            continue
        for e in g.out.get(n, ()):
            nd = d + float(g.length[e])
            if nd > budget:
                continue
            w = int(g.v[e])
            if w not in dist or nd < dist[w][0] - 1e-9:
                dist[w] = (nd, e)
                heapq.heappush(pq, (nd, w))
    g._dij[key] = dist
    return dist


def edge_path(g, src_node, dst_node, budget=DIJ_BUDGET):
    """Directed-edge list realising the shortest path src_node -> dst_node, or None."""
    if src_node == dst_node:
        return []
    dist = dijkstra(g, src_node, budget)
    if dst_node not in dist:
        return None
    out, n = [], dst_node
    while n != src_node:
        d, e = dist[n]
        if e < 0:
            return None
        out.append(e)
        n = int(g.u[e])
        if len(out) > 200:
            return None
    return out[::-1]


def graph_dist(g, a, b, budget=DIJ_BUDGET):
    """Travel distance from (edge, s) a to (edge, s) b along legal directions, or inf."""
    (ei, si), (ej, sj) = a, b
    if ei == ej:
        if sj >= si - 1e-6:
            return sj - si
        return np.inf
    head = float(g.length[ei]) - si
    if head > budget:
        return np.inf
    dist = dijkstra(g, int(g.v[ei]), budget)
    uj = int(g.u[ej])
    if uj not in dist:
        return np.inf
    return head + dist[uj][0] + sj


def match(g, query, radius=RADIUS, topk=TOPK, step=STEP):
    """Match a polyline onto the graph. Returns dict with the directed-edge sequence + geometry,
    or None if the query is too short or no candidate edge is found near it."""
    q = resample(np.asarray(query, dtype=float), step)
    if len(q) < 2:
        return None
    psi = headings(q)

    states = []
    for p, a in zip(q, psi):
        cand = []
        for e in g.candidates(p, radius):
            cost, s, d, dpsi = emission(g, e, p, a)
            cand.append((cost, e, s, d, dpsi))
        cand.sort(key=lambda t: t[0])
        states.append(cand[:topk])
    if not states[0] or not any(states):
        return None
    states = [s for s in states if s]
    q = q[: len(states)] if len(states) == len(q) else q

    # ---- Viterbi: DAG shortest path over (sample, candidate) states --------
    V = [[c[0] for c in states[0]]]
    B = [[-1] * len(states[0])]
    for k in range(1, len(states)):
        prev, cur = states[k - 1], states[k]
        euclid = float(np.linalg.norm(q[min(k, len(q) - 1)] - q[min(k - 1, len(q) - 1)]))
        row, back = [], []
        for cost_j, ej, sj, _dj, _pj in cur:
            best, arg = np.inf, -1
            for i, (_ci, ei, si, _di, _pi) in enumerate(prev):
                if not np.isfinite(V[k - 1][i]):
                    continue
                gd = graph_dist(g, (ei, si), (ej, sj))
                if not np.isfinite(gd) or gd > 3.0 * euclid + 60.0:
                    continue
                t = abs(gd - euclid) / BETA
                v = V[k - 1][i] + t
                if v < best:
                    best, arg = v, i
            row.append(best + cost_j)
            back.append(arg)
        if not np.isfinite(min(row)):  # broken chain: restart the trellis here
            row = [c[0] for c in cur]
            back = [-1] * len(cur)
        V.append(row)
        B.append(back)

    k = len(states) - 1
    i = int(np.argmin(V[k]))
    seq = []
    while k >= 0:
        cost, e, s, d, dpsi = states[k][i]
        seq.append((k, e, s, d, dpsi))
        i = B[k][i]
        k -= 1
        if i < 0 and k >= 0:
            i = int(np.argmin(V[k]))
    seq = seq[::-1]

    # ---- stitch the matched states into one continuous directed route -----
    edges, gaps = [seq[0][1]], 0
    for _k, e, _s, _d, _p in seq[1:]:
        if e == edges[-1]:
            continue
        p = edge_path(g, int(g.v[edges[-1]]), int(g.u[e]))
        if p is None:
            gaps += 1
        else:
            edges.extend(p)
        edges.append(e)
    s_start = seq[0][2]
    s_end = seq[-1][2]

    parts = []
    for n, e in enumerate(edges):
        a = s_start if n == 0 else 0.0
        b = s_end if n == len(edges) - 1 and e == seq[-1][1] else float(g.length[e])
        sub = g.slice(e, a, b)
        if len(sub):
            parts.append(sub if not parts else sub[1:] if np.allclose(sub[0], parts[-1][-1], atol=1e-6) else sub)
    geom = np.vstack(parts) if parts else np.empty((0, 2))

    return {
        "edges": edges,
        "geom": geom,
        "gaps": gaps,
        "n_states": len(seq),
    }
