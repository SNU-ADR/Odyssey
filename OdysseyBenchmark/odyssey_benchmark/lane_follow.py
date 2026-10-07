#!/usr/bin/env python3
"""Lane-following score, computed after the rollout: at stop lines where a lane must be chosen, was
the ego in a lane that leads to the exit?

What it measures
----------------
Wherever the route has a roadblock followed by a connector, the lanes of that roadblock that
**continue into the next connector** are fixed. The map already records them as `exit_lanes`;
we do not decide them.

    lane L is exit-feasible  <=>  exit_lanes(L) ∩ lanes(next connector) != empty set

A choice stop line is one where only **some** of the lanes are exit-feasible, and only those are
scored. At a stop line where every lane leads to the same result there is nothing to judge. This
is not a metric for wrong-way driving or leaving the route (that is the job of DS offroad and the
SD route) -- it only checks whether the right lane was chosen when a lane had to be chosen.

Rules
-----
Stop line
    Choice stop lines only. If the route lists several connectors in a row, all of them are exit
    targets. The stop-line point is the end of the lane that ends first along the direction of
    travel (laterally, the middle of the lane ends); the stop line is the segment through that
    point perpendicular to the lane direction, HALF_WIDTH (25 m) to each side -- covering the full
    road width, including the opposite carriageway.
Crossing
    The first moment the ego centre crosses that segment in the lane direction (ego heading
    within 90 degrees of the lane direction). Stop lines are taken in route order, and only
    crossings after the previous stop line was judged count. If the rollout starts already past
    the line inside that roadblock's lanes, the first sample is judged (`started_past`); if the
    trajectory ends within NEAR_TOL before the line without crossing it, the last sample is judged
    (`ended_near`).
Verdict
    Pass if, at the sample just before the crossing, the ego centre is inside an exit-feasible
    lane polygon (with matching heading). Another lane of the same roadblock (`wrong_lane`), a road
    off the route (`off_route`, e.g. the opposite carriageway) or off road (`off_road`) fails. If
    that sample is off-lane or already inside the next connector, walk back up to NEAR_TOL and
    use the last lane the ego was in.
Late entry
    Even on a pass, if any sample during the EARLY_M (10 m) driven before reaching the stop line
    was outside a correct lane, the ego entered late (`late`). Correct lanes are the exit-feasible
    lanes plus the lanes of earlier roadblocks that lead into them without a lane change (when the
    roadblock is short). Distance is the path length driven by the ego centre -- straight-line
    coordinates along the stop-line direction diverge from the real distance when walking back
    through curves and intersections. If the rollout starts inside that stretch and is in a
    correct lane from the first sample, it is not late -- the model did not drive before that.
Log baseline
    The log (human) trajectory is judged with the same map and rules. Stop lines where the log
    does not get full credit (pass, 1) are not judged in this run either and are removed from the
    number of choice stop lines (N) (`excluded`). This removes stop lines the log never crossed
    (the rollout starts past the line), places where map fragments disagree with human driving
    (2-6 m roadblocks), places where the log ends before the stop line, and places where the
    human entered late. The log decides which stop lines are "valid" -- the scorer does not
    re-decide it with a proxy such as roadblock length.
Score
    Per stop line: pass 1, late pass LATE_CREDIT (0.5), fail 0.
    lane_score = sum of credits / stop lines reached.

The previous rule (v1, 0.7 ** number of fails) assigned the ego only to route lanes and judged
at the last sample driven in that roadblock. An ego that approached a choice stop line on the
opposite carriageway or off road therefore "never drove that roadblock" and had no judgement
point -- choosing the wrong lane went unscored. This rule judges at the moment of crossing a stop
line that spans the full road width, so such approaches also count as fails. Validation: on human
logs, 175 of 177 stop lines pass (the other 2 are roadblocks with lanes shorter than 5 m).

Lane assignment uses polygons
-----------------------------
Do not use "nearest lane within N m of its centreline". Lanes are 3.4 m wide, so a 4 m threshold
**assigns poses more than 2 m outside a lane to that lane** (measured: in one run, a departure
2.3-2.9 m away was assigned to the lane). The map's lane polygons tile the roadblock without gaps
(area of union == sum of areas), so polygon containment is the map's own answer, and a pose inside
no polygon stays **off-lane**. In practice, inside a polygon the distance to the centreline was at
most 1.83 m.

**Verdicts do not use lane numbers.** `left_neighbor`/`right_neighbor` cannot be trusted (see the
`_apply_start_lane_shift` docstring in ego_agent.py: some lanes list themselves as a neighbour
because UNSORTED `interior_edges` is sliced with SORTED indices), and when numbers collide an
infeasible lane takes a feasible lane's number. Verdicts use **lane ids**; numbers are for
reporting.

None when not measurable
------------------------
If no choice stop line was reached, lane_score is **None**, not 0. 0 is a measurement ("all
wrong"); None means "could not be measured".

RouteDS multiplies in `lane_penalty` (P_PLC, below); run as a script, this file only computes the
score and writes it to json/csv.

What it reads
-------------
`rollout_trajectory.npz` alone is enough.

    ds_states / ds_sim_steps         dense ego (rear axle, global UTM)
    route_roadblock_ids              route roadblock order
    driving_inputs_json.map          lane_graph (route lane graph, baked by metric_manager)
                                     and roadblock polygons (where a failure was, for reporting)
    expert_xy/_origin/_heading       log trajectory (rear axle, 0.5 s). global = expert_xy + expert_origin
    initial_ego_center                 + initial_ego_center (metric_manager's definition). Steps come
    driving_inputs_json.rc           from rc.gt -- rc.gt[k] is the log at step k·stride, so find the i0
                                     where expert matches rc.gt[i0:] and count from i0·stride.

**Older runs** without `lane_graph` fall back to the scenario pkl's `map_features`
(path given with `--scenarios`).

Usage
-----
    python -m odyssey_benchmark.lane_follow <run_dir or glob> [--csv out.csv]

Requires numpy and shapely. Needs no nuplan, map or simulator env.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

#: Extent (m) of the stop line to each side of the stop-line point. From the middle of the route
#: lanes across the median to the outer edge of the opposite carriageway is about 25 m on a
#: divided road. Human logs crossed at most about 11 m to the side of the point. Any wider and
#: crossings of the line's extension while driving on a neighbouring road would count.
HALF_WIDTH = 25.0

#: A trajectory that ends within this distance before the line counts as reaching the stop line.
#: It is also how far to walk back when the judged sample is off-lane or already in the next
#: connector.
NEAR_TOL = 5.0

#: Full credit requires being in a correct lane from this distance before the stop line (m).
EARLY_M = 10.0

#: Credit for a stop line where the ego was in a correct lane at the line but entered it within EARLY_M.
LATE_CREDIT = 0.5

#: P_PLC (the lane term of RouteDS): for each judged (reached) stop line, a fail multiplies by
#: FAIL_FACTOR, a late pass by LATE_FACTOR and a pass by 1. 1.0 when no stop line was judged. Stop
#: lines not reached are not multiplied -- RC already penalises the distance not driven.
FAIL_FACTOR = 0.7
LATE_FACTOR = 0.9

#: Lane fragments shorter than this do not set the stop-line position; a sub-metre fragment
#: would otherwise pull the stop line back to the start of the roadblock.
MIN_LANE_LEN = 1.0

#: Consecutive samples required before an assignment is trusted, so that one or two frames jumping
#: sideways while crossing a lane boundary are not counted as a lane change (0.3 s at 10 Hz).
MIN_RUN = 3

#: Tolerance that only absorbs the shared boundary of two adjacent lanes. Not a threshold.
EDGE_BUFFER = 0.25

#: Excludes lanes running in the opposite direction.
MAX_HEADING_DIFF = np.pi / 3

#: nuPlan ego rear axle -> box centre. Lane verdicts are made at the centre.
REAR_TO_CENTRE = 1.461

STREET = "LANE_SURFACE_STREET"
CONNECTOR = "LANE_SURFACE_UNSTRUCTURE"


@dataclass
class Gate:
    """One choice stop line on the route. Coordinates are global UTM."""

    rb: str
    next_rb: str
    lanes: List[str]
    feasible: List[str]
    p: List[float]               # stop-line point
    t: List[float]               # unit lane direction (the stop line is perpendicular to it)
    approach: List[str]          # correct lanes before the stop line: feasible + upstream lanes
                                 # that lead into them without a lane change


@dataclass
class Verdict:
    """Verdict for one stop line. If `reached` is False the other fields are meaningless."""

    rb: str
    next_rb: str
    n_lanes: int
    n_feasible: int
    reached: bool                # whether the stop line was reached
    how: str                     # "" | "passed" | "ended_near" | "started_past"
    step: Optional[int]          # sim step of the judged sample
    lane: Optional[str]          # lane of the judged sample (among route lanes)
    where: str                   # "" | feasible | wrong_lane | other_route_lane | off_lane | off_route | off_road
    passed: Optional[bool]       # None if not reached
    offset_m: Optional[float]    # lateral offset of the judged sample from the stop-line point,
                                 # left + (reporting)
    entered_m: Optional[float]   # on a pass, distance driven to the stop line after last entering a
                                 # correct lane (m). None: in a correct lane from the first sample
    credit: Optional[float]      # 1 | LATE_CREDIT | 0. None if not reached


@dataclass
class Result:
    """Result of one rollout. lane_score None means it could not be measured."""

    token: str
    lane_score: Optional[float]
    n_stops: int                 # choice stop lines on the route
    n_reached: int               # of those, the ones reached (score denominator)
    n_pass: int                  # in an exit-feasible lane at the line (incl. late passes)
    n_late: int                  # of those, entered within EARLY_M
    n_fail: int
    n_ended_near: int            # reached ones where the trajectory ended before the line
    n_no_feasible: int           # stop lines with no exit-feasible lane (map defect) -- not scored
    source: str                  # "npz" | "scenario_pkl"
    reason: str                  # why there is no value; "ok" when there is one
    stops: List[Verdict] = field(default_factory=list)
    # stop lines dropped because the log did not get full credit (the log's verdicts)
    excluded: List[Verdict] = field(default_factory=list)


# --------------------------------------------------------------------------- geometry
def _arc(p):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def _project(polyline, xy):
    """(distance to the centreline, heading there, arc length left to the roadblock end)."""
    P = np.asarray(polyline, dtype=float)
    if len(P) < 2:
        return float("inf"), 0.0, float("inf")
    s = _arc(P)
    a, d = P[:-1], np.diff(P, axis=0)
    dd = np.maximum(np.einsum("ij,ij->i", d, d), 1e-9)
    t = np.clip(np.einsum("ij,ij->i", xy - a, d) / dd, 0.0, 1.0)
    foot = a + t[:, None] * d
    dist = np.linalg.norm(foot - xy, axis=1)
    j = int(np.argmin(dist))
    at = s[j] + t[j] * (s[j + 1] - s[j])
    return float(dist[j]), float(np.arctan2(d[j, 1], d[j, 0])), float(s[-1] - at)


def _smooth(seq, k=MIN_RUN):
    out = list(seq)
    i = 0
    while i < len(out):
        j = i
        while j < len(out) and out[j] == out[i]:
            j += 1
        if j - i < k and i > 0:
            for t in range(i, j):
                out[t] = out[i - 1]
        i = j
    return out


class LaneGraph:
    """Lanes on the route. Assigned by polygon containment; verdicts use lane ids."""

    def __init__(self, lanes: Dict[str, dict], keep_rbs=None):
        """If `keep_rbs` is given, only lanes of those roadblocks are candidates.

        The verdict is about the route, so candidates are limited to the route. Anything off the
        route is not assigned but `None` -- off-lane -- and whether that spot is the opposite
        carriageway or off road is told separately by the roadblock polygons (`_where`).
        """
        from shapely.geometry import Polygon
        from shapely.strtree import STRtree
        if keep_rbs is not None:
            keep = {str(x) for x in keep_rbs}
            lanes = {k: v for k, v in lanes.items() if v["rb"] in keep}
        self.lanes = lanes
        self.by_rb: Dict[str, List[str]] = {}
        for lid, lane in lanes.items():
            self.by_rb.setdefault(lane["rb"], []).append(lid)
        self._ids, polys = [], []
        for lid, lane in lanes.items():
            P = np.asarray(lane["polygon"], dtype=float)
            if len(P) < 4:
                continue
            g = Polygon(P)
            if not g.is_valid:
                g = g.buffer(0)
            if g.is_empty:
                continue
            self._ids.append(lid)
            polys.append(g)
        self._polys = polys
        self._tree = STRtree(polys) if polys else None

    def rb_type(self, rb: str) -> str:
        ids = self.by_rb.get(rb, [])
        return self.lanes[ids[0]]["type"] if ids else "?"

    def lane_index(self, lid: str) -> Optional[int]:
        """Index counted from the left. For reporting only -- map neighbour lists can be wrong."""
        n = self.lanes[lid].get("left_count")
        return None if n is None else int(n)

    def assign(self, xy, heading):
        """(lane id, distance to centreline, distance left to roadblock end). Outside: (None, inf, inf)."""
        from shapely.geometry import Point
        if self._tree is None:
            return None, float("inf"), float("inf")
        pt = Point(float(xy[0]), float(xy[1]))
        hits = [int(i) for i in self._tree.query(pt)]
        inside = [i for i in hits if self._polys[i].covers(pt)]
        if not inside:
            near = [(self._polys[i].distance(pt), i) for i in hits]
            near = [(d, i) for d, i in near if d <= EDGE_BUFFER]
            if not near:
                return None, float("inf"), float("inf")
            inside = [min(near)[1]]
        best, bd, br = None, 1e18, float("inf")
        for i in inside:
            lid = self._ids[i]
            dist, ang, rem = _project(self.lanes[lid]["polyline"], xy)
            if abs((ang - float(heading) + np.pi) % (2 * np.pi) - np.pi) > MAX_HEADING_DIFF:
                continue
            if dist < bd:
                best, bd, br = lid, dist, rem
        return (best, bd, br) if best is not None else (None, float("inf"), float("inf"))


# --------------------------------------------------------------------------- inputs
def lane_graph_from_npz(z, keep_rbs=None) -> Optional[LaneGraph]:
    """Route lane graph baked by metric_manager. None if absent (older runs)."""
    if "driving_inputs_json" not in z.files:
        return None
    inputs = json.loads(str(z["driving_inputs_json"]))
    raw = (inputs.get("map") or {}).get("lane_graph")
    if not raw:
        return None
    lanes = {}
    for lid, lane in raw.items():
        lanes[str(lid)] = {
            "rb": str(lane["rb"]), "type": str(lane["type"]),
            "polygon": np.asarray(lane["polygon"], dtype=float),
            "polyline": np.asarray(lane["polyline"], dtype=float),
            "exit": [str(x) for x in lane.get("exit", ())],
            "left_count": lane.get("left_count"),
        }
    return LaneGraph(lanes, keep_rbs)


#: (pkl path, origin) -> lane dict. Several arms share one scene, so reopening a 12 MB pkl for every
#: run makes backfills tens of times slower. The origin is part of the key because it is used to
#: lift coordinates to UTM -- the same scene with a different origin has different coordinates.
#: Per-route filtering is not cached (only the STRtree is rebuilt; unpickling is the costly part).
_GRAPH_CACHE: Dict[tuple, Optional[Dict[str, dict]]] = {}


def lane_graph_from_scenario(scenario_pkl: str, origin, keep_rbs=None) -> Optional[LaneGraph]:
    """Fallback for older runs. The scenario pkl is a UTM-parallel frame with the t=0 ego rear
    axle as origin.

    Checked: subtracting REAR_TO_CENTRE along the heading from the pkl's ego centre matches the
    sidecar's gt_xy with a residual of 0.000 m. Hence UTM = local + initial_ego_center.
    """
    origin = np.asarray(origin, dtype=float)
    key = (scenario_pkl, round(float(origin[0]), 3), round(float(origin[1]), 3))
    lanes = _GRAPH_CACHE.get(key)
    if lanes is not None:
        return LaneGraph(lanes, keep_rbs) if lanes else None
    with open(scenario_pkl, "rb") as fh:
        scene = list(pickle.load(fh).values())[0]
    lanes = {}
    for lid, v in scene["map_features"].items():
        if "roadblock_id" not in v:
            continue
        lanes[str(lid)] = {
            "rb": str(v["roadblock_id"]), "type": str(v["type"]),
            "polygon": np.asarray(v["polygon"], dtype=float) + origin,
            "polyline": np.asarray(v["polyline"], dtype=float) + origin,
            "exit": [str(x) for x in v["exit_lanes"]],
            "left_count": len(v["left_neighbor"]),
        }
    _GRAPH_CACHE[key] = lanes
    return LaneGraph(lanes, keep_rbs) if lanes else None


def ego_centre(ds_states):
    xy = np.asarray(ds_states, dtype=float)[:, :2]
    h = np.asarray(ds_states, dtype=float)[:, 2]
    return xy + REAR_TO_CENTRE * np.stack([np.cos(h), np.sin(h)], 1), h


# --------------------------------------------------------------------------- stop lines
def gates(graph: LaneGraph, route: List[str]) -> tuple:
    """-> ([Gate], stop lines dropped for having no exit-feasible lane). Choice stop lines only."""
    out, no_feasible = [], 0
    for i, rb in enumerate(route[:-1]):
        nxt = route[i + 1]
        if graph.rb_type(rb) != STREET or graph.rb_type(nxt) != CONNECTOR:
            continue
        lanes = graph.by_rb.get(rb, [])
        # If the route lists several connectors in a row after this stop line (an intersection at
        # the end of the route), a lane that enters a later connector directly also follows the
        # route (measured: 4 of 5 human-log fails had this shape -- the middle lane skips
        # route[i+1] and continues straight into route[i+2]).
        nxt_lanes = set()
        for r in route[i + 1:]:
            if graph.rb_type(r) != CONNECTOR:
                break
            nxt_lanes |= set(graph.by_rb.get(r, []))
        feas = [l for l in lanes if set(graph.lanes[l]["exit"]) & nxt_lanes]
        if not lanes:
            continue
        if not feas:
            no_feasible += 1
            continue
        if len(feas) == len(lanes):                      # every lane continues -- nothing to choose
            continue
        ends, tans, full = [], [], []
        for l in lanes:
            P = np.asarray(graph.lanes[l]["polyline"], float)
            if len(P) < 2 or np.linalg.norm(P[-1] - P[-2]) < 1e-6:
                continue
            d = P[-1] - P[-2]
            ends.append(P[-1])
            tans.append(d / np.linalg.norm(d))
            full.append(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) >= MIN_LANE_LEN)
        if not ends:
            continue
        ends, t = np.asarray(ends), np.mean(tans, axis=0)
        t /= np.linalg.norm(t)
        # End of the lane that ends first. At skewed intersections lane ends are staggered by
        # several metres (measured up to 3.9 m) -- with a later line, an ego in the earliest-ending
        # lane would already be inside the connector when it crossed.
        c = ends.mean(0)
        use = ends[np.asarray(full)] if any(full) else ends
        p = c + float(np.min((use - c) @ t)) * t
        # Correct lanes before the stop line. If the roadblock is shorter than EARLY_M, the stretch
        # before it lies in the previous roadblock -- there, the correct lanes are those that reach
        # an exit-feasible lane without a lane change (following exits only).
        approach = set(feas)
        for r in reversed(route[:i]):
            up = {l for l in graph.by_rb.get(r, []) if set(graph.lanes[l]["exit"]) & approach}
            if not up:
                break
            approach |= up
        out.append(Gate(rb=rb, next_rb=nxt, lanes=list(lanes), feasible=feas,
                        p=p.tolist(), t=t.tolist(), approach=sorted(approach)))
    return out, no_feasible


# --------------------------------------------------------------------------- verdicts
def _where(pt, blocks, route_set):
    from shapely.geometry import Point
    P = Point(float(pt[0]), float(pt[1]))
    ids = {rb for rb, poly in blocks if poly.covers(P)}
    if not ids:
        return "off_road"
    return "off_lane" if ids & route_set else "off_route"


def judge(gs: List[Gate], graph: LaneGraph, xy, heading, steps, blocks, route) -> List[Verdict]:
    """One Verdict per stop line. The visualisation uses this function too."""
    xy, heading, steps = np.asarray(xy, float), np.asarray(heading, float), np.asarray(steps)
    lids = _smooth([graph.assign(xy[i], heading[i])[0] for i in range(len(xy))])
    arc = _arc(xy)
    last = len(xy) - 1
    route_set = set(route)
    out, start = [], 0
    for G in gs:
        base = dict(rb=G.rb, next_rb=G.next_rb, n_lanes=len(G.lanes), n_feasible=len(G.feasible))
        p, t = np.asarray(G.p), np.asarray(G.t)
        n = np.array([-t[1], t[0]])
        along, side = (xy - p) @ t, (xy - p) @ n              # along (+ once past the line), lateral
        fwd = np.cos(heading - np.arctan2(t[1], t[0])) > 0.0  # moving in the lane direction
        j, how = None, ""
        if start == 0 and along[0] >= 0.0 and abs(side[0]) <= HALF_WIDTH and lids[0] in G.lanes:
            # Warm-up handed over past the line. Judge it if still in that roadblock's lane; if
            # already in the connector, the warm-up made the choice, so it is not counted.
            j, how = 0, "started_past"
        if j is None:
            i0 = max(start, 1)
            for i in np.flatnonzero((along[i0 - 1:-1] < 0.0) & (along[i0:] >= 0.0) & fwd[i0:]) + i0:
                f = -along[i - 1] / (along[i] - along[i - 1])  # interpolate lateral position at the crossing
                if abs(side[i - 1] + f * (side[i] - side[i - 1])) <= HALF_WIDTH:
                    j, how = int(i) - 1, "passed"
                    break
            if j is None and last >= start and -NEAR_TOL <= along[last] < 0.0 \
                    and abs(side[last]) <= HALF_WIDTH:
                j, how = last, "ended_near"
        if j is None:
            out.append(Verdict(**base, reached=False, how="", step=None, lane=None, where="",
                               passed=None, offset_m=None, entered_m=None, credit=None))
            continue
        start = j + 1
        # If the judged sample is off-lane or already in the next connector (it passed the end of
        # its own lane first), walk back up to NEAR_TOL and judge by the last lane it was in.
        nxt = set(graph.by_rb.get(G.next_rb, []))
        k, back = j, 0.0
        while k > 0 and (lids[k] is None or lids[k] in nxt) and back <= NEAR_TOL:
            back += float(np.linalg.norm(xy[k] - xy[k - 1]))
            k -= 1
        if lids[k] is not None and lids[k] not in nxt:
            j = k
        lane = lids[j]
        if lane in G.feasible:
            where = "feasible"
        elif lane in G.lanes:
            where = "wrong_lane"
        elif lane is not None:
            where = "other_route_lane"
        else:
            where = _where(xy[j], blocks, route_set)
        entered, credit = None, 0.0
        if where == "feasible":
            # Walk back from the judged sample to the first sample of the stretch spent in a correct
            # lane. Do not stop at the previous stop line -- if the ego stayed in a correct lane since
            # before it, the stretch starts there.
            approach = set(G.approach)
            k = j
            while k > 0 and lids[k - 1] in approach:
                k -= 1
            # distance driven from the entry sample to the judged sample + judged sample to the line
            entered = None if k == 0 else round(float(arc[j] - arc[k] + max(-along[j], 0.0)), 2)
            credit = 1.0 if entered is None or entered >= EARLY_M else LATE_CREDIT
        out.append(Verdict(**base, reached=True, how=how, step=int(steps[j]), lane=lane,
                           where=where, passed=where == "feasible", offset_m=round(float(side[j]), 2),
                           entered_m=entered, credit=credit))
    return out


def score(d, xy, heading, steps, token="", source="npz") -> Result:
    """`d` is the output of load_npz. The trajectory is passed separately so that the human log
    can be scored with the same map.

    `d["log"]` is the log trajectory (xy, heading, steps). Stop lines where the log does not get
    full credit are not judged and are removed from N. Without a log there is no way to decide
    which stop lines count, so no value is produced (`no_log`)."""
    gs = d["gates"]
    keep, excluded = [], []

    def empty(reason, **kw):
        base = dict(token=token, lane_score=None, n_stops=len(keep), n_reached=0, n_pass=0,
                    n_late=0, n_fail=0, n_ended_near=0, n_no_feasible=d["no_feasible"],
                    source=source, reason=reason, excluded=excluded)
        base.update(kw)
        return Result(**base)

    if d["graph"] is None:
        return empty("no_lane_graph")
    if not d["route"]:
        return empty("no_route")
    if d.get("log") is None:
        return empty("no_log")
    lxy, lh, lst = d["log"]
    log = judge(gs, d["graph"], lxy, lh, lst, d["blocks"], d["route"])
    keep = [i for i, v in enumerate(log) if v.reached and v.credit == 1.0]
    excluded = [v for i, v in enumerate(log) if i not in set(keep)]
    if xy is None or len(xy) < MIN_RUN:
        return empty("no_ego")
    # Judge every stop line in route order, then select -- excluded stop lines still take part in
    # the crossing order (start).
    judged = judge(gs, d["graph"], xy, heading, steps, d["blocks"], d["route"])
    stops = [judged[i] for i in keep]
    reached = [v for v in stops if v.reached]
    if not reached:
        return empty("no_stop_reached", stops=stops)
    n_pass = sum(1 for v in reached if v.passed)
    return Result(token=token, lane_score=sum(v.credit for v in reached) / len(reached),
                  n_stops=len(stops), n_reached=len(reached), n_pass=n_pass,
                  n_late=sum(1 for v in reached if v.passed and v.credit < 1.0),
                  n_fail=len(reached) - n_pass,
                  n_ended_near=sum(1 for v in reached if v.how == "ended_near"),
                  n_no_feasible=d["no_feasible"], source=source, reason="ok", stops=stops,
                  excluded=excluded)


def load_npz(npz_path: str, scenarios: str = ""):
    """Everything scoring needs: map, route and ego. Shared by the validation panel and the CLI."""
    with np.load(npz_path, allow_pickle=True) as z:
        return load_pinned(z, scenarios)


def load_pinned(z, scenarios: str = ""):
    """Body of load_npz. z is an open npz or the same-shaped input that end-of-scene scoring is
    about to save (driving_metrics.PinnedInputs) -- end-of-scene scoring and re-scoring read the
    same input with the same code."""
    from shapely import wkb
    route = ([str(x) for x in z["route_roadblock_ids"]]
             if "route_roadblock_ids" in z.files else [])
    graph, source = lane_graph_from_npz(z, keep_rbs=route), "npz"
    if graph is None and scenarios:
        pkl = _scenario_pkl(z, scenarios)
        if pkl:
            graph = lane_graph_from_scenario(pkl, z["initial_ego_center"], keep_rbs=route)
            source = "scenario_pkl"
    inputs = json.loads(str(z["driving_inputs_json"])) if "driving_inputs_json" in z.files else {}
    m = inputs.get("map") or {}
    blocks = [(str(t), wkb.loads(g, hex=True))
              for t, ty, g in zip(m.get("tokens", ()), m.get("types", ()), m.get("wkb", ()))
              if ty == "ROADBLOCK"]
    if "ds_states" in z.files:
        xy, heading = ego_centre(z["ds_states"])
        steps = np.asarray(z["ds_sim_steps"], int)
    else:
        xy = heading = steps = None
    log = log_track(z, inputs.get("rc") or {})
    gs, no_feasible = gates(graph, route) if graph is not None else ([], 0)
    return dict(route=route, graph=graph, blocks=blocks, gates=gs, no_feasible=no_feasible,
                xy=xy, heading=heading, steps=steps, source=source, log=log)


def log_track(z, rc):
    """Log (human) trajectory -> (centre xy, heading, sim step). None if absent.

    Position and heading are the expert (0.5 s, rear axle) baked by metric_manager. Steps come
    from rc.gt: rc.gt[k] is the log rear axle at step k·stride (sdroute_metric:
    reference_gt_dense[::stride]), and the expert starts at step rc.handoff. Check that both are
    the same points; if not, fail rather than guess the steps. Interpolate linearly to every step
    so the log is judged the same way as the dense ego.
    """
    need = ("expert_xy", "expert_origin", "expert_heading", "initial_ego_center")
    if any(k not in z.files for k in need) or not rc.get("gt") or not rc.get("stride"):
        return None
    n = int(z["expert_source_pose_count"]) if "expert_source_pose_count" in z.files \
        else len(z["expert_xy"])
    ex = (np.asarray(z["expert_xy"], float)[:n] + np.asarray(z["expert_origin"], float)[:2]
          + np.asarray(z["initial_ego_center"], float)[:2])
    eh = np.unwrap(np.asarray(z["expert_heading"], float)[:n])
    gt, stride = np.asarray(rc["gt"], float), int(rc["stride"])
    i0 = int(rc["handoff"]) // stride
    m = min(n, len(gt) - i0)
    if m < MIN_RUN or not np.allclose(ex[:m], gt[i0:i0 + m], atol=1e-3):
        raise ValueError("expert trajectory differs from rc.gt[handoff/stride:] -- "
                         "cannot determine log steps")
    s0 = (i0 + np.arange(m)) * stride
    dense = np.arange(s0[0], s0[-1] + 1)
    st = np.stack([np.interp(dense, s0, ex[:m, 0]), np.interp(dense, s0, ex[:m, 1]),
                   np.interp(dense, s0, eh[:m])], 1)
    cxy, ch = ego_centre(st)
    return cxy, ch, dense


def score_npz(npz_path: str, token: str = "", scenarios: str = "") -> Result:
    d = load_npz(npz_path, scenarios)
    return score(d, d["xy"], d["heading"], d["steps"], token=token or _token_of(npz_path),
                 source=d["source"])


def score_pinned(z, token: str = "") -> Result:
    """Score one open scoring input (called by driving_metrics.apply_rules). Does not rebuild the
    map from the scenario pkl -- what is not in the pinned input is not scored."""
    d = load_pinned(z)
    return score(d, d["xy"], d["heading"], d["steps"],
                 token=token or (str(z["scene"]) if "scene" in z.files else ""), source=d["source"])


#: Rollout scene name. Log names themselves contain '-' (veh-14), so split on the trailing window.
_SCENE_RE = re.compile(r"^(?P<log>.+)_w(?P<start>\d+)-(?P<end>\d+)$")


def _scenario_pkl(z, scenarios: str) -> str:
    """Find an older run's lane graph in the scenario pkl.

    The window start frame must match too. One log is cut into several scenes, so picking by log
    name alone grabs another window's map (measured: 41 of 46 runs attached to the wrong pkl that
    way, leaving only 6 judgement points). Returns an empty string unless there is exactly one
    candidate -- not scoring is better than scoring against the wrong map.
    """
    scene = str(z["scene"]) if "scene" in z.files else ""
    m = _SCENE_RE.match(scene)
    if not m:
        return ""
    pattern = "*_%s_f%04d_*" % (m.group("log"), int(m.group("start")))
    hits = sorted(glob.glob(os.path.join(scenarios, pattern, "all_scenarios.pkl")))
    return hits[0] if len(hits) == 1 else ""


def _token_of(path: str) -> str:
    m = re.search(r"exp_([0-9a-f]{16})_", path)
    if m:
        return m.group(1)
    m = re.search(r"([0-9a-f]{16})", os.path.basename(os.path.dirname(path)))
    return m.group(1) if m else os.path.basename(path)


def find_npz(target: str) -> List[str]:
    if target.endswith(".npz"):
        return sorted(glob.glob(target))
    hits: List[str] = []
    for root in sorted(glob.glob(target)):
        hits.extend(glob.glob(os.path.join(root, "**", "rollout_trajectory.npz"),
                              recursive=True))
    return sorted(set(hits))


ROW_FIELDS = ("token", "lane_score", "n_stops", "n_reached", "n_pass", "n_late", "n_fail",
              "n_ended_near", "n_no_feasible", "source", "reason")


def lane_penalty(r: Result) -> float:
    """P_PLC = FAIL_FACTOR ** fails x LATE_FACTOR ** late passes. 1.0 when no stop line was judged
    (including runs without a value).

    Example: 2 passes, 2 fails, 1 late -> 0.7 ** 2 x 0.9 = 0.441. n_fail and n_late count reached
    stop lines only.
    """
    return float(FAIL_FACTOR ** r.n_fail * LATE_FACTOR ** r.n_late)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="run_dir, glob, or .npz path")
    ap.add_argument("--csv", help="csv file to write results to")
    ap.add_argument("--scenarios", default="",
                    help="scenario directory for older runs without lane_graph")
    a = ap.parse_args(argv)

    paths = find_npz(a.target)
    if not paths:
        print(f"no npz found: {a.target}", file=sys.stderr)
        return 1

    rows: List[Result] = [score_npz(p, scenarios=a.scenarios) for p in paths]

    ok = [r for r in rows if r.lane_score is not None]
    print(f"npz {len(paths)} files, scored {len(ok)}/{len(rows)}")
    if ok:
        reached = sum(r.n_reached for r in ok)
        passed = sum(r.n_pass for r in ok)
        print(f"  choice stop lines reached {reached}, passed {passed} ({100 * passed / reached:.1f} %)"
              f", late (entered within {EARLY_M:g} m) {sum(r.n_late for r in ok)}"
              f", ended before the line {sum(r.n_ended_near for r in ok)}")
    for reason in sorted({r.reason for r in rows if r.lane_score is None}):
        print(f"  no value, {reason}: {sum(1 for r in rows if r.reason == reason)}")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ROW_FIELDS))
            w.writeheader()
            for r in rows:
                w.writerow({k: getattr(r, k) for k in ROW_FIELDS})
        print(f"wrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
