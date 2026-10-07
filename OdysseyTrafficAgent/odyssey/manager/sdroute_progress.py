"""Ordered SD-polyline geometry and the GT -> SD lobe match behind RC's denominator.

`build_reference` answers one question: which occurrence of the SD route did this
GT track actually drive, and over what ``[s_start, s_end]``? A U/P-turn route passes
the same place twice, so a nearest-point start attaches to the return lobe. Instead
of thresholding per-point distance, it walks the WHOLE GT against each of up to 20
start candidates and takes the lowest total cost (mean/p90 offset plus a span-vs-GT
length term) -- sequence consistency, not a corridor gate, resolves the ambiguity.

`sdroute_reference.build_coverage_reference` wraps this for production scoring.
Whether the MODEL stayed on route is a separate question answered by the SDF matcher
(`sdroute_prefix.py`) against the directed SD graph, not by this polyline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt


Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ProgressConfig:
    """Only the fields the surviving reference builder reads. Distances in m, angles in rad.

    This module has no corridor/heading thresholds. Whether the model stayed on route is decided
    by the SDF matcher (`sdroute_prefix.py`) against the directed SD graph.
    """

    # Multi-start matcher that picks the first lobe by looking at the whole GT track.
    gt_initial_candidates: int = 20
    gt_candidate_separation_m: float = 8.0
    gt_candidate_extra_distance_m: float = 40.0
    gt_length_cost_weight: float = 8.0
    gt_distance_cost_clip_m: float = 30.0
    gt_profile_bin_m: float = 0.25

    # Ordered projection window that only searches around the previous match.
    forward_window_m: float = 60.0
    initial_backward_m: float = 15.0
    initial_forward_m: float = 40.0
    backward_window_m: float = 15.0
    max_step_s_slack_m: float = 5.0

    min_reference_length_m: float = 0.5


@dataclass(frozen=True)
class Projection:
    s: float
    d: float
    segment: int
    heading: float
    xy: Array


class RouteGeometry:
    """Ordered polyline and continuous point-to-segment projection onto it."""

    def __init__(self, xy: Array):
        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
            raise ValueError("route_xy must have shape (N, 2), N >= 2")
        if not bool(np.isfinite(xy).all()):
            raise ValueError("route_xy contains non-finite coordinates")

        # Zero-length segments are excluded from projection candidates, but the original
        # ordered geometry is kept.
        self.xy = xy
        self.a = xy[:-1]
        self.v = xy[1:] - xy[:-1]
        self.length = np.linalg.norm(self.v, axis=1)
        if not bool((self.length > 1e-9).any()):
            raise ValueError("route has no non-zero segment")
        self.s = np.concatenate([[0.0], np.cumsum(self.length)])
        self.heading = np.arctan2(self.v[:, 1], self.v[:, 0])

    @property
    def total_length(self) -> float:
        return float(self.s[-1])

    def _all_projections(self, point: Array) -> Tuple[Array, Array, Array, Array]:
        """(s, signed-d, projected-xy, raw-t) of the projection onto every segment."""
        p = np.asarray(point, dtype=np.float64).reshape(2)
        den = np.maximum(self.length * self.length, 1e-12)
        t = np.einsum("ij,ij->i", p - self.a, self.v) / den
        t = np.clip(t, 0.0, 1.0)
        q = self.a + t[:, None] * self.v
        dist = np.linalg.norm(p - q, axis=1)
        cross = self.v[:, 0] * (p[1] - q[:, 1]) - self.v[:, 1] * (p[0] - q[:, 0])
        signed = np.where(cross < 0.0, -dist, dist)
        return self.s[:-1] + t * self.length, signed, q, t

    def project(self, point: Array, s_lo: Optional[float] = None,
                s_hi: Optional[float] = None) -> Projection:
        """Nearest projection onto the part of the route overlapping ``[s_lo, s_hi]``.

        Projecting and then clamping s would make d and s refer to different points. Here each
        segment's allowed t range is cut first, so the returned s/d/xy always come from one point.
        """
        p = np.asarray(point, dtype=np.float64).reshape(2)
        den = np.maximum(self.length * self.length, 1e-12)
        raw_t = np.einsum("ij,ij->i", p - self.a, self.v) / den
        t_lo = np.zeros_like(raw_t)
        t_hi = np.ones_like(raw_t)
        valid = self.length > 1e-9

        if s_lo is not None:
            valid &= self.s[1:] >= float(s_lo)
            t_lo = np.maximum(t_lo, (float(s_lo) - self.s[:-1]) /
                              np.maximum(self.length, 1e-12))
        if s_hi is not None:
            valid &= self.s[:-1] <= float(s_hi)
            t_hi = np.minimum(t_hi, (float(s_hi) - self.s[:-1]) /
                              np.maximum(self.length, 1e-12))
        valid &= t_lo <= t_hi

        if not bool(valid.any()):
            # Never silently jump globally on a bad window. Only when the window lies entirely past
            # the route end/start, snap to the nearest valid endpoint.
            target = self.total_length if s_lo is not None and s_lo >= self.total_length else 0.0
            return self.at_s(target, point=p)

        t = np.minimum(np.maximum(raw_t, t_lo), t_hi)
        q = self.a + t[:, None] * self.v
        dist = np.linalg.norm(p - q, axis=1)
        i = int(np.argmin(np.where(valid, dist, np.inf)))
        cross = self.v[i, 0] * (p[1] - q[i, 1]) - self.v[i, 1] * (p[0] - q[i, 0])
        signed_d = -float(dist[i]) if cross < 0.0 else float(dist[i])
        return Projection(
            s=float(self.s[i] + t[i] * self.length[i]),
            d=signed_d,
            segment=i,
            heading=float(self.heading[i]),
            xy=q[i].copy(),
        )

    def at_s(self, s: float, point: Optional[Array] = None) -> Projection:
        """Point at route arc length ``s``. If point is given, also compute signed-d from it."""
        value = float(np.clip(s, 0.0, self.total_length))
        i = int(np.searchsorted(self.s, value, side="right") - 1)
        i = min(max(i, 0), len(self.length) - 1)
        seg_len = max(float(self.length[i]), 1e-12)
        t = float(np.clip((value - self.s[i]) / seg_len, 0.0, 1.0))
        q = self.a[i] + t * self.v[i]
        signed_d = 0.0
        if point is not None:
            p = np.asarray(point, dtype=np.float64).reshape(2)
            dist = float(np.linalg.norm(p - q))
            cross = self.v[i, 0] * (p[1] - q[1]) - self.v[i, 1] * (p[0] - q[0])
            signed_d = -dist if cross < 0.0 else dist
        return Projection(value, signed_d, i, float(self.heading[i]), q.copy())

    def initial_candidates(self, point: Array, config: ProgressConfig) -> List[Projection]:
        """Distinct route-lobe candidates around the first point."""
        ss, dd, qq, _ = self._all_projections(point)
        order = np.argsort(np.abs(dd))
        best_d = float(abs(dd[order[0]]))
        out: List[Projection] = []
        for raw_i in order:
            i = int(raw_i)
            if abs(dd[i]) > best_d + config.gt_candidate_extra_distance_m:
                break
            if self.length[i] <= 1e-9:
                continue
            if any(abs(ss[i] - previous.s) < config.gt_candidate_separation_m
                   for previous in out):
                continue
            out.append(Projection(float(ss[i]), float(dd[i]), i,
                                  float(self.heading[i]), qq[i].copy()))
            if len(out) >= config.gt_initial_candidates:
                break
        return out


def _angle_error(a: float, b: float) -> float:
    return abs(float(np.arctan2(np.sin(a - b), np.cos(a - b))))


def _path_headings(xy: Array) -> Array:
    """Path heading where stationary points inherit the actual travel direction nearby."""
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 2:
        return np.zeros(len(xy), dtype=np.float64)
    delta = np.gradient(xy, axis=0)
    speed = np.linalg.norm(delta, axis=1)
    heading = np.arctan2(delta[:, 1], delta[:, 0])
    good = np.flatnonzero(speed > 1e-3)
    if len(good) == 0:
        return np.zeros(len(xy), dtype=np.float64)
    for i in np.flatnonzero(speed <= 1e-3):
        heading[i] = heading[good[int(np.argmin(abs(good - i)))]]
    return heading


def _ordered_match(points: Array, route: RouteGeometry, config: ProgressConfig,
                   first: Optional[Projection] = None, anchor_s: Optional[float] = None,
                   monotonic: bool = False) -> List[Projection]:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return []
    if first is None:
        lo = anchor_s - config.initial_backward_m if anchor_s is not None else None
        hi = anchor_s + config.initial_forward_m if anchor_s is not None else None
        first = route.project(points[0], lo, hi)
    matches = [first]
    last_s = first.s
    for i in range(1, len(points)):
        travelled = float(np.linalg.norm(points[i] - points[i - 1]))
        back = 0.0 if monotonic else config.backward_window_m
        # While baking the reference, the cost of the whole GT sequence selects the correct lobe.
        # The SD centerline's turning radius is larger than the actual drive, so some P/U-turns
        # need s to advance 20-30 m in one GT step; a physical gate here would trap the GT itself
        # in the middle of the route.
        forward = max(config.forward_window_m,
                      3.0 * travelled + config.max_step_s_slack_m)
        current = route.project(points[i], last_s - back, last_s + forward)
        matches.append(current)
        last_s = current.s
    return matches


@dataclass(frozen=True)
class RouteReference:
    route: RouteGeometry
    gt_xy: Array
    gt_matches: Tuple[Projection, ...]
    profile_s: Array
    profile_d: Array
    profile_heading: Array
    profile_gt_arc: Array
    s_start: float
    s_end: float
    goal_route_xy: Array
    goal_gt_xy: Array
    gt_distance_m: float
    gt_route_distance_median: float
    gt_route_distance_p90: float
    gt_route_distance_max: float
    span_to_gt_length_ratio: float
    naive_s_start: float
    match_cost: float

    @property
    def length(self) -> float:
        return float(self.s_end - self.s_start)



def build_reference(route_xy: Array, gt_xy: Array,
                    config: ProgressConfig = ProgressConfig()) -> RouteReference:
    """Determine the correct route lobe and ``[s_start, s_end]`` from the whole GT sequence."""
    route = RouteGeometry(route_xy)
    gt = np.asarray(gt_xy, dtype=np.float64)
    if gt.ndim != 2 or gt.shape[1] != 2 or len(gt) < 2:
        raise ValueError("gt_xy must have shape (N, 2), N >= 2")
    gt_step = np.linalg.norm(np.diff(gt, axis=0), axis=1)
    gt_distance = float(gt_step.sum())
    if gt_distance < config.min_reference_length_m:
        raise ValueError("GT trajectory is too short to define route progress")

    naive = route.project(gt[0])
    candidates = route.initial_candidates(gt[0], config)
    if not candidates:
        candidates = [naive]

    best_cost = float("inf")
    best_matches: Optional[List[Projection]] = None
    for first in candidates:
        matches = _ordered_match(gt, route, config, first=first, monotonic=True)
        s = np.array([m.s for m in matches])
        d = np.abs(np.array([m.d for m in matches]))
        span = max(float(s[-1] - s[0]), 0.01)
        ratio = span / max(gt_distance, 0.01)
        clipped = np.minimum(d, config.gt_distance_cost_clip_m)
        # Pick the candidate whose whole future stays close and whose s-span is similar to the
        # length GT travelled. Start and end are not projected independently, which keeps the
        # start from attaching to the return lobe of a U/P-turn.
        cost = (float(clipped.mean()) + 0.25 * float(np.quantile(clipped, 0.9))
                + config.gt_length_cost_weight * abs(float(np.log(np.clip(ratio, 0.05, 20.0)))))
        if cost < best_cost:
            best_cost = cost
            best_matches = matches
    assert best_matches is not None

    matches = tuple(best_matches)
    match_s = np.array([m.s for m in matches], dtype=np.float64)
    match_d = np.array([m.d for m in matches], dtype=np.float64)
    gt_heading = _path_headings(gt)
    gt_arc = np.concatenate([[0.0], np.cumsum(gt_step)])

    # Merge stationary/duplicate s into one bin to get the strict ordering np.interp requires.
    bins = np.round(match_s / config.gt_profile_bin_m).astype(np.int64)
    profile = []
    for value in np.unique(bins):
        mask = bins == value
        circular_heading = float(np.arctan2(np.mean(np.sin(gt_heading[mask])),
                                            np.mean(np.cos(gt_heading[mask]))))
        profile.append((float(np.median(match_s[mask])),
                        float(np.median(match_d[mask])), circular_heading,
                        float(np.median(gt_arc[mask]))))
    profile_np = np.asarray(profile, dtype=np.float64)
    profile_heading = np.unwrap(profile_np[:, 2])

    s_start = float(match_s[0])
    s_end = float(match_s[-1])
    if s_end - s_start < config.min_reference_length_m:
        raise ValueError("GT does not span enough distance on the selected SD route")
    abs_d = np.abs(match_d)
    ratio = (s_end - s_start) / gt_distance
    return RouteReference(
        route=route,
        gt_xy=gt,
        gt_matches=matches,
        profile_s=profile_np[:, 0],
        profile_d=profile_np[:, 1],
        profile_heading=profile_heading,
        profile_gt_arc=profile_np[:, 3],
        s_start=s_start,
        s_end=s_end,
        goal_route_xy=route.at_s(s_end).xy,
        goal_gt_xy=gt[-1].copy(),
        gt_distance_m=gt_distance,
        gt_route_distance_median=float(np.median(abs_d)),
        gt_route_distance_p90=float(np.quantile(abs_d, 0.9)),
        gt_route_distance_max=float(abs_d.max()),
        span_to_gt_length_ratio=float(ratio),
        naive_s_start=float(naive.s),
        match_cost=float(best_cost),
    )
