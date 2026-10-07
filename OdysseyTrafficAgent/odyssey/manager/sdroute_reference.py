"""SD-route scoring reference: the GT-cropped span that RC divides by.

This is the ONLY part of the former coverage metric that production scoring
(`hmm_prefix_1m`) still uses. It answers one question: given the baked SD
polyline and the GT rear-axle track, which SD span is this scenario scored over?

`build_coverage_reference` picks the correct route occurrence by scoring whole-GT
candidates against each other, not by thresholding a per-point distance: a U/P-turn
route passes the same place twice, so a nearest-point start attaches to the return
lobe. The winning candidate supplies `s_start`/`s_end` (RC's denominator) and
`crop_xy()`, which `base_env` uses both as the arrival predicate's reference
(`_sd_goal_state`) and as the departure guard's (`_sd_route_distance`).

Corner spans get a monotone GT-arc -> SD-arc correspondence so a coarse SD vertex
does not read as a jump in progress. Distances are ORIGINAL SD metres throughout.

There is no corridor or heading gate here. Deciding whether the MODEL stayed on
route is a separate step and belongs to the SDF matcher (`sdroute_prefix.py`),
which matches against the directed SD graph rather than this polyline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    from .sdroute_progress import RouteGeometry, build_reference
except ImportError:  # standalone offline tools
    from sdroute_progress import RouteGeometry, build_reference  # noqa: F401


@dataclass(frozen=True)
class ReferenceConfig:
    """Only the fields `build_coverage_reference` actually reads."""
    turn_angle_deg: float = 45.0
    turn_heading_baseline_m: float = 4.0
    turn_margin_m: float = 8.0
    turn_gap_m: float = 3.0
    min_length_m: float = 0.5

@dataclass(frozen=True)
class Maneuver:
    start_index: int
    end_index: int
    u_start: float
    u_end: float
    s_start: float
    s_end: float


@dataclass
class CoverageReference:
    route: RouteGeometry
    gt: np.ndarray
    u: np.ndarray
    raw_s: np.ndarray
    mapped_s: np.ndarray
    maneuvers: Tuple[Maneuver, ...]
    quality: dict

    @property
    def s_start(self):
        return float(self.mapped_s[0])

    @property
    def s_end(self):
        return float(self.mapped_s[-1])

    @property
    def length(self):
        return self.s_end - self.s_start

    def crop_xy(self, end_s=None):
        """Original SD geometry between GT-matched endpoints, including partial edges."""
        end = self.s_end if end_s is None else float(end_s)
        if not self.s_start <= end <= self.s_end:
            raise ValueError('crop end must lie in the scored SD span')
        inside = (self.route.s > self.s_start) & (self.route.s < end)
        return np.vstack((self.route.at_s(self.s_start).xy,
                          self.route.xy[inside], self.route.at_s(end).xy))

def _xy(value, name, min_points=2):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < min_points:
        raise ValueError(f"{name} must have shape (N, 2), N >= {min_points}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return a


def _arc(xy):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]

def build_coverage_reference(route_xy, gt_xy, config=ReferenceConfig(), *,
                             gt_start_index=0, gt_end_index=None):
    """Match FULL GT first, then clip the scored window; never reselect a lobe.

    Index bounds refer to the supplied GT samples; end is inclusive. The handoff
    pose belongs to both the reference and the driven trajectory.
    """
    gt_full = _xy(gt_xy, "gt_xy")
    end = len(gt_full) - 1 if gt_end_index is None else int(gt_end_index)
    start = int(gt_start_index)
    if not 0 <= start < end < len(gt_full):
        raise ValueError("invalid GT evaluation window")
    base = build_reference(route_xy, gt_full)
    # Keep endpoint occurrences of exact stationary runs. No median-of-duplicate-s
    # profile: it loses phase while stopped at a coarse route vertex.
    gt = gt_full[start:end + 1].copy()
    raw = np.array([m.s for m in base.gt_matches])[start:end + 1]
    keep = np.r_[True, np.linalg.norm(np.diff(gt, axis=0), axis=1) > 1e-7]
    gt, raw = gt[keep], raw[keep]
    if len(gt) < 2:
        raise ValueError("stationary GT: no coverage denominator")
    u = _arc(gt)
    if raw[-1] - raw[0] < config.min_length_m:
        raise ValueError("GT spans too little SD route to define coverage")

    # Find coarse corners AND anomalous projection advances. Intersection polygons
    # are not blanket exemptions; rounded turns with no projection issue need none.
    seeds = []
    for i in range(1, len(gt)):
        du, ds = u[i] - u[i - 1], raw[i] - raw[i - 1]
        gap = ds > 2.0 * du + config.turn_gap_m
        before = base.route.at_s(raw[i] - config.turn_heading_baseline_m).heading
        after = base.route.at_s(raw[i] + config.turn_heading_baseline_m).heading
        angle = abs(np.arctan2(np.sin(after - before), np.cos(after - before)))
        if gap or angle >= np.deg2rad(config.turn_angle_deg):
            seeds.append((max(0, int(np.searchsorted(u, u[i - 1] - config.turn_margin_m)) - 1),
                          min(len(gt) - 1, int(np.searchsorted(u, u[i] + config.turn_margin_m)))))
    # Also catch a sharp vertex jumped over between two otherwise distant anchors.
    for i in range(1, len(base.route.heading)):
        angle = abs(np.arctan2(np.sin(base.route.heading[i] - base.route.heading[i - 1]),
                              np.cos(base.route.heading[i] - base.route.heading[i - 1])))
        s = base.route.s[i]
        if angle < np.deg2rad(config.turn_angle_deg) or not raw[0] < s < raw[-1]:
            continue
        k = min(int(np.searchsorted(raw, s)), len(gt) - 1)
        seeds.append((max(0, int(np.searchsorted(u, u[max(k - 1, 0)] - config.turn_margin_m)) - 1),
                      min(len(gt) - 1, int(np.searchsorted(u, u[k] + config.turn_margin_m)))))
    spans = []
    for a, b in sorted(seeds):
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    mapped = raw.copy()
    maneuvers = []
    for a, b in spans:
        if b <= a or raw[b] - raw[a] <= 1e-8:
            continue
        mapped[a:b + 1] = raw[a] + ((u[a:b + 1] - u[a]) / (u[b] - u[a])) * (raw[b] - raw[a])
        maneuvers.append(Maneuver(a, b, float(u[a]), float(u[b]), float(raw[a]), float(raw[b])))
    quality = dict(gt_route_p90_m=base.gt_route_distance_p90,
                   gt_route_max_m=base.gt_route_distance_max,
                   sd_to_gt_length_ratio=base.span_to_gt_length_ratio,
                   suspect=bool(base.gt_route_distance_p90 > 10.0
                                or not 0.8 <= base.span_to_gt_length_ratio <= 1.2),
                   gt_start_index=start, gt_end_index=end)
    return CoverageReference(base.route, gt, u, raw, mapped, tuple(maneuvers), quality)
