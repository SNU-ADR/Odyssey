"""SD route (120, 5) -> SimLingo-style navigation text for the VLM prompt.

Output is the next maneuver with its distance plus the one after it, e.g.
    "turn right in 50 m, then turn left"
    "turn left now, then follow the road"
    "follow the road"
following SimLingo's `Command: {command} in {d} meter then {next_command}.`
(simlingo_training/dataloader/dataset_base.py get_navigational_conditioning), with ReCogDrive's own
command words ("turn left" / "turn right") and one straight/turn threshold of 35 deg as in CARLA's
RoadOption.

Input is the sdroute_target cache: route_centerline (P, 5) in the current ego frame (x forward, y left,
~1 m spacing) and mask (P,), a valid prefix. Only the positions are used:
* column 4 (heading) is not angle-wrapped, so the heading is recomputed from the positions and
  unwrapped;
* masked points are zero-filled; keeping them adds a fake U-turn at the end;
* the route is an OSM road centerline, not ego's lane, so the lateral offset is never turned into
  lane-change text;
* the first metres come from projecting ego onto the route and are noisy, so the heading there is
  ramped from the ego heading (0 in the ego frame) to the route heading just after them. "Ego already
  mid-turn" then shows up as a maneuver at distance 0 ("... now").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class RouteTextConfig:
    smooth_m: int = 7              # moving-average window on the 1 m resampled positions [m]
    start_skip_m: float = 3.0      # heading over the first metres is ramped from the ego heading (0)
    rate_deg_per_m: float = 1.0    # |dphi/ds| above this counts as turning (10 deg over 10 m)
    merge_gap_m: float = 8.0       # same-direction turning runs closer than this are one maneuver
    min_turn_deg: float = 15.0     # runs below this are ignored before the jog check
    turn_deg: float = 35.0         # below -> "follow the road"; CARLA/SimLingo straight threshold
    uturn_deg: float = 135.0       # at or above -> U-turn
    max_turn_span_m: float = 40.0  # a maneuver is sized by its heading change over this much route only
    jog_gap_m: float = 25.0        # an opposite-direction pair closer than this ...
    jog_net_deg: float = 20.0      # ... whose net change is below this is an S-shaped jog -> dropped
    now_m: float = 5.0             # a maneuver starting closer than this is "now"
    min_points: int = 10           # fewer valid points, or a route shorter than this many metres -> no route text


@dataclass
class Maneuver:
    kind: str          # "left" | "right" | "U-turn"
    start_m: float     # distance along the route where the heading starts to change
    end_m: float
    delta_deg: float   # signed total heading change, + = left


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _heading_profile(route, mask, cfg: RouteTextConfig) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(s, phi_deg): heading relative to ego along the route, 1 m apart. None = unusable route."""
    route, mask = _to_numpy(route), _to_numpy(mask).astype(bool)
    n = int(mask.sum())
    if n < cfg.min_points or not mask[:n].all():
        return None
    pos = route[:n, :2].astype(np.float64)
    s_raw = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
    if s_raw[-1] < cfg.min_points:
        return None
    s = np.arange(0.0, s_raw[-1], 1.0)
    pos = np.stack([np.interp(s, s_raw, pos[:, 0]), np.interp(s, s_raw, pos[:, 1])], 1)
    if cfg.smooth_m > 1 and len(s) > 2 * cfg.smooth_m:
        pad = cfg.smooth_m // 2
        padded = np.pad(pos, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(cfg.smooth_m) / cfg.smooth_m
        pos = np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(2)], 1)[: len(s)]
    d = np.diff(pos, axis=0)
    phi = np.degrees(np.unwrap(np.arctan2(d[:, 1], d[:, 0])))
    s = s[:-1]
    # Reference = ego heading (0). Shift the unwrapped profile by whole turns so it starts near 0,
    # then ramp the noisy first metres from 0 to the heading just after them.
    k0 = min(int(np.searchsorted(s, cfg.start_skip_m)), len(phi) - 1)
    phi = phi - 360.0 * np.round(phi[k0] / 360.0)
    phi[:k0] = np.linspace(0.0, phi[k0], k0, endpoint=False)
    return s, phi


def extract_maneuvers(route, mask, cfg: Optional[RouteTextConfig] = None) -> Optional[List[Maneuver]]:
    """All maneuvers of at least cfg.turn_deg along the route, nearest first. None = unusable route."""
    cfg = cfg or RouteTextConfig()
    prof = _heading_profile(route, mask, cfg)
    if prof is None:
        return None
    s, phi = prof
    if len(s) < 2:
        return []
    rate = np.gradient(phi, s)
    active = np.abs(rate) > cfg.rate_deg_per_m
    sign = np.sign(rate)

    runs = []  # contiguous same-sign turning runs, [first, last] index
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(active) and active[j + 1] and sign[j + 1] == sign[i]:
            j += 1
        if runs and sign[i] == sign[runs[-1][0]] and s[i] - s[runs[-1][1]] < cfg.merge_gap_m:
            runs[-1][1] = j
        else:
            runs.append([i, j])
        i = j + 1

    cands = []
    for a, b in runs:
        a0, b1 = max(0, a - 1), min(len(phi) - 1, b + 1)
        # A circular drive or roundabout turns steadily for 60-110 m and would add up to a 140-250 deg
        # "U-turn". Size every maneuver by its first max_turn_span_m instead; a real U-turn fits in that.
        bc = min(b1, int(np.searchsorted(s, s[a0] + cfg.max_turn_span_m)))
        delta = float(phi[bc] - phi[a0])
        if abs(delta) >= cfg.min_turn_deg:
            cands.append((float(s[a0]), float(s[b1]), delta))

    mans: List[Maneuver] = []
    k = 0
    while k < len(cands):
        if (k + 1 < len(cands) and np.sign(cands[k][2]) != np.sign(cands[k + 1][2])
                and cands[k + 1][0] - cands[k][1] < cfg.jog_gap_m
                and abs(cands[k][2] + cands[k + 1][2]) < cfg.jog_net_deg):
            k += 2  # S-shaped jog: the road shifts sideways but keeps its direction
            continue
        start, end, delta = cands[k]
        if abs(delta) >= cfg.uturn_deg:
            mans.append(Maneuver("U-turn", start, end, delta))
        elif abs(delta) >= cfg.turn_deg:
            mans.append(Maneuver("left" if delta > 0 else "right", start, end, delta))
        k += 1
    return mans


def _round_m(x: float) -> int:
    q = 5 if x < 50 else 10
    return int(max(q, round(x / q) * q))


def _phrase(m: Maneuver) -> str:
    # A U-turn cannot be told apart from a parking/driveway loop using the SD route geometry alone
    # (both just bend the route back ~180 deg), so it is folded into left/right by its turn direction
    # (delta sign, + = left), matching driving_command's one-hot which has no U-turn category.
    if m.kind == "U-turn":
        return "turn left" if m.delta_deg > 0 else "turn right"
    return f"turn {m.kind}"


def route_to_text(route, mask, cfg: Optional[RouteTextConfig] = None) -> Optional[str]:
    """SimLingo-style text: next maneuver + distance, then the one after it. None = unusable route."""
    cfg = cfg or RouteTextConfig()
    mans = extract_maneuvers(route, mask, cfg)
    if mans is None:
        return None
    if not mans:
        return "follow the road"
    first = mans[0]
    when = "now" if first.start_m < cfg.now_m else f"in {_round_m(first.start_m)} m"
    nxt = _phrase(mans[1]) if len(mans) > 1 else "follow the road"
    return f"{_phrase(first)} {when}, then {nxt}"
