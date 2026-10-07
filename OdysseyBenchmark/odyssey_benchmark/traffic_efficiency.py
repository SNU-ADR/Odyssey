#!/usr/bin/env python3
"""Ego driving speed relative to surrounding traffic. A port of Bench2Drive Efficiency.

What it measures
----------------
"How fast did the ego go compared with the other vehicles in the same scene." It is a relative
speed, not an absolute one: 100% means the speed of the traffic flow, 50% means it crawled at
half that speed.

    efficiency = Σ_{t∈K} ego_speed[t] / Σ_{t∈K} bg_speed[t] × 100
    coverage   = |K| / |W|        W = frames inside the GT window (handoff ~ min(run end, GT end))
                                  K = frames of W with at least one background vehicle

Three differences from the original (Bench2Drive MinimumSpeedRouteTest)
-----------------------------------------------------------------------
1. **Denominator set.** The original selects vehicles by the `role_name == 'background'` tag.
   We have no such tag, and unlike CARLA, which only spawns vehicles meant to drive, nuPlan
   logs contain every roadside parked car as VEHICLE (measured: 75.3% of background-vehicle
   cells have speed 0). So the denominator is **every VEHICLE that exceeded MOVER_MPS at least
   once in the run**.

   It matters that this is decided **per vehicle**, not per frame. Dropping stopped vehicles
   frame by frame would remove cars waiting at a signal from the denominator, yet their
   stopping is the real state of the traffic flow.

2. **No chunking.** The original splits the route into 20 chunks and averages the per-chunk %.
   But even the original is Σego/Σbg *within* a chunk (the shared speed_points cancel out),
   and our runs range from 17 to 158 frames, so 20 chunks would be 1 frame each.

3. **Measured only in the GT window.** The run budget is 2x the GT driving time, so the ego keeps
   driving after the log ends, when there are no log vehicles and hence no traffic flow.
   Filling empty frames with 100% as the original does would give the whole second half full
   marks. Merely dropping empty frames while dividing coverage by the whole run length makes
   coverage measure how slow the ego is rather than how much traffic the scene has (measured
   on 2,249 runs: within a token, the rank correlation of run length and coverage was -0.86;
   of 336 runs dropped by COV_GATE, 129 passed when measured against the GT window and 117 of
   those had exhausted the budget; slow arms had their TE median inflated by up to 10 pp).

   So the input is cut to the GT window W from the start: the mover decision, numerator,
   denominator and coverage all see the same window. The GT end is read only from the RC
   snapshot pinned in the scoring input (gt_window_end). Most runs were already measured over
   this window in effect, but only because log vehicles happen to disappear at the end of the
   log. IDM keeps driving vehicles whose log has ended, so R runs were already off
   (p95 16 pp against the explicitly cut value).

   The original's 1000% clipping is not needed -- excluding parked vehicles removes the
   divergent tail.

None when not measurable
------------------------
With no background vehicles, or fewer than MIN_SCORED_FRAMES frames, the result is **None**,
not 0. 0 is a measurement ("drove at 0x the traffic speed"); None means "could not be
measured". Collapsing the two into one value contaminates aggregates. Runs where RC could not
build a GT baseline have no defined window either, so they are None (reason "no_gt_window").

How to read it
--------------
- **Read it only through medians and paired tests.** Do not use means -- on short runs the
  denominator is unstable and can yield thousands of %.
- Tokens with low coverage are weak evidence. Excluding those below COV_GATE from aggregates is
  recommended. coverage is relative to the GT window, not to the whole run.
- **The denominator reacts to the ego.** In reactive (IDM) runs background vehicles follow
  behind the ego and slow down with it, so absolute slowness is not fully captured.
- Do not draw conclusions from a single-run A/B: closed-loop runs are not reproducible.

Usage
-----
    python -m odyssey_benchmark.traffic_efficiency <run_dir or glob> [--csv out.csv]
    python -m odyssey_benchmark.traffic_efficiency 'experiments/simulation/*'

Needs no nuplan, map or simulator env -- only numpy.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

#: Only vehicles that exceed this speed at least once enter the denominator. This is the port's
#: only free parameter. At 0.5 / 1.0 / 2.0 the median moves gently (77 / 75 / 69) -- the
#: cut-off is not sensitive, so 1.0 is used.
MOVER_MPS = 1.0

#: Minimum valid frames for legacy 0.5 s records. Denser records keep the same minimum
#: observation time (MIN_SCORED_SECONDS below).
MIN_SCORED_FRAMES = 3
# The legacy 3 samples were 0.5 s apart. Preserve the minimum measured time
# when current runs provide five times as many 0.1 s samples.
MIN_SCORED_SECONDS = MIN_SCORED_FRAMES * 0.5

#: Coverage (fraction of GT-window frames with a background vehicle) below this is weak
#: evidence: the value is still reported but flagged. Not chosen by a sensitivity study.
COV_GATE = 0.3


#: The type whose npz velocity is meaningful. StaticObjects (cones, barriers) have no velocity
#: attribute in nuPlan and are filled with 0, so they must be filtered out.
VEHICLE_TYPE = "VEHICLE"


@dataclass
class Result:
    """Result of one rollout. efficiency None means it could not be measured."""

    token: str
    efficiency: Optional[float]
    coverage: float
    n_bg_mean: float
    bg_speed_mean: Optional[float]
    ego_speed_mean: Optional[float]
    n_vehicles: int
    n_movers: int
    n_frames: int
    low_coverage: bool
    reason: str  # why there is no value; "ok" when there is one


def gt_window_end(rc: dict) -> Optional[int]:
    """Last simulator step of the GT window. None when RC could not build a GT baseline.

    The only source is the RC snapshot pinned in the scoring input (SDRouteMetric.snapshot).
    rc["gt"][k] is the log rear axle at step k·stride (sdroute_metric:
    reference_gt_dense[::stride]), so the last GT step is (len(gt) - 1)·stride. It is not
    re-derived from the run budget (num_future = 2x GT) or expert_source_pose_count -- that
    would be a second source for the same window.
    """
    if rc["gt"] is None:
        return None
    return (len(rc["gt"]) - 1) * int(rc["stride"])


def _pinned(npz_path: str) -> Optional[dict]:
    """Saved scoring input (driving_inputs_json) -> the same arguments as re-scoring.

    None for an old npz without it. The GT window is defined only by the saved RC snapshot. The
    old format, which kept only 0.5 s grid arrays, would require guessing the window from other
    values, so it is no longer scored.

    Actor rows are cleaned with the same function as re-scoring (driving_metrics.replay). Using
    the saved rows as-is leaves the final row that jumped to the start point without a pose
    (velocity 0) in the denominator; measured on 2,249 runs, this made the CLI read up to
    2.2 pp above the recorded value in 295 of them.
    """
    from .pinned_actors import drop_unset_pose_frames
    with np.load(npz_path, allow_pickle=True) as d:
        if "driving_inputs_json" not in d.files:
            return None
        if "ds_states" not in d.files or "ds_sim_steps" not in d.files:
            raise ValueError("pinned scoring archive lacks dense ego states or steps")
        inputs = json.loads(str(d["driving_inputs_json"]))
        actors, _ = drop_unset_pose_frames(inputs["actors"])
        return dict(
            states=np.asarray(d["ds_states"], dtype=np.float64), actors=actors,
            steps=np.asarray(d["ds_sim_steps"]), sim_dt=float(inputs["sim_dt"]),
            gt_end_step=gt_window_end(inputs["rc"]),
            scene=str(d["scene"]) if "scene" in d.files else "",
            map_location=str(d["map_location"]) if "map_location" in d.files else "",
        )


def _load(npz_path: str) -> Optional[dict]:
    """GT-window frame arrays for visualisation. None for an old npz without scoring inputs."""
    p = _pinned(npz_path)
    if p is None:
        return None
    if p["gt_end_step"] is None:
        raise ValueError(f"{npz_path}: RC snapshot has no GT baseline -- no window to measure TE")
    result = dense_frame_data(p["states"], p["actors"], p["steps"], p["sim_dt"],
                              p["gt_end_step"])
    n = len(result["ego_velocity"])
    result.update(
        agent_source_mode=np.full(result["agent_heading"].shape, "unknown", dtype="<U7"),
        ego_xy=p["states"][:n, :2], ego_heading=p["states"][:n, 2],
        scene=p["scene"], map_location=p["map_location"],
    )
    return result


def compute(z: dict, mover_mps: float = MOVER_MPS) -> dict:
    """Build the mask and per-frame series. The visualisation uses this function too.

    Of the returned values, ``mask`` is an (A, F) boolean "did this vehicle enter this frame's
    denominator", and ``ego_speed``/``bg_speed`` are length-F series, so what was selected in
    each frame can be plotted directly.
    """
    xy = z["agent_xy"]
    vel = z["agent_velocity"]
    n_a, n_f = xy.shape[0], xy.shape[1]

    # Ego history starts at scene frame 0, detections at the closed-loop handoff. The difference
    # is the offset (measured: ego_len - agent_len == offset in all 1,390 runs).
    ego_speed_all = np.linalg.norm(z["ego_velocity"], axis=-1)
    idx = np.arange(n_f) + z["offset"]
    if n_f and idx[-1] > len(ego_speed_all) - 1:
        # Reaching here means the two npz arrays disagree. Do not truncate silently.
        raise ValueError(
            f"ego({len(ego_speed_all)}) and agent({n_f})+offset({z['offset']}) do "
            "not line up -- the dump is broken or the meaning of offset changed")
    ego_speed = ego_speed_all[idx] if n_f else np.zeros(0)

    speed = np.linalg.norm(vel, axis=-1)            # (A, F)
    # With no actors, numpy returns a scalar False for `empty array == str`, so force shape (A,).
    is_vehicle = np.asarray(z["agent_types"]).reshape(-1).astype(str) == VEHICLE_TYPE   # (A,)
    present = ~np.isnan(xy[..., 0])                 # (A, F) NaN = absent in that frame
    veh_present = present & is_vehicle[:, None]

    # Per-vehicle decision: did it exceed mover_mps at least once in the run.
    # A vehicle never seen in any frame is NaN throughout and nanmax warns -- that is a normal
    # path and it becomes 0 just below, so only the warning is suppressed (errstate does not).
    peak = np.where(veh_present, speed, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ever = np.nanmax(peak, axis=1) if n_f else np.zeros(n_a)
    ever = np.nan_to_num(ever, nan=0.0)
    is_mover = ever > mover_mps                     # (A,)

    mask = veh_present & is_mover[:, None]          # (A, F) cells in the denominator
    n_bg = mask.sum(axis=0)                         # (F,)
    scored = n_bg > 0                               # K

    # Frames with no background vehicle are all NaN and nanmean warns. scored already excludes
    # those frames, so the values are unused -- only the warning is suppressed.
    bg_speed = np.where(mask, speed, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bg_mean = np.nanmean(bg_speed, axis=0) if n_a else np.full(n_f, np.nan)

    return {
        "mask": mask, "veh_present": veh_present, "is_vehicle": is_vehicle,
        "is_mover": is_mover, "n_bg": n_bg, "scored": scored,
        "ego_speed": ego_speed, "bg_mean": bg_mean, "speed": speed,
        "n_frames": n_f, "n_agents": n_a,
    }


def score(z: dict, token: str = "", mover_mps: float = MOVER_MPS) -> Result:
    """One rollout -> Result. When not measurable, returns efficiency=None and a reason."""
    dt = float(z.get("score_dt", 0.5))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"invalid Traffic Efficiency frame spacing: {dt}")
    minimum_frames = max(MIN_SCORED_FRAMES,
                         math.ceil(MIN_SCORED_SECONDS / dt - 1e-9))
    c = compute(z, mover_mps)
    n_f = c["n_frames"]
    scored, n_bg, bg_mean, ego_speed = c["scored"], c["n_bg"], c["bg_mean"], c["ego_speed"]
    coverage = float(scored.mean()) if n_f else 0.0
    n_bg_mean = float(n_bg[scored].mean()) if scored.any() else 0.0
    n_veh = int(c["is_vehicle"].sum())
    n_mov = int((c["is_mover"] & c["is_vehicle"]).sum())

    def empty(reason: str) -> Result:
        return Result(token, None, coverage, n_bg_mean, None, None,
                      n_veh, n_mov, n_f, coverage < COV_GATE, reason)

    if n_f == 0:
        return empty("no_frames")
    if n_veh == 0:
        return empty("no_vehicles")
    if n_mov == 0:
        # Background vehicles existed but were all parked: "not measurable", not 0%.
        return empty("all_parked")
    if int(scored.sum()) < minimum_frames:
        return empty("too_few_scored_frames")

    num = float(np.nansum(ego_speed[scored]))
    den = float(np.nansum(bg_mean[scored]))
    if not den > 1e-6:
        return empty("zero_denominator")

    return Result(
        token=token,
        efficiency=num / den * 100.0,
        coverage=coverage,
        n_bg_mean=n_bg_mean,
        bg_speed_mean=den / int(scored.sum()),
        ego_speed_mean=num / int(scored.sum()),
        n_vehicles=n_veh,
        n_movers=n_mov,
        n_frames=n_f,
        low_coverage=coverage < COV_GATE,
        reason="ok",
    )


def dense_frame_data(states, actor_frames, steps, sim_dt, gt_end_step: int) -> dict:
    """Align ego and all observed actors on the consecutive simulator frames of the GT window."""
    states = np.asarray(states, dtype=np.float64)
    steps = np.asarray(steps, dtype=np.int64)
    if (states.ndim != 2 or states.shape[1] < 5 or steps.ndim != 1
            or len(states) != len(actor_frames) or len(states) != len(steps)
            or (len(steps) > 1 and not np.all(np.diff(steps) == 1))):
        raise ValueError("Traffic Efficiency requires aligned consecutive dense frames")
    # Frames outside the GT window are treated as never existing, so every computation below
    # (including the mover decision) sees the same window. Steps are consecutive, so what
    # remains is a leading prefix.
    n = int(np.sum(steps <= gt_end_step))
    states, actor_frames, steps = states[:n], actor_frames[:n], steps[:n]
    tokens = sorted({str(actor[0]) for frame in actor_frames for actor in frame})
    order = {name: i for i, name in enumerate(tokens)}
    xy = np.full((len(tokens), len(steps), 2), np.nan)
    velocity = np.full_like(xy, np.nan)
    heading = np.full((len(tokens), len(steps)), np.nan)
    size = np.zeros((len(tokens), 2), dtype=np.float64)
    types = np.full(len(tokens), "", dtype=object)
    for frame_idx, actors in enumerate(actor_frames):
        for actor in actors:
            index = order[str(actor[0])]
            types[index] = str(actor[1])
            xy[index, frame_idx] = actor[2:4]
            heading[index, frame_idx] = actor[4]
            size[index] = actor[5:7]
            velocity[index, frame_idx] = actor[8:10]
    return dict(agent_xy=xy, agent_velocity=velocity, agent_heading=heading,
                agent_size=size, agent_tokens=np.asarray(tokens),
                agent_types=types, ego_velocity=states[:, 3:5], offset=0,
                score_dt=float(sim_dt))


def score_dense_frames(states, actor_frames, steps, sim_dt, gt_end_step: Optional[int],
                       token: str = "", mover_mps: float = MOVER_MPS) -> Result:
    """Score every observed simulator frame of the GT window (0.1 s for 0.1 s runs)."""
    if gt_end_step is None:
        return Result(token, None, 0.0, 0.0, None, None, 0, 0, 0, True, "no_gt_window")
    return score(dense_frame_data(states, actor_frames, steps, sim_dt, gt_end_step),
                 token, mover_mps)


def score_npz(npz_path: str, token: str = "", mover_mps: float = MOVER_MPS) -> Optional[Result]:
    """Re-score a saved scoring input with the same function as live scoring (score_dense_frames)."""
    p = _pinned(npz_path)
    if p is None:
        return None
    return score_dense_frames(p["states"], p["actors"], p["steps"], p["sim_dt"],
                              p["gt_end_step"], token or _token_of(npz_path), mover_mps)


def _token_of(path: str) -> str:
    m = re.search(r"exp_([0-9a-f]{16})_", path)
    if m:
        return m.group(1)
    m = re.search(r"([0-9a-f]{16})", os.path.basename(os.path.dirname(path)))
    return m.group(1) if m else os.path.basename(path)


def find_npz(target: str) -> List[str]:
    """Turn a single run_dir or a glob into a list of npz files."""
    if target.endswith(".npz"):
        return sorted(glob.glob(target))
    hits: List[str] = []
    for root in sorted(glob.glob(target)):
        hits.extend(glob.glob(os.path.join(
            root, "**", "rollout_trajectory.npz"), recursive=True))
    return sorted(set(hits))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="run_dir, glob, or .npz path")
    ap.add_argument("--csv", help="csv file to write results to")
    ap.add_argument("--mover-mps", type=float, default=MOVER_MPS,
                    help=f"minimum peak speed for a vehicle to enter the denominator (default {MOVER_MPS})")
    a = ap.parse_args(argv)

    paths = find_npz(a.target)
    if not paths:
        print(f"no npz found: {a.target}", file=sys.stderr)
        return 1

    rows: List[Result] = []
    skipped_old = 0
    for p in paths:
        r = score_npz(p, mover_mps=a.mover_mps)
        if r is None:
            skipped_old += 1
            continue
        rows.append(r)

    ok = [r for r in rows if r.efficiency is not None]
    gated = [r for r in ok if not r.low_coverage]
    print(f"npz {len(paths)} files, old format skipped {skipped_old}, scored {len(ok)}/{len(rows)}")
    if ok:
        v = np.array([r.efficiency for r in ok])
        g = np.array([r.efficiency for r in gated]) if gated else v
        print(f"  efficiency  median {np.median(v):6.1f}%   "
              f"p25 {np.percentile(v, 25):.1f}  p75 {np.percentile(v, 75):.1f}")
        print(f"  coverage>={COV_GATE} only  n={len(gated)}  median {np.median(g):6.1f}%")
        print(f"  coverage    median {np.median([r.coverage for r in ok]):.2f}")
        print("  * Read medians, not means. Compare models with paired tests.")
    for reason in sorted({r.reason for r in rows if r.efficiency is None}):
        n = sum(1 for r in rows if r.reason == reason)
        print(f"  no value, {reason}: {n}")

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))
        print(f"wrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
