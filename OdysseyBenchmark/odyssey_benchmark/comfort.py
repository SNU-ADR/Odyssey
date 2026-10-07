#!/usr/bin/env python3
"""Post-rollout 0.1 s comfort from pinned observed ego states.

This is intentionally separate from PDM's 0.5 s ``comfort`` and ``score``.
It uses the same six physical limits, but local Savitzky-Golay windows instead
of fitting a polynomial across the whole episode. Thus a brief acceleration
or yaw event remains observable at the simulator's 0.1 s cadence.

The required rear-axle heading and ego-frame velocity/acceleration are already
saved as ``ds_states`` with consecutive ``ds_sim_steps`` in current NPZs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


METHOD = "observed_0p1_local_savgol_v1"
LIMITS = {
    "lon_accel_max": 2.40,       # m/s²
    "lon_accel_min": -4.05,      # m/s²
    "lat_accel_abs": 4.89,      # m/s²
    "jerk_abs": 8.37,           # m/s³, derivative of acceleration magnitude
    "lon_jerk_abs": 4.13,       # m/s³
    "yaw_accel_abs": 1.93,      # rad/s²
    "yaw_rate_abs": 0.95,       # rad/s
}
MIN_POSES = 7                 # enough for the 0.7 s yaw-acceleration window


def score_states(states, steps, dt: float) -> dict:
    """Return separate episode comfort and interpretable violation diagnostics."""
    values = np.asarray(states, dtype=np.float64)
    sim_steps = np.asarray(steps, dtype=np.int64)
    dt = float(dt)
    if (values.ndim != 2 or values.shape[1] < 7 or sim_steps.ndim != 1
            or len(values) != len(sim_steps)
            or (len(sim_steps) > 1 and not np.all(np.diff(sim_steps) == 1))):
        raise ValueError("Comfort requires aligned consecutive dense ego states")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"invalid comfort dt: {dt}")
    result = dict(Comf=None, Comf_method=METHOD,
                  Comf_dt_s=dt, Comf_pose_count=len(values),
                  Comf_first_violation_step=None,
                  Comf_violation_types="")
    if not np.isclose(dt, 0.1, atol=1e-9, rtol=0):
        result["Comf_status"] = "requires_0p1_rollout"
        return result
    if len(values) < MIN_POSES:
        result["Comf_status"] = "insufficient_poses"
        return result
    if not np.isfinite(values[:, [2, 5, 6]]).all():
        result["Comf_status"] = "nonfinite_states"
        return result

    # ds_states[:, 5:7] is already in the ego frame, not global XY. Smooth
    # locally over 0.5 s; the PDM implementation's full-episode polynomial
    # can erase even a three-frame 20 m/s² impulse on a 10 s drive.
    lon_acc = savgol_filter(values[:, 5], 5, 2, mode="interp")
    lat_acc = savgol_filter(values[:, 6], 5, 2, mode="interp")
    lon_jerk = savgol_filter(lon_acc, 5, 2, deriv=1, delta=dt, mode="interp")
    # Match PDM's six metric definitions: its "magnitude jerk" differentiates
    # the acceleration norm, rather than taking the norm of vector jerk.
    acceleration_magnitude = savgol_filter(np.hypot(values[:, 5], values[:, 6]),
                                           5, 2, mode="interp")
    jerk = savgol_filter(acceleration_magnitude, 5, 2, deriv=1, delta=dt,
                         mode="interp")
    yaw = np.unwrap(values[:, 2])
    yaw_rate = savgol_filter(yaw, 5, 2, deriv=1, delta=dt, mode="interp")
    yaw_accel = savgol_filter(yaw, 7, 3, deriv=2, delta=dt, mode="interp")
    bad = {
        "lon_accel": (lon_acc <= LIMITS["lon_accel_min"])
                     | (lon_acc >= LIMITS["lon_accel_max"]),
        "lat_accel": np.abs(lat_acc) >= LIMITS["lat_accel_abs"],
        "jerk": jerk >= LIMITS["jerk_abs"],
        "lon_jerk": np.abs(lon_jerk) >= LIMITS["lon_jerk_abs"],
        "yaw_accel": np.abs(yaw_accel) >= LIMITS["yaw_accel_abs"],
        "yaw_rate": np.abs(yaw_rate) >= LIMITS["yaw_rate_abs"],
    }
    union = np.logical_or.reduce(list(bad.values()))
    result.update(
        Comf=float(not np.any(union)), Comf_status="ok",
        Comf_first_violation_step=(int(sim_steps[np.flatnonzero(union)[0]])
                                           if np.any(union) else None),
        Comf_violation_types="|".join(name for name, mask in bad.items()
                                               if np.any(mask)),
        Comf_max_lon_accel=float(np.max(lon_acc)),
        Comf_min_lon_accel=float(np.min(lon_acc)),
        Comf_max_abs_lat_accel=float(np.max(np.abs(lat_acc))),
        Comf_max_jerk=float(np.max(jerk)),
        Comf_max_abs_lon_jerk=float(np.max(np.abs(lon_jerk))),
        Comf_max_abs_yaw_accel=float(np.max(np.abs(yaw_accel))),
        Comf_max_abs_yaw_rate=float(np.max(np.abs(yaw_rate))),
    )
    return result


def score_npz(path: str) -> dict:
    """Recompute the exact live comfort field without scene, map, or PDM."""
    with np.load(path, allow_pickle=True) as saved:
        if not {"ds_states", "ds_sim_steps", "driving_inputs_json"}.issubset(saved.files):
            raise ValueError("archive lacks pinned dense comfort inputs")
        inputs = json.loads(str(saved["driving_inputs_json"]))
        result = score_states(saved["ds_states"], saved["ds_sim_steps"], inputs["sim_dt"])
        result["token"] = str(saved["scene"]) if "scene" in saved.files else Path(path).stem
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="rollout NPZ, run directory, or glob")
    parser.add_argument("--csv", help="write per-scene results")
    args = parser.parse_args(argv)
    targets = glob.glob(args.target)
    paths = sorted({str(file) for target in targets for file in
                    ([Path(target)] if target.endswith(".npz") else
                     Path(target).rglob("rollout_trajectory.npz"))})
    if not paths:
        parser.error(f"no rollout archives found: {args.target}")
    rows = [score_npz(path) for path in paths]
    for row in rows:
        print(f"{row['token']}: Comf={row['Comf']} "
              f"status={row['Comf_status']} "
              f"first={row['Comf_first_violation_step']} "
              f"limits={row['Comf_violation_types']}")
    if args.csv:
        fields = sorted({key for row in rows for key in row})
        with open(args.csv, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
