"""The simulator's adapter onto the repository-wide constants, plus the cadence helpers.

The numbers live in `OdysseyTrafficAgent/defines.py` -- one declaration, so they
cannot drift. This module puts that file on the path and re-exports it, so the simulator code
imports from its own tree:

    from odyssey.utils.cadence import CONTROL_DT

Anything numeric that belongs to the simulator's clocks goes in defines.py, not here.
What lives here is the logic that turns those constants plus the config into the values a
run actually uses.
"""
import logging
import math
import os
import sys
from dataclasses import dataclass

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isfile(os.path.join(_root, "defines.py")):
    _root = os.path.dirname(_root)
if not os.path.isfile(os.path.join(_root, "defines.py")):
    raise ImportError(
        f"cannot find defines.py above {_here}. It holds every constant in this repo; "
        f"without it the values below would have to be re-typed here, which is the "
        f"failure mode defines.py exists to prevent.")
if _root not in sys.path:
    sys.path.insert(0, _root)

from defines import (  # noqa: E402  (path must be set first)
    CONTROL_DT,
    GT_GOAL_END_DIST_M,
    GT_GOAL_PROGRESS_RATIO,
    HEADING_CHORD_M,
    NUPLAN_LIDAR_DT,
    PLAN_FILE_DT,
    PLANNER_POSE_DT,
    SCORE_DT,
    SD_GOAL_END_DIST_M,
    SD_GOAL_PROGRESS_RATIO,
    SD_ROUTE_MAX_DIST_M,
    SIM_DT_CHOICES,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Cadence", "resolve_cadence",
    "CONTROL_DT", "HEADING_CHORD_M", "NUPLAN_LIDAR_DT", "PLAN_FILE_DT", "PLANNER_POSE_DT", "SCORE_DT",
    "SIM_DT_CHOICES", "SUBSTEP_KEY_BY_CONTROLLER", "scene_dt",
    "SD_ROUTE_MAX_DIST_M", "SD_GOAL_PROGRESS_RATIO", "SD_GOAL_END_DIST_M",
    "GT_GOAL_PROGRESS_RATIO", "GT_GOAL_END_DIST_M",
    "source_frame_at_elapsed_time", "resolve_plant_substeps",
    "plan_file_stride", "synchronized_gt_warmup_steps",
]


# Config key holding the sub-step count, per ego controller. One physical quantity, three
# spellings -- the three controllers were written at different times and each invented its
# own. They are folded into a single derived value below.
SUBSTEP_KEY_BY_CONTROLLER = {
    'two_stage_controller': 'two_stage_substeps',
}


def scene_dt(scene):
    """Frame spacing of a scenario, in seconds. `sample_rate` is a count, not a rate."""
    return float(scene['sample_rate']) * NUPLAN_LIDAR_DT


def source_frame_at_elapsed_time(scene, elapsed_s):
    """Map elapsed seconds to an un-resampled OpenScene source-frame index.

    ``sample_rate`` is a count of 0.05 s nuPlan lidar periods.  It is not Hz: a
    value of 10 means a 0.5 s frame interval (2 Hz).  Centralizing this conversion
    prevents renderers and offline metrics from silently applying the inverse unit.
    """
    dt = scene_dt(scene)
    elapsed_s = float(elapsed_s)
    if dt <= 0 or not math.isfinite(dt):
        raise ValueError(f"invalid source frame spacing: {dt!r}")
    if not math.isfinite(elapsed_s):
        raise ValueError(f"invalid elapsed time: {elapsed_s!r}")
    return max(0, int(math.floor(elapsed_s / dt + 1e-9)))


def synchronized_gt_warmup_steps(config):
    """One GT-history boundary shared by the ego planner and reactive traffic.

    ``num_history`` includes the current sample, so a history tensor of length N needs
    N-1 transitions before a learned planner can take over.  The legacy config key keeps
    its name for CLI compatibility, but it now gates the whole simulator handoff rather
    than PDM alone.  Centralizing the derivation prevents ego and traffic from silently
    switching at different ticks.
    """
    if not bool(config.get("pdm_gt_warmup_enabled", True)):
        return 0
    return max(int(config["num_history"]) - 1, 0)


def resolve_plant_substeps(configured, plant_dt, controller='the ego controller'):
    """Sub-steps per outer step, so that the plant integrates at CONTROL_DT.

    Sub-stepping controllers set the motion model's dt to plant_dt/N, integrate N times
    and restore it, so N does not change how far the ego travels per outer step -- it
    changes only the integration period. There is therefore exactly one correct N, and it
    is derived, not chosen: plant_dt / CONTROL_DT.

    An explicit config value still wins, because tuning experiments need to vary it. The
    default is derived: a typed value such as 1 against a 0.5 s step would integrate the
    bicycle in a single 0.5 s Euler step while the LQR gains were tuned for 0.1 s.
    """
    if configured:
        return max(int(configured), 1)
    n = float(plant_dt) / CONTROL_DT
    n_int = int(round(n))
    if n_int < 1 or abs(n - n_int) > 1e-9:
        raise ValueError(
            f"{controller}: cannot derive sub-steps -- a {plant_dt}s outer step is not a "
            f"whole number of {CONTROL_DT}s controller periods ({n:.4f}). Either use an "
            f"outer step in {SIM_DT_CHOICES}, or set the sub-step count explicitly and "
            f"accept that the plant will not integrate at the controller period.")
    return n_int


def plan_file_stride(sim_dt):
    """Rows of a PLAN_FILE_DT-spaced trajectory that one outer step covers.

    Consumers that read a trajectory BY INDEX -- LogPlayController takes "one sim step
    ahead" -- need this; consumers that read it BY TIME (the trackers) do not, which is
    why the trajectory keeps its native spacing and the stride lives at the index.
    """
    n = float(sim_dt) / PLAN_FILE_DT
    n_int = int(round(n))
    if n_int < 1 or abs(n - n_int) > 1e-9:
        raise ValueError(
            f"an outer step of {sim_dt}s is not a whole number of {PLAN_FILE_DT}s planner "
            f"rows ({n:.4f}), so no index in the trajectory lands one step ahead. Use an "
            f"outer step in {SIM_DT_CHOICES}.")
    return n_int


# Tolerance on a scene's declared sample_rate against its own timestamps. Wide enough for
# jitter in a real log, narrow enough that an off-by-one declaration (1 vs 2) cannot pass.
SAMPLE_RATE_TOLERANCE = 0.02


@dataclass(frozen=True)
class Cadence:
    """Every period a run uses, derived once from the one knob and the scene.

    Deriving every field from one place keeps them consistent: separate config keys set in
    lockstep can disagree without anything noticing.
    """

    sim_dt: float             # outer step: render == replan == one scenario frame
    scene_dt: float           # the scene's own frame spacing, before any upsampling
    upsample_n: int           # GT frames inserted per scene frame to reach sim_dt
    plant_substeps: int       # plant integrations per outer step, at CONTROL_DT each
    step_stride: int          # rows of a PLAN_FILE_DT trajectory one outer step covers
    score_stride_steps: int   # outer steps per SCORE_DT scoring frame


def _exact_ratio(numerator, denominator, what, hint=""):
    """numerator/denominator as an int, or a refusal that names both."""
    ratio = float(numerator) / float(denominator)
    n = int(round(ratio))
    if n < 1 or abs(ratio - n) > 1e-9:
        raise ValueError(
            f"{what}: {numerator}s does not divide into {denominator}s "
            f"({ratio:.6f}). {hint}".rstrip())
    return n


def check_sample_rate(scene, scene_id="<scene>"):
    """Confirm a scene's declared sample_rate against its own timestamps.

    `sample_rate` is a COUNT of NUPLAN_LIDAR_DT periods, and getting it wrong is not a
    local error: engine.sim_dt comes from it, so a scene declaring half its true
    spacing doubles the angular velocity derived for every track that lacks one, mistimes
    HybridReplay's agent interaction, and halves the GT-duration rollout budget. Three
    OmniRe scenes shipped declaring 1 for 0.1 s data.

    Scenes without timestamps (the MTGS sets) cannot be checked and are left alone.
    """
    declared = float(scene["sample_rate"]) * NUPLAN_LIDAR_DT
    ts = (scene.get("metadata") or {}).get("omnire_timestamps_us")
    if ts is None or len(ts) < 2:
        return declared
    import numpy as np
    measured = float(np.median(np.diff(np.asarray(ts, dtype=np.float64))) / 1e6)
    if abs(measured - declared) > SAMPLE_RATE_TOLERANCE * max(measured, 1e-9):
        raise ValueError(
            f"{scene_id}: sample_rate={scene['sample_rate']} declares "
            f"{declared:.4f}s frames, but the scene's own omnire_timestamps_us say "
            f"{measured:.4f}s. sample_rate is a count of {NUPLAN_LIDAR_DT}s lidar periods, "
            f"not a rate -- this scene should declare "
            f"{measured / NUPLAN_LIDAR_DT:.0f}. "
            f"Fix the data with projects/Reconstruction/omnire/fix_sample_rate.py; do not "
            f"work around it, every duration in the run comes from this field.")
    return declared


def resolve_cadence(cfg, scene, scene_id="<scene>") -> Cadence:
    """Resolve every period a run uses, from `rollout_dt` and the scene.

    Call this on the scene as loaded, BEFORE any upsampling -- _resample_scene divides
    sample_rate by the factor it applied, so afterwards scene_dt has already become sim_dt.
    """
    scene_dt = check_sample_rate(scene, scene_id)

    rollout_dt = cfg.get("rollout_dt", None)
    # No rollout_dt means one outer step per scene frame. It does NOT mean 0.5: that was
    # assumed in four places and is wrong for every scene whose sample_rate is not 10.
    sim_dt = float(rollout_dt) if rollout_dt else scene_dt

    if not any(abs(sim_dt - c) < 1e-9 for c in SIM_DT_CHOICES):
        raise ValueError(
            f"{scene_id}: outer step {sim_dt}s is not one of {SIM_DT_CHOICES}. "
            f"Rendering, replanning and the scenario all advance once per outer step, and "
            f"the step has to divide by {CONTROL_DT}s (the controller period) and into "
            f"{SCORE_DT}s (the scoring grid).")

    return Cadence(
        sim_dt=sim_dt,
        scene_dt=scene_dt,
        upsample_n=_exact_ratio(
            scene_dt, sim_dt, f"{scene_id}: scene frames vs outer step",
            "the GT tracks are upsampled by this factor, so it must be a whole number."),
        plant_substeps=_exact_ratio(
            sim_dt, CONTROL_DT, f"{scene_id}: outer step vs controller period"),
        step_stride=_exact_ratio(
            sim_dt, PLAN_FILE_DT, f"{scene_id}: outer step vs planner file grid",
            "consumers that read the plan by index need a row exactly one step ahead."),
        score_stride_steps=_exact_ratio(
            SCORE_DT, sim_dt, f"{scene_id}: scoring grid vs outer step"),
    )
