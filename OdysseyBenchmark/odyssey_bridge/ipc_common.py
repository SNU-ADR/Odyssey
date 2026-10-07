"""The simulator <-> planner contract, shared by every planner backend.

Every planner backend imports the same constants and frame helpers from here, so no backend
can drift from the contract the simulator depends on.

The simulator's ``PlannerClient.get_trajectory`` computes velocity as ``diff / 0.1``, i.e. it
ASSUMES the plan it receives is sampled at 0.1 s. Every navsim planner here emits 0.5 s poses,
so ``upsample_trajectory`` resamples 0.5 s -> 0.1 s before the hand-off. Getting this wrong does
not raise; it silently drives the ego at 5x the intended speed.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np

# navsim standard history length. The navsim dataclasses index frames [1,2,3] of this
# list for a 3-frame model, so the list MUST be length 4 regardless of how many the
# model actually consumes.
# Declared in OdysseyTrafficAgent/defines.py -- a number that lives in two files drifts,
# and these two in particular are a contract between two processes. The names here are this
# module's vocabulary and are kept; only the declaration moved.
_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "OdysseyTrafficAgent")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from defines import (  # noqa: E402
    NUM_HISTORY_FRAMES,
    PLANNER_POSE_DT as PLAN_DT,   # seconds between navsim planner poses
    PLAN_FILE_DT as IPC_DT,       # seconds the simulator assumes for the plan rows
)


# --------------------------------------------------------------------------- #
# Trajectory format bridging
# --------------------------------------------------------------------------- #
def upsample_trajectory(traj_05s: np.ndarray, plan_dt: float = PLAN_DT) -> np.ndarray:
    """(N,3) @0.5s ego-local (x, y, heading) -> (M,3) @0.1s for the simulator.

    Linear on x/y; angle-continuous (unwrapped) on heading. The planner poses are the
    strictly-future samples t=0.5..N*0.5s (t=0 is the ego origin, added here only as an
    interpolation anchor); PlannerClient prepends the zero row itself.
    """
    traj_05s = np.asarray(traj_05s, dtype=np.float64)
    n = traj_05s.shape[0]
    src_t = np.concatenate([[0.0], np.arange(1, n + 1) * plan_dt])
    src_xy = np.concatenate([np.zeros((1, 2)), traj_05s[:, :2]], axis=0)
    src_h = np.concatenate([[0.0], np.unwrap(traj_05s[:, 2])], axis=0)

    n_out = int(round(n * plan_dt / IPC_DT))         # 40 for 8 poses / 4 s
    dst_t = np.arange(1, n_out + 1) * IPC_DT          # 0.1..4.0 s (future only)
    out = np.zeros((n_out, 3), dtype=np.float64)
    out[:, 0] = np.interp(dst_t, src_t, src_xy[:, 0])
    out[:, 1] = np.interp(dst_t, src_t, src_xy[:, 1])
    out[:, 2] = np.interp(dst_t, src_t, src_h)
    return out


def scene_prefix(log_token: str) -> str:
    """The per-scene file prefix of a --record run's plan_traj/{prefix}_{step}.npy."""
    parts = log_token.split("-")
    if len(parts[-1]) == 3:                # synthetic, e.g. bb4f37403cea5b0e-001
        return "-".join(parts[-2:])
    return parts[-1]


# --------------------------------------------------------------------------- #
# Camera-only frames: lidar_path is None
# --------------------------------------------------------------------------- #
_LIDAR_PATCHED: Dict[str, bool] = {}


def patch_lidar_for_camera_only(module_name: str = "navsim.common.dataclasses") -> None:
    """Make navsim tolerate the ``lidar_path=None`` the simulator writes.

    The simulator renders cameras only, so every frame dict carries ``lidar_path=None``.
    Stock navsim does ``Path(scene_dict_list[i]["lidar_path"])`` BEFORE
    ``Lidar.from_paths``'s own ``"lidar_pc" in sensor_names`` check, and ``Path(None)``
    raises TypeError. Some forks (DrivoR) wrap that in a bare ``except``; others do not.

    Rather than guess per fork, make ``Path(None)`` itself survive: it yields ``None``,
    which ``Lidar.from_paths``'s own ``"lidar_pc" in sensor_names`` guard then turns into
    an empty ``Lidar()``.

    Subclassing ``type(real_path())`` rather than ``real_path`` is deliberate: ``Path`` is
    abstract and instantiating it returns the concrete OS flavour (PosixPath), which is the
    class that must actually be subclassed.
    """
    if _LIDAR_PATCHED.get(module_name):
        return
    import importlib

    dc = importlib.import_module(module_name)
    real_path = dc.Path

    class _NoneTolerantPath(type(real_path())):             # type: ignore[misc,valid-type]
        """Path, except ``Path(None)`` yields None instead of raising."""

        def __new__(cls, *args, **kw):
            if len(args) == 1 and args[0] is None:
                return None                      # from_paths' own guard then returns Lidar()
            return real_path(*args, **kw)

    dc.Path = _NoneTolerantPath
    _LIDAR_PATCHED[module_name] = True


def inject_ay_enabled() -> bool:
    """Is the lateral-accel reconstruction switched on for this run? (env ODYSSEY_INJECT_AY=1)"""
    return str(os.environ.get("ODYSSEY_INJECT_AY", "0")).strip() == "1"


def inject_ay(scene_list: List[dict]) -> List[dict]:
    """Refill ``ego_dynamic_state[3]`` (ay) with the centripetal term ``v_x * omega``.

    WHY THIS EXISTS. What the rear-axle kinematic bicycle sets to zero is the rear axle's LATERAL
    VELOCITY -- the no-slip constraint -- not its lateral acceleration. Differentiate the body-frame
    velocity of a yawing rigid body, ``a = dv/dt|body + omega x v`` with ``v = (v_x, 0)``, and the
    lateral component is ``a_y = v_x * omega``, nonzero whenever the car turns. The plant simply
    never writes it down: kinematic_bicycle.py stores ``new_rear_acceleration = [accel, 0]`` and
    vehicle_utils.get_acceleration_shifted adds only terms along ``displacement = [d, 0]``, so the
    field the planner reads is an exact zero (|ay| ~ 1e-16 in two_stage_controller runs). navsim TRAINING reads the very same field from the OpenScene logs
    -- ``ego_acceleration = ego_dynamic_state[2:]``, byte-identical code -- where ay is real:
    |ay| mean 0.2517, std 0.484, 14% of frames above 0.5. The models were therefore trained on a
    channel that is identically zero at inference, and ``ay == 0.000000`` is a value essentially
    absent from the training distribution.

    WHAT THIS IS. Inside the plant it is an IDENTITY, not a guess: ``v_x * omega`` is this model's
    own lateral acceleration, recomputed from two state variables the plant does maintain (body
    v_x, and the vehicle's yaw rate in can_bus[15]). Against LOGGED GT it correlates at r = 0.903
    with a mean residual of 0.103 (15% of frames off by more than 0.2) -- but that gap measures the
    kinematic plant against a real vehicle, whose ay also carries tire-slip and sideslip-rate
    terms, not an error in the reconstruction. The scale is what matters here: |v*omega| mean
    0.2371 against GT 0.2517, i.e. the channel is back inside the distribution the model was
    trained on instead of sitting at a value the model never saw.

    DELIBERATELY CONSUMER-SIDE. Applied to the planner's input copy only, never to
    data_manager's ego_dynamic_state. The plant's state array declares ay = 0; deriving it in one
    consumer keeps the declaration the single source, where writing the derived value back would
    leave two. The cost is that every OTHER consumer keeps reading zero,
    and one of them scores: PDM comfort takes lateral acceleration and lateral jerk straight from
    ACCELERATION_Y (pdm_comfort_metrics._extract_ego_acceleration; the rear-axle-to-centre shift
    adds only a yaw-acceleration term), so its |ay| <= 4.89 m/s^2 bound is never actually exercised.
    The simulator state, the saved frames and the scoring all keep ay = 0; only what the model
    reads changes.

    Caller is responsible for having deep-copied, or for not caring: this mutates in place and
    returns the same list.
    """
    for f in scene_list:
        eds = f.get("ego_dynamic_state")
        cb = f.get("can_bus")
        # No silent fallback: both sources must be present. A frame missing either is
        # left exactly as it was rather than being filled from a guess.
        if eds is None or len(eds) < 4 or cb is None or len(cb) < 16:
            continue
        _eds = list(eds)
        # can_bus[15] is the vehicle model's own yaw rate (data_manager.py writes
        # rear_vehicle.current_angular_velocity there); eds[0] is body-frame v_x.
        _eds[3] = float(_eds[0]) * float(cb[15])
        f["ego_dynamic_state"] = (np.asarray(_eds, dtype=np.asarray(eds).dtype)
                                  if isinstance(eds, np.ndarray) else type(eds)(_eds))
    return scene_list


def pad_history(frames: List[dict], history_stride: int = 1,
                num_history_frames: int = NUM_HISTORY_FRAMES) -> List[dict]:
    """Return exactly ``num_history_frames`` dicts, oldest..newest, sampled at
    ``history_stride`` and front-padded.

    stride>1 (a fine rollout) picks frames 0.5 s apart out of the dense 0.1 s stream so the
    model keeps its trained 0.5 s history spacing; stride 1 = consecutive. Early in a rollout
    there are fewer frames than the model wants, so the oldest available is repeated -- navsim
    indexes this list positionally and would IndexError on a short one.
    """
    stride = max(1, int(history_stride))
    # newest-first, stride apart: frames[-1], frames[-1-S], ...; drop picks off the front
    picks = [frames[-1 - k * stride] for k in range(num_history_frames)
             if 1 + k * stride <= len(frames)]
    buf = list(reversed(picks))             # -> oldest..newest
    while len(buf) < num_history_frames:
        buf.insert(0, buf[0])               # repeat oldest at rollout start
    return buf
