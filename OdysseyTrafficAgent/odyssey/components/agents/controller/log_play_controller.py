# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
import logging

import numpy as np

from odyssey.components.agents.controller.abstract_controller import AbstractController
from odyssey.engine.engine_utils import get_engine
from odyssey.utils.cadence import plan_file_stride

logger = logging.getLogger(__name__)


class LogPlayController(AbstractController):
    """
    Assume tracking controller is absolutely perfect, and just follow a trajectory:
    teleport the ego to the planner's predicted next waypoint each step.

    Temporal-window fusion (global_config) damps the jerky accel/decel seen in a 0.1 s true closed
    loop, where step-to-step disagreement between successive fresh predictions of the SAME future pose
    makes the ego lurch. Fuse the predictions the last `fusion_window` steps each made for the SAME
    absolute target (the next sim step). Waypoints are world/absolute, so at step t the newest traj's
    waypoints[stride] targets t+dt, the traj from t-1 targets it at waypoints[2*stride], ..., the
    traj k steps old at waypoints[(1+k)*stride].

    fusion_mode  = none | mean | weighted   (weighting of the aligned estimates)
        none     -> execute waypoints[stride] only (original behaviour).
        mean     -> equal weights.
        weighted -> more weight on the newest (w_k = fusion_decay ** k, k=0 newest).
    fusion_space = position | velocity      (WHAT is averaged)
        position -> average the absolute predicted positions. Faithful to "average the t+0.1 points",
                    but blends STALE far-horizon predictions (made from older ego states) so the target
                    accumulates positional drift -> can leave the drivable area / collide.
        velocity -> average each prediction's per-step DISPLACEMENT (one stride apart) and
                    apply it to the ego's CURRENT real position. Same jerk damping (a velocity
                    low-pass), but re-anchored every step so there is NO positional drift.

    Warm-up (fewer than `fusion_window` steps, or an older prediction too short to reach t+dt) uses
    whatever aligned estimates exist. Every step is a planning step, so every entry is fresh.
    """

    # Below this speed the steering angle is not observable from the motion (yaw_dot / v is 0/0),
    # so the previous angle is held. 0.05 m/s is well under the slowest crawl these rollouts show.
    _V_EPS = 0.05
    _LOG_EVERY = 50

    # Envelope the teleport is allowed to move inside. A predicted pose that needs more than this
    # is not something the car could have reached, so executing it would credit the planner with
    # motion no vehicle can produce. Defaults are the usual comfort/traction figures:
    #   -0.5 g braking, +0.3 g drive.  Overridable per run (lp_accel_min / lp_accel_max).
    _A_MIN = -0.5 * 9.81
    _A_MAX = 0.3 * 9.81

    def __init__(self, agent):
        super().__init__(agent)
        self._traj_hist = []        # newest-last recent Trajectory objects (len <= fusion_window)
        self._last_ep_step = None
        self._prev_v_long = None    # for the finite differences the plant would otherwise produce
        self._prev_omega = None
        self._n_steps = 0
        self._n_clipped = 0         # steps whose demanded steering exceeded MAX_STEERING
        self._sum_abs_delta = 0.0
        self._max_abs_delta = 0.0
        self._n_fb_long = 0         # steps handed to the plant because accel was out of envelope
        self._n_fb_lat = 0          # ... because steering was
        self._mm = None             # fallback plant, built on first use

    def _bounds(self):
        c = self.agent.config
        return (float(c.get('lp_accel_min', self._A_MIN) or self._A_MIN),
                float(c.get('lp_accel_max', self._A_MAX) or self._A_MAX))

    def _plant(self):
        """The bicycle model used when a predicted pose is out of envelope.

        ACTUATOR LAG IS DISABLED HERE, deliberately. The two modes have to share one bandwidth or
        the switch itself becomes a disturbance: a teleport step reaches the demanded state within
        the step (that is what teleporting means), while the plant's first-order lags at
        rollout_dt=0.1 realise only dt/(dt+tau) of a command -- 0.1/(0.1+0.2) = 33 % of an
        acceleration, 0.1/(0.1+0.05) = 67 % of a steering angle. Mixing them would make every
        fallback step a sudden loss of authority, and a trace that dips in and out of the envelope
        would show ripples that belong to the mode switch rather than to the planner.

        The KBM exposes exactly these switches for this reason (see its __init__: they force the
        filter gain to 1, which is the tau -> 0 identity, rather than approximating with a small
        tau). Scoped to this private instance, so two_stage / hipad keep their lags. Set
        lp_fallback_lag=true to keep the lag instead and measure the mixed-bandwidth version.
        """
        if self._mm is None:
            from odyssey.components.agents.controller.motion_model.kinematic_bicycle import (
                KinematicBicycleModel,
            )
            self._mm = KinematicBicycleModel(self.agent)
            if not bool(self.agent.config.get('lp_fallback_lag', False)):
                self._mm._no_accel_lag = True
                self._mm._no_steering_lag = True
        self._mm._step_dt = self._dt()
        return self._mm

    def _dt(self):
        """Seconds per executed step.

        The resolved outer step, from the scene ScenarioManager already upsampled to match.
        Do not fall back to 0.5: that is the scenario's native period only when its frames
        happen to be 0.5 s apart.
        """
        return float(get_engine().sim_dt)

    def _step_stride(self):
        """Rows of the plan that one executed step covers.

        The planner's trajectory arrives on its own PLAN_FILE_DT grid, so "the pose one sim
        step ahead" is index `stride`, not index 1. The client does not decimate the array,
        so its spacing stays the constant the trackers read it by.
        """
        return plan_file_stride(self._dt())

    def _log_stats(self):
        """Report how much steering the executed plan actually demanded.

        A teleport hides this entirely: the body simply appears at the next pose, so a
        physically impossible plan and a comfortable one look identical. With the angle
        inverted from the motion, `clip` counts the steps the plan asked for more lock than the
        car has -- the direct answer to "could this trajectory have been driven at all".
        """
        if self._n_steps % self._LOG_EVERY:
            return
        n = max(self._n_steps, 1)
        a_min, a_max = self._bounds()
        logger.info(
            "[log_play] n=%d  |delta| mean %.4f max %.4f rad (MAX %.3f)  |  "
            "plant fallback: long %.0f%% lat %.0f%%  envelope [%.2f, %.2f] m/s^2  lag %s",
            self._n_steps, self._sum_abs_delta / n, self._max_abs_delta, self.agent.MAX_STEERING,
            100.0 * self._n_fb_long / n, 100.0 * self._n_fb_lat / n, a_min, a_max,
            "on" if bool(self.agent.config.get('lp_fallback_lag', False)) else "off")

    def _fusion_params(self):
        cfg = get_engine().global_config
        mode = str(cfg.get("fusion_mode", "none"))
        space = str(cfg.get("fusion_space", "position"))
        window = max(1, int(cfg.get("fusion_window", 5)))
        decay = float(cfg.get("fusion_decay", 0.5))
        return mode, space, window, decay

    def step(self):
        """Inherited, see superclass."""
        traj = self.agent.trajectory
        mode, space, window, decay = self._fusion_params()

        # Reset the window on a new episode: the same controller instance may be reused, so a
        # non-increasing episode_step means a scenario reset -- drop the stale history.
        ep_step = getattr(get_engine(), "episode_step", None)
        if self._last_ep_step is not None and ep_step is not None and ep_step <= self._last_ep_step:
            self._traj_hist = []
            # the finite differences and the steering tally belong to the finished episode
            self._prev_v_long = self._prev_omega = None
            self._n_steps = self._n_clipped = 0
            self._n_fb_long = self._n_fb_lat = 0
            self._sum_abs_delta = self._max_abs_delta = 0.0
        self._last_ep_step = ep_step

        self._traj_hist.append(traj)
        if len(self._traj_hist) > window:
            self._traj_hist = self._traj_hist[-window:]

        # Original behaviour: single trajectory (start), fusion off, or degenerate 1-point traj.
        _s = self._step_stride()
        if mode not in ("mean", "weighted") or len(traj.waypoints) <= _s:
            idx = 0 if len(traj.waypoints) <= _s else _s
            self._apply(traj.waypoints[idx], traj.velocities[idx],
                        traj.headings[idx], traj.angular_velocities[idx])
            return

        # Gather aligned estimates of the next sim step: the traj recorded k steps ago contributes
        # its index (1+k)*stride. For 'velocity' the per-step displacement spans one stride.
        pos, disp, vel, head, angv, wts = [], [], [], [], [], []
        for k in range(min(window, len(self._traj_hist))):
            t_k = self._traj_hist[-1 - k]
            j = (1 + k) * _s
            if len(t_k.waypoints) <= j:
                continue                                  # too short to reach t+dt -> skip
            pos.append(np.asarray(t_k.waypoints[j], dtype=float))
            disp.append(np.asarray(t_k.waypoints[j], dtype=float)
                        - np.asarray(t_k.waypoints[j - _s], dtype=float))
            vel.append(np.asarray(t_k.velocities[j], dtype=float))
            head.append(float(t_k.headings[j]))
            angv.append(float(t_k.angular_velocities[j]))
            wts.append(1.0 if mode == "mean" else decay ** k)

        if not pos:                                        # nothing aligned -> original behaviour
            self._apply(traj.waypoints[_s], traj.velocities[_s],
                        traj.headings[_s], traj.angular_velocities[_s])
            return

        w = np.asarray(wts, dtype=float)
        w = w / w.sum()
        f_vel = np.average(np.stack(vel), axis=0, weights=w)
        head = np.asarray(head, dtype=float)               # circular mean for the angle
        f_head = float(np.arctan2((w * np.sin(head)).sum(), (w * np.cos(head)).sum()))
        f_angv = float((w * np.asarray(angv, dtype=float)).sum())
        if space == "velocity":
            # re-anchor to the ego's actual current position (traj.waypoints[0]) + fused displacement
            f_pos = np.asarray(traj.waypoints[0], dtype=float) + np.average(np.stack(disp), axis=0, weights=w)
        elif space == "longitudinal":
            # Smooth ONLY the along-track (speed) component; keep the current prediction's DIRECTION,
            # heading and lateral path untouched. The jerk is longitudinal (accel/decel); averaging the
            # full 2D displacement also smooths the lateral path, which drifts the ego off already-good
            # scenes. Project each aligned displacement onto the current predicted direction, fuse only
            # that forward magnitude, and step the ego along the current direction by it.
            cur0 = np.asarray(traj.waypoints[0], dtype=float)
            cur_disp = np.asarray(traj.waypoints[_s], dtype=float) - cur0
            n = float(np.linalg.norm(cur_disp))
            dirn = cur_disp / (n + 1e-9)
            fwd = float(np.dot(np.average(np.stack(disp), axis=0, weights=w), dirn))
            f_pos = cur0 + dirn * fwd
            f_head = float(traj.headings[_s])                                 # keep current path heading
            f_vel = np.asarray(traj.velocities[_s], dtype=float) * (fwd / (n + 1e-9))  # scale by smoothed speed
            f_angv = float(traj.angular_velocities[_s])
        else:  # position
            f_pos = np.average(np.stack(pos), axis=0, weights=w)
        self._apply(f_pos, f_vel, f_head, f_angv)

    def _apply(self, position, velocity, heading, angular_velocity):
        """Place the ego at the planner's pose AND bring the bicycle state with it.

        The four setters below move the body but leave the plant's own state behind: the KBM
        also carries tire_steering, acceleration, angular acceleration and the rear-axle frame
        (kinematic_bicycle.py update_center_agent). Teleporting without them left
        `tire_steering` pinned at its initial value for the whole rollout -- the plant believed
        the wheels were straight while the body drove a curve -- and left the whole rear-axle
        state untouched.

        Frames: the incoming waypoint is a CENTRE pose. planner_client builds each state with
        EgoState.build_from_rear_axle() but then stores `ego_state.waypoint`, and nuplan's
        .waypoint wraps the car_footprint, i.e. the centre -- measured, 1.461 m ahead of the rear
        axle for the Pacifica. (hipad_pid_controller's aim-distance note calls these waypoints
        rear-axle; that is wrong, and only its lookahead floor rests on it.) update_center_agent
        takes REAR-axle inputs and does the shift itself, so the centre is converted back here.
        Getting this backwards would push the ego 1.461 m forward every single step.

        So instead of writing the body directly, invert the plant's own kinematics and hand the
        result to the same entry point the KBM uses. Everything downstream -- the centre pose,
        the shifted velocity/acceleration, ego status reported to the planner -- is then derived
        exactly as it is on a controller step, and cannot disagree with it by construction.

            delta = atan(L * yaw_dot / v)        <- inverse of the plant's yaw_dot = v tan(delta)/L
            a     = (v - v_prev) / dt            <- signed longitudinal, the plant's own convention

        `delta` is what makes this a KINEMATIC teleport rather than a jump: it is the steering
        angle the predicted motion actually demands, so a plan the car could not physically
        execute now shows up as a clip against MAX_STEERING instead of passing silently. The
        clipped count is logged (see _log_stats) -- that number is the reason to do this at all.
        """
        # Background / static / adversary agents are plain BaseAgents replayed on their logged path:
        # they carry no rear-vehicle / kinematic-bicycle model (rear_vehicle, MAX_STEERING,
        # vehicle.wheel_base, current_tire_steering are all EgoAgent-only). Set the centre state
        # directly -- the committed pre-fusion behaviour -- and skip the ego-only rear-axle refinement
        # below. Without this the first background agent crashes the whole simulation.
        rv = getattr(self.agent, "rear_vehicle", None)
        if rv is None:
            self.agent.set_position(position)
            self.agent.set_velocity(velocity, in_local_frame=False)
            self.agent.set_heading_theta(heading)
            self.agent.set_angular_velocity(angular_velocity)
            return
        pos = np.asarray(position, dtype=float).reshape(-1)[:2]
        heading = float(heading)
        omega = float(angular_velocity)
        dt = self._dt()

        # Longitudinal speed in the plant's convention: KBM keeps velocity as [v_long, 0] in the
        # body frame, so project rather than taking a norm -- a norm cannot express reversing.
        v = np.asarray(velocity, dtype=float).reshape(-1)[:2]
        v_long = float(v[0] * np.cos(heading) + v[1] * np.sin(heading))

        # Steering angle demanded by this step's motion. Guard the low-speed end: yaw_dot/v blows
        # up as v -> 0 and a stopped car's steering angle is unobservable from its (zero) motion,
        # so hold the previous angle there instead of inventing a huge one.
        prev_delta = float(self.agent.current_tire_steering)
        if abs(v_long) < self._V_EPS:
            delta = prev_delta
            clipped = False
        else:
            raw = np.arctan(self.agent.vehicle.wheel_base * omega / v_long)
            delta = float(np.clip(raw, -self.agent.MAX_STEERING, self.agent.MAX_STEERING))
            clipped = abs(raw) > self.agent.MAX_STEERING
        self._n_steps += 1
        self._n_clipped += int(clipped)
        self._sum_abs_delta += abs(delta)
        self._max_abs_delta = max(self._max_abs_delta, abs(delta))

        accel = (v_long - self._prev_v_long) / dt if self._prev_v_long is not None else 0.0
        ang_acc = (omega - self._prev_omega) / dt if self._prev_omega is not None else 0.0

        # ---- envelope gate -------------------------------------------------------------
        # Teleporting means asserting the car got to that pose. Assert it only when it could:
        # both the longitudinal effort and the steering the motion implies have to be inside the
        # vehicle's envelope. Longitudinal alone is not enough -- a pose can be perfectly
        # reachable in speed while demanding more lock than the wheels have, and executing that
        # leaves the recorded steering angle disagreeing with the motion it supposedly produced.
        a_min, a_max = self._bounds()
        long_bad = self._prev_v_long is not None and not (a_min <= accel <= a_max)
        if long_bad or clipped:
            self._n_fb_long += int(long_bad)
            self._n_fb_lat += int(clipped)
            # Hand the CLAMPED demand to the plant and let it produce the pose. The ego lands
            # short of the prediction -- that is the point: the gap is the part of the plan the
            # car could not do, and it is now visible in the trajectory instead of absorbed.
            a_cmd = float(np.clip(accel, a_min, a_max))
            rate_cmd = (delta - prev_delta) / dt
            self._prev_v_long, self._prev_omega = v_long, omega
            self._plant().propagate_state(a_cmd, rate_cmd)
            self._log_stats()
            return

        self._prev_v_long, self._prev_omega = v_long, omega

        # Centre -> rear axle (see the frame note above). Only the position needs it: the lever arm
        # lies along the body x-axis, so omega x d is purely lateral and the LONGITUDINAL speed is
        # the same at both points -- which is all the plant stores ([v_long, 0]).
        rear_pos = pos - rv.rear_axle_to_center_dist * np.array([np.cos(heading), np.sin(heading)])

        rv.update_center_agent(
            new_rear_pos=rear_pos,
            new_rear_heading=heading,
            new_rear_velocity=[v_long, 0.0],          # local frame, per update_center_agent's note
            new_rear_acceleration=[accel, 0.0],
            new_rear_angular_velocity=omega,
            new_rear_angular_acc=ang_acc,
            new_tire_steering=delta,
            new_action=[(delta - prev_delta) / dt, accel],   # steering RATE, as the plant records
        )
        self._log_stats()
