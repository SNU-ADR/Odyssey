# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
"""
Base class for vehicle / pedestrian / etc. objects.
"""
import copy
import math
import numpy as np
import logging
from collections import deque

# controllers.
from odyssey.components.agents.controller import build_controller
from odyssey.components.agents.vehicle_model.vehicle_utils import RearVehicle

from odyssey.components.agents.base_agent import BaseAgent
from odyssey.common.dataclasses import Trajectory

from odyssey.scenario.scenarios.scenario_description import ScenarioDescription as SD
from odyssey.utils import math_utils


logger = logging.getLogger(__name__)


class EgoAgent(BaseAgent):
    """
    BaseAgent is something interacting with game engine.
    """

    def __init__(self, object_id, object_track, name, random_seed=None, config=None):

        assert object_id == 'ego'
        super(EgoAgent, self).__init__(
            object_id=object_id,
            object_track=object_track,
            name=name,
            random_seed=random_seed,
            config=config,)

        self.rear_vehicle = RearVehicle(self)

        self._length = self.vehicle.length
        self._height = self.vehicle.height
        self._width = self.vehicle.width

    def reset(self,):
        super(EgoAgent, self).reset()

        # other info
        #  control information.
        self.throttle_brake = 0.0
        self.steering = 0
        self.last_current_action = deque([(0.0, 0.0), (0.0, 0.0)], maxlen=2)

        # vehicle information
        self.tire_steering = 0
        self.acceleration = [0, 0]
        self.angular_acceleration = 0

        self._apply_start_lane_shift()
        self._apply_start_speed()
        self._apply_start_accel()

    # ---------------------------------------------------------------- start-lane shift
    #: Translation from the logged start pose to the shifted one, in scene-local metres. Zero
    #: unless ego_start_lane_shift moved the ego. planner_client adds it to the warm-up replay
    #: so the reference the tracker follows before the planner takes over moves with the ego.
    start_lane_shift_delta = np.zeros(2, dtype=np.float64)
    #: The shift that was ACTUALLY applied, which is not the requested one when the road has
    #: fewer lanes than asked for (see the FALLBACK branch). Read this, never the config, when
    #: labelling a result.
    start_lane_shift_applied = 0
    #: Largest |shift| the dashboard offers. Beyond this it is a typo, not a road.
    _LANE_SHIFT_MAX_N = 2

    #: Constant speed the warm-up replays the logged path at, in m/s, or None when the warm-up
    #: keeps the log's own timing. planner_client reads it; see ego_start_speed_delta.
    start_speed_warmup = None
    #: The speed delta ACTUALLY applied -- 0.0 when the requested one would have gone negative
    #: and the run fell back to the log. Read this, never the config, when labelling a result.
    start_speed_delta_applied = 0.0
    #: (v0, a) for a warm-up that replays the logged path at speed v0 + a*t, or None when the
    #: warm-up keeps the log's timing. planner_client reads it; see ego_start_accel_delta.
    start_accel_warmup = None
    #: The acceleration delta ACTUALLY applied -- 0.0 after a fallback.
    start_accel_delta_applied = 0.0

    #: A lane whose centerline sits closer than this to the ego's own is the same lane under a
    #: different id (nuPlan splits lanes at roadblock boundaries), not a neighbour to move into.
    _LANE_SHIFT_MIN_SEPARATION_M = 1.0
    #: How far the nearest point on a candidate may sit ALONG the ego's heading. A candidate whose
    #: perpendicular foot is metres fore or aft is not beside the ego at all.
    _LANE_SHIFT_MAX_LONGITUDINAL_M = 2.0
    #: A candidate must run roughly the way the ego is pointing. Without this the crossing arms of
    #: an intersection qualify as "the next lane over".
    _LANE_SHIFT_MAX_HEADING_DEG = 20.0

    def _apply_start_lane_shift(self):
        """Move the ego to a neighbouring lane at spawn, keeping its logged speed.

        Runs at reset only, so the shift is an INITIAL CONDITION: from step 1 the rollout is
        ordinary closed loop and the planner has to recover on its own. See
        `ego_start_lane_shift` in default_runner.yaml for why object_track is left alone.

        Neighbours are found GEOMETRICALLY, not from map topology, because neither topological
        route is usable here:

        `left_neighbor` / `right_neighbor` are wrong. nuplan_utils.extract_map_features builds them
        by slicing the UNSORTED `block.interior_edges` with an index taken from the SORTED copy, so
        a lane can list itself as its own neighbour and single-entry lists come back empty.

        Grouping by `roadblock_id` does not work either, and that is nuPlan's design rather than a
        bug: the ego often starts INSIDE AN INTERSECTION, on a roadblock CONNECTOR. Parallel
        connectors through one intersection sit in different roadblocks, so same-roadblock
        grouping can find a single lane and report no neighbour where several exist.

        So: any lane feature that runs roughly parallel to the ego, whose perpendicular foot is
        beside the ego rather than fore or aft, ordered by how far to the side it lies. The
        endpoint check is what makes the foot meaningful -- argmin against a lane that ended
        behind the ego returns its last vertex, which could teleport the ego up the road
        instead of across it.
        """
        raw = self.config.get('ego_start_lane_shift', 0)
        try:
            shift = int(str(raw).strip() or 0)
        except (TypeError, ValueError):
            raise ValueError(
                f"ego_start_lane_shift must be an integer in "
                f"[{-self._LANE_SHIFT_MAX_N}, {self._LANE_SHIFT_MAX_N}], got {raw!r}")
        if shift == 0:
            return
        # Out of range is a TYPO and raises; a map that simply has fewer lanes is the world being
        # what it is and falls back below. Conflating the two would either hide a mistyped knob or
        # kill a legitimate scenario.
        if abs(shift) > self._LANE_SHIFT_MAX_N:
            raise ValueError(
                f"ego_start_lane_shift={shift} is out of range "
                f"[{-self._LANE_SHIFT_MAX_N}, {self._LANE_SHIFT_MAX_N}]")
        side = 'left' if shift > 0 else 'right'
        n = abs(shift)

        from odyssey.engine.engine_utils import get_engine
        scene = get_engine().managers['scenario_manager'].current_scene

        p0 = np.asarray(self.current_position, dtype=np.float64)[:2]
        h0 = float(self.current_heading)
        # +lateral is to the ego's LEFT, matching openloop_replay_controller's convention.
        fwd = np.array([math.cos(h0), math.sin(h0)], dtype=np.float64)
        left = np.array([-math.sin(h0), math.cos(h0)], dtype=np.float64)

        want_positive = (side == 'left')
        candidates = []
        for k, v in scene[SD.MAP_FEATURES].items():
            if not str(v.get('type', '')).startswith('LANE'):
                continue
            pl = np.asarray(v.get('polyline', ()), dtype=np.float64)
            if pl.ndim != 2 or len(pl) < 2:
                continue
            pl = pl[:, :2]
            j = int(np.argmin(np.linalg.norm(pl - p0, axis=1)))
            if j == 0 or j == len(pl) - 1:
                continue                                    # ended before/after the ego
            delta = pl[j] - p0
            if abs(float(delta @ fwd)) > self._LANE_SHIFT_MAX_LONGITUDINAL_M:
                continue
            tangent = pl[min(j + 1, len(pl) - 1)] - pl[max(j - 1, 0)]
            if np.linalg.norm(tangent) < 1e-6:
                continue
            dtheta = abs(math.degrees(math_utils.wrap_to_pi(
                math.atan2(tangent[1], tangent[0]) - h0)))
            if dtheta > self._LANE_SHIFT_MAX_HEADING_DEG:
                continue
            lat = float(delta @ left)
            if abs(lat) < self._LANE_SHIFT_MIN_SEPARATION_M:
                continue                                    # the ego's own lane, or a re-id of it
            if (lat > 0) == want_positive:
                candidates.append((abs(lat), k, pl, j))
        candidates.sort(key=lambda c: c[0])
        if len(candidates) < n:
            # The road does not have the lane that was asked for. Fall back to the logged start
            # rather than failing the scenario, so a campaign that sweeps -2..+2 over many scenes
            # still produces a row for every one.
            #
            # The cost is that this row is now indistinguishable from a shift=0 row by its numbers
            # alone, so the warning has to be the record: it names the requested shift and what
            # was actually applied. `start_lane_shift_applied` carries the same thing in-process.
            logger.warning(
                "[start-lane-shift] FALLBACK: requested %+d (%s x%d) but only %d parallel "
                "lane(s) run alongside the ego's start pose on that side %s. "
                "Starting in the LOGGED lane instead -- this rollout is a shift=0 run.",
                shift, side, n, len(candidates),
                [(k, round(l, 2)) for l, k, _, _ in candidates])
            self.start_lane_shift_applied = 0
            return

        _, target, pl, j = candidates[n - 1]
        # Heading from the target lane's local tangent, so the ego points along the new lane
        # rather than keeping a heading that was tangent to the old one.
        k1 = min(j + 1, len(pl) - 1)
        k0 = max(k1 - 1, 0)
        tangent = pl[k1] - pl[k0]
        heading = float(math.atan2(tangent[1], tangent[0])) if np.linalg.norm(tangent) > 1e-6 else h0

        moved = float(np.linalg.norm(pl[j] - p0))
        self.set_position(np.array([pl[j][0], pl[j][1]], dtype=np.float64))
        self.set_heading_theta(heading)
        # reset() already latched these off the un-shifted pose; a stale `last_position` would
        # show up as a phantom first-step displacement in every speed/heading estimate built from
        # the difference.
        self.last_position = self._cur_pos
        self.last_heading_dir = self._cur_heading_theta
        # The rollout does not begin under the planner. For the first `num_history - 1` steps
        # planner_client replays object_track as ABSOLUTE waypoints, so a shifted ego is simply
        # off that reference and the tracker hauls it back to the logged lane before the planner
        # ever gets a say. So the warm-up reference has to move with the ego. A rigid translation is enough: the
        # target lane was accepted only if it runs parallel to the ego (<= 20 deg), so the logged
        # path translated onto it stays in the new lane for the ~8 m the warm-up covers.
        self.start_lane_shift_delta = np.asarray(self._cur_pos, dtype=np.float64)[:2] - p0
        self.start_lane_shift_applied = shift

        logger.info(
            "[start-lane-shift] applied %+d (ego %s x%d) -> lane %s, moved %.2f m, "
            "heading %+.1f -> %+.1f deg, speed %.2f m/s unchanged; "
            "warm-up reference translated by (%+.2f, %+.2f)",
            shift, side, n, target, moved,
            math.degrees(h0), math.degrees(heading),
            float(np.linalg.norm(np.asarray(self._cur_velocity, dtype=np.float64)[:2])),
            self.start_lane_shift_delta[0], self.start_lane_shift_delta[1])

    def _warmup_steps(self):
        """Number of steps the warm-up covers. Fails instead of guessing if num_history is missing.

        Falling back to a default of 1 would shrink the warm-up window to one step and silently
        give wrong mean speed/acceleration. num_history always comes from the config.
        """
        if 'num_history' not in self.config:
            raise KeyError("num_history is missing from the config; the warm-up window length "
                           "cannot be made up here.")
        return max(int(self.config['num_history']) - 1, 1)

    def _apply_start_speed(self):
        """Hand the rollout to the planner at a constant speed other than the log's.

        The warm-up (the first num_history-1 steps, before the planner is queried) replays the
        logged path. With ego_start_speed_delta set it replays the same path at a CONSTANT speed,
        mean(log speed over the warm-up) + delta, and the ego spawns already at that speed. At
        handoff the planner therefore sees a history and an ego_dynamic_state that both say
        "faster/slower than the log", on the log's own path.

        Setting the velocity here is enough for the plant: RearVehicle keeps no state of its own
        and the bicycle model reads rear_vehicle.current_velocity, which is derived from
        agent.current_velocity every call. What is NOT enough is leaving last_velocity alone --
        see below.
        """
        raw = self.config.get('ego_start_speed_delta', 0.0)
        try:
            delta = float(str(raw).strip() or 0.0)
        except (TypeError, ValueError):
            raise ValueError(f"ego_start_speed_delta must be a number in m/s, got {raw!r}")
        if not math.isfinite(delta):
            raise ValueError(f"ego_start_speed_delta must be finite, got {raw!r}")
        if delta == 0.0:
            return

        # The mean is taken over the steps the warm-up actually covers, not a hardcoded 1.5 s:
        # num_history is 16 at rollout_dt=0.1 and 4 at 0.5, and the window must follow it.
        warm = self._warmup_steps()
        v_log = np.linalg.norm(
            np.asarray(self.object_track['velocity'], dtype=np.float64)[:warm + 1, :2], axis=1)
        v_mean = float(v_log.mean())
        target = v_mean + delta
        if target < 0.0:
            # Slower than a log that is already (nearly) stopped. There is no such speed; run the
            # log's own warm-up and say so. A log whose ego is stopped for the whole warm-up
            # lands here for every negative delta.
            logger.warning(
                "[start-speed] FALLBACK: requested %+.2f m/s but the log's mean over the %d-step "
                "warm-up is %.2f m/s, so the target would be %.2f m/s. Running the LOG's warm-up "
                "instead -- this rollout is a delta=0 run.",
                delta, warm, v_mean, target)
            self.start_speed_delta_applied = 0.0
            return

        h = float(self.current_heading)
        self.set_velocity(np.array([math.cos(h), math.sin(h)], dtype=np.float64), value=target)
        # set_acceleration() falls back to (current - last) / dt whenever the vehicle model has not
        # set a value, and reset() latched last_velocity off the LOG's speed. Left alone, the first
        # step would report (target - v_log0) / dt to the planner as the ego's acceleration --
        # +20 m/s^2 for a +2 m/s delta at 0.1 s -- through ego_dynamic_state. Constant-speed
        # entry means zero acceleration at handoff, so say exactly that.
        self.last_velocity = self._cur_velocity
        self.acceleration = np.zeros(2, dtype=np.float64)
        self.start_speed_warmup = target
        self.start_speed_delta_applied = delta
        logger.info(
            "[start-speed] applied %+.2f m/s: warm-up replays the logged path at a constant "
            "%.2f m/s (log mean over %d steps %.2f, log v0 %.2f)",
            delta, target, warm, v_mean, float(v_log[0]))

    def _apply_start_accel(self):
        """Hand the rollout to the planner mid-acceleration: warm-up at v0 + a*t on the log's path.

        The logged start speed v0 is kept; the warm-up replays the logged PATH with the speed
        ramping at a = mean(log acceleration over the warm-up) + ego_start_accel_delta. At handoff
        the planner therefore sees a different speed (v0 + a*T) AND a non-zero acceleration in
        ego_dynamic_state. The acceleration is what separates this from ego_start_speed_delta,
        whose handoff acceleration is zero by construction.

        Mutually exclusive with ego_start_speed_delta: both rewrite the warm-up speed profile and
        there is no single reading of "both".
        """
        raw = self.config.get('ego_start_accel_delta', 0.0)
        try:
            delta = float(str(raw).strip() or 0.0)
        except (TypeError, ValueError):
            raise ValueError(f"ego_start_accel_delta must be a number in m/s^2, got {raw!r}")
        if not math.isfinite(delta):
            raise ValueError(f"ego_start_accel_delta must be finite, got {raw!r}")
        if delta == 0.0:
            return
        try:
            speed_delta = float(str(self.config.get('ego_start_speed_delta', 0.0)).strip() or 0.0)
        except (TypeError, ValueError):
            speed_delta = 0.0
        if speed_delta != 0.0:
            raise ValueError(
                f"ego_start_accel_delta={delta} and ego_start_speed_delta={speed_delta} are both "
                f"set. Each rewrites the warm-up speed profile; set one.")

        from odyssey.engine.engine_utils import get_engine
        dt = float(get_engine().sim_dt)
        warm = self._warmup_steps()
        v_log = np.linalg.norm(
            np.asarray(self.object_track['velocity'], dtype=np.float64)[:warm + 1, :2], axis=1)
        v0 = float(v_log[0])
        a_mean = float(np.mean(np.diff(v_log) / dt))
        a = a_mean + delta
        T = warm * dt
        if v0 + a * T < 0.0:
            # The ramp would reverse the car before the planner even takes over (any negative
            # delta on a log that sits at 0 m/s). Same rule as the other
            # initial-condition knobs: warn, run the log's warm-up, and record what was applied.
            logger.warning(
                "[start-accel] FALLBACK: requested %+.2f m/s^2, but v0 %.2f + a %.2f * %.1f s = "
                "%.2f m/s < 0 before handoff. Running the LOG's warm-up instead -- this rollout "
                "is a delta=0 run.", delta, v0, a, T, v0 + a * T)
            self.start_accel_delta_applied = 0.0
            return

        h = float(self.current_heading)
        # v0 is the log's own, so reset() already holds it; set it explicitly anyway so the plant,
        # the warm-up profile and ego_dynamic_state are guaranteed to start from the same number.
        self.set_velocity(np.array([math.cos(h), math.sin(h)], dtype=np.float64), value=v0)
        self.last_velocity = self._cur_velocity
        # agent.acceleration is GLOBAL-frame (RearVehicle rotates it into the body frame), so the
        # ramp goes along the heading. From step 1 the bicycle model reports the acceleration it
        # actually achieved; this only sets what the step-0 frame says.
        self.acceleration = np.array([a * math.cos(h), a * math.sin(h)], dtype=np.float64)
        self.start_accel_warmup = (v0, a)
        self.start_accel_delta_applied = delta
        logger.info(
            "[start-accel] applied %+.2f m/s^2: warm-up replays the logged path at "
            "%.2f %+.2f*t m/s (log mean accel over %d steps %+.2f); handoff at %.2f m/s",
            delta, v0, a, warm, a_mean, v0 + a * T)

    def _preprocess_action(self, action):
        if action is None:
            return None, {"raw_action": None}
        return action, {'raw_action': (action[0], action[1])}

    def before_step(self, action=None):
        """
        Save info and make decision before action
        """
        if action is not None:
            assert len(action) == 2
        action, step_info = self._preprocess_action(action)

        self.last_position = self._cur_pos
        self.last_heading_dir = self._cur_heading_theta
        self.last_velocity = self._cur_velocity  # 2D vector
        self._acc_from_model = False  # reset per step; see set_acceleration()
        if action is not None:
            self.set_action(action)
        return step_info

    def step(self):

        self.commit_step_plan(self.compute_step_plan())

    def compute_step_plan(self):
        """Compute this tick's ego plan without changing the vehicle state.

        Keeping planning separate from state propagation lets the agent manager run a
        GPU/external ego planner alongside the CPU background-traffic solver.  Both sides then
        observe the same state at the beginning of the tick; neither can see the other's
        partially committed next state.

        The boolean preserves the old distinction between an invalid policy step (do nothing)
        and a valid policy which deliberately returns ``None`` (the controller may hold its
        previous command).
        """
        if not self.policy.is_current_step_valid:
            return False, None
        return True, self.policy.act()

    def commit_step_plan(self, result):
        """Apply a plan returned by :meth:`compute_step_plan` to the ego vehicle."""

        valid, trajectory = result
        if not valid:
            return

        self.trajectory = trajectory

        # controller to update the position.
        self.controller.step()
        # set acceleration
        self.set_acceleration(None)
        self._traj_step += 1

    def after_step(self):
        if self.navigation is not None:
            self.navigation.update_localization()

        step_info = {
            "position": self.current_position,
            "heading": self.current_heading,
            "velocity": float(self.current_speed),
            "length": self.length,
            "width": self.width,
            "height": self.height,
        } # TODO: valid judgement

        return step_info

    def set_action(self, action):
        if action is None:
            return
        self.steering = action[0]
        self.throttle_brake = action[1]

        self.last_current_action.append(action)  # the real step of physics world is implemented in taskMgr.step()

    @property
    def current_steering(self):
        return self.steering

    def set_acceleration(self, acc):
        """Record the ego acceleration reported to the planner.

        Restores nuPlan semantics. nuplan-devkit's two_stage_controller takes the ego state --
        acceleration included -- straight from KinematicBicycleModel.propagate_state(), where
        the acceleration is the low-pass-filtered value the model itself produced; nuPlan never
        finite-differences velocity to obtain it (there is no `velocity_diff`/`last_velocity`
        anywhere in its controller). Odyssey's fork added an override here that discarded
        the model's physical value in favour of an unbounded 0.5 s finite difference, which
        amplified controller velocity noise into non-physical accelerations (-470 m/s^2
        observed) -- garbage input for a planner that leans on ego-status.

        Teleport/replay controllers are a Odyssey-only feature (nuPlan has no equivalent)
        and never invoke the vehicle model, so the finite difference is kept as a fallback for
        those paths only.
        """
        if acc is not None:
            self.acceleration = np.asarray(acc, dtype=float).reshape(-1)[:2]
            self._acc_from_model = True
        elif getattr(self, '_acc_from_model', False):
            return                      # vehicle model already set a physical value this step
        elif getattr(self, 'last_velocity', None) is not None:
            # One outer step's velocity change over one outer step. sim_dt is the resolved
            # step, because ScenarioManager already upsampled the scene to match it; a fixed
            # 0.5 fallback would be wrong on 0.1 s scenes. This is planner INPUT, not just a
            # logged metric.
            _dt = float(self.engine.sim_dt)
            self.acceleration = (np.asarray(self._cur_velocity, dtype=float)
                                 - np.asarray(self.last_velocity, dtype=float)) / _dt
        else:
            self.acceleration = np.zeros(2, dtype=float)

    @property
    def current_acceleration(self):
        return self.acceleration

    def set_angular_acceleration(self, angular_acc):
        self.angular_acceleration = angular_acc

    @property
    def current_angular_acceleration(self):
        return self.angular_acceleration

    def set_tire_steering(self, tire_steering):
        self.tire_steering = tire_steering

    @property
    def current_tire_steering(self):
        """
        return the steering of the car.
        """
        return self.tire_steering

    @property
    def last_and_current_action(self):
        """
        return the last and current action of the car.
        """
        return self.last_current_action
