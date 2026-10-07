# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
import logging
import warnings
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple
import traceback
from collections import deque

import numpy as np
from omegaconf import DictConfig

from odyssey.manager import frenet
from odyssey.utils import cadence
from odyssey.manager.sdroute_metric import load_scene_route
from odyssey.utils import merge_dicts, concat_step_infos
from odyssey.engine.engine_utils import get_engine, close_engine, \
    engine_initialized, initialize_engine, initialize_global_config

from odyssey.manager.scenario_manager import ScenarioManager
from odyssey.manager.render_manager import RenderManager
from odyssey.manager.map_manager import ScenarioMapManager
from odyssey.manager.agent_manager import BaseAgentManager
from odyssey.manager.data_manager import DataManager
from odyssey.manager.metric_manager import MetricManager
from odyssey.runner.utils import RunnerReport
from odyssey.components.agents.planner.outside_planner import OutsidePlanner

class BaseEnv:

    # ===== Intialization =====
    def __init__(self, config: DictConfig, name: str, data: Dict):
        if config is None:
            config = {}
        self.logger = logging.getLogger("Odyssey")
        self.config = config
        self.name = name
        self.info_dicts = data

        # Whether the simulation is still running
        self._is_simulation_running = False
        self.max_step = self.config.max_step

        # SD route departure guard. None = unresolved, False = no sidecar for this scene
        # (guard off), ndarray = baked polyline. _sd_route_distance refreshes it on scene change.
        # _sd_route_crop is the scored span it actually measures distance to (RC's GT crop).
        self._sd_route_xy = None
        self._sd_route_crop = None

        # Stall guard for long audit runs. Off by default -- benchmark termination rules do not
        # change unless an experiment opts in. The deque is updated once per step:
        # done_function() can be called by both MetricManager and the env in the same tick.
        self._stall_scene = None
        self._stall_last_step = None
        self._stall_samples = deque()
        self._stall_cumulative_distance_m = 0.0

        self.outside_planner = None

    def lazy_init(self):
        """
        Only init once in runtime, variable here exists till the close_env is called
        :return: None
        """
        # The real init happens here (creates the ego and its modules).
        if engine_initialized():
            return
        initialize_global_config(self.config)
        initialize_engine(self.config)
        # Let managers reach back to the env. metric_manager uses done_function() to know whether
        # this step is the final one, so a run that ends early is still scored instead of falling
        # outside the window. Must precede setup_engine() -- it must be set at manager registration.
        self.engine.env = self
        self.setup_engine()
        self._after_lazy_init()

    def _after_lazy_init(self):
        pass

    @property
    def engine(self):
        return get_engine()

    # ===== Run-time =====
    def run(self) -> List[RunnerReport]:

        self.reset(seed=0)
        scenario_manager = self.engine.managers['scenario_manager']
        if len(scenario_manager.available_scenario_indices) == 0:
            self.logger.info("No scenarios to process. Returning empty reports.")
            return []

        reports = []
        start_time = time.perf_counter()
        current_scene_length = self.engine.global_config['num_history'] + self.engine.global_config['num_future'] - 1

        for step in range(self.max_step):
            try:
                self.logger.info(f"Step {self.engine.episode_step}/{current_scene_length} for scenario "
                                    f"({self.engine.managers['scenario_manager'].current_scene_index+1}/{self.engine.managers['scenario_manager'].num_scenarios}) "
                                    f"{self.engine.managers['scenario_manager'].current_scene_id}")
                actions = self._get_actions(self.engine.episode_step)
                _, _, termination, truncation = self.step(actions)

                if truncation or step == self.max_step - 1:
                    # append the last report
                    current_time = time.perf_counter()
                    completed_scenario_id = self.engine.managers['scenario_manager'].current_scene_id
                    report = RunnerReport(
                        succeeded=True,
                        error_message=None,
                        start_time=start_time,
                        end_time=current_time,
                        scenario_name=completed_scenario_id,
                        planner_name=None,
                        log_name=self.name,
                    )
                    reports.append(report)


                    # all scenarios done
                    if termination:
                        break

                    # next scenario
                    _ = self._reset_scenario()
                    start_time = time.perf_counter()

            except Exception as e:
                error = traceback.format_exc()

                self.logger.exception("----------- Simulation failed")

                # Score what was actually driven before the failure. Frames up to this step were
                # rendered, planned and stepped normally and only the failing step is missing;
                # discarding them would turn a nearly complete drive into zero data (a planner
                # stalling mid-run would lose the whole token).
                #
                # MetricManager scores from its own cache, so it needs no state the exception may
                # have corrupted, and _scored makes this a no-op for runs already scored at normal
                # termination. The report below is still succeeded=False -- this salvages the
                # score, it does not turn a failure into a success.
                try:
                    _mm = self.engine.managers.get('metric_manager')
                    if _mm is not None and not getattr(_mm, "_scored", False):
                        _mm._score_and_save({"token": _mm.current_scene["id"],
                                             "step": _mm.current_step})
                        self.logger.warning(
                            "Scored the %s steps completed before the failure.", _mm.current_step)
                except Exception:
                    # Never let salvage scoring mask the original failure.
                    self.logger.exception("Could not score the partial rollout; reporting the "
                                          "failure as-is.")

                # TODO: modify scene related materials
                failed_scenes = f"[{self.engine.managers['scenario_manager'].current_scene_id}, {self.name}]\n"
                self.logger.warning(f"\nFailed simulation [log,token]:\n {failed_scenes}")

                self.logger.warning("----------- Simulation failed!")
                
                if self.config.exit_on_failure:
                    raise RuntimeError('Simulation failed')
                
                current_time = time.perf_counter()
                report = RunnerReport(
                    succeeded=False,
                    error_message=error,
                    start_time=start_time,
                    end_time=current_time,
                    scenario_name=self.engine.managers['scenario_manager'].current_scene_id,
                    planner_name=None,
                    log_name=self.name,
                )
                reports.append(report)

                # Mark failed scenario as attempted
                scenario_manager.coverage[scenario_manager.current_scene_index] = 1
                # Check if all scenarios have been attempted
                if scenario_manager.all_scenes_completed:
                    break

                # next scenario
                _ = self._reset_scenario()
                start_time = time.perf_counter()

        return reports

        
    def step(self, actions):
        # prepare for stepping the simulation
        scene_manager_before_step_infos = self.engine.before_step(actions)
        # step all entities and the simulator
        self.engine.step(self.config.decision_repeat)
        # update states, if restore from episode data, position and heading will be force set in update_state() function
        scene_manager_after_step_infos = self.engine.after_step()

        engine_info = merge_dicts(scene_manager_after_step_infos, scene_manager_before_step_infos,
                                  allow_new_keys=True, without_copy=True)
        
        return self._get_step_return(actions, engine_info=engine_info)  # collect observation, reward, termination

    # Arrival uses the RC reference's GT-cropped SD span when it is available.
    # The GT endpoint remains a fallback for scenes without a usable sidecar.
    # Neither arrival predicate proves the post-rollout SDF-prefix RC.
    # The benchmark's arrival rule (defines.py); not configurable.
    GT_GOAL_PROGRESS_RATIO = cadence.GT_GOAL_PROGRESS_RATIO
    GT_GOAL_END_DIST_M = cadence.GT_GOAL_END_DIST_M
    SD_GOAL_PROGRESS_RATIO = cadence.SD_GOAL_PROGRESS_RATIO
    SD_GOAL_END_DIST_M = cadence.SD_GOAL_END_DIST_M

    def _sd_goal_state(self):
        """Project the executed rear axle onto the exact SD span used as RC's denominator."""
        scene = self.engine.managers['scenario_manager'].current_scene
        step = int(self.engine.episode_step)
        metric_manager = self.engine.managers.get('metric_manager')
        metric = getattr(metric_manager, '_rc_metric', None)
        reference = getattr(metric, 'reference', None)
        if (metric is None or metric_manager.current_scene is not scene
                or reference is None or reference.quality['suspect']):
            return None
        if step < metric.handoff:
            return None
        if getattr(self, '_sd_goal_scene', None) is not scene:
            crop = reference.crop_xy()
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(crop, axis=0), axis=1))]
            if arc[-1] < frenet.MIN_REF_LENGTH_M:
                return None
            self._sd_goal_ref = (crop, arc)
            self._sd_goal_last = 0
            self._sd_goal_scene = scene
            self._sd_goal_cache = None
        cached = self._sd_goal_cache
        if cached is not None and cached[0] == step:
            return cached[1]
        agent = self.engine.managers['agent_manager'].ego_agent
        origin = scene['metadata'].get('old_origin_in_current_coordinate')
        if origin is None:
            return None
        here = (np.asarray(agent.rear_vehicle.current_position, dtype=np.float64)[:2]
                - np.asarray(origin, dtype=np.float64).reshape(2) + metric.offset)
        crop, arc = self._sd_goal_ref
        lo = self._sd_goal_last
        hi = min(len(crop), max(lo + 2, int(np.searchsorted(
            arc, arc[lo] + frenet.WIN_FWD_M, side='right')) + 1))
        s_here, _, self._sd_goal_last = frenet._segment_project(here, crop, arc, lo, hi)
        state = (float(s_here / arc[-1]), float(np.linalg.norm(here - crop[-1])), float(arc[-1]))
        self._sd_goal_cache = (step, state)
        return state

    def _gt_goal_state(self):
        """GT endpoint arrival, independent of offline SD-route coverage.

        Both reference and query are rear-axle positions in scene-local coordinates.
        Exclude GT warm-up, preserve the score-grid reference endpoint, and update
        the walking cursor once per simulation step (multiple callers are readers).
        """
        agent = self.engine.managers['agent_manager'].ego_agent
        scene = self.engine.managers['scenario_manager'].current_scene
        step = int(self.engine.episode_step)
        handoff = int(self.engine.global_config['num_history']) - 1
        if step < handoff:
            return None
        if getattr(self, "_gt_goal_scene", None) is not scene:
            gt = np.asarray(agent.object_track['position'], dtype=np.float64)[:, :2]
            heading = np.asarray(agent.object_track['heading'], dtype=np.float64)
            stride = int(scene["cadence"].score_stride_steps)
            gt, heading = gt[handoff::stride], heading[handoff::stride]
            if len(gt) < 2 or len(heading) != len(gt):
                return None
            rac = float(agent.rear_vehicle.rear_axle_to_center_dist)
            rear = gt - rac * np.column_stack((np.cos(heading), np.sin(heading)))
            self._gt_goal_ref = frenet.arclen(rear)
            self._gt_goal_last = 0
            self._gt_goal_scene = scene
            self._gt_goal_cache = None
        cached = getattr(self, "_gt_goal_cache", None)
        if cached is not None and cached[0] == step:
            return cached[1]
        ref, ref_s = self._gt_goal_ref
        total = float(ref_s[-1])
        if total < frenet.MIN_REF_LENGTH_M:
            self._gt_goal_cache = (step, None)
            return None
        here = np.asarray(agent.rear_vehicle.current_position, dtype=np.float64)[:2]
        lo = self._gt_goal_last
        hi = min(len(ref), max(lo + 2, int(np.searchsorted(
            ref_s, ref_s[lo] + frenet.WIN_FWD_M, side="right")) + 1))
        if hi - lo < 2:
            progress = 1.0
        else:
            s_here, _, self._gt_goal_last = frenet._segment_project(here, ref, ref_s, lo, hi)
            progress = float(s_here / total)
        state = (progress, float(np.linalg.norm(here - ref[-1])), step)
        self._gt_goal_cache = (step, state)
        return state

    # An ego this far from the baked SD route is no longer following the given route.
    # The rollout ends (see done_function for how it is scored).
    SD_ROUTE_MAX_DIST_M = cadence.SD_ROUTE_MAX_DIST_M

    def _hopeless_stall_state(self):
        """Diagnostics if the ego is in a clearly hopeless stall, else None.

        Uses only executed ego poses and the current speed -- it neither reads planner output nor
        predicts future motion. It requires a long observation window, so ordinary signal waits
        and congestion do not trigger it; it is only useful as an opt-in to avoid spending the
        tail of a 160 s audit on an ego that has already been stopped for a minute.
        """
        cfg = self.engine.global_config
        if not bool(cfg.get("hopeless_stall_early_stop_enabled", False)):
            return None
        # With the guard on, these values must exist. Returning None would be indistinguishable
        # from the opted-in feature silently switching off.
        scene = self.engine.managers["scenario_manager"].current_scene
        agent = self.engine.managers["agent_manager"].ego_agent
        step = int(self.engine.episode_step)
        dt = float(cfg.get("rollout_dt", 0.0) or 0.0)
        if dt <= 0.0:
            dt = float(scene["cadence"].sim_dt)
        position = np.asarray(agent.current_position, dtype=np.float64)[:2]
        velocity = np.asarray(agent.current_velocity, dtype=np.float64)[:2]

        if self._stall_scene is not scene:
            self._stall_scene = scene
            self._stall_last_step = None
            self._stall_samples.clear()
            self._stall_cumulative_distance_m = 0.0

        if self._stall_last_step != step:
            if self._stall_samples:
                self._stall_cumulative_distance_m += float(np.linalg.norm(
                    position - self._stall_samples[-1][1]))
            self._stall_samples.append(
                (step, position.copy(), self._stall_cumulative_distance_m))
            self._stall_last_step = step

        min_elapsed_s = float(cfg.get("hopeless_stall_min_elapsed_s", 100.0))
        window_s = float(cfg.get("hopeless_stall_window_s", 60.0))
        max_travel_m = float(cfg.get("hopeless_stall_max_travel_m", 1.0))
        max_speed_mps = float(cfg.get("hopeless_stall_max_speed_mps", 0.2))
        min_step = int(round(min_elapsed_s / dt))
        window_steps = max(int(round(window_s / dt)), 1)
        while (len(self._stall_samples) > 1
               and self._stall_samples[1][0] <= step - window_steps):
            self._stall_samples.popleft()

        if step < min_step or not self._stall_samples:
            return None
        oldest_step, _oldest_position, oldest_distance = self._stall_samples[0]
        if step - oldest_step < window_steps:
            return None
        travelled = self._stall_cumulative_distance_m - oldest_distance
        speed = float(np.linalg.norm(velocity))
        if travelled > max_travel_m or speed > max_speed_mps:
            return None

        # End only on the scoring cadence -- the final state must land in the scored trajectory.
        # There is no default falling back to stride 1: 1 means "every step is a scoring frame",
        # which would cause exactly what this branch prevents (ending off the grid).
        stride = int(scene["cadence"].score_stride_steps)
        if stride > 1 and step % stride:
            return None
        return {
            "term_reason": "hopeless_stall",
            "stall_window_s": window_s,
            "stall_travel_m": float(travelled),
            "stall_speed_mps": speed,
        }

    def _sd_route_distance(self):
        """Rear-axle distance to the SCORED SD span; absence disables only the guard."""
        scene = self.engine.managers['scenario_manager'].current_scene
        if getattr(self, '_sd_route_scene', None) is not scene:
            self._sd_route_scene = scene
            self._sd_route_crop = None
            route, self._sd_route_path = load_scene_route()
            self._sd_route_xy = route if route is not None else False
        if self._sd_route_xy is False:
            return None
        origin = scene['metadata'].get('old_origin_in_current_coordinate')
        if origin is None:
            return None
        # Departure is measured against the span RC divides by, i.e. the SD span cropped by GT.
        # The full sidecar can continue well past the GT end point, so an ego that leaves the
        # scored span along that tail can stay close to the full route while far from the
        # scored one. Without a reference (sidecar only, GT does not cover the SD route) or with
        # a suspect one, use the full route: ending a run on a wrong crop is worse than ending
        # it late.
        manager = self.engine.managers.get('metric_manager')
        reference = getattr(getattr(manager, '_rc_metric', None), 'reference', None)
        route = self._sd_route_xy
        if (reference is not None and manager.current_scene is scene
                and not reference.quality['suspect']):
            if self._sd_route_crop is None:
                self._sd_route_crop = reference.crop_xy()
            route = self._sd_route_crop
        agent = self.engine.managers['agent_manager'].ego_agent
        here = np.asarray(agent.rear_vehicle.current_position, dtype=float)[:2]
        here = here - np.asarray(origin, dtype=float).reshape(2)
        seg = np.diff(route, axis=0)
        den = np.maximum(np.einsum('ij,ij->i', seg, seg), 1e-12)
        t = np.clip(np.einsum('ij,ij->i', here - route[:-1], seg) / den, 0., 1.)
        projected = route[:-1] + t[:, None] * seg
        return float(np.linalg.norm(projected - here, axis=1).min())

    def done_function(self) -> Tuple[bool, Dict]:
        """Departure first, then SD-span arrival (GT fallback); never read offline RC."""
        # Check route departure first: an ego this far off is no longer driving this scenario,
        # and letting it continue only accumulates distance route_completion reads as progress.
        # route_deviation zeroes the two route-related penalties, sending DS to 0 -- collision
        # terms keep the values measured until then.
        if bool(self.engine.global_config.get("sd_route_departure_guard_enabled", True)):
            _sd = self._sd_route_distance()
            if _sd is not None and _sd > self.SD_ROUTE_MAX_DIST_M:
                self.logger.info("Rollout ended at step %s: ego is %.1f m from the SD route (limit %.0f m)",
                            self.engine.episode_step, _sd, self.SD_ROUTE_MAX_DIST_M)
                return True, {"term_reason": "route_deviation",
                              "sd_route_distance_m": _sd}

        # Exhausting the baked route is the same kind of driving outcome -- from this step the
        # planner has no route to condition on and cannot answer (planner_client sets this flag
        # after seeing the sentinel and stopping the ego). The branches below assume an ego that
        # keeps receiving plans, so end here immediately. As with departure, this does not wait
        # for the scoring cadence: while waiting for the grid the ego would only stand still.
        if getattr(self, "_route_exhausted", False):
            self.logger.info("Rollout ended at step %s: the baked SD route is exhausted",
                             self.engine.episode_step)
            return True, {"term_reason": "route_exhausted"}

        state = self._sd_goal_state()
        if state is not None:
            progress_ratio, end_dist_m, route_m = state
            if (progress_ratio >= self.SD_GOAL_PROGRESS_RATIO
                    and end_dist_m <= self.SD_GOAL_END_DIST_M):
                return True, {"term_reason": "destination_arrival", "goal_source": "sd_route",
                              "sd_goal_progress_ratio": progress_ratio,
                              "sd_goal_remaining_m": (1.0 - progress_ratio) * route_m,
                              "sd_goal_end_dist_m": end_dist_m}
        else:
            state = self._gt_goal_state()
            if state is None:
                return False, {}
            progress_ratio, end_dist_m, _step = state
            if (progress_ratio >= self.GT_GOAL_PROGRESS_RATIO
                    and end_dist_m <= self.GT_GOAL_END_DIST_M):
                return True, {"term_reason": "destination_arrival", "goal_source": "gt_fallback",
                              "gt_progress_ratio": progress_ratio,
                              "gt_remaining_m": (1.0 - progress_ratio) * float(self._gt_goal_ref[1][-1]),
                              "gt_end_dist_m": end_dist_m}
        stall = self._hopeless_stall_state()
        if stall is not None:
            self.logger.info(
                "Rollout ended at step %s: ego travelled %.3f m over the last %.1f s",
                self.engine.episode_step, stall["stall_travel_m"], stall["stall_window_s"])
            return True, stall
        # Budget exhaustion is handled by truncateds in _get_step_return; repeating it here would
        # only add another place to keep in sync.
        return False, {}

    def reset(self, seed: Union[None, int] = None):
        """
        Reset the env, scene can be restored and replayed by giving episode_data
        Reset the environment or load an episode from episode data to recover is
        :param seed: The seed to set the env. It is actually the scene index you intend to choose
        :return: None
        """

        # Start simulation
        self._is_simulation_running = True
        self.lazy_init()

        self.seed(seed)

        if len(self.engine.managers['scenario_manager'].available_scenario_indices) == 0:
            self.logger.info("No scenarios remaining after filtering.")
            return None

        return self._reset_scenario(from_reset=True)

    def _reset_scenario(self, from_reset=False):
        """
        Reset the scenario to the initial state.
        """
        # Replaying a scene may reuse the same scene object. ScenarioManager hands back
        # `self.scenes[scenario_id]` -- the SAME object, not a copy -- so every one of these
        # caches is guarded by an identity check that a second episode of the same scene
        # passes. The SD goal cursor is the one that bites: `_sd_goal_last` is the forward-only
        # walking index into the reference, so an un-reset cursor still points at the end of
        # the previous rollout, the first post-handoff projection lands at arc[-1], and the
        # episode terminates at the handoff step with progress_ratio ~= 1.0. That is a
        # `destination_arrival` label on a 1.5 s rollout whose RC is 0 and whose DS is 0 -- a success
        # label on a failed run, invisible in the CSV. The GT twin below was always reset; the
        # SD four were missed when they were added.
        self._gt_goal_scene = None
        self._gt_goal_cache = None
        self._sd_goal_scene = None
        self._sd_goal_ref = None
        self._sd_goal_last = 0
        self._sd_goal_cache = None
        self._sd_route_scene = None
        self._sd_route_xy = None
        self._sd_route_crop = None
        # Route exhaustion does not carry over to the next scene -- left set, the next rollout
        # would end with route_exhausted on its first step.
        self._route_exhausted = False
        if not from_reset:
            # switch to the next scenario
            self.engine.managers['scenario_manager'].next_scene()

        reset_info = self.engine.reset()

        if self.config.use_planner_actions:
            ego_agent = self.engine.managers['agent_manager'].ego_agent
            if from_reset:
                self.outside_planner = OutsidePlanner(ego_agent)
            else:
                self.outside_planner.reset(ego_agent)

        return self._get_reset_return(reset_info)

    def _get_actions(self, step: int):
        if self.config.use_planner_actions:
            actions = self.outside_planner.get_trajectory(step)
        else:
            actions = None
        return actions

    def _get_reset_return(self, reset_info):
        # TODO: figure out how to get the information of the before step
        scene_manager_before_step_infos = reset_info
        # Do rendering at the first frame
        obses = self.engine.get_sensor()
        obses_infos = {'render': obses}
        scene_manager_after_step_infos = self.engine.after_step()

        done_infos = {}

        engine_info = merge_dicts(
            scene_manager_after_step_infos, scene_manager_before_step_infos, allow_new_keys=True, without_copy=True
        )

        step_infos = concat_step_infos([engine_info, done_infos, obses_infos])
        return step_infos

    def _get_step_return(self, actions, engine_info):
        # get observations
        obses = self.engine.get_sensor()

        # get done info
        done, done_info = self.done_function()
        done_infos = done_info

        # merge all info
        step_infos = concat_step_infos([engine_info, done_infos])

        # Truncation flag. This is what finishes a scene and moves on to the next, so even an
        # early finish must go through here; done alone is not enough.
        #
        # horizon = num_history + num_future - 1. num_future is not the log length but 2x the GT
        # drive duration (the old launcher's 2x GT budget switch) -- the closed-loop ego is slower
        # than the recording, so bounding by log length would cut runs that are still driving.
        # Hence gt_budget_multiplier is 1.0: otherwise the 2x would be applied twice.
        _horizon = (self.engine.global_config['num_history']
                    + self.engine.global_config['num_future'] - 1)
        _budget = int(_horizon * float(
            self.engine.global_config.get('gt_budget_multiplier', 1.0) or 1.0))
        # The final step must be scoreable. MetricManager scores only when
        # step % score_stride == 0, so cutting off the grid leaves the pdm csv unwritten and DS is
        # lost with it. Round the budget down to the grid. This is the same number as
        # MetricManager._step_budget(), read from the same source -- a comment saying
        # "same derivation" does not prevent the two from diverging.
        #
        # Do not cut at log_length. Running past it is fine downstream: DataManager supplies
        # frames up to the same budget, the 3DGS background ignores timestamps, and the GT
        # baseline forming RC's denominator is built once at handoff, so it does not shrink as
        # the run gets longer.
        _cad = (self.engine.managers['scenario_manager'].current_scene or {}).get("cadence")
        if _cad is None:
            raise RuntimeError("scene['cadence'] is missing; it is required to align the budget "
                               "to the scoring grid.")
        _stride = _cad.score_stride_steps
        _budget = (_budget // _stride) * _stride
        truncateds = self.engine.episode_step >= _budget
        if done:
            truncateds = True       # goal reached or budget spent: this scene is over

        # set termination flag
        terminates = done
        if self.engine.managers['scenario_manager'].all_scenes_completed and truncateds:
            terminates = True
            step_infos['all_scenes_completed'] = True

        # Score once more after termination is decided, because the manager already ran for this
        # step.
        #
        # engine.step() runs MetricManager.step() before this method, so on the step where the ego
        # crosses the goal, done_function() was still False when the scorer asked
        # _rollout_ending(). The loop then breaks on the done computed here -- no next step, no
        # scoring call, no csv row. This is a missed ordering, not a missed grid point, so
        # MetricManager's off-grid branch cannot catch it.
        #
        # Scoring uses already-cached frames, so this step's state is not needed, and _scored
        # makes it a no-op for runs already scored. MIN_SCORED_POSES still decides the length check.
        if truncateds or terminates:
            _mm = self.engine.managers.get('metric_manager')
            if _mm is not None and not getattr(_mm, "_scored", False):
                # Do not swallow scoring failures. Logging a single line would reproduce the very
                # symptom this exists to fix (no row written) while hiding the cause.
                _mm._score_and_save({"token": _mm.current_scene["id"],
                                     "step": _mm.current_step})

        return obses, step_infos, terminates, truncateds

    def close(self):
        if self.engine is not None:
            close_engine()

    def setup_engine(self):
        """
        Engine setting after launching
        """
        self.engine.register_manager("scenario_manager", ScenarioManager(self.info_dicts))
        if self.config.with_render_manager:
            self.engine.register_manager("render_manager", RenderManager())
        self.engine.register_manager('map_manager', ScenarioMapManager())
        self.engine.register_manager('agent_manager', BaseAgentManager())
        if self.config.with_data_manager:
            if not self.config.with_render_manager:
                raise ValueError("Data manager requires render manager to be enabled")
            self.engine.register_manager('data_manager', DataManager())
        if self.config.with_metric_manager:
            self.engine.register_manager('metric_manager', MetricManager())

    def seed(self, seed=None):
        if seed is not None:
            self.engine.seed(seed)

    @property
    def current_seed(self):
        return self.engine.global_random_seed
