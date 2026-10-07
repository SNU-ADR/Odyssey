# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
import importlib
import json
import logging
import os
import csv
import numpy as np
from typing import Any, Dict

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

from odyssey.manager.base_manager import BaseManager
from odyssey.utils import math_utils
from odyssey.utils.cadence import SCORE_DT
from odyssey.utils.nuplan_map_utils import (
    resolve_map_location,
    route_map_radius,
    route_roadblock_ids,
    MAP_RADIUS_MARGIN_M,
)
from odyssey.scenario.scenarios.scenario_description import ScenarioDescription as SD
from odyssey.components.agents.policy.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from odyssey.components.agents.policy.pdm_planner.reference_config import (
    LATERAL_OFFSETS,
    build_reference_idm_policy,
)
from odyssey.components.agents.policy.pdm_planner.utils.odyssey_to_pdm_utils import OdysseyToNuPlanConverter
from odyssey.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap

logger = logging.getLogger(__name__)

#: The config key that names the scorer: a dotted module path, imported with importlib. The
#: simulator records a run; scoring it is the benchmark's, and the benchmark launcher always passes
#: scorer=odyssey_benchmark.scorer. The module provides these names.
SCORER_KEY = 'scorer'
SCORER_API = ('rules_from_config', 'check_rules_drivable', 'pack_frame', 'capture_tlc_signal_snapshot',
              'pack_map', 'pack_tlc', 'replay', 'apply_termination', 'PinnedInputs', 'SDRouteMetric')


def load_scorer(global_config):
    """The configured scorer module. Loud when it is unset or incomplete: without it a run cannot be
    scored, and a run that is not scored must not look like one that was."""
    path = global_config.get(SCORER_KEY) if hasattr(global_config, 'get') else None
    if path in (None, ''):
        raise ValueError(f"with_metric_manager needs a scorer: set {SCORER_KEY}=<module> "
                         "(the benchmark launcher passes odyssey_benchmark.scorer)")
    module = importlib.import_module(str(path))
    missing = [name for name in SCORER_API if not hasattr(module, name)]
    if missing:
        raise ValueError(f"scorer {path} does not provide {missing}")
    return module


def _json_numpy(value):
    """Preserve NumPy scoring inputs in the JSON pinned for offline replay."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


# route_map_radius / _gt_ego_xy now live in odyssey.utils.nuplan_map_utils (imported
# above) so the PDM *policy* can size its planner identically without importing a manager.
# Re-exported here: this module was their original home and other code may import them.


#: The per-run result file (routeds_{NR|R}.csv): the benchmark's reported quantities and their
#: components (paper Appendix C), identification and scoring provenance. Everything else the
#: scorer computes stays in memory and in the pinned inputs (rollout_trajectory.npz).
RESULT_COLUMNS = ("scene", "react", "steps", "term_reason",
                  "RouteDS", "RC", "P_SD", "P_col", "P_off", "P_TL", "P_PLC", "PLCA", "PLCS", "Eff", "Comf",
                  "collision_count", "tl_violation_count",
                  "plc_stops", "plc_reached", "plc_pass", "plc_late", "plc_fail",
                  "tl_set", "plc_rule", "scoring_error")


def pre_lane_change_scores(lane):
    """(PLCA, PLCS) from a lane_follow result: credit 1 per clean pass, 0.5 per late pass, 0 per
    failure. PLCA averages over the stop lines reached; PLCS over every evaluation stop line on the
    route, so stop lines never reached count 0. None when the run has no evaluation stop lines."""
    if lane is None or not getattr(lane, 'n_stops', 0):
        return None, None
    credit = lane.n_pass - 0.5 * lane.n_late
    plca = credit / lane.n_reached if lane.n_reached else None
    return plca, credit / lane.n_stops


def result_row(row, scene, react, lane=None):
    """Project the scorer's row onto RESULT_COLUMNS."""
    plca, plcs = pre_lane_change_scores(lane)
    out = dict(row, scene=scene, react=react, steps=row.get('step'), PLCA=plca, PLCS=plcs,
               plc_stops=getattr(lane, 'n_stops', None), plc_pass=getattr(lane, 'n_pass', None),
               scoring_error=row.get('scoring_error') or '')
    return {k: out.get(k) for k in RESULT_COLUMNS}


def _csv_value(value):
    if value is None:
        return ''
    if isinstance(value, float):
        return '' if value != value else f'{value:.5f}'
    if isinstance(value, (np.floating,)):
        return f'{float(value):.5f}'
    if isinstance(value, (np.integer,)):
        return int(value)
    return value

class MetricManager(BaseManager):
    """Manager for evaluating autonomous driving system performance metrics"""
    
    PRIORITY = 20

    def __init__(self):
        """Derive the scoring window from num_future and convert it to OUTER (0.5 s) frames.

        Why derive the window from num_future: a static buffer_size/sampling_poses would score
        long rollouts on their first few steps only, so a run could report a perfect score while
        the ego froze and was hit by cars.

        The stride is read from scene["cadence"], where ScenarioManager put it. Recomputing it here
        from rollout_dt would be a second source for the same number -- if the scoring grid is
        wrong, the whole score is wrong.

        Config values are in SIM (0.1 s) steps and are converted to OUTER below. With S=1 everything
        behaves exactly like the original 0.5 s setup.
        """
        super(MetricManager, self).__init__()
        self._scorer = load_scorer(self.engine.global_config)
        self._score_stride = self._cadence().score_stride_steps
        _S = self._score_stride
        _nh_sim = int(self.engine.global_config['num_history'])
        _nf_sim = int(self.engine.global_config['num_future'])

        # Scoring starts here, so it must land on an outer boundary. Flooring would put the handoff
        # at sim step 0, which is never scored, so initialization would be skipped entirely.
        if (_nh_sim - 1) % _S:
            raise ValueError(
                f"num_history={_nh_sim} sim steps does not divide into {SCORE_DT}s scoring "
                f"frames at stride {_S} (from scene['cadence']). It must be "
                f"(outer_frames - 1) * {_S} + 1; the nearest valid values are "
                f"{(_nh_sim - 1) // _S * _S + 1} and {((_nh_sim - 1) // _S + 1) * _S + 1}.")
        _nh = (_nh_sim - 1) // _S + 1

        # num_future comes from the GT driving-time budget, so it is an arbitrary number of sim
        # steps. Round up: the window must cover the rollout, and rounding down would drop up to
        # S-1 steps at the end from scoring.
        _nf = -(-_nf_sim // _S)                  # ceil
        if _nf_sim % _S:
            logger.info(
                "[SCORE] num_future=%d sim steps is not a whole number of %.1fs frames at "
                "stride %d; grading %d frames (%d sim steps) so the window covers the rollout",
                _nf_sim, SCORE_DT, _S, _nf, _nf * _S)

        _sampling_poses = max(int(_nf) - 1, 1)
        self._future_sampling = TrajectorySampling(num_poses=_sampling_poses + 1, interval_length=SCORE_DT)
        self._proposal_sampling = TrajectorySampling(num_poses=_sampling_poses, interval_length=SCORE_DT)
        self._map_radius = 100
        self._pdm_closed = PDMClosedPlanner(
            trajectory_sampling=self._future_sampling,
            proposal_sampling=self._proposal_sampling,
            # Keep the scoring helper's proposal family identical to reference PDM-Closed.
            # Its trajectory sampling and wider map remain evaluator adapters: unlike the
            # driving policy, this instance scores one complete saved rollout at 0.5 s.
            idm_policies=build_reference_idm_policy(),
                lateral_offsets=list(LATERAL_OFFSETS),
                map_radius=self._map_radius,
        ) # register planner

        # Scoring unit in OUTER (0.5 s) frames. At native steps S=1, so these equal the config values.
        self.num_history = _nh                      # history frames (0.5 s)
        self.num_future = _nf                       # future frames (0.5 s)
        
        self.current_scene = None
        self.converter = None
        # The route being scored against. Resolved once per scene (get_route_roadblocks_ids).
        self._route_roadblocks_ids = None
        self.ego_states_list = []
        self.detection_tracks_list = []
        self.agent_source_mode_list = []
        self.agent_source_row_list = []
        self._idm_lifecycle_sim_steps = []
        self._idm_lifecycle_source_steps = []
        self.score_rows = []

        self._cached_roadblock_ids = []
        # One-shot latch for the handoff-frame init below. See the note at its `if`.
        self._graded_init_done = False
        self.buffer_size = self.num_future  # OUTER (0.5 s) future frames -- see __init__ note

    def before_reset(self):
        """Reset all metrics"""
        reset_info = super().before_reset()
        self.agent = None
        return reset_info

    def reset(self):
        """Reset the planner"""
        self.current_step = self.engine.episode_step
        self.current_scene = self.engine.managers['scenario_manager'].current_scene
        self.agent = self.engine.managers['agent_manager'].ego_agent
        self.converter = OdysseyToNuPlanConverter(self.current_scene, self.agent, self.engine)

        # The scoring stride was derived in __init__ from rollout_dt alone, because
        # current_scene is not available there. That is right whenever a caller sets
        # rollout_dt, and wrong when it does not: the fallback assumed one outer step per
        # SCORE_DT frame, which holds only for scenes whose frames are already SCORE_DT
        # apart. Now that the scene is here, re-derive and refuse to score on a stride that
        # does not match the step the world is actually taking -- grading every 0.1 s step
        # as if it were a 0.5 s frame reports a fifth of the horizon it claims.
        # A cross-check, not a derivation: _score_stride is read from scene["cadence"], and here we
        # only check that it matches the step the engine is actually taking. If they disagree, the
        # scoring grid would be silently wrong, so fail.
        _scene_stride = int(round(SCORE_DT / float(self.engine.sim_dt)))
        if _scene_stride != self._score_stride:
            raise ValueError(
                f"scoring stride {self._score_stride} (from scene['cadence']) does not match "
                f"this engine's step: sim_dt={self.engine.sim_dt}s gives {_scene_stride} steps "
                f"per {SCORE_DT}s scoring frame. The cadence and the engine step disagree.")

        nuplan_map_root = self.engine.global_config['nuplan_map_root']
        map_location = resolve_map_location(self.current_scene, nuplan_map_root)
        self._map_location = map_location
        self.map_api = get_maps_api(nuplan_map_root, "nuplan-maps-v1.0", map_location)
        # Size the PDM map queries to THIS scene's drive. __init__ cannot do it -- current_scene
        # is only set here -- so the planner's own copy is updated alongside ours.
        self._map_radius = route_map_radius(self.current_scene)
        self._pdm_closed._map_radius = self._map_radius
        # NB: module logger, not self._logger -- MetricManager has no such attribute (the two
        # existing self._logger.error() calls sit on never-taken branches and would AttributeError).
        logger.info("[pdm] map_radius %.0f m (GT drive extent + %.0f m margin)",
                    self._map_radius, MAP_RADIUS_MARGIN_M)
        self.ego_states_list = []
        self.detection_tracks_list = []
        self.agent_source_mode_list = []
        self.agent_source_row_list = []
        self._idm_lifecycle_sim_steps = []
        self._idm_lifecycle_source_steps = []
        self._cached_roadblock_ids = []
        self._graded_init_done = False
        self._scored = False        # this scene has not been written yet
        self._rc_metric = self._scorer.SDRouteMetric(
            self.current_scene, self.agent.object_track,
            int(self.engine.global_config['num_history']) - 1, self._score_stride,
            float(self.agent.rear_vehicle.rear_axle_to_center_dist), self.map_api)
        self._rc_fields = {}
        self._route_roadblocks_ids = None
        # Pinned scoring inputs. Must be cleared on scene change: the _graded_init_done latch refills
        # them, but if a run ends before scoring, the previous scene's curve would remain and be
        # dumped. That is worse than no pin (it looks as if measured with another scene's ruler).
        self._cached_expert_traj = None
        self._expert_reference_pos = None
        self._ds_states, self._ds_actors, self._ds_steps = [], [], []
        self._tlc_signal_snapshots = []
        self._driving_inputs = None
        self._pinned = None
        self._dense_series = None
        self._tlc_series = None
        # Record-level scoring rules (tl_set, plc_rule). Pinned into driving_inputs at scene end and
        # applied by replay (driving_metrics.apply_rules). Scene-end scoring would reject a
        # designated-signal scene with a timetable rule driven without that timetable anyway, so
        # stop here before spending the rollout.
        self._scoring_rules = self._scorer.rules_from_config(self.engine.global_config)
        self._scorer.check_rules_drivable(self._scoring_rules, self.engine.global_config,
                                          self.current_scene['id'])


    def after_reset(self):
        """Get ego vehicle reference after reset"""
        reset_info = super().after_reset()
        return reset_info
    
    def before_step(self):
        self.current_step = self.engine.episode_step
        if self.current_scene is None:
            logger.error("No scene available for scoring.")
            return None
        elif self.current_step == 0:
            self._record_rc_pose()
            self.ego_states_list.append(self.converter.convert_to_current_ego_state(self.current_step))
    
    def step(self):
        """Accumulate and score only on OUTER (0.5 s) frames.

        All accumulation below is indexed by OUTER frame o, so inserting off-grid sub-frame state
        would break the 0.5 s sampling the scorer assumes. Sub-frames therefore take only the
        scoring branch and leave accumulation untouched. That branch is needed because the step on
        which done_function() fires is not guaranteed to be a stride multiple, and it is safe to
        reach because it only reads what is already cached.
        """
        self.current_step = self.engine.episode_step
        self._record_rc_pose()
        if self.current_step % self._score_stride != 0:
            if self._rollout_ending():
                self._score_and_save({"token": self.current_scene["id"],
                                      "step": self.current_step})
            return
        o = self.current_step // self._score_stride     # OUTER (0.5 s) frame index
        score_row: Dict[str, Any] = {
            "token": self.current_scene["id"],
            "step": self.current_step,
        }

        if self.current_scene is None:
            logger.error("No scene available for scoring.")
            return None
        self.ego_states_list.append(self.converter.convert_to_current_ego_state(self.current_step))

        # Initialize once, on the first scored frame after handoff. A latch rather than equality
        # (`o ==`) because if num_history collapses to 1, equality hits o == 0, but base_engine.step()
        # increments episode_step before the managers run, so step 0 is never scored -- init would
        # be skipped entirely, leaving _map_api / _pdm_closed / _cached_expert_traj unset.
        if not self._graded_init_done and o >= self.num_history - 1:
            # The route the run is scored against: the scorer's route dicts (lane graph, on-route
            # polygons) come from these roadblocks, in route order and without repeats.
            self._cached_roadblock_ids = list(dict.fromkeys(self.get_route_roadblocks_ids()))
            self._pdm_closed.initialize({"route_roadblock_dict_ids": self._cached_roadblock_ids,
                                         "map_api": self.map_api})
            self._map_api = self.map_api

        if o >= self.num_history - 1:
            # update detection tracks and traffic light data
            detection_tracks = self.converter.convert_to_detections_tracks_from_agent_input(self.current_step)
            self.detection_tracks_list.append(detection_tracks)
            idm_batch = getattr(self.engine, "_nuplan_idm_batch", None)
            source_mode_for = getattr(idm_batch, "source_mode_for", None)
            agent_manager = self.engine.managers.get('agent_manager')
            source_row_for = getattr(agent_manager, 'object_source_row', None)
            frame_rows = {}
            frame_modes = {}
            tracked_container = getattr(detection_tracks, 'tracked_objects', None)
            for obj in getattr(tracked_container, "tracked_objects", []) or []:
                token = str(getattr(obj, "track_token", "") or "")
                if not token:
                    continue
                sector_replay = getattr(
                    agent_manager, '_uses_reactive_sector_replay', lambda _t: False
                )(token)
                mode = ('sector_replay' if sector_replay else
                        source_mode_for(token) if source_mode_for is not None else 'replay')
                frame_modes[token] = mode
                # An IDM pose is integrated state, not a sampled source row. -1 says N/A;
                # replay modes name the exact PKL row that produced the archived pose.
                frame_rows[token] = (
                    -1 if mode == 'idm' else
                    int(source_row_for(token, self.current_step))
                    if source_row_for is not None else int(self.current_step)
                )
            self.agent_source_mode_list.append(frame_modes)
            if not hasattr(self, 'agent_source_row_list'):     # partial managers in tests
                self.agent_source_row_list = []
            self.agent_source_row_list.append(frame_rows)
        
        # Same latch as the PDM init above -- both must initialize on the same scored frame, so the
        # flag is set only after both have run (at the end of this block).
        if not self._graded_init_done and o >= self.num_history - 1:
            # Sample the expert (GT) at the 0.5 s cadence: every S steps from the current step.
            _S = self._score_stride
            expert_pos = self.agent.object_track['position'][self.current_step::_S][:self.buffer_size, :2]
            expert_headings = self.agent.object_track['heading'][self.current_step::_S][:self.buffer_size]
            # center to real-axle
            expert_pos_rear = expert_pos.copy()
            for i in range(len(expert_pos)):
                expert_pos_rear[i] = math_utils.translate_longitudinally(
                    expert_pos_rear[i], 
                    expert_headings[i], 
                    -self.agent.rear_vehicle.rear_axle_to_center_dist
                ).reshape(2)
            reference_pos = expert_pos_rear[0].copy()
            for i in range(len(expert_pos)):
                expert_pos_rear[i] = expert_pos_rear[i] - reference_pos
            self._cached_expert_traj = np.concatenate([expert_pos_rear, expert_headings[:, None]], axis=1)
            # The origin subtracted above. The dump must store it next to the relative coordinates
            # so readers can map back to global (expert_origin in _dump_rollout_trajectory).
            self._expert_reference_pos = np.asarray(reference_pos, dtype=np.float64)
            # Number of real GT poses. Long rollouts pad the expert (_score_and_save); the centerline
            # is cut here so repeated end points do not create zero-length interpolation knots.
            self._gt_source_pose_count = len(self._cached_expert_traj)
            self._graded_init_done = True
        
        # Finalize only when the rollout ends, not at the unmultiplied log horizon.
        if self._rollout_ending():
            self._score_and_save(score_row)

    def _record_rc_pose(self):
        agent_manager = getattr(self.engine, "managers", {}).get("agent_manager")
        source_step = getattr(agent_manager, "idm_source_step", None)
        if source_step is not None and (
            not getattr(self, "_idm_lifecycle_sim_steps", [])
            or self._idm_lifecycle_sim_steps[-1] != self.current_step
        ):
            if not hasattr(self, "_idm_lifecycle_sim_steps"):
                self._idm_lifecycle_sim_steps = []
                self._idm_lifecycle_source_steps = []
            self._idm_lifecycle_sim_steps.append(int(self.current_step))
            self._idm_lifecycle_source_steps.append(int(source_step))
        rear = np.asarray(self.agent.rear_vehicle.current_position, dtype=float)[:2]
        rear = rear + np.asarray(self.converter.initial_ego_center, dtype=float)[:2]
        self._rc_metric.observe(self.current_step, rear, self.agent.current_heading)
        if self.current_step >= self._rc_metric.handoff and hasattr(self, '_ds_steps'):
            if not self._ds_steps or self._ds_steps[-1] != self.current_step:
                state, actors = self._scorer.pack_frame(
                    self.converter.convert_to_current_ego_state(self.current_step),
                    self.converter.convert_to_detections_tracks_from_agent_input(self.current_step))
                self._ds_steps.append(self.current_step)
                self._ds_states.append(state)
                self._ds_actors.append(actors)
                # Capture the signal at the same live tick as the ego/actor snapshot. TLC
                # finalization is forbidden from asking the scenario timetable or replay clock
                # what would have happened at this old step.
                self._tlc_signal_snapshots.append(self._scorer.capture_tlc_signal_snapshot(
                    self.current_scene, self.converter, self.current_step))

    def _score_and_save(self, score_row: Dict[str, Any]) -> None:
        """Score the accumulated OUTER frames and write the csv row.

        Split out of step() so that all three ways a rollout ends (full window reached, early
        off-grid termination, after base_env confirms termination) are scored by the same code. It
        only reads state already accumulated by completed OUTER frames, so it is safe to call on a
        step where accumulation was deliberately skipped.
        """
        if getattr(self, '_scored', False):
            return
        # Independent of PDM's minimum length, measure the actual travelled RC
        # from observed poses, including the last non-grid pose. The common
        # finalizer may award the 99% SD-goal completion tolerance afterwards.
        term_reason = self._term_reason()
        departure_m = float(getattr(self.engine.env, 'SD_ROUTE_MAX_DIST_M', 30.))
        # Scoring happens once, in replay() over the pinned inputs below. The live RC matcher runs
        # only where replay cannot: a run that ended before any dense frame was pinned, or a
        # scoring failure (the row then still carries RC and the termination).
        self._rc_fields = {}
        score_row["term_reason"] = term_reason
        # The legacy name destination_arrival is kept, so also record what the arrival decision
        # was actually based on.
        _done, goal_info = self.engine.env.done_function()
        if _done:
            for key in ("goal_source", "sd_goal_progress_ratio", "sd_goal_remaining_m",
                        "sd_goal_end_dist_m", "gt_progress_ratio", "gt_remaining_m",
                        "gt_end_dist_m"):
                if key in goal_info:
                    score_row[key] = goal_info[key]
        scorer = self._scorer
        pack_map, apply_termination, replay, pack_tlc = (
            scorer.pack_map, scorer.apply_termination, scorer.replay, scorer.pack_tlc)
        if getattr(self, '_ds_steps', None):
            self._driving_inputs = None
            try:
                initial = self.ego_states_list[self.num_history - 1]
                drivable = PDMDrivableMap.from_simulation(self.map_api, initial, self._map_radius)
                self._pdm_closed._load_route_dicts(self._cached_roadblock_ids)
                map_data = pack_map(drivable, self._pdm_closed._route_lane_dict,
                                    initial.car_footprint.vehicle_parameters,
                                    route_roadblock_dict=self._pdm_closed._route_roadblock_dict,
                                    map_api=self.map_api)
                # TLC checks contact at every step, so GT is loaded at those same steps rather than at
                # 0.5 s (where several steps would share one GT pose).
                dense_gt = getattr(self._rc_metric, 'reference_gt_dense', None)
                gt_rear = gt_heading = None
                if self._rc_metric.reference_gt is not None and dense_gt is not None:
                    idx = np.clip(np.asarray(self._ds_steps, int), 0, len(dense_gt) - 1)
                    gt_rear = dense_gt[idx]
                    gt_heading = self._rc_metric.gt_heading_dense[idx]
                cadence = self._cadence()
                render_manager = self.engine.managers.get('render_manager')
                signal_source_overrides = getattr(
                    render_manager, 'traffic_light_signal_source_overrides', None)
                tlc_data = pack_tlc(
                    self.current_scene, self._pdm_closed._route_lane_dict,
                    self._ds_steps, float(self.engine.sim_dt), int(cadence.upsample_n),
                    gt_rear, gt_heading, map_data['vehicle'],
                    render_override=self.engine.global_config.get('tlc_render_sanity_override_path'),
                    apply_gt_filter=self.engine.global_config.get('tlc_apply_gt_filter', False),
                    route_roadblock_dict=self._pdm_closed._route_roadblock_dict,
                    map_location=self._map_location,
                    apply_nearside_turn_filter=self.engine.global_config.get(
                        'tlc_apply_nearside_turn_filter', True),
                    signal_source_overrides=signal_source_overrides,
                    signal_snapshots=self._tlc_signal_snapshots)
                goal = {key: score_row[key] for key in (
                    'goal_source', 'sd_goal_progress_ratio', 'sd_goal_remaining_m',
                    'sd_goal_end_dist_m', 'gt_progress_ratio', 'gt_remaining_m',
                    'gt_end_dist_m') if key in score_row}
                self._driving_inputs = dict(map=map_data, actors=self._ds_actors,
                    sim_dt=float(self.engine.sim_dt), rc=self._rc_metric.snapshot(),
                    tlc=tlc_data,
                    goal=goal,
                    term_reason=term_reason, departure_distance_m=float(getattr(self.engine.env, 'SD_ROUTE_MAX_DIST_M', 30.)),
                    rules=self._scoring_rules)
            except Exception as exc:                 # noqa: BLE001 - record the reason, keep saving
                # Packing the scoring inputs failed, so there is nothing to replay. The row carries RC
                # and the termination from the live matcher plus scoring_error, and the npz is still
                # saved (without driving_inputs_json), so the run is reported instead of lost.
                logger.exception('[score] packing the scoring inputs failed')
                self._driving_inputs = None
                self._rc_fields = self._rc_metric.score(term_reason=term_reason, departure_distance_m=departure_m)
                score_row.update(self._rc_fields)
                score_row.update(scoring_error=f'{type(exc).__name__}: {exc}', RouteDS=None)
                self._pinned = self._scoring_pins()
            else:
                # Scoring. The just-pinned inputs are passed to replay in the exact shape they are saved
                # in (PinnedInputs) -- the same function a rescore calls on the saved npz. The scene-end
                # score and a rescore are therefore equal by definition, and unpinned simulator state
                # cannot enter either. The static-obstacle criterion (driving_metrics.STATIC_MODE:
                # static classes the reconstruction does not bake, so the model cannot see them, are not
                # scored) and the record-level rules (apply_rules) are also decided once in there.
                #
                # On failure, record scoring_error in the row and clear DS (the record gets
                # scoring_status=incomplete). The inputs are still saved -- once the cause is fixed, the
                # same npz must be rescorable.
                self._pinned = self._scoring_pins()
                capture = {}
                try:
                    score_row.update(replay(self._pinned, capture))
                except Exception as exc:                 # noqa: BLE001 - record the reason, keep saving
                    logger.exception('[score] scene-end scoring failed')
                    self._rc_fields = self._rc_metric.score(term_reason=term_reason, departure_distance_m=departure_m)
                    score_row.update(self._rc_fields)
                    score_row.update(scoring_error=f'{type(exc).__name__}: {exc}', RouteDS=None)
                self._lane_result = capture.get('lane_result')      # pre-lane-change counts for the result row
                # The per-tick series behind the scalars. Saved rather than left to be recovered:
                # see driving_metrics.dense_series.
                self._tlc_series = capture.get('tlc_series') or {}
                self._dense_series = {key: capture[key] for key in (
                    'contact_step', 'contact_token', 'contact_at_fault', 'ego_area_flags')
                    if key in capture}
        else:
            self._rc_fields = self._rc_metric.score(term_reason=term_reason, departure_distance_m=departure_m)
            score_row.update(self._rc_fields)
            apply_termination(score_row, term_reason)
        self.score_rows.append(score_row)
        self.save_scores()
        self._scored = True

    def _step_budget(self) -> int:
        """Grid-aligned step budget. Must equal the value at which base_env sets truncateds.

        base_env, _rollout_ending and _term_reason must all derive it identically. If they diverge,
        a run ended by one is recorded under a different name by another, and term_reason no longer
        matches the actual reason for termination.

        Fails if missing rather than returning 0: 0 means "no budget", so the truncation branch
        would never fire and runs that never reached the goal would silently run to the end.
        """
        cfg = self.engine.global_config
        horizon = int(cfg['num_history']) + int(cfg['num_future']) - 1
        mult = float(cfg['gt_budget_multiplier'])
        budget = int(horizon * mult)
        return (budget // self._score_stride) * self._score_stride

    def _rollout_ending(self) -> bool:
        """True on the step where env has decided to end this rollout early.

        Read from env rather than recomputed here -- done_function owns the goal decision, and a copy
        would be one more thing to keep in sync. engine.env is attached by lazy_init() before
        setup_engine(), so it always exists when the managers run.

        Finalizes termination regardless of PDM's minimum length. Short drives also save RC and
        termination metadata.
        """
        if getattr(self, "_scored", False):
            return False
        done, _info = self.engine.env.done_function()
        # Exhausting the budget ends the rollout just like done_function. It is the only way a run
        # that never reaches the goal ends, so derive the same cap base_env truncates at.
        if not done:
            _cap = self._step_budget()
            if _cap and self.current_step >= _cap:
                done = True
        return bool(done)

    def _term_reason(self) -> str:
        """Why the run stopped: destination_arrival / time_limit / log_end / stopped.

        This column exists to tell whether a low route completion means "finished the route" or
        "ran out of steps", so it must mean exactly what its name says.

        The budget is horizon x gt_budget_multiplier. Comparing with the unmultiplied horizon would,
        with a multiplier above 1, label both runs still driving and runs force-stopped by the
        budget as log_end -- exactly what this column is meant to distinguish.
        """
        # If done_function fired, that is the real reason. Falling through to the budget estimate
        # below would record a run that ended in destination_arrival as time_limit.
        done, info = self.engine.env.done_function()
        if done:
            return str(info.get("term_reason") or "done")
        o = self.current_step // self._score_stride
        horizon = self.num_history + self.num_future - 1        # the log's own length
        mult = float(self.engine.global_config['gt_budget_multiplier'])
        budget = int(horizon * mult)
        # The same cap base_env truncates at, in OUTER frames. Without it, runs stopped by the budget
        # are reported as log_end -- both sides must derive it identically for this column to hold.
        _cap = self._step_budget()
        if _cap:
            budget = min(budget, _cap // self._score_stride)
        if o >= budget:
            return "time_limit"      # hard stop; never reached the goal
        if o >= horizon:
            return "log_end"         # drove past the log length but still within budget
        # Neither: it stopped for another reason (dead planner, truncated scene). "budget" would be
        # false.
        return "stopped"

    def _cadence(self):
        """This scene's cadence, resolved once by ScenarioManager via resolve_cadence."""
        # Also called from __init__, when self.current_scene does not exist yet, so ask
        # ScenarioManager directly (PRIORITY -10, so it is created first).
        scene = getattr(self, "current_scene", None)
        if scene is None:
            scene = self.engine.managers['scenario_manager'].current_scene
        c = (scene or {}).get("cadence")
        if c is None:
            raise RuntimeError(
                "scene['cadence'] is missing. ScenarioManager is expected to set it from "
                "resolve_cadence; the scoring grid must not be re-derived here.")
        return c

    def _log_visibility_summary(self):
        """Log one line on how actor visibility was applied in scoring.

        A score alone cannot tell whether it was lowered by a car that was not visible. A setup
        that ran for weeks without traffic lights, and one where 3.0% of actor frames were not on
        screen, were both found late because nothing reported them. Logged on every run.
        """
        st = getattr(self.converter, "visibility_stats", None)
        if not st or not st["frames"]:
            return
        # Actors that come out at the origin without a pose are excluded from scoring (see the note
        # in odyssey_to_pdm_utils). Log how many -- a nonzero count that keeps growing means more
        # actors are failing to get a pose.
        if st.get("skipped_unset_pose"):
            logger.info("[actors] excluded %d actor-frames that came out at the origin without a pose "
                        "(the last frame of an actor whose source ended, just before it disappears)",
                        st["skipped_unset_pose"])
        if not st["frames_filtered"]:
            logger.info("[actors] visibility filter not applied (no renderer) -- scoring all %d actor frames",
                        st["scored"])
            return
        logger.info("[actors] scored %d frame-actors (everything the simulator passed). %d of them were "
                    "not in the previous frame's render list -- a one-tick lag on appearance is normal; "
                    "a larger build-up means scored cars are not on screen.",
                    st["scored"], st["skipped_invisible"])

    def get_route_roadblocks_ids(self):
        """The route this run is SCORED against, in the global frame.

        It does not read ``self.agent.navigation.checkpoint_lanes``, which is picked out on
        the simulator's SCENE-LOCAL map: the route roadblocks become the scorer's on-route polygon
        set, so a route picked in the wrong frame charges a perfectly legal drive as off-route
        (see route_roadblock_ids). pdm_policy asks the same question of the same map.

        Cached: the logged ego track, the converter's initial_ego_center and the map api are
        all constant for a scene, and this is called once per graded frame.
        """
        if self._route_roadblocks_ids is None:
            positions = np.asarray(self.agent.object_track[SD.POSITION], dtype=np.float64)
            self._route_roadblocks_ids = route_roadblock_ids(
                positions, self.converter.initial_ego_center, self.map_api, logger)
            logger.info("[SCORE] route resolved from %d logged ego poses -> %d roadblocks",
                        len(positions), len(self._route_roadblocks_ids))
        return self._route_roadblocks_ids

    def save_scores(self):
        """Write the run's result row (routeds_{NR|R}.csv) and the pinned scoring inputs."""
        if not self.score_rows:
            logger.warning("No scores to save.")
            return
        react = 'R' if self.engine.global_config.get('agent_policy') == 'nuplan_idm_policy' else 'NR'
        out_dir = self.engine.global_config['data_output_dir']
        # The scores are this manager's output, so it creates the directory itself. Relying on
        # DataManager would lose the result to an OSError on the last line in runs that disable the
        # renderer/data manager.
        os.makedirs(out_dir, exist_ok=True)
        new_score_row = self.score_rows[-1]
        scene = os.environ.get('ODYSSEY_SCENE_NAME') or str(new_score_row.get('token'))
        row = result_row(new_score_row, scene, react.lower(), getattr(self, '_lane_result', None))
        path = os.path.join(out_dir, f'routeds_{react}.csv')
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            writer.writerow({k: _csv_value(v) for k, v in row.items()})
        logger.info(f"Saved result to {path} successfully.")
        self._log_visibility_summary()

        self._dump_rollout_trajectory(out_dir)

    def _scoring_pins(self):
        """Every input scoring reads, shaped like the npz (PinnedInputs); the dump stores this object.

        Scene-end scoring (_score_and_save) passes it to replay -- the same function and values a
        rescore uses with the saved npz. Anything not in here therefore enters neither scoring.

        Pins the "rulers" used for scoring (route roadblocks and the expert curve). Rebuilding them
        from the scenario pkl + map at every scoring would let any change to those inputs silently
        diverge from past results. Storing the values makes rescoring deterministic.

        When absent, nothing is written (not an empty array) -- readers must be able to tell a
        "run without pins" from a "run with an empty route".
        """
        pins = self._scorer.PinnedInputs()
        if getattr(self, '_driving_inputs', None) is not None:
            pins.update(driving_inputs_json=json.dumps(self._driving_inputs, allow_nan=False,
                                                       default=_json_numpy),
                        ds_states=np.asarray(self._ds_states, dtype=float),
                        ds_sim_steps=np.asarray(self._ds_steps, dtype=int))
        # Pin **_cached_roadblock_ids**, not _route_roadblocks_ids -- the two differ:
        #
        #   route_roadblock_ids(...)     original, built by walking the map (+ trailing successor margin)
        #     -> _route_roadblocks_ids   cached once per scene
        #       -> step()'s accumulation loop   collected inside the scoring window, without repeats
        #         -> _cached_roadblock_ids  <- exactly what _load_route_dicts() receives
        #
        # Pinning the original would make rescoring re-apply the window filter, i.e. reinterpret a
        # value it was handed. Store the list the scorer actually saw.
        rb_ids = getattr(self, "_cached_roadblock_ids", None)
        if rb_ids:
            pins["route_roadblock_ids"] = np.array([str(i) for i in rb_ids], dtype=object)
        # The curve used as the normalizing denominator of ego_progress: the GT (expert) rear-axle
        # trajectory.
        #
        # Note: _cached_expert_traj is **relative** to its first point, and that origin is itself
        # scene-local (reference_pos above is in the object_track frame). Leaving readers to guess
        # the frame invites mistaking it for global, so the origin is stored too. Mapping back to
        # global (UTM) takes **two steps** -- relation verified by measurement:
        #
        #     global_xy = expert_xy + expert_origin + initial_ego_center[:2]
        #
        # (ego_xy is already global, so the two must not simply be subtracted.)
        gt = getattr(self, "_cached_expert_traj", None)
        origin = getattr(self, "_expert_reference_pos", None)
        if gt is not None and len(gt) and origin is not None:
            gt = np.asarray(gt, dtype=np.float64)
            pins["expert_xy"] = gt[:, :2]
            pins["expert_heading"] = gt[:, 2]
            pins["expert_origin"] = np.asarray(origin, dtype=np.float64)[:2]
            pins["expert_source_pose_count"] = int(
                getattr(self, "_gt_source_pose_count", len(gt)))
        rc_steps, rc_xy, rc_heading, rc_time = self._rc_metric.arrays()
        pins.update(rc_ego_xy=rc_xy, rc_ego_heading=rc_heading, rc_time_s=rc_time, rc_sim_steps=rc_steps,
                    map_location=str(getattr(self, "_map_location", "")),
                    scene=str(self.current_scene.get("id", "")),
                    initial_ego_center=np.asarray(self.converter.initial_ego_center,
                                                  dtype=np.float64)[:2])
        return pins

    def _dump_rollout_trajectory(self, out_dir):
        """Save the actual drive so trajectories can be plotted without a renderer.

        Privileged / vector-map runs disable the render and data managers, so neither vis_data nor
        frames are left -- only scores. This manager runs independently of both and already holds
        ego and actor states at the scoring cadence, so the dump costs just one file.

        Global UTM, one row per scoring frame (0.5 s). The GT (object_track) it is compared against
        is not stored here.

        The scores are already on disk via to_csv before this call, so a failure here does not lose
        the scoring result -- the ordering is the guarantee, not a try.
        """
        ego = self.ego_states_list
        if not ego:
            return
        ego_xy = np.array([[e.rear_axle.x, e.rear_axle.y] for e in ego], dtype=np.float64)
        ego_h = np.array([e.rear_axle.heading for e in ego], dtype=np.float64)
        ego_v = np.array([[e.dynamic_car_state.rear_axle_velocity_2d.x,
                           e.dynamic_car_state.rear_axle_velocity_2d.y] for e in ego],
                         dtype=np.float64)
        # Store acceleration too. Without it, rescoring fills it by differentiating velocity at the
        # scoring cadence, whereas the simulator uses the actual dynamic_car_state value. The
        # slightly different state perturbed polygon-boundary tests and shifted the off-road
        # column by 1-2%.
        ego_a = np.array([[e.dynamic_car_state.rear_axle_acceleration_2d.x,
                           e.dynamic_car_state.rear_axle_acceleration_2d.y] for e in ego],
                         dtype=np.float64)

        # Background actors, keyed by track token so a viewer can follow them across frames.
        frames = self.detection_tracks_list
        order, rows, types = {}, [], {}
        for fi, dt in enumerate(frames):
            for obj in getattr(dt, "tracked_objects", []) or []:
                tok = str(getattr(obj, "track_token", "") or getattr(obj, "metadata", None)
                          and getattr(obj.metadata, "track_token", "") or "")
                if not tok:
                    continue
                order.setdefault(tok, len(order))
                types.setdefault(tok, str(getattr(obj, "tracked_object_type", "UNKNOWN"))
                                 .rsplit(".", 1)[-1])
                c = obj.center
                box = obj.box if hasattr(obj, "box") else None
                velocity = getattr(obj, "velocity", None)
                rows.append((order[tok], fi, c.x, c.y, c.heading,
                             getattr(box, "length", 0.0) if box else 0.0,
                             getattr(box, "width", 0.0) if box else 0.0,
                             getattr(velocity, "x", 0.0) if velocity else 0.0,
                             getattr(velocity, "y", 0.0) if velocity else 0.0))
        n_a, n_f = len(order), len(frames)
        # Ego history starts at scene frame 0, detections at the closed-loop handoff. Preserve that
        # offset instead of pretending the two arrays share a time origin.
        agent_frame_offset = max(len(ego_xy) - n_f, 0)
        a_xy = np.full((n_a, n_f, 2), np.nan)
        a_h = np.full((n_a, n_f), np.nan)
        a_v = np.full((n_a, n_f, 2), np.nan)
        a_size = np.zeros((n_a, 2))
        a_mode = np.full((n_a, n_f), "unknown", dtype="<U16")
        a_source_row = np.full((n_a, n_f), -1, dtype=np.int64)
        for ai, fi, x, y, h, ln, wd, vx, vy in rows:
            a_xy[ai, fi] = (x, y)
            a_h[ai, fi] = h
            a_v[ai, fi] = (vx, vy)
            a_size[ai] = (ln, wd)
        for fi, modes in enumerate(self.agent_source_mode_list[:n_f]):
            for token, mode in modes.items():
                if token in order:
                    a_mode[order[token], fi] = mode
        for fi, source_rows in enumerate(
                getattr(self, 'agent_source_row_list', [])[:n_f]):
            for token, source_row in source_rows.items():
                if token in order:
                    a_source_row[order[token], fi] = int(source_row)

        idm_batch = getattr(self.engine, "_nuplan_idm_batch", None)
        path_predrop_tokens = {
            str(token) for token, reason in
            (getattr(idm_batch, "predropped_vehicles", {}) or {}).items()
            if str(reason).startswith("source_path_le_")
        }
        leaked = sorted(set(order) & path_predrop_tokens)
        if leaked:
            raise RuntimeError(
                "source-path predropped vehicle(s) reached the scored NPZ snapshot: "
                + ",".join(leaked)
            )
        im_stats = getattr(idm_batch, "intersection_manager_stats", None) or {}
        hold_stats = im_stats.get("hold_reason_ticks", {})
        hold_reason_names = np.array(
            ["active_npc", "downstream_blocked", "ego_safety_envelope", "fifo_wait"]
        )
        hold_reason_ticks = np.array(
            [int(hold_stats.get(reason, 0)) for reason in hold_reason_names], dtype=np.int64
        )
        agent_manager = self.engine.managers.get("agent_manager")
        source_end_stats = (
            getattr(agent_manager, "source_end_liveness_stats", None) or {}
        )
        parallel_runtime = (
            getattr(agent_manager, "parallel_runtime_stats", None) or {}
        )
        spawn_overlap_stats = (
            getattr(agent_manager, "spawn_overlap_stats", None) or {}
        )
        spawn_events = list(getattr(agent_manager, 'spawn_gate_events', ()) or ())

        # Scoring inputs: store the exact object scene-end scoring passed to replay (_scoring_pins).
        # Rebuilding it here would create two copies: "what scoring saw" and "what was saved".
        pinned = getattr(self, '_pinned', None)
        pins = dict(pinned if pinned is not None else self._scoring_pins())
        # Per-tick collision contacts and ego-area flags. They are an OUTPUT of scoring, not an
        # input to it, so they are not in driving_inputs_json: a replay recomputes them from the
        # pinned inputs and must be free to disagree if the rules change. They are here so a
        # viewer or an audit can read the timeline without re-running the scorer (and without
        # the monkey-patch that was the only way to get the contacts).
        pins.update(getattr(self, '_dense_series', None) or {})
        # Exact 10 Hz TLC trace from the very same score_tlc invocation that
        # produced the CSV scalars.  Consumers must read this instead of
        # re-intersecting the legacy 0.5 s ego_xy trajectory with map polygons.
        pins.update(getattr(self, '_tlc_series', None) or {})
        # Opt-in signal patch (tl_signal_patch_path): its sha, the applied rows, and per scored step x signal
        # connector whether the recorded colour came from the patch. Absent keys = no patch.
        patch = self.current_scene.get('tl_signal_patch') if hasattr(self.current_scene, 'get') else None
        if patch:
            snaps = getattr(self, '_tlc_signal_snapshots', None) or []
            ids = sorted({c for row in snaps for c in (row.get('patched') or {})})
            pins['tl_patch_sha256'] = str(patch['sha256'])
            pins['tl_patch_json'] = json.dumps(patch, allow_nan=False, separators=(',', ':'))
            pins['tlc_signal_patched_connector_ids'] = np.asarray(ids, dtype=str)
            pins['tlc_signal_patched_sim_steps'] = np.asarray([int(row['step']) for row in snaps], dtype=np.int64)
            pins['tlc_signal_patched'] = np.asarray(
                [[bool((row.get('patched') or {}).get(c, False)) for c in ids] for row in snaps], dtype=bool
            ).reshape(len(snaps), len(ids))
        # Exact render-side representative-frame decisions.  This is present only when
        # tl_control_path was explicitly supplied; an absent key therefore still means
        # the legacy natural-replay renderer, rather than an empty control policy.
        render_manager = self.engine.managers.get('render_manager')
        tl_render_trace = getattr(render_manager, 'traffic_light_control_trace', None)
        if tl_render_trace:
            pins['tl_render_control_json'] = json.dumps(
                tl_render_trace, allow_nan=False, separators=(',', ':'))

        path = os.path.join(out_dir, "rollout_trajectory.npz")
        np.savez_compressed(
            path,
            ego_xy=ego_xy, ego_heading=ego_h, ego_velocity=ego_v,
            ego_acceleration=ego_a,
            **pins,
            rc_handoff_step=self._rc_metric.handoff,
            rc_origin_offset=self._rc_metric.offset,
            rc_method=str(self._rc_fields.get('rc_method', '')),
            agent_tokens=np.array(list(order), dtype=object),
            agent_types=np.array([types[token] for token in order], dtype=object),
            agent_xy=a_xy, agent_heading=a_h, agent_velocity=a_v, agent_size=a_size,
            agent_source_mode=a_mode,
            agent_source_row=a_source_row,
            agent_frame_offset=agent_frame_offset,
            idm_source_path_predrop_max_m=float(
                getattr(idm_batch, "_source_path_predrop_max_m", 0.0)
            ),
            idm_source_path_predrop_tokens=np.asarray(
                sorted(path_predrop_tokens), dtype="U64"
            ),
            idm_excluded_connector_ids=np.asarray(
                sorted(getattr(idm_batch, "_excluded_idm_connector_ids", ())), dtype="U32"
            ),
            idm_route_intent_tokens=np.asarray(
                im_stats.get("route_intent_tokens", []), dtype="U64"
            ),
            idm_route_intent_source_rows=np.asarray(
                im_stats.get("route_intent_source_rows", []), dtype=np.int64
            ),
            idm_route_intent_from_edges=np.asarray(
                im_stats.get("route_intent_from_edges", []), dtype="U32"
            ),
            idm_route_intent_stock_edges=np.asarray(
                im_stats.get("route_intent_stock_edges", []), dtype="U32"
            ),
            idm_route_intent_chosen_edges=np.asarray(
                im_stats.get("route_intent_chosen_edges", []), dtype="U32"
            ),
            idm_route_intent_match_m=np.asarray(
                im_stats.get("route_intent_match_m", []), dtype=np.float64
            ),
            idm_route_intent_margin_m=np.asarray(
                im_stats.get("route_intent_margin_m", []), dtype=np.float64
            ),
            idm_route_intent_changed_branches=int(
                im_stats.get("route_intent_changed_branches", 0)
            ),
            idm_initial_speed_tokens=np.asarray(
                im_stats.get("initial_speed_tokens", []), dtype=object
            ),
            idm_initial_speed_source_rows=np.asarray(
                im_stats.get("initial_speed_source_rows", []), dtype=np.int64
            ),
            idm_initial_speed_raw_mps=np.asarray(
                im_stats.get("initial_speed_raw_mps", []), dtype=np.float64
            ),
            idm_initial_speed_derived_mps=np.asarray(
                im_stats.get("initial_speed_derived_mps", []), dtype=np.float64
            ),
            idm_initial_speed_applied_mps=np.asarray(
                im_stats.get("initial_speed_applied_mps", []), dtype=np.float64
            ),
            idm_initial_speed_source_kind=np.asarray(
                im_stats.get("initial_speed_source_kind", []), dtype=object
            ),
            idm_initial_speed_clamp_reasons=np.asarray(
                im_stats.get("initial_speed_clamp_reason", []), dtype=object
            ),
            idm_initial_speed_seeded_vehicles=int(
                im_stats.get("initial_speed_seeded_vehicles", 0)
            ),
            idm_initial_speed_zero_fallbacks=int(
                im_stats.get("initial_speed_zero_fallbacks", 0)
            ),
            idm_initial_speed_applied_p50_mps=float(
                im_stats.get("initial_speed_applied_p50_mps", 0.0)
            ),
            idm_initial_speed_applied_p95_mps=float(
                im_stats.get("initial_speed_applied_p95_mps", 0.0)
            ),
            idm_initial_speed_applied_max_mps=float(
                im_stats.get("initial_speed_applied_max_mps", 0.0)
            ),
            idm_lifecycle_sim_steps=np.asarray(
                getattr(self, "_idm_lifecycle_sim_steps", []), dtype=np.int64
            ),
            idm_lifecycle_source_steps=np.asarray(
                getattr(self, "_idm_lifecycle_source_steps", []), dtype=np.int64
            ),
            score_dt=SCORE_DT, num_history=self.num_history,
            idm_intersection_manager_enabled=bool(
                im_stats.get("intersection_manager_enabled", False)
            ),
            idm_im_updates=int(im_stats.get("updates", 0)),
            idm_im_hold_reason_names=hold_reason_names,
            idm_im_hold_reason_ticks=hold_reason_ticks,
            idm_im_stall_events=int(im_stats.get("stall_events", 0)),
            idm_im_preexisting_active_stall_events=int(
                im_stats.get("preexisting_active_stall_events", 0)
            ),
            idm_im_active_stall_retirements=int(
                im_stats.get("active_stall_retirements", 0)
            ),
            idm_im_intersection_stopped_seconds=float(
                im_stats.get("intersection_stopped_seconds", 0.0)
            ),
            idm_im_downstream_reservation_conflicts=int(
                im_stats.get("downstream_reservation_conflicts", 0)
            ),
            idm_im_downstream_reservation_ttl_revocations=int(
                im_stats.get("downstream_reservation_ttl_revocations", 0)
            ),
            idm_im_peak_downstream_reservations=int(
                im_stats.get("peak_downstream_reservations", 0)
            ),
            idm_im_candidate_ticks=int(im_stats.get("candidate_ticks", 0)),
            idm_im_contention_candidate_ticks=int(
                im_stats.get("contention_candidate_ticks", 0)
            ),
            idm_im_safety_candidate_ticks=int(
                im_stats.get("safety_candidate_ticks", 0)
            ),
            idm_im_unmanaged_candidate_ticks=int(
                im_stats.get("unmanaged_candidate_ticks", 0)
            ),
            idm_im_downstream_moving_same_flow_ignored_ticks=int(
                im_stats.get("downstream_moving_same_flow_ignored_ticks", 0)
            ),
            idm_im_downstream_transient_stop_ignored_ticks=int(
                im_stats.get("downstream_transient_stop_ignored_ticks", 0)
            ),
            idm_im_downstream_persistent_blocker_ticks=int(
                im_stats.get("downstream_persistent_blocker_ticks", 0)
            ),
            idm_im_downstream_cross_direction_blocker_ticks=int(
                im_stats.get("downstream_cross_direction_blocker_ticks", 0)
            ),
            idm_im_downstream_insufficient_path_ticks=int(
                im_stats.get("downstream_insufficient_path_ticks", 0)
            ),
            idm_im_downstream_unclassified_blocker_ticks=int(
                im_stats.get("downstream_unclassified_blocker_ticks", 0)
            ),
            idm_im_rear_ego_hold_suppressions=int(
                im_stats.get("rear_ego_hold_suppressions", 0)
            ),
            idm_spawn_crossing_deferrals=int(
                im_stats.get("spawn_crossing_deferrals", 0)
            ),
            idm_spawn_sector_waiting_vehicles=int(
                im_stats.get("spawn_sector_waiting_vehicles", 0)
            ),
            idm_spawn_ego_lane_clearance_deferrals=int(
                im_stats.get("spawn_ego_lane_clearance_deferrals", 0)
            ),
            idm_initial_ego_lane_clearance_deferrals=int(
                im_stats.get("initial_ego_lane_clearance_deferrals", 0)
            ),
            idm_parked_vehicle_fallbacks=int(
                im_stats.get("parked_vehicle_fallbacks", 0)
            ),
            idm_source_motion_open_loop_vehicles=int(
                im_stats.get("source_motion_open_loop_vehicles", 0)
            ),
            idm_source_motion_spawned_vehicles=int(
                im_stats.get("source_motion_spawned_vehicles", 0)
            ),
            idm_source_motion_spawn_deferrals=int(
                im_stats.get("source_motion_spawn_deferrals", 0)
            ),
            idm_unknown_signal_fallback_events=int(
                im_stats.get("unknown_signal_fallback_events", 0)
            ),
            idm_unknown_signal_fallback_intersection_seconds=float(
                im_stats.get("unknown_signal_fallback_intersection_ticks", 0)
            ) * float(self.engine.sim_dt),
            idm_unknown_signal_fallback_connector_ticks=int(
                im_stats.get("unknown_signal_fallback_connector_ticks", 0)
            ),
            # Kept without an idm_ prefix because the same publication gate now covers both
            # trajectory replay and reactive-sector actors.
            spawn_overlap_deferral_ticks=int(
                spawn_overlap_stats.get("deferral_ticks", 0)
            ),
            spawn_overlap_deferral_seconds=float(
                spawn_overlap_stats.get("deferral_ticks", 0)
            ) * float(self.engine.sim_dt),
            spawn_overlap_deferred_actors=int(
                spawn_overlap_stats.get("deferred_actors", 0)
            ),
            spawn_overlap_dropped_actors=int(
                spawn_overlap_stats.get("dropped_actors", 0)
            ),
            # Which actors, not just how many. A dropped actor has no box anywhere in this
            # file -- that is the point of dropping it -- so without the tokens a later
            # reader cannot tell a gated rollout from one the log simply never populated.
            # A fixed-width string dtype, not object, so this field reads back without
            # allow_pickle.
            spawn_overlap_dropped_tokens=np.asarray(
                spawn_overlap_stats.get("dropped_tokens", ()), dtype="U40"
            ),
            spawn_gate_event_steps=np.asarray(
                [event['simulation_step'] for event in spawn_events], dtype=np.int64),
            spawn_gate_event_tokens=np.asarray(
                [event['token'] for event in spawn_events], dtype='U64'),
            spawn_gate_event_source_rows=np.asarray(
                [event['source_row'] for event in spawn_events], dtype=np.int64),
            spawn_gate_event_decisions=np.asarray(
                [event['decision'] for event in spawn_events], dtype='U16'),
            spawn_gate_event_reasons=np.asarray(
                [event['reason'] for event in spawn_events], dtype='U32'),
            idm_source_end_stall_retirements=int(
                source_end_stats.get("stall_retirements", 0)
            ),
            idm_source_end_retained_agent_ticks=int(
                source_end_stats.get("retained_agent_ticks", 0)
            ),
            idm_source_end_retained_agent_seconds=float(
                source_end_stats.get("retained_agent_ticks", 0)
            ) * float(self.engine.sim_dt),
            idm_source_end_route_extensions=int(
                im_stats.get("source_end_route_extensions", 0)
            ),
            idm_source_end_route_exhaustion_despawns=int(
                im_stats.get("source_end_route_exhaustion_despawns", 0)
            ),
            idm_dead_end_route_despawns=int(
                im_stats.get("dead_end_route_despawns", 0)
            ),
            idm_dead_end_queue_retirements=int(
                im_stats.get("dead_end_queue_retirements", 0)
            ),
            idm_source_ended_active_stall_retirements=int(
                im_stats.get("source_ended_active_stall_retirements", 0)
            ),
            idm_wait_cycle_events=int(im_stats.get("wait_cycle_events", 0)),
            idm_wait_cycle_agent_seconds=float(
                im_stats.get("wait_cycle_agent_seconds", 0.0)
            ),
            idm_rear_lead_rejections=int(
                im_stats.get("rear_lead_rejections", 0)
            ),
            idm_adjacent_lead_rejections=int(
                im_stats.get("adjacent_lead_rejections", 0)
            ),
            idm_ego_deadlock_breaker_events=int(
                im_stats.get("ego_deadlock_breaker_events", 0)
            ),
            idm_ego_deadlock_breaker_ticks=int(
                im_stats.get("ego_deadlock_breaker_ticks", 0)
            ),
            idm_scoped_non_vehicle_lead_suppressions=int(
                im_stats.get("scoped_non_vehicle_lead_suppressions", 0)
            ),
            idm_handoff_prebuild_seconds=float(
                im_stats.get("handoff_prebuild_seconds", 0.0)
            ),
            idm_handoff_source_vehicles=int(
                im_stats.get("handoff_source_vehicles", 0)
            ),
            idm_handoff_built_idm_vehicles=int(
                im_stats.get("handoff_built_idm_vehicles", 0)
            ),
            idm_handoff_routable_not_built=int(
                im_stats.get("handoff_routable_not_built", 0)
            ),
            idm_handoff_gt_fallback_vehicles=int(
                im_stats.get("handoff_gt_fallback_vehicles", 0)
            ),
            idm_handoff_smooth_merge_vehicles=int(
                im_stats.get("handoff_smooth_merge_vehicles", 0)
            ),
            idm_handoff_dropped_routable_vehicles=int(
                im_stats.get("handoff_dropped_routable_vehicles", 0)
            ),
            idm_handoff_snap_distance_p95_m=float(
                im_stats.get("handoff_snap_distance_p95_m", 0.0)
            ),
            idm_handoff_snap_distance_max_m=float(
                im_stats.get("handoff_snap_distance_max_m", 0.0)
            ),
            idm_handoff_snap_heading_p95_deg=float(
                im_stats.get("handoff_snap_heading_p95_deg", 0.0)
            ),
            idm_handoff_new_snap_overlap_pairs=int(
                im_stats.get("handoff_new_snap_overlap_pairs", 0)
            ),
            idm_handoff_candidate_new_snap_overlap_pairs=int(
                im_stats.get("handoff_candidate_new_snap_overlap_pairs", 0)
            ),
            runtime_parallel_steps=int(parallel_runtime.get("steps", 0)),
            runtime_ego_planner_seconds=float(
                parallel_runtime.get("ego_planner_s", 0.0)
            ),
            runtime_idm_seconds=float(parallel_runtime.get("idm_s", 0.0)),
            runtime_parallel_critical_path_seconds=float(
                parallel_runtime.get("critical_path_s", 0.0)
            ),
            runtime_parallel_overlap_seconds=float(
                parallel_runtime.get("overlap_s", 0.0)
            ),
        )
        logger.info("[DUMP] executed rollout -> %s  (%d graded frames, %d agents)",
                    path, len(ego_xy), n_a)
