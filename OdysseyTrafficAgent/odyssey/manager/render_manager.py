# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
import numpy as np
import hashlib
import json
import re
from pathlib import Path
from hydra.utils import instantiate

from odyssey.manager.base_manager import BaseManager
from odyssey.manager import tlc_timetable_set
from odyssey.scenario.scenarios.scenario_description import ScenarioDescription as SD
from odyssey_renderer.base_renderer import RenderState
from odyssey_renderer.omnire.tl_control import (
    TL_CONTROL_SCENE_EXCLUSIONS,
    TrafficLightController,
    excluded_scene_reason,
)
import logging
logger = logging.getLogger(__name__)

class RenderManager(BaseManager):

    PRIORITY = 10000    # lowest priority
    # tl_control_allow_excluded entries already reported as having no effect (logged once per process).
    _TL_ALLOW_WARNED = set()

    def __init__(self):
        super(RenderManager, self).__init__()

        self.current_scene: SD = self.engine.managers['scenario_manager'].current_scene
        self.current_scene_id = self.current_scene[SD.ID]
        self.base_timestamp = self.current_scene[SD.BASE_TIMESTAMP]
        self.local2global_translation_xy = -np.array(self.current_scene[SD.METADATA][SD.OLD_ORIGIN_IN_CURRENT_COORDINATE])

        self.renderer = instantiate(self.global_config.renderer)
        self.rendering_results = {}
        self._tl_ctl = None
        self._tl_control_path = None
        self._tl_control_sha256 = None
        self._tl_control_exclusion_reason = None
        self._tl_control_exclusion_overridden = None
        self._tl_log = {}
        self._tl_entry_lane = None

        # render_every=N: rasterise and restore only at steps 0, N, 2N ...
        self._render_every = max(int(self.global_config.get('render_every', 1)), 1)

    def _configure_traffic_light_control(self):
        """Bind one explicit control file to this scene and the scoring signal clock."""
        self._tl_ctl = None
        self._tl_control_path = None
        self._tl_control_sha256 = None
        self._tl_control_exclusion_reason = None
        self._tl_control_exclusion_overridden = None
        self._tl_log = {}
        self._tl_entry_lane = None
        self._tl_signal_patch = None
        # Opt-in preset (tlc_timetable_set, default null): for a scene in the set it supplies
        # tl_control_pin_uncontrolled and tl_control_allow_excluded (the label stays tl_control_path, the scene's
        # own); explicit keys win. None = read the keys as always.
        preset = tlc_timetable_set.resolve(self.global_config, self.current_scene_id)
        self._tlc_timetable_set = preset if preset and preset.get('in_set') else None
        configured = tlc_timetable_set.value_for(self.global_config, 'tl_control_path', preset)
        if configured in (None, ''):
            return

        match = re.search(r'odyssey_scene\d{3}', str(self.current_scene_id))
        actual = match.group(0) if match else str(self.current_scene_id)
        exclusion = excluded_scene_reason(actual)
        # Opt-in per scene: tl_control_allow_excluded lists scenes that may use tl_control
        # although they are on the audited denylist. Absent or empty = the denylist applies.
        allowed = tlc_timetable_set.value_for(self.global_config, 'tl_control_allow_excluded', preset) or ()
        allowed = {str(allowed)} if isinstance(allowed, str) else {str(s) for s in allowed}
        for entry in sorted(allowed - RenderManager._TL_ALLOW_WARNED):
            if not re.fullmatch(r'odyssey_scene\d{3}', entry) or entry not in TL_CONTROL_SCENE_EXCLUSIONS:
                RenderManager._TL_ALLOW_WARNED.add(entry)
                logger.warning(
                    "tl_control_allow_excluded: entry %r has no effect (%s)", entry,
                    'not a published scene name (odyssey_sceneNNN)' if not re.fullmatch(r'odyssey_scene\d{3}', entry)
                    else 'scene is not on the tl_control exclusion list')
        if exclusion is not None and actual in allowed:
            logger.warning(
                "tl_control: scene %s is on the exclusion list (%s) but explicitly allowed "
                "by tl_control_allow_excluded", actual, exclusion)
        elif exclusion is not None:
            self._tl_control_path = str(configured)
            self._tl_control_exclusion_reason = exclusion
            logger.warning(
                "[tl-control] disabled by audited scene denylist: scene=%s reason=%s path=%s",
                actual, exclusion, configured)
            return

        path = Path(str(configured)).expanduser().resolve()
        payload = path.read_bytes()
        control = json.loads(payload)
        expected = str(control.get('scene_id', ''))
        if expected and expected != actual:
            raise ValueError(
                f"tl_control scene mismatch: {expected!r} != {actual!r} "
                f"for {self.current_scene_id!r}")

        agent_manager = self.engine.managers.get('agent_manager')
        source_row = getattr(agent_manager, 'traffic_light_source_row', None)
        if not callable(source_row):
            raise RuntimeError(
                "tl_control requires agent_manager.traffic_light_source_row; "
                "a separate render clock would disagree with TLC")
        patch = self.current_scene.get('tl_signal_patch') if hasattr(self.current_scene, 'get') else None
        if patch:
            # Signal patch on (tl_signal_patch_path): rendering-only edge rules (spec v2 section 5, D2/D3).
            # A sector that has not opened (row -1) draws its spawn-row colour; a row past the log holds the
            # last row. Scoring and planning keep the plain source row (-1 / past the log = UNKNOWN).
            source_row = self._patched_render_row(agent_manager, source_row)
            self._tl_signal_patch = dict(patch)
            logger.info("[tl-patch] render: scene=%s sha=%s spans=%d (row -1 -> spawn row, past log -> hold last)",
                        patch.get('scene'), patch.get('sha256'), patch.get('spans'))
        # Opt-in (default off): heads with representative frames are always drawn from a pinned frame,
        # never from natural playback or hold_frame (tl_control rule 6).
        pin_uncontrolled = bool(tlc_timetable_set.value_for(self.global_config, 'tl_control_pin_uncontrolled', preset,
                                                            False))
        if pin_uncontrolled:
            self._tl_ctl = TrafficLightController.from_scenario(
                control, self.current_scene, source_row=source_row, pin_uncontrolled=True)
            logger.info("[tl-control] pin_uncontrolled=on")
        else:
            self._tl_ctl = TrafficLightController.from_scenario(
                control, self.current_scene, source_row=source_row)
        self._tl_control_path = str(path)
        self._tl_control_sha256 = hashlib.sha256(payload).hexdigest()
        self._tl_control_exclusion_overridden = exclusion
        logger.info(
            "[tl-control] scene=%s path=%s sha256=%s heads=%d missing_connectors=%d",
            actual, path, self._tl_control_sha256, len(self._tl_ctl.node_heads),
            len(self._tl_ctl.missing_connectors))

    def _patched_render_row(self, agent_manager, source_row):
        """Render-only row clock for the signal patch: -1 -> spawn row, past the log -> last row.

        The spawn row exists only with the sector-replay clock (agent_manager.traffic_light_spawn_row); with the
        identity clock no row is -1 and a connector without a sector keeps -1 (the controller's usual handling)."""
        lengths = {}
        for item in (self.current_scene.get('dynamic_map_states') or {}).values():
            if item.get('type') == 'TRAFFIC_LIGHT' and item.get('traffic_light_lane') is not None:
                c = str(item['traffic_light_lane'])
                lengths[c] = max(lengths.get(c, 0), len(item['state']['traffic_light_state']))
        spawn_row = getattr(agent_manager, 'traffic_light_spawn_row', None)

        def row(connector, step):
            r = int(source_row(connector, step))
            if r < 0 and callable(spawn_row):
                s = spawn_row(connector)
                if s is not None:
                    r = int(s)
            n = lengths.get(str(connector))
            if n and r >= n:
                r = n - 1
            return r
        return row

    def _traffic_light_step(self, step, route, ego_xy, ego_heading):
        """Resolve one step, log it, and return the TL_SOURCE_FRAMES mapping (None = omit the key)."""
        self._tl_unobserved_ok = {}
        if route is None:
            if getattr(self._tl_ctl, 'pin_uncontrolled', False):
                # Opt-in: before the scoring planner has a route (warm-up), nothing is in scope, but
                # pinned heads are still drawn from pinned frames (tl_control rule 6) from step 0.
                resolved = self._tl_ctl.resolve(step, route_connectors=[])
                self._tl_log[step] = {
                    'step': step,
                    'mapping': dict(resolved.mapping),
                    'counts': resolved.counts(),
                    'report': dict(resolved.report),
                    'route_connectors': [],
                    'signal_scope': [],
                    'skipped': 'route_unavailable',
                }
                self._note_unobserved(step, resolved)
                return dict(resolved.mapping)
            # Do not guess a route before the scoring planner has built the one TLC uses.
            # Omitting the render-state key preserves natural replay for this warm-up frame.
            self._tl_log[step] = {
                'step': step, 'mapping': {}, 'counts': {},
                'skipped': 'route_unavailable',
            }
            return None
        signal_scope, lane_match = self._tl_ctl.connector_scope(
            route, ego_xy=ego_xy, ego_heading=ego_heading,
            previous_entry_lane=self._tl_entry_lane)
        if lane_match.get('entry_lane') is not None:
            self._tl_entry_lane = str(lane_match['entry_lane'])
        resolved = self._tl_ctl.resolve(step, route_connectors=signal_scope)
        self._tl_log[step] = {
            'step': step,
            'mapping': dict(resolved.mapping),
            'counts': resolved.counts(),
            'report': dict(resolved.report),
            'route_connectors': list(route),
            'signal_scope': list(signal_scope),
            'lane_match': lane_match,
        }
        self._note_unobserved(step, resolved)
        return dict(resolved.mapping)

    def _note_unobserved(self, step, resolved):
        """Label-allowed unobserved representative frames (tl_control allow_unobserved scenes only)."""
        allowed = getattr(resolved, 'unobserved_ok', None)
        if allowed:
            self._tl_unobserved_ok = dict(allowed)
            self._tl_log[step]['unobserved_ok'] = dict(allowed)

    def _ego_route_connector_ids(self):
        """Return the exact route dictionary used by TLC, or None before it exists."""
        metric_manager = self.engine.managers.get('metric_manager')
        planner = getattr(metric_manager, '_pdm_closed', None)
        route = getattr(planner, '_route_lane_dict', None)
        if not route:
            return None
        return tuple(str(connector_id) for connector_id in route.keys())

    @property
    def traffic_light_control_trace(self):
        """JSON-safe audit payload embedded into rollout_trajectory.npz."""
        if self._tl_ctl is None:
            if self._tl_control_exclusion_reason is not None:
                return {
                    'schema': 'tl_render_control_trace/2',
                    'scene_id': str(self.current_scene_id),
                    'control_path': self._tl_control_path,
                    'control_sha256': None,
                    'excluded': True,
                    'exclusion_reason': self._tl_control_exclusion_reason,
                    'missing_connectors': [],
                    'steps': [],
                }
            return None
        trace = {
            'schema': 'tl_render_control_trace/2',
            'scene_id': str(self._tl_ctl.control.get('scene_id', '')),
            'control_path': self._tl_control_path,
            'control_sha256': self._tl_control_sha256,
            'missing_connectors': list(self._tl_ctl.missing_connectors),
            'steps': [self._tl_log[step] for step in sorted(self._tl_log)],
        }
        if getattr(self._tl_ctl, 'pin_uncontrolled', False):
            trace['pin_uncontrolled'] = True           # only when tl_control_pin_uncontrolled is on
        if getattr(self, '_tl_signal_patch', None):
            trace['tl_signal_patch'] = {k: self._tl_signal_patch[k] for k in ('scene', 'sha256', 'path', 'spans')}
        if getattr(self, '_tl_control_exclusion_overridden', None) is not None:
            trace['exclusion_overridden'] = True       # only when tl_control_allow_excluded bypassed it
            trace['exclusion_reason'] = self._tl_control_exclusion_overridden
        preset = getattr(self, '_tlc_timetable_set', None)
        if preset:                                     # only when the tlc_timetable_set preset covered this scene
            trace['tlc_timetable_set'] = {k: preset[k] for k in ('name', 'dir', 'sha256')}
        return trace

    @property
    def traffic_light_signal_source_overrides(self):
        """Per-step planned-contact connector -> actual-lane signal connector.

        TLC deliberately keeps its original route connector polygons as the intersection-entry
        contact gate.  Only the signal source changes: when ego approaches from a sibling lane,
        the colour it faces is that lane's connector colour.  Returning the decision made for
        rendering keeps scoring from independently map-matching the trajectory a second time.
        """
        if self._tl_ctl is None:
            return None
        out = {}
        for step, record in self._tl_log.items():
            overrides = (record.get('lane_match') or {}).get(
                'signal_source_overrides') or {}
            if overrides:
                out[int(step)] = {str(target): str(source)
                                  for target, source in overrides.items()}
        return out

    def render(self):
        render_state = RenderState()

        # get dynamic agents state. (x, y, heading)
        agents_state = {}
        ego_local_xy = None
        ego_heading = None
        # Reactive rendering also needs the stationary vehicle proxies. They are kept
        # in _static_agents and would otherwise disappear when checkpoint poses are disabled.
        agent_manager = self.engine.managers['agent_manager']
        use_scenario_actors = getattr(self.renderer, 'render_simulated_vehicles', False)
        agents = agent_manager.all_agents if use_scenario_actors else agent_manager.get_dynamic_agents
        # A parked car's policy becomes invalid once the log ends, but the car only stops at its
        # last log pose and stays in the world, where it is scored. Rendering must see the same world.
        #
        # The set read is _parked_vehicle_ids. Reading the R-only _static_pinned_vehicle_ids would
        # give an always-empty set under NR (trajectory_policy; see
        # agent_manager._static_pin_and_predrop_enabled), so parked cars whose log ended would stay
        # in scoring but vanish from the image -- the model would be penalized for hitting cars it
        # cannot see. _parked_vehicle_ids is filled regardless of policy (AgentManager.reset) and
        # is the same set scoring counts.
        pinned = getattr(agent_manager, '_parked_vehicle_ids', frozenset())
        for obj_id, agent in agents.items():
            if not agent.policy.is_current_step_valid and obj_id not in pinned:
                continue

            if obj_id == 'ego':
                # pass rear axle position and heading
                ego_local_xy = np.asarray(agent.rear_vehicle.current_position[:2], dtype=float)
                ego_heading = float(agent.current_heading)
                bbox = np.array([agent.rear_vehicle.current_position[0], agent.rear_vehicle.current_position[1], 0.0, 0.0, 0.0, agent.current_heading])
            else:
                bbox = agent.bounding_box
            # transform to nuplan global coordinate system.
            bbox[:2] += self.local2global_translation_xy
            agents_state[obj_id] = bbox

        render_state[RenderState.AGENT_STATE] = agents_state
        # The log row each actor is replaying. On the log clock this equals the sim step, exactly
        # what the renderer used before. If a replay clock moved it, that row is the source of the
        # position the simulator and scorer see, and the renderer must see the same -- the
        # simulator decides who placed an actor where.
        render_state[RenderState.AGENT_SOURCE_ROW] = {
            obj_id: int(agent.traj_step)
            for obj_id, agent in agents.items()
            if obj_id != 'ego' and obj_id in agents_state}
        if self._tl_ctl is not None:
            frames = self._traffic_light_step(int(self.engine.episode_step), self._ego_route_connector_ids(),
                                              ego_local_xy, ego_heading)
            if frames is not None:
                render_state[RenderState.TL_SOURCE_FRAMES] = frames
                allowed = getattr(self, '_tl_unobserved_ok', None)
                if allowed:      # unobserved representative frames the labels allow
                    render_state[RenderState.TL_ALLOW_UNOBSERVED] = dict(allowed)
        # Actors not admitted are in neither agents_state nor the row list above, but the renderer
        # draws actors at their checkpoint poses, so that alone does not remove them from the image.
        # Pass the decision through so the rendered world and the scored world stay the same.
        render_state[RenderState.SUPPRESSED_ACTORS] = getattr(
            agent_manager, 'spawn_deferred_tokens', frozenset())
        render_state[RenderState.HELD_AT_LOG_POSE] = self._held_at_log_pose(agents_state)

        # get timestamp
        global_step = self.engine.episode_step
        timestamp = self.base_timestamp + global_step * self.engine.sim_dt * 1e6
        render_state[RenderState.TIMESTAMP] = timestamp

        # get sensor parameters
        # TODO: pertube camera location by timeshift.
        render_state[RenderState.SKIP_CAMERAS] = (
            self._render_every > 1 and (self.engine.episode_step % self._render_every) != 0)
        render_state[RenderState.CAMERAS] = self.current_scene[SD.CAMERAS]
        render_state[RenderState.LIDAR] = self.current_scene[SD.LIDAR]
        
        return self.renderer.render(render_state)

    def _held_at_log_pose(self, agents_state):
        """-> frozenset of the tokens the simulator keeps on their logged pose this step.

        Static agents replay the log, and under nuplan_idm_policy every vehicle is an IDM agent
        unless the batch handed it back to its source poses (is_gt_replay_vehicle). Whether a car
        stood still in the log does not say which: IDM drives a car parked on a lane.
        """
        agent_manager = self.engine.managers['agent_manager']
        dynamic = agent_manager.get_dynamic_agents
        batch = getattr(self.engine, '_nuplan_idm_batch', None)
        is_gt_replay = getattr(batch, 'is_gt_replay_vehicle', None)
        return frozenset(
            token for token in agents_state
            if token != 'ego' and (token not in dynamic
                                   or (is_gt_replay is not None and is_gt_replay(token))))

    def before_reset(self):
        pass

    def reset(self):
        self.current_scene: SD = self.engine.managers['scenario_manager'].current_scene
        self.current_scene_id = self.current_scene[SD.ID]
        self.base_timestamp = self.current_scene[SD.BASE_TIMESTAMP]
        self.local2global_translation_xy = -np.array(self.current_scene[SD.METADATA][SD.OLD_ORIGIN_IN_CURRENT_COORDINATE])
        self.renderer.reset(self.current_scene_id)
        self._configure_traffic_light_control()

    def after_reset(self):
        pass

    def before_step(self):
        pass

    def step(self):
        pass

    def get_observations(self):
        step = self.engine.episode_step
        logger.debug(f"Rendering step {step} for scenario {self.current_scene_id}")
        render_results = self.render()

        self.rendering_results[step] = render_results
        # DataManager consumes only this step. Bound image ownership here even when
        # its frame-supply cap prevents consumption, so old frames cannot accumulate.
        for _stale in [k for k in self.rendering_results if k != step]:
            del self.rendering_results[_stale]

        # trigger data saving after getting the render results
        self.engine.managers['data_manager'].save_current_frame_data()

        return render_results

    def after_step(self):
        pass

    def close(self):
        pass

    def destroy(self):
        pass
