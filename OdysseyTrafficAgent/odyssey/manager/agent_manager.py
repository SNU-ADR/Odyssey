# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
"""
This manager allows one to use object like vehicles/traffic lights as agent with multi-agent support.
You would better make your own agent manager based on this class.
"""

import concurrent.futures
import copy
import json
import os
import time
import numpy as np

from odyssey.manager.base_manager import BaseManager

from odyssey.utils import math_utils
from odyssey.utils.type import OdysseyObjectType
from odyssey.utils.agent_utils import BEV_visualizer
from odyssey.scenario.scenarios.scenario_description import ScenarioDescription as SD
from odyssey.scenario.scenarios.parse_scenario_state import parse_object_track
from odyssey.scenario import hybrid_replay
from odyssey.scenario.hybrid_replay import EgoProgressClock, HybridReplay, SectorReplay

from odyssey.components.agents.base_agent import BaseAgent
from odyssey.components.agents.ego_agent import EgoAgent

import logging
logger = logging.getLogger(__name__)


def _nuplan_idm_controls_vehicle(config, obj_track):
    """Whether nuPlan, rather than source displacement, owns this vehicle's motion."""
    return config.get('agent_policy') == 'nuplan_idm_policy' and obj_track.get('type') == 'VEHICLE'


def _static_pin_and_predrop_enabled(config):
    """R-only switch (nuplan_idm_static_pin_and_predrop). NR never reads this key."""
    return (config.get('agent_policy') == 'nuplan_idm_policy'
            and bool(config.get('nuplan_idm_static_pin_and_predrop', False)))


def _is_pinned_static_vehicle(obj_track):
    """A VEHICLE valid in every frame with its xy fixed at one point -- a car parked in the PKL.

    The exporter does not write a flag yet, so this is decided from the track itself. An explicit
    metadata control_mode='static' is followed as is.
    """
    if obj_track.get('type') != 'VEHICLE':
        return False
    if obj_track.get('metadata', {}).get('control_mode') == 'static':
        return True
    state = obj_track.get('state', {})
    valid = np.asarray(state.get('valid', []), dtype=bool).reshape(-1)
    positions = np.asarray(state.get('position', []), dtype=np.float64)
    if (not len(valid) or not valid.all() or positions.ndim != 2
            or len(positions) != len(valid) or positions.shape[1] < 2):
        return False
    return float(np.ptp(positions[:, :2], axis=0).max()) < 1e-3


def _simulation_enabled(obj_track):
    """Whether this PKL actor participates in world simulation.

    Old PKLs have no classification field and remain enabled.  New checkpoint-pinned
    PKLs retain zero-Gaussian actors for provenance while marking them disabled.
    """
    return bool(obj_track.get('metadata', {}).get('simulation_enabled', True))


def _is_source_replayed_actor(obj_track):
    """Actors whose checkpoint/source trajectory owns motion instead of vehicle IDM."""
    mode = obj_track.get('metadata', {}).get('control_mode')
    if mode is not None:
        return mode == 'replay'
    return obj_track.get('type') in ('PEDESTRIAN', 'CYCLIST', 'BICYCLE')


def _footprint(center_xy, heading, length, width):
    """The actor's BEV footprint: the same four corners the scorer collides on.

    Planar by construction -- collision here, in the PDM scorer and in the renderer is a
    2D question, and no elevation is consulted anywhere along that path.
    """
    from shapely.geometry import Polygon

    cos_h, sin_h = np.cos(heading), np.sin(heading)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    half_l, half_w = length / 2.0, width / 2.0
    corners = ((half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w))
    return Polygon([(cx + x * cos_h - y * sin_h, cy + x * sin_h + y * cos_h)
                    for x, y in corners])


class BaseAgentManager(BaseManager):
    PRIORITY = 10
    STATIC_THRESHOLD = 3  # m, static if moving distance < 3

    # Overlap area which counts as "the actor would appear inside the ego". Same number as
    # the reactive fleet's footprint_overlap_only gate, so both paths mean the same thing
    # by a spawn collision and a shared touch does not trip either.
    SPAWN_OVERLAP_AREA_M2 = 0.01

    # Filled only for R + nuplan_idm_static_pin_and_predrop (reset). Empty keeps the original behaviour.
    _static_pinned_vehicle_ids = frozenset()

    # A fact stated by the track: a car parked for the whole clip. Unlike the set above, it is filled
    # regardless of policy -- NR also needs to know "this car is parked" to keep it in place after
    # the log ends (_survives_source_end).
    _parked_vehicle_ids = frozenset()

    def _is_source_replayed_token(self, obj_id):
        scene = getattr(self.engine, 'current_scene', None)
        if scene is None:
            scene = self.engine.managers['scenario_manager'].current_scene
        return _is_source_replayed_actor(
            scene.get(SD.OBJECT_TRACKS, {}).get(obj_id, {}))

    def __init__(self):
        """
        Each agent has the properties of:
            * observation.
            * actions.
        """
        super(BaseAgentManager, self).__init__()

        # Dynamic agents:
        #  use their policies and update position and each frame.
        self._dynamic_agents = {}  # {object.id: BaseAgent}

        # Static agents:
        #  objects without policy, like barriers and cones.
        self._static_agents = {}  # {object.id: BaseAgent}

        # Dead agents:
        #  once the agent is unavailable or out-of-visible areas,
        #  remove.
        self._dead_agents = {}  # {object.id: BaseAgent}

        self._BEV_vis = None
        self._trajectory_buffer = {}
        self._replay = None
        self._traffic_light_clock = None
        self._idm_lifecycle_clock = None
        self._idm_source_step = 0
        self._idm_spawn_clock = None
        self._idm_spawn_eligible = set()
        self._reactive_sector_rows = {}
        self._replay_rows, self._replay_rows_step = {}, None
        self._source_end_progress = {}
        self._source_end_stall_retirements = 0
        self._source_end_retained_agent_ticks = 0
        self._spawn_overlap_deferred = set()
        self._spawn_overlap_deferred_ever = set()
        self._spawn_overlap_dropped = set()
        self._spawn_overlap_deferral_ticks = 0
        self._spawn_gate_events = []
        # One worker is enough: the simulation thread performs IDM while this worker runs the
        # ego planner.  A persistent worker avoids creating a thread at every 10 Hz tick.
        self._ego_planner_executor = None
        self._parallel_runtime = {
            'steps': 0,
            'ego_planner_s': 0.0,
            'idm_s': 0.0,
            'critical_path_s': 0.0,
        }

    def _get_ego_planner_executor(self):
        if getattr(self, '_ego_planner_executor', None) is None:
            self._ego_planner_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ego-planner")
            logger.info(
                "[EGO-IDM-PARALLEL] enabled: ego planning worker + IDM simulation thread; "
                "commit after join"
            )
        return self._ego_planner_executor

    def _parallel_ego_idm_enabled(self):
        return (
            self.engine.global_config.get('agent_policy') == 'nuplan_idm_policy'
            and bool(self.engine.global_config.get('parallel_ego_idm', True))
            and isinstance(self._dynamic_agents.get('ego'), EgoAgent)
        )

    def _compute_ego_and_idm(self, ego, nuplan_batch, current_step):
        """Plan ego and propagate traffic concurrently from one immutable tick snapshot.

        Only planning is sent to the worker.  Ego control/state propagation is deliberately
        committed after IDM joins, while nuPlan's internal traffic poses are not copied into
        Odyssey proxies until afterwards.  Consequently both computations read the states
        from the start of the tick and cannot observe a half-updated world.
        """
        def timed_ego_plan():
            started = time.perf_counter()
            return ego.compute_step_plan(), time.perf_counter() - started

        wall_started = time.perf_counter()
        future = self._get_ego_planner_executor().submit(timed_ego_plan)
        idm_started = time.perf_counter()
        try:
            nuplan_batch.prepare_step(current_step)
        except BaseException:
            # Do not let a still-running GPU planner mutate its private policy history while the
            # failed environment is being torn down/reset.  cancel() handles a task which has not
            # started; result() joins one which has.
            if not future.cancel():
                try:
                    future.result()
                except BaseException:
                    pass
            raise
        idm_s = time.perf_counter() - idm_started
        plan_result, ego_s = future.result()
        critical_path_s = time.perf_counter() - wall_started

        runtime = getattr(self, '_parallel_runtime', None)
        if runtime is None:
            runtime = self._parallel_runtime = {
                'steps': 0,
                'ego_planner_s': 0.0,
                'idm_s': 0.0,
                'critical_path_s': 0.0,
            }
        runtime['steps'] += 1
        runtime['ego_planner_s'] += ego_s
        runtime['idm_s'] += idm_s
        runtime['critical_path_s'] += critical_path_s
        if runtime['steps'] % 100 == 0:
            overlap_s = max(
                runtime['ego_planner_s'] + runtime['idm_s']
                - runtime['critical_path_s'],
                0.0,
            )
            logger.info(
                "[EGO-IDM-PARALLEL] steps=%d ego=%.3fs idm=%.3fs "
                "critical-path=%.3fs overlap=%.3fs",
                runtime['steps'], runtime['ego_planner_s'], runtime['idm_s'],
                runtime['critical_path_s'], overlap_s,
            )

        # State mutation remains on the simulation thread and is outside both compute timers.
        ego.commit_step_plan(plan_result)

    # set up agents in the scenarios.
    def before_reset(self):  # remove agents.
        ret = super(BaseAgentManager, self).before_reset()

        for k, v in self._dead_agents.items():
            v.destroy()
        self._dead_agents = {}

        for k, v in self._static_agents.items():
            v.destroy()
        self._static_agents = {}

        for k, v in self._dynamic_agents.items():
            v.destroy()
        self._dynamic_agents = {}

        return ret

    def _process_agent_track(self, obj_id, obj_track):
        """
        processing agent track data.
        Args:
            obj_id: agent id
            obj_track: agent track data
        Returns:
            state: dict, agent state
            valid_periods: list, valid periods of the agent
        """
        time_idx = np.arange(len(obj_track[SD.STATE][SD.POSITION]))
        state = parse_object_track(obj_track, time_idx, sim_dt=self.engine.sim_dt)
        
        # Extract valid time periods
        valid_array = state[SD.VALID]
        valid_periods = []
        start_idx = None
        
        for i in range(len(valid_array)):
            if valid_array[i] and start_idx is None:
                start_idx = i
            elif not valid_array[i] and start_idx is not None:
                valid_periods.append((start_idx, i-1))
                start_idx = None
        
        if start_idx is not None:
            valid_periods.append((start_idx, len(valid_array)-1))
            
        return state, valid_periods

    def _is_static_track(self, obj_id, obj_track, state):
        """Whether this actor stands still for the whole clip -- a static agent.

        spawn_agent makes a static agent out of it, and the sector replay clock leaves it
        on the log clock. Both read this one decision.
        """
        # Parked cars (valid throughout, zero motion) are not handed to IDM; like cones and
        # barriers they are pinned to the log pose as static agents. This is the only exception
        # to the "all vehicles dynamic under R" rule below.
        if obj_id in self._static_pinned_vehicle_ids:
            return True
        # Replay-controlled pedestrians/bicycles must stay dynamic even when their
        # source displacement is small; only dynamic agents advance pose per frame.
        source_replayed = _is_source_replayed_actor(obj_track)
        if not source_replayed and not OdysseyObjectType.is_participant(obj_track['type']):
            return True
        valid_points = state[SD.POSITION][np.where(state[SD.VALID])]
        # nuPlan's reference builder treats every VEHICLE on a lane as a smart IDM agent,
        # including a car which barely moved in the source clip. Classifying that same proxy
        # as static here makes it replay GT while the shared IDM batch propagates a different,
        # lane-snapped pose for the same token. Other IDM agents avoid the internal pose and
        # can drive through the visible/PDM-observed one. Keep all vehicles dynamic for this
        # policy; genuinely unroutable vehicles still receive the batch's GT-fallback pose.
        nuplan_idm_vehicle = _nuplan_idm_controls_vehicle(self.engine.global_config, obj_track)
        moving = (source_replayed or nuplan_idm_vehicle
                  or np.max(np.std(valid_points, axis=0)[:2]) > self.STATIC_THRESHOLD
                  or obj_id == 'ego')
        return not moving

    def _spawn_gate_settings(self):
        """(check for overlap?, drop permanently on a hit?) -- **both policies read the same values.**

        Reading them in one place keeps NR and R in the same world: if only one policy read them,
        gate=false would let overlapping spawns in under one arm while the other kept dropping
        them. Both read gate (default false) and action (default drop) from here.

        defer exists only in trajectory_policy (the deferred set and retry path live only there).
        If R is asked for it, fail instead of silently running drop -- code moved to remove a
        silent asymmetry must not create a new one.
        """
        cfg = self.engine.global_config
        gate = bool(cfg.get('trajectory_spawn_ego_overlap_gate', False))
        # What an overlapping spawn costs the actor: the rollout ('drop', the default,
        # it is gone for good) or only its turn ('defer', reconsidered every step).
        drops = str(cfg.get('trajectory_spawn_ego_overlap_action', 'drop')) != 'defer'
        if not drops and cfg.agent_policy == 'nuplan_idm_policy':
            raise ValueError(
                "trajectory_spawn_ego_overlap_action=defer is not implemented for "
                "nuplan_idm_policy; use drop or run with agent_policy=trajectory_policy")
        return gate, drops

    def _survives_source_end(self, obj_id):
        """Whether this is a parked car -- an actor that must stay in place after its source ends.

        Source end means "the log stopped observing", not that the car vanished. The simulation
        runs longer than the log by horizon_extension_factor (renderer engine._tick_index), so
        removing it would leave the road beyond the log length empty -- in model input and in
        scoring alike. R (nuplan_idm_policy) already kept parked cars (despawn branch below); only
        NR removed them.

        Only tracks valid in every frame enter this set (_is_pinned_static_vehicle), so inside the
        log window should_be_active is always True. This exception therefore applies **only outside
        the log**, and existing behaviour inside the window does not change by a single step.
        """
        return obj_id in self._parked_vehicle_ids and obj_id in self._static_agents

    def spawn_agent(self, obj_id, obj_track, traj_step=None):
        """
        Create and initialize an agent.

        `traj_step` seeds the actor's log row. It defaults to the engine's step, which
        keys the actor on the log's own clock; the hybrid replay clock passes the row
        ego progress calls for instead, so a mid-episode spawn appears in the pose the
        recording had at that point of the route rather than at that wall-clock time.
        """
        state, valid_periods = self._process_agent_track(obj_id, obj_track)

        source_replayed = _is_source_replayed_actor(obj_track)
        if obj_id in self._static_agents:
            is_static = True
        elif obj_id in self._dynamic_agents:
            is_static = False
        else:
            is_static = self._is_static_track(obj_id, obj_track, state)
        set_to_add = self._static_agents if is_static else self._dynamic_agents

        # for static agents, set agent_policy == 'trajectory_policy'.
        # ALSO keep the injected ADVERSARY on its scripted trajectory even when background agents are
        # reactive (agent_policy=nuplan_idm_policy), so background log-replay artifacts (a follower rear-ending
        # the planner-driven ego) go away while OUR model adversary still follows its generated path.
        try:
            adv_id = self.engine.managers['scenario_manager'].current_scene.get('adv_object_id')
        except Exception:
            adv_id = None
        if source_replayed or is_static or (adv_id is not None and obj_id == adv_id):
            static_config = self.engine.global_config.copy()
            static_config['agent_policy'] = 'trajectory_policy'
            static_config['agent_controller'] = 'log_play_controller'
            agent_config = static_config
        else:
            agent_config = self.engine.global_config

        # Spawn agent
        if obj_id == 'ego':
            set_to_add[obj_id] = EgoAgent(
                obj_id, state, name=obj_id, config=agent_config)
        else:
            step0 = self.engine.episode_step if traj_step is None else int(traj_step)
            set_to_add[obj_id] = BaseAgent(
                obj_id, state, name=obj_id, config=agent_config, traj_step=step0)

        # Initialize agent
        set_to_add[obj_id].reset()
        
        # Update valid_periods
        if obj_id not in self._agent_valid_periods:
            self._agent_valid_periods[obj_id] = valid_periods
            
        return state

    def reset(self):
        super(BaseAgentManager, self).reset()
        self._trajectory_buffer = {}
        self._agent_valid_periods = {}
        self._replay = None
        self._traffic_light_clock = None
        self._idm_lifecycle_clock = None
        self._idm_source_step = int(self.engine.episode_step)
        self._idm_spawn_clock = None
        self._idm_spawn_eligible = set()
        self._reactive_sector_rows = {}
        self._replay_rows, self._replay_rows_step = {}, None
        self._source_end_progress = {}
        self._source_end_stall_retirements = 0
        self._source_end_retained_agent_ticks = 0
        self._spawn_overlap_deferred = set()
        self._spawn_overlap_deferred_ever = set()
        self._spawn_overlap_dropped = set()
        self._spawn_overlap_deferral_ticks = 0
        self._spawn_gate_events = []
        self._parallel_runtime = {
            'steps': 0,
            'ego_planner_s': 0.0,
            'idm_s': 0.0,
            'critical_path_s': 0.0,
        }
        
        if self.engine.global_config.visualize_BEV:
            if self._BEV_vis is None:
                self._BEV_vis = BEV_visualizer()
            else:
                self._BEV_vis.clear()
        
        sim_length = self.engine.global_config.max_step + 1

        # Spawning, static classification and the sector clock all read these sets, so decide them
        # once beforehand. The classification (which cars are parked) is kept apart from the R
        # policy switch (whether to take those cars out of IDM and predrop them). The
        # classification is a fact about the track, so NR needs it too -- recomputing it every
        # step costs a ptp over all frames per actor, so it is built once here.
        self._parked_vehicle_ids = frozenset(
            obj_id for obj_id, obj_track in self.current_agent_data.items()
            if obj_id != 'ego' and _simulation_enabled(obj_track)
            and _is_pinned_static_vehicle(obj_track))
        self._static_pinned_vehicle_ids = (
            self._parked_vehicle_ids
            if _static_pin_and_predrop_enabled(self.engine.global_config) else frozenset())

        disabled = 0
        rendered = bool(self.engine.global_config.get('with_render_manager', False))
        pose_source = self.engine.current_scene.get('metadata', {}).get('actor_pose_source')
        skipped_unverified = 0
        for obj_id, obj_track in self.current_agent_data.items():
            if not _simulation_enabled(obj_track):
                disabled += 1
                continue
            if (rendered and pose_source != 'checkpoint'
                    and _is_source_replayed_actor(obj_track)):
                skipped_unverified += 1
                continue

            _, valid_periods = self._process_agent_track(obj_id, obj_track)
            self._agent_valid_periods[obj_id] = valid_periods
            
            should_spawn = (
                obj_id == 'ego' or  
                any(start == 0 for start, _ in valid_periods)  
            )
            
            if should_spawn:
                self.spawn_agent(obj_id, obj_track)
                logger.debug(f"Spawned agent {obj_id} at reset")

            self._trajectory_buffer[obj_id] = {
                "type": obj_track.get('type'),
                "metadata": {
                    "track_length": sim_length,
                    "type": obj_track.get('type'),
                    "object_id": obj_id
                },
                "state": {
                    "position": np.zeros([sim_length, 2], dtype=np.float32),
                    "heading": np.zeros([sim_length, 1], dtype=np.float32),
                    "velocity": np.zeros([sim_length, 1], dtype=np.float32),
                    "valid": np.zeros([sim_length, 1], dtype=np.float32),
                    "length": np.zeros([sim_length, 1], dtype=np.float32),
                    "width": np.zeros([sim_length, 1], dtype=np.float32),
                    "height": np.zeros([sim_length, 1], dtype=np.float32),
                }
            }

        if disabled:
            logger.info(
                'Retained %d simulation-disabled checkpoint actors in the PKL; '
                'BaseAgentManager did not materialize them', disabled)
        if skipped_unverified:
            logger.warning(
                'Skipped %d replay actors: rendered scene has no verified checkpoint '
                'actor poses; rebuild with --actor-pose checkpoint', skipped_unverified)

        replay_mode = str(self.engine.global_config.get('agent_replay', 'absolute'))
        if replay_mode not in {'absolute', 'hybrid', 'sector', 'sector_lead'}:
            raise ValueError(
                'agent_replay must be absolute, hybrid, sector or sector_lead, got '
                f'{replay_mode!r}'
            )
        if replay_mode != 'absolute':
            self._replay = self._build_replay_clock(replay_mode)
            if isinstance(self._replay, SectorReplay):
                self._traffic_light_clock = self._replay

        lifecycle_clock = str(
            self.engine.global_config.get('nuplan_idm_lifecycle_clock', 'time')
        )
        if lifecycle_clock not in {'time', 'ego_progress', 'sector'}:
            raise ValueError(
                'nuplan_idm_lifecycle_clock must be time, ego_progress or sector, got '
                f'{lifecycle_clock!r}'
            )
        self._idm_lifecycle_mode = lifecycle_clock
        if (
            self.engine.global_config.agent_policy == 'nuplan_idm_policy'
            and lifecycle_clock == 'ego_progress'
        ):
            self._idm_lifecycle_clock = self._build_idm_lifecycle_clock()
            self._update_idm_source_step(int(self.engine.episode_step))

        spawn_eligibility = str(self.engine.global_config.get(
            'nuplan_idm_spawn_eligibility', 'source'
        ))
        if spawn_eligibility not in {'source', 'intersection_sector'}:
            raise ValueError(
                'nuplan_idm_spawn_eligibility must be source or intersection_sector, got '
                f'{spawn_eligibility!r}'
            )
        if (self.engine.global_config.agent_policy == 'nuplan_idm_policy'
                and lifecycle_clock == 'sector'
                and spawn_eligibility != 'intersection_sector'):
            raise ValueError(
                'nuplan_idm_lifecycle_clock=sector requires '
                'nuplan_idm_spawn_eligibility=intersection_sector'
            )
        if (
            self.engine.global_config.agent_policy == 'nuplan_idm_policy'
            and spawn_eligibility == 'intersection_sector'
        ):
            self._idm_spawn_clock = self._build_idm_spawn_clock()
            self._traffic_light_clock = self._idm_spawn_clock
            self._update_idm_spawn_eligibility(int(self.engine.episode_step))

        # Align Odyssey's initial proxy population with nuPlan's own builder before the ego's
        # first observation. Preparing step 0 has a zero time span, so it does not propagate any
        # vehicle; it only routes/admission-checks the t=0 population. Without this, a builder-
        # refused overlapping vehicle is visible to PDM for one step and removed only afterwards.
        if self.engine.global_config.agent_policy == 'nuplan_idm_policy':
            from odyssey.components.agents.policy.nuplan_idm_policy import _batch
            nuplan_batch = _batch(self.engine, self.engine.global_config)
            nuplan_batch.prepare_step(int(self.engine.episode_step))
            self._drop_nuplan_predropped(nuplan_batch, int(self.engine.episode_step))
            self._prune_unmaterialized_nuplan_proxies(
                nuplan_batch, int(self.engine.episode_step))
            for obj_id in list(self.all_agents):
                self._sync_proxy_to_nuplan_pose(
                    nuplan_batch, obj_id, int(self.engine.episode_step))
            self._reconcile_initial_publication(
                nuplan_batch, int(self.engine.episode_step))

    def _reconcile_initial_publication(self, nuplan_batch, current_step):
        """Apply sector presence and the ego spawn gate before frame zero is observable."""
        rows = dict(getattr(self, '_reactive_sector_rows', {}))
        self._write_replay_rows({
            token: row for token, row in rows.items()
            if self._uses_reactive_sector_replay(token)
        })
        for obj_id in list(self.all_agents):
            if obj_id == 'ego':
                continue
            source_row = self.object_source_row(obj_id, current_step)
            if self._uses_reactive_sector_replay(obj_id) and source_row < 0:
                for store in (self._dynamic_agents, self._static_agents):
                    agent = store.pop(obj_id, None)
                    if agent is not None:
                        agent.destroy()
                continue
            conflict = self._spawn_conflict_with_ego(obj_id, source_row)
            if conflict is None:
                continue
            self._drop_spawn(obj_id, current_step, source_row, conflict)
            for store in (self._dynamic_agents, self._static_agents):
                agent = store.pop(obj_id, None)
                if agent is not None:
                    agent.destroy()
            remove = getattr(nuplan_batch, 'remove_agent', None)
            if remove is not None:
                remove(obj_id)

    def _write_replay_rows(self, rows):
        """Put each actor on the log row the replay clock gave it.

        A negative row is the clock saying the actor is not on the road -- it has no row.
        Clamping that to 0 hands it the log's first row instead, and an actor absent at
        row 0 has a zero-filled pose there, so it materialises at the local origin for a
        frame before presence destroys it. Measured on one 100 s sector run: 520 such
        appearances, up to 7 in a single frame, and one of them vanishing at each sector
        boundary as the row stopped being negative. Presence already reads the same rows
        and drops anything below zero, so leaving the row alone is enough.
        """
        for obj_id, row in rows.items():
            if obj_id == 'ego' or row < 0:
                continue
            agent = self.all_agents.get(obj_id)
            if agent is not None:
                agent.set_traj_step(int(round(row)))

    def _intersection_polygons(self):
        """Polygons of the lanes inside intersections.

        In nuPlan, the lanes of a ROADBLOCK_CONNECTOR are the connecting lanes through an
        intersection, and the scene builder marks them LANE_SURFACE_UNSTRUCTURE (the layer branch
        in nuplan_utils.py). They are in the same local frame as the ego track.

        The nuPlan map has a separate INTERSECTION layer, and the scene builder queries it
        (nuplan_utils.py:338) but does not store it in map_features -- it is only unioned with the
        block polygons to build road boundaries and then discarded (:477-479). So scene PKLs have
        no intersection polygons, and using them would require re-extracting the scenes. The
        connecting lanes are already there, and the ego actually drives on them when crossing an
        intersection, so they are the more direct choice for a crossing test.
        """
        scene = self.engine.current_scene
        polys = [feature['polygon']
                 for feature in scene.get(SD.MAP_FEATURES, {}).values()
                 if feature.get('type') == OdysseyObjectType.LANE_SURFACE_UNSTRUCTURE
                 and feature.get('polygon') is not None]
        if not polys:
            raise ValueError(
                'sector_by=inter but this scene has no intersection lanes in map_features -- '
                'there is nothing to place boundaries on, so it does not silently fall back to time')
        return polys

    def _traffic_light_positions(self):
        """Connector representative points from the scenario PKL, in route coordinates."""
        signals = {}
        for light in self.engine.current_scene.get(SD.DYNAMIC_MAP_STATES, {}).values():
            if light.get('type') != OdysseyObjectType.TRAFFIC_LIGHT:
                continue
            connector_id = light.get('traffic_light_lane')
            position = light.get(SD.TRAFFIC_LIGHT_POSITION)
            if connector_id is None or position is None:
                continue
            connector_id = str(connector_id)
            point = np.asarray(position, dtype=np.float64).reshape(-1)
            if len(point) < 2 or not np.all(np.isfinite(point[:2])):
                continue
            point = point[:2]
            previous = signals.get(connector_id)
            if previous is not None and not np.allclose(previous, point):
                raise ValueError(
                    f'conflicting traffic-light positions for connector {connector_id}'
                )
            signals[connector_id] = point
        return signals

    def _build_replay_clock(self, mode):
        """Key the background on ego route progress rather than on the log's clock.

        `mode` picks how the offset moves with that progress: `hybrid` follows it
        continuously, `sector` holds it fixed between route sectors. Returns None -- and
        so leaves today's behaviour alone -- if the scene has no ego track to anchor
        progress on.
        """
        cfg = self.engine.global_config
        tracks = {}
        static = set()
        route_xy = route_heading = None
        for obj_id, obj_track in self.current_agent_data.items():
            if not _simulation_enabled(obj_track):
                continue
            state, _ = self._process_agent_track(obj_id, obj_track)
            valid = np.where(state[SD.VALID])[0]
            if len(valid) < 2:
                continue
            if obj_id == 'ego':
                route_xy = state[SD.POSITION][:, :2]
                route_heading = np.asarray(state[SD.HEADING]).reshape(-1)
                continue
            tracks[obj_id] = (valid, state[SD.POSITION][valid, :2])
            if self._is_static_track(obj_id, obj_track, state):
                static.add(obj_id)

        if route_xy is None or not tracks:
            logger.warning('agent_replay=%s: no ego track to anchor progress on; '
                           'falling back to the log clock', mode)
            return None

        if mode in ('sector', 'sector_lead'):
            # The sector clock never freezes with the ego, so it has no use for the
            # ambient/interactive split the replay_* knobs tune.
            split = str(cfg.get('sector_by', 'time') or 'time')
            if split not in ('time', 'inter'):
                raise ValueError(f'sector_by must be time or inter, got {split!r}')
            # `x or default` swallows 0 -- 0.0 or 15.0 is 15.0. Submitting sector_lead_m=0
            # ("appear the moment the boundary is reached", the baseline for lead spawning) that
            # way silently runs as 15 and scores exactly like 15; an experiment was lost that way.
            # Only `is None` separates an empty value from 0.
            def _knob(key, default):
                v = cfg.get(key, None)
                return float(default) if v is None else float(v)

            # Stationary objects (parked cars, cones) are kept out of sectors. The original viewer
            # draws them as static_always for the whole clip, independent of clocks and triggers.
            # Putting them in sectors makes parked cars appear all at once when the ego crosses a
            # spawn line -- in one measured run, 34 of the 57 cars created at spawn lines had never
            # moved. Actors without a clock run on the log clock (_presence_step's default).
            clock = SectorReplay(
                route_xy, {k: v for k, v in tracks.items() if k not in static},
                signals=self._traffic_light_positions(),
                sector_len_s=_knob('sector_len_s', hybrid_replay.SECTOR_LEN_S),
                lead_m=(_knob('sector_lead_m', hybrid_replay.SECTOR_LEAD_M)
                        if mode == 'sector_lead' else 0.0),
                intersections=(self._intersection_polygons() if split == 'inter' else None),
                inter_min_m=_knob('sector_inter_min_m', hybrid_replay.SECTOR_INTER_MIN_M),
                inter_approach_m=_knob('sector_inter_approach_m',
                                       hybrid_replay.SECTOR_INTER_APPROACH_M),
                src_dt=self.engine.sim_dt)
            return clock
        return HybridReplay(
            route_xy, route_heading, tracks,
            tau_interact=cfg.get('replay_tau_interact', None) or hybrid_replay.TAU_INTERACT,
            static_min=cfg.get('replay_static_min', None) or hybrid_replay.STATIC_MIN,
            pet_s=cfg.get('replay_pet_s', None) or hybrid_replay.PET_S,
            pet_r=cfg.get('replay_pet_r', None) or hybrid_replay.PET_R,
            src_dt=self.engine.sim_dt)

    def _build_idm_lifecycle_clock(self):
        """Build the IDM source clock exclusively from the scenario PKL ego track."""
        ego_track = self.current_agent_data.get(self.sdc_track_id)
        if ego_track is None:
            raise ValueError('ego-progress IDM lifecycle requires an ego track in scenario PKL')
        state, _ = self._process_agent_track(self.sdc_track_id, ego_track)
        valid = np.flatnonzero(np.asarray(state[SD.VALID]).reshape(-1))
        if len(valid) < 2:
            raise ValueError('ego-progress IDM lifecycle requires two valid ego PKL rows')
        clock = EgoProgressClock(state[SD.POSITION][valid, :2], rows=valid)
        logger.info(
            '[NUPLAN-IDM] lifecycle clock=ego_progress source=scenario_pkl rows=%d',
            len(valid),
        )
        return clock

    def _build_idm_spawn_clock(self):
        """Build the shared intersection-sector clock for reactive simulation.

        Geometry, ego route and complete valid actor trajectories come from the scenario PKL.
        IDM vehicles use the opening time plus their own valid source duration;
        vulnerable actors and rejected RigidNodes consume the shifted sector row;
        signal connectors consume the sector's shared delta.
        """
        ego_track = self.current_agent_data.get(self.sdc_track_id)
        if ego_track is None:
            raise ValueError('intersection-sector IDM spawn requires an ego scenario-PKL track')
        ego_state, _ = self._process_agent_track(self.sdc_track_id, ego_track)
        ego_valid = np.flatnonzero(np.asarray(ego_state[SD.VALID]).reshape(-1))
        if len(ego_valid) < 2:
            raise ValueError('intersection-sector IDM spawn requires two valid ego PKL rows')

        tracks = {}
        for obj_id, obj_track in self.current_agent_data.items():
            if (
                obj_id == self.sdc_track_id
                or not _simulation_enabled(obj_track)
                or not OdysseyObjectType.is_participant(obj_track.get('type'))
                # Parked cars stay out of sectors, like NR's stationary objects -- they stand for
                # the whole clip on the log clock.
                or obj_id in self._static_pinned_vehicle_ids
            ):
                continue
            state, _ = self._process_agent_track(obj_id, obj_track)
            valid = np.flatnonzero(np.asarray(state[SD.VALID]).reshape(-1))
            if len(valid) == 0:
                continue
            tracks[str(obj_id)] = (valid, state[SD.POSITION][valid, :2])
        if not tracks:
            raise ValueError('intersection-sector reactive replay found no dynamic actor tracks')

        cfg = self.engine.global_config
        lead = cfg.get('nuplan_idm_intersection_spawn_lead_m', None)
        if lead is None:
            lead = cfg.get('sector_lead_m', None) or hybrid_replay.SECTOR_LEAD_M
        clock = SectorReplay(
            ego_state[SD.POSITION][ego_valid, :2],
            tracks,
            signals=self._traffic_light_positions(),
            lead_m=float(lead),
            intersections=self._intersection_polygons(),
            inter_min_m=(cfg.get('sector_inter_min_m', None)
                         or hybrid_replay.SECTOR_INTER_MIN_M),
            inter_approach_m=(cfg.get('sector_inter_approach_m', None)
                              or hybrid_replay.SECTOR_INTER_APPROACH_M),
            src_dt=self.engine.sim_dt,
        )
        logger.info(
            '[NUPLAN-IDM] shared reactive clock=intersection_sector source=scenario_pkl '
            'actors=%d sectors=%d lead=%.1fm',
            len(tracks), len(clock.bound_rows), clock.lead_m,
        )
        return clock

    def _update_idm_spawn_eligibility(self, simulation_step):
        """Open sector spawn gates from monotone live-ego progress."""
        clock = getattr(self, '_idm_spawn_clock', None)
        if clock is None:
            self._idm_spawn_eligible = set()
            return
        ego = self._dynamic_agents.get('ego')
        if ego is None:
            raise RuntimeError('intersection-sector IDM spawn has no materialized ego')
        previous = set(self._idm_spawn_eligible)
        self._reactive_sector_rows = clock.step(
            int(simulation_step), ego.current_position[:2])
        eligible = {
            token for token in clock.sector if clock.is_eligible(token)
        }
        opened = eligible - previous
        self._idm_spawn_eligible = eligible
        if opened:
            logger.info(
                '[NUPLAN-IDM] opened intersection-sector spawn gate for %d vehicle(s) '
                'at tick=%d (eligible=%d/%d)',
                len(opened), int(simulation_step), len(eligible), len(clock.sector),
            )

    def _update_idm_source_step(self, simulation_step):
        """Update the lifecycle row; traffic-light time remains ``simulation_step``."""
        if getattr(self, '_idm_lifecycle_clock', None) is None:
            self._idm_source_step = int(simulation_step)
            return self._idm_source_step
        ego = self._dynamic_agents.get('ego')
        if ego is None:
            raise RuntimeError('ego-progress IDM lifecycle has no materialized ego')
        self._idm_source_step = max(
            0,
            int(round(self._idm_lifecycle_clock.source_row(ego.current_position[:2]))),
        )
        return self._idm_source_step

    def _presence_step(self, obj_id, simulation_step, replay_rows):
        """Source row controlling this actor's spawn/despawn eligibility."""
        if self._uses_reactive_sector_replay(obj_id):
            return int(round(self._reactive_sector_rows.get(obj_id, -1)))
        if self._uses_idm_sector_lifecycle(obj_id):
            return self._idm_spawn_clock.actor_lifecycle_row(obj_id, simulation_step)
        if (
            getattr(self, '_idm_lifecycle_clock', None) is not None
            and _nuplan_idm_controls_vehicle(
                self.engine.global_config, self.current_agent_data[obj_id]
            )
        ):
            return self._idm_source_step
        return int(round(replay_rows.get(obj_id, simulation_step)))

    def _uses_reactive_sector_replay(self, obj_id):
        """Whether this actor is executed from the shared sector-shifted source track."""
        if (getattr(self, '_idm_spawn_clock', None) is None or obj_id == 'ego'
                or obj_id in self._static_pinned_vehicle_ids):
            return False
        track = self.current_agent_data.get(obj_id, {})
        if _is_source_replayed_actor(track):
            return True
        batch = getattr(self.engine, '_nuplan_idm_batch', None)
        is_gt_replay = getattr(batch, 'is_gt_replay_vehicle', None)
        return bool(is_gt_replay is not None and is_gt_replay(obj_id))

    def _uses_idm_sector_lifecycle(self, obj_id):
        """Whether this R vehicle uses sector-open time plus its source duration."""
        if (getattr(self, '_idm_lifecycle_mode', 'time') != 'sector'
                or getattr(self, '_idm_spawn_clock', None) is None
                or obj_id == 'ego'
                or obj_id in self._static_pinned_vehicle_ids
                or self._uses_reactive_sector_replay(obj_id)):
            return False
        track = self.current_agent_data.get(obj_id)
        return bool(track is not None
                    and _nuplan_idm_controls_vehicle(self.engine.global_config, track))

    def object_source_row(self, obj_id, simulation_step):
        """Exact source row used for an open-loop object in the current execution."""
        if str(obj_id) in getattr(self, '_spawn_overlap_dropped', set()):
            return -1
        if self._uses_reactive_sector_replay(str(obj_id)):
            return int(round(self._reactive_sector_rows.get(str(obj_id), -1)))
        if self._uses_idm_sector_lifecycle(str(obj_id)):
            return self._idm_spawn_clock.actor_lifecycle_row(str(obj_id), simulation_step)
        return int(simulation_step)

    def source_rows(self, simulation_step):
        """{track id: log row} for every non-ego track whose replay row is set at this step.

        For export_source_rows (DataManager writes it into each planner frame): a planner-side
        oracle has to know where the open-loop actors will be, and only this manager knows
        which row each one is on. It is the row whose pose the actor shows in this frame: the
        presence row (_presence_step) for an actor not on the road yet, and for a replayed
        actor its traj_step, which its step() has already moved one past the clock's row
        (the controller drives it to the next logged pose). The track is on the road iff its
        scenario-PKL ``valid`` is 1 there. Once set, a row advances one per step -- the log
        clock does, and so does a sector's (SectorReplay) -- so the actor's future is row + j. A track left out has no row: spawn-dropped or
        deferred, or in a sector the ego has not opened yet, which is exactly as unknown to the
        simulator.

        Only for clocks that advance one row per step. HybridReplay and the ego-progress IDM
        lifecycle clock follow the ego's progress (a stopped ego freezes them), so row + j would
        be a wrong future -- they raise instead.
        """
        if (self._replay is not None and not isinstance(self._replay, SectorReplay)) \
                or getattr(self, '_idm_lifecycle_clock', None) is not None:
            raise RuntimeError(
                'source_rows needs a replay clock that advances one row per step (absolute, '
                'sector, sector_lead); this run follows the ego\'s progress')
        if self._replay_rows_step not in (None, simulation_step):
            raise RuntimeError(
                f'source_rows({simulation_step}) after the replay rows of step '
                f'{self._replay_rows_step}; the frame would describe another step')
        # Before the first step() only reset() has run, which spawns on the log clock.
        rows = self._replay_rows if self._replay_rows_step == simulation_step else {}
        dropped = self.spawn_deferred_tokens
        out = {}
        for obj_id in self._agent_valid_periods:
            if obj_id == 'ego' or obj_id in dropped:
                continue
            row = self._presence_step(obj_id, simulation_step, rows)
            agent = self._dynamic_agents.get(obj_id)
            if agent is not None and row >= 0:
                row = agent.traj_step
            if row >= 0:
                out[str(obj_id)] = int(row)
        return out

    def _spawn_conflict_with_ego(self, obj_id, row):
        """Would this actor enter the world already intersecting the ego's footprint?

        The log decided when this actor appears, but it decided that against the logged
        ego. Ours is model-driven, so the pair of poses the two sources produce together
        can be one that never occurred and, without physics, cannot resolve: the actor
        materializes inside the ego and the scorer charges the ego for the contact.

        Overlap alone -- no clearance, braking or crossing envelope. "Too close" is a
        judgement the log never made, and making it here would rewrite the recording;
        "already inside" is not a judgement, it is a pose that cannot exist.
        """
        ego = self._dynamic_agents.get('ego')
        if ego is None:
            return None
        track = self.current_agent_data[obj_id][SD.STATE]
        positions = track[SD.POSITION]
        idx = int(np.clip(int(row), 0, len(positions) - 1))
        actor = _footprint(positions[idx][:2],
                           float(np.asarray(track[SD.HEADING]).reshape(-1)[idx]),
                           float(np.asarray(track['length']).reshape(-1)[idx]),
                           float(np.asarray(track['width']).reshape(-1)[idx]))
        # current_position is the box centre; rear_vehicle carries the rear axle.
        ego_box = _footprint(ego.current_position[:2], float(ego.current_heading),
                             float(ego.length), float(ego.width))
        if actor.intersection(ego_box).area > self.SPAWN_OVERLAP_AREA_M2:
            return 'overlap'

        cfg = self.engine.global_config
        if not bool(cfg.get('spawn_ego_tight_ahead_gate', False)):
            return None
        delta = np.asarray(positions[idx][:2], dtype=float) - np.asarray(
            ego.current_position[:2], dtype=float)
        forward = np.array([np.cos(ego.current_heading), np.sin(ego.current_heading)])
        lateral = np.array([-forward[1], forward[0]])
        longitudinal = float(delta @ forward)
        lateral_distance = abs(float(delta @ lateral))
        actor_length = float(np.asarray(track['length']).reshape(-1)[idx])
        actor_width = float(np.asarray(track['width']).reshape(-1)[idx])
        bumper_gap = longitudinal - (float(ego.length) + actor_length) / 2.0
        lateral_limit = ((float(ego.width) + actor_width) / 2.0
                         + float(cfg.get('spawn_ego_tight_ahead_lateral_margin_m', 0.5)))
        if (longitudinal > 0.0
                and bumper_gap < float(cfg.get('spawn_ego_tight_ahead_gap_m', 5.0))
                and lateral_distance <= lateral_limit):
            return 'tight_ahead'
        return None

    def _drop_spawn(self, obj_id, current_step, source_row, reason):
        """Permanently reject one impossible spawn and archive the exact decision."""
        if not hasattr(self, '_spawn_overlap_dropped'):
            self._spawn_overlap_dropped = set()
        if not hasattr(self, '_spawn_overlap_deferred_ever'):
            self._spawn_overlap_deferred_ever = set()
        self._spawn_overlap_dropped.add(str(obj_id))
        self._spawn_overlap_deferred_ever.add(str(obj_id))
        if not hasattr(self, '_spawn_gate_events'):
            self._spawn_gate_events = []
        self._spawn_gate_events.append({
            'simulation_step': int(current_step),
            'token': str(obj_id),
            'source_row': int(source_row),
            'decision': 'drop',
            'reason': str(reason),
        })
        logger.info('Dropped agent %s at step %s row=%s: spawn_%s',
                    obj_id, current_step, source_row, reason)

    @property
    def spawn_deferred_tokens(self):
        """Actors kept out of the world because their spawn pose overlaps the ego.

        They are absent from `all_agents`, so simulation and scoring already do not see
        them. The renderer draws actors from checkpoint poses rather than from this
        manager, so it would still draw them; it is handed the same set instead, and one
        decision keeps one meaning across all three.

        Both outcomes are in it. Under `defer` a token is here only while it is waiting;
        under `drop` it never leaves, because a dropped actor stays out for the rest of
        the rollout and would otherwise reappear in the image alone.
        """
        return frozenset(self._spawn_overlap_deferred | self._spawn_overlap_dropped)

    @property
    def spawn_overlap_stats(self):
        """Spawn-overlap gate counts for the rollout record.

        `deferred_actors` is every actor the gate stopped at least once, whichever action
        was taken; `dropped_actors` is how many of those never made it in. Under the
        default `drop` the two are equal by construction, and the difference under
        `defer` is what waiting recovered.
        """
        return {
            'deferral_ticks': int(self._spawn_overlap_deferral_ticks),
            'deferred_actors': len(self._spawn_overlap_deferred_ever),
            'dropped_actors': len(self._spawn_overlap_dropped),
            # The tokens, not just how many: an actor the gate kept out leaves no box in the
            # rollout, so the only way to say afterwards WHICH one is missing is to name it
            # here. Sorted so the record does not reorder between otherwise equal runs.
            'dropped_tokens': sorted(str(t) for t in self._spawn_overlap_dropped),
        }

    @property
    def spawn_gate_events(self):
        """Immutable copy of every permanent spawn rejection in execution order."""
        return tuple(dict(event) for event in getattr(self, '_spawn_gate_events', ()))

    @property
    def idm_source_step(self):
        """Scenario-PKL row used by the current IDM lifecycle tick."""
        return int(self._idm_source_step)

    def is_idm_spawn_eligible(self, obj_id):
        """Whether nuPlan may attempt this token's first admission this tick."""
        if getattr(self, '_idm_spawn_clock', None) is None:
            return True
        return str(obj_id) in self._idm_spawn_eligible

    def traffic_light_source_row(self, connector_id, simulation_step):
        """Return the scenario-PKL signal row exposed at one simulation step.

        Sector replay uses the same released cohort clock for traffic lights and actors.
        Outside a sector mode this is the identity mapping.  An unassigned connector or a
        sector which had not opened at the queried historical step returns ``-1`` and is
        therefore observed as UNKNOWN by every consumer.
        """
        clock = getattr(self, '_traffic_light_clock', None)
        if clock is None:
            return int(simulation_step)
        return int(round(clock.signal_row(str(connector_id), int(simulation_step))))

    def traffic_light_spawn_row(self, connector_id):
        """Scenario-PKL row at which this connector's sector opens (its spawn-line row), or None.

        None with the identity clock (no sector) or for an unassigned connector. Used only by the
        signal-patch renderer rule "row -1 draws the spawn-row colour" (render_manager); scoring
        and planning keep ``traffic_light_source_row`` (-1 = UNKNOWN)."""
        clock = getattr(self, '_traffic_light_clock', None)
        if clock is None:
            return None
        sector = (getattr(clock, 'signal_sector', None) or {}).get(str(connector_id))
        rows = getattr(clock, 'spawn_rows', None)
        if sector is None or rows is None or sector >= len(rows):
            return None
        return int(round(float(rows[sector])))

    def _get_current_lane(self, obj_id):
        """
        Used to find current lane information.
        """
        agent = self._dynamic_agents[obj_id]
        map = self.engine.current_map
        possible_lanes = map.road_network.get_closest_lane_index(
            agent.current_position, return_all=True)
        possible_lanes = possible_lanes[:3] # extract the most possible 3 lanes
    
        min_heading_diff = float('inf')
        best_lane = None
        
        for lane_info in possible_lanes:
            lane = lane_info[2]
            long, _ = lane.local_coordinates(agent.current_position)
            lane_heading = lane.heading_theta_at(long)
            heading_diff = abs(agent.current_heading - lane_heading)
            
            if heading_diff < min_heading_diff:
                min_heading_diff = heading_diff
                best_lane = lane
        
        if best_lane is None:
            return None
        
        return best_lane
    
    def get_surrounding_agents(self, obj):
        """
        Get the agents in front of and behind the given agent.
        
        Args:
            obj: The agent.
        
        Returns:
            tuple: A tuple containing four lists:
                - List of IDs of agents in front of the given agent.
                - List of distances of agents in front of the given agent.
                - List of IDs of agents behind the given agent.
                - List of distances of agents behind the given agent.
        """

        if obj.id not in self.all_agents:
            raise ValueError(f"Agent with ID {obj.id} not found in dynamic agents.")
        
        agent = self.all_agents[obj.id]
        agent_position = agent.current_position
        front_agents, front_distances, back_agents, back_distances = [], [], [], []
        
        for other_id, other_agent in self.all_agents.items():
            if other_id == obj.id:
                continue
            if other_agent.height == 0.0 or other_agent.width == 0.0 or other_agent.length == 0.0:
                continue
            ref_lane = agent.navigation.current_lane
            agent_long, agent_lat = ref_lane.local_coordinates(agent_position)
            other_position = other_agent.current_position
            other_long, other_lat = ref_lane.local_coordinates(other_position)
            distance = np.linalg.norm(np.array(agent_position) - np.array(other_position))
            heading_diff = abs(math_utils.wrap_to_pi(other_agent.current_heading - agent.current_heading))
            if abs(agent_lat - other_lat) > 2.0 or heading_diff > np.pi / 2:
                continue
            if other_long > agent_long:
                front_agents.append(other_id)
                front_distances.append(distance)
            else:
                back_agents.append(other_id)
                back_distances.append(distance)
        
        return front_agents, front_distances, back_agents, back_distances

    ##### Step Function #####
    def before_step(self, *args, **kwargs):
        for v in self._dynamic_agents.values():
            v.before_step(None)

    def step(self):
        current_step = self.engine.episode_step
        self._update_idm_source_step(current_step)
        self._update_idm_spawn_eligibility(current_step)

        # The replay clock decides which log row each actor should be showing before the
        # actors read it. Written onto the agents rather than passed to the policy, so
        # that presence -- spawn, despawn and the policy's own validity test -- all key
        # off the same row.
        rows = {}
        if self._replay is not None:
            ego = self.all_agents.get('ego')
            if ego is not None:
                rows = self._replay.step(current_step, ego.current_position[:2])
                self._write_replay_rows(rows)
        elif getattr(self, '_idm_spawn_clock', None) is not None:
            rows = dict(getattr(self, '_reactive_sector_rows', {}))
            self._write_replay_rows({
                token: row for token, row in rows.items()
                if self._uses_reactive_sector_replay(token)
            })
        self._replay_rows, self._replay_rows_step = rows, current_step

        # Despawn anything that has driven off the end of the map. IDMNavigation raises this
        # flag when the agent passes the last lane the scene's map extract holds -- there is no
        # successor lane to hand it, and the lane geometry extrapolates rather than clamping, so
        # left alone it would keep driving straight out of the world. Same treatment as a
        # log-replay agent whose valid period has ended: destroy and forget.
        for obj_id in [k for k, v in self._dynamic_agents.items()
                       if getattr(getattr(v, 'navigation', None), 'route_exhausted', False)]:
            agent = self._dynamic_agents[obj_id]
            remove_from_batch = getattr(agent.policy, 'remove_from_batch', None)
            if remove_from_batch is not None:
                remove_from_batch()
            agent.destroy()
            del self._dynamic_agents[obj_id]
            logger.info("Removed agent %s at step %s: drove past the edge of the scene's map",
                        obj_id, current_step)

        # A source track ending means the log stopped observing that vehicle; it does not mean
        # the closed-loop vehicle physically vanished. Keep nearby admitted IDM vehicles alive
        # so they cannot disappear in front of the ego, but retire source-ended vehicles once
        # they leave the configured safety bubble. remove_from_batch() makes this one-way, so a
        # retired vehicle cannot flicker back in at the distance boundary. The legacy switch
        # reproduces immediate source-end despawn. Open-loop objects still follow source validity.
        if self.engine.global_config.agent_policy == 'nuplan_idm_policy':
            nuplan_batch = getattr(self.engine, '_nuplan_idm_batch', None)
            despawn_idm_on_source_end = bool(
                self.engine.global_config.get('nuplan_idm_despawn_on_source_end', False))
            source_end_keep_radius = float(
                self.engine.global_config.get('nuplan_idm_source_end_keep_radius', 50.0))
            source_end_liveness = bool(
                self.engine.global_config.get('nuplan_idm_source_end_liveness_enabled', True))
            source_end_stall_timeout_s = float(
                self.engine.global_config.get('nuplan_idm_source_end_stall_timeout_s', 8.0))
            source_end_progress_epsilon_m = float(
                self.engine.global_config.get(
                    'nuplan_idm_source_end_progress_epsilon_m', 0.5))
            source_end_stall_ticks = None
            if source_end_liveness:
                if source_end_stall_timeout_s <= 0 or source_end_progress_epsilon_m <= 0:
                    raise ValueError(
                        'source-end stall timeout and progress epsilon must be positive')
                source_end_stall_ticks = max(
                    1,
                    int(round(
                        source_end_stall_timeout_s
                        / float(getattr(self.engine, 'sim_dt', 0.1))
                    )),
                )
            source_end_progress = getattr(self, '_source_end_progress', {})
            self._source_end_progress = source_end_progress
            ego = self._dynamic_agents.get('ego')
            for obj_id, valid_periods in self._agent_valid_periods.items():
                if obj_id == 'ego':
                    continue
                step_for_presence = self._presence_step(obj_id, current_step, rows)
                should_be_active = step_for_presence >= 0 and any(
                    start <= step_for_presence <= end for start, end in valid_periods)
                source_ended_dynamic = not should_be_active and obj_id in self._dynamic_agents
                # Hybrid replay rows intentionally pause with the ego.  That is correct for
                # ordinary reactive traffic, but a temporally corrupt track rejected from IDM
                # must not freeze forever at its last GT pose while the ego is stopped behind
                # it.  Such tracks are a narrow, precomputed exception and follow wall-clock
                # source validity instead of the route-relative replay row.
                requires_source_validity = getattr(
                    nuplan_batch, 'requires_source_validity', None
                )
                source_validity_limited = bool(
                    requires_source_validity is not None
                    and requires_source_validity(obj_id)
                )
                if source_validity_limited and obj_id in self._dynamic_agents:
                    valid = np.asarray(
                        self.current_agent_data[obj_id]['state']['valid']
                    ).reshape(-1)
                    validity_row = (
                        step_for_presence
                        if self._uses_idm_sector_lifecycle(obj_id) else current_step
                    )
                    source_ended_dynamic = not (
                        0 <= validity_row < len(valid) and bool(valid[validity_row])
                    )
                    if source_ended_dynamic:
                        nuplan_batch.expire_source_limited_track(obj_id)
                is_idm_vehicle = False
                outside_keep_radius = False
                stalled_source_end = False
                if source_ended_dynamic:
                    is_idm_vehicle = _nuplan_idm_controls_vehicle(
                        self.engine.global_config, self.current_agent_data[obj_id])
                    # Retention belongs only to vehicles actually propagated by IDM.  Vehicle
                    # tracks demoted/refused into GT replay keep their source lifecycle and are
                    # removed immediately at source end, just like replayed pedestrians and
                    # bicycles.  Otherwise the 50 m rule freezes their last GT pose near ego.
                    is_gt_replay_vehicle = getattr(
                        nuplan_batch, 'is_gt_replay_vehicle', None
                    )
                    if (
                        is_idm_vehicle
                        and is_gt_replay_vehicle is not None
                        and is_gt_replay_vehicle(obj_id)
                    ):
                        is_idm_vehicle = False
                    if is_idm_vehicle:
                        agent_position = np.asarray(
                            self._dynamic_agents[obj_id].current_position[:2], dtype=float)
                    if is_idm_vehicle and ego is not None:
                        ego_position = np.asarray(ego.current_position[:2], dtype=float)
                        outside_keep_radius = (
                            np.linalg.norm(agent_position - ego_position) > source_end_keep_radius)
                    if (is_idm_vehicle and not despawn_idm_on_source_end
                            and not outside_keep_radius and source_end_liveness):
                        first_tick, anchor = source_end_progress.get(
                            obj_id, (current_step, agent_position.copy()))
                        if np.linalg.norm(agent_position - anchor) >= source_end_progress_epsilon_m:
                            first_tick, anchor = current_step, agent_position.copy()
                        source_end_progress[obj_id] = (first_tick, anchor)
                        stalled_source_end = current_step - first_tick >= source_end_stall_ticks
                else:
                    source_end_progress.pop(obj_id, None)
                if (source_ended_dynamic
                        and (not is_idm_vehicle
                             or despawn_idm_on_source_end
                             or outside_keep_radius
                             or stalled_source_end
                             or source_validity_limited)):
                    agent = self._dynamic_agents[obj_id]
                    remove_from_batch = getattr(agent.policy, 'remove_from_batch', None)
                    if remove_from_batch is not None:
                        remove_from_batch()
                    agent.destroy()
                    del self._dynamic_agents[obj_id]
                    source_end_progress.pop(obj_id, None)
                    if stalled_source_end:
                        self._source_end_stall_retirements = getattr(
                            self, '_source_end_stall_retirements', 0) + 1
                    reason = ("non-IDM source ended" if not is_idm_vehicle
                              else "legacy immediate source-end despawn"
                              if despawn_idm_on_source_end
                              else (f"source ended and made <"
                                    f"{source_end_progress_epsilon_m:.2f} m progress for "
                                    f"{source_end_stall_timeout_s:.1f} s")
                              if stalled_source_end
                              else f"source ended and outside {source_end_keep_radius:.1f} m")
                    log = logger.info if stalled_source_end else logger.debug
                    log("Removed agent %s at step %s: %s", obj_id, current_step, reason)
                elif source_ended_dynamic and is_idm_vehicle:
                    self._source_end_retained_agent_ticks = getattr(
                        self, '_source_end_retained_agent_ticks', 0) + 1
                # Parked cars stay in place after source end (the log's last row) until the episode
                # ends -- treated like IDM cars kept by despawn_on_source_end=false, but not moving.
                if (not should_be_active and obj_id in self._static_agents
                        and obj_id not in self._static_pinned_vehicle_ids):
                    self._static_agents[obj_id].destroy()
                    del self._static_agents[obj_id]
                    logger.debug("Removed nuPlan fallback object %s at step %s: source ended",
                                 obj_id, current_step)

        parallel_ego_done = False
        if self._parallel_ego_idm_enabled():
            from odyssey.components.agents.policy.nuplan_idm_policy import _batch
            ego = self._dynamic_agents['ego']
            nuplan_batch = _batch(self.engine, self.engine.global_config)
            self._compute_ego_and_idm(ego, nuplan_batch, current_step)
            parallel_ego_done = True

        for obj_id, v in self.all_agents.items():
            # TODO: set action to None for now.
            if parallel_ego_done and obj_id == 'ego':
                continue
            v.step()

        overlap_gate, overlap_drops = self._spawn_gate_settings()

        if self.engine.global_config.agent_policy == 'trajectory_policy':
            # Rebuilt every step: this is the renderer's view of *this* tick's decision, not
            # a queue. Retry needs no memory -- an actor the gate held back is simply still
            # absent from all_agents next step, so the loop below reconsiders it.
            deferred_last_step = self._spawn_overlap_deferred
            self._spawn_overlap_deferred = set()
            for obj_id, valid_periods in self._agent_valid_periods.items():
                if obj_id == 'ego':
                    continue

                # On the hybrid clock an actor's presence follows the row it is being
                # replayed at, not the wall clock: ambient traffic appears where the
                # recording put it along the route, and an interactive actor whose
                # bundle has not been released yet is not on the road at all.
                step_for_presence = int(round(rows.get(obj_id, current_step)))
                if step_for_presence < 0:
                    should_be_active = False
                else:
                    should_be_active = any(start <= step_for_presence <= end
                                        for start, end in valid_periods)
                

                # Despawn and spawn are separate outcomes of the same presence test, so they
                # are written as one branch rather than an if/if/elif chain: the `elif` used
                # to bind to the static-agent removal above it, and any condition inserted
                # between them would have silently swallowed every spawn.
                if not should_be_active:
                    # Parked cars stay in place after the log ends (see _survives_source_end).
                    # Spawning is handled by the branch below, so actors whose window has not
                    # opened yet are not here.
                    if self._survives_source_end(obj_id):
                        continue
                    for store in (self._dynamic_agents, self._static_agents):
                        if obj_id in store:
                            store[obj_id].destroy()
                            del store[obj_id]
                            logger.debug(f"Removed agent {obj_id} at step {current_step}")
                    # Its window closed while it was still being held back, so it never
                    # appears at all. The log said it was there; the log also said the ego
                    # was somewhere else, and only one of those can be honoured.
                    if obj_id in deferred_last_step:
                        self._spawn_overlap_dropped.add(obj_id)
                    continue

                if obj_id in self.all_agents or current_step <= 0:
                    continue

                # Dropped once means dropped for the rollout: its first appearance was the
                # one the log placed, and it landed inside the ego. A later entry would be
                # a moment the recording never contains, so there is nothing to retry.
                if obj_id in self._spawn_overlap_dropped:
                    continue

                conflict = (self._spawn_conflict_with_ego(obj_id, step_for_presence)
                            if overlap_gate else None)
                if conflict is not None:
                    if overlap_drops:
                        self._drop_spawn(obj_id, current_step, step_for_presence, conflict)
                        continue
                    self._spawn_overlap_deferred.add(obj_id)
                    self._spawn_overlap_deferred_ever.add(obj_id)
                    self._spawn_overlap_deferral_ticks += 1
                    logger.debug("Deferred agent %s at step %s: spawn pose overlaps ego",
                                 obj_id, current_step)
                    continue

                # traj_step stays whatever the replay clock says, exactly as an undeferred
                # actor's would. Holding the row it was deferred from would enter it into a
                # past it never occupied, and the renderer now draws each actor at the row
                # the manager reports (RenderState.AGENT_SOURCE_ROW), so the stale row would
                # be drawn as well as scored -- consistent, and consistently wrong.
                self.spawn_agent(obj_id, self.current_agent_data[obj_id],
                                 traj_step=rows.get(obj_id))
                logger.debug(f"Spawned agent {obj_id} at step {current_step}")
        elif self.engine.global_config.agent_policy == 'nuplan_idm_policy':
            # The first existing traffic policy normally advances the shared batch above, after
            # ego has taken its established place in Odyssey's update order. If the scene has
            # no materialized traffic yet, advance it here so a late source track can still be
            # considered for admission. Do not pre-advance: using the previous ego pose changes
            # the fleet dynamics and is not part of nuPlan's reference behavior.
            from odyssey.components.agents.policy.nuplan_idm_policy import _batch
            nuplan_batch = _batch(self.engine, self.engine.global_config)
            nuplan_batch.prepare_step(current_step)
            self._drop_nuplan_predropped(nuplan_batch, current_step)

            # Initial-frame builder refusals can already have Odyssey proxies. Remove them
            # after the batch makes its decision; late tracks never enter all_agents until the
            # corresponding collision-safe pose exists.
            self._prune_unmaterialized_nuplan_proxies(nuplan_batch, current_step)

            for obj_id, valid_periods in self._agent_valid_periods.items():
                if (obj_id == 'ego' or obj_id in self.all_agents
                        or obj_id in getattr(self, '_spawn_overlap_dropped', set())):
                    continue
                step_for_presence = self._presence_step(obj_id, current_step, rows)
                should_be_active = step_for_presence >= 0 and any(
                    start <= step_for_presence <= end for start, end in valid_periods)
                source_replayed = self._is_source_replayed_token(obj_id)
                if should_be_active and (
                        source_replayed or nuplan_batch.can_materialize(obj_id)):
                    spawn_row = (
                        step_for_presence
                        if (getattr(self, '_idm_lifecycle_clock', None) is not None
                            or self._uses_idm_sector_lifecycle(obj_id))
                        and _nuplan_idm_controls_vehicle(
                            self.engine.global_config, self.current_agent_data[obj_id]
                        )
                        else rows.get(obj_id)
                    )
                    # Read the same gate as NR (see the block above). With the defaults it drops
                    # unconditionally, as before.
                    conflict = (self._spawn_conflict_with_ego(obj_id, step_for_presence)
                                if overlap_gate else None)
                    if conflict is not None:
                        self._drop_spawn(
                            obj_id, current_step, step_for_presence, conflict)
                        remove = getattr(nuplan_batch, 'remove_agent', None)
                        if remove is not None:
                            remove(obj_id)
                        continue
                    self.spawn_agent(obj_id, self.current_agent_data[obj_id],
                                     traj_step=spawn_row)
                    if not source_replayed:
                        self._sync_proxy_to_nuplan_pose(nuplan_batch, obj_id, current_step)
                    logger.debug("Materialized nuPlan IDM proxy %s at step %s",
                                 obj_id, current_step)
        else:
            for obj_id, valid_periods in self._agent_valid_periods.items():
                if obj_id == 'ego':
                    continue

                # An IDM actor drives itself, but it still has to be *put on the road*
                # somewhere, and on the log clock that happens at a wall-clock time the
                # ego may be nowhere near. With the hybrid clock it appears where the
                # recording first showed it along the route instead; from that pose IDM
                # takes over.
                step_for_presence = int(round(rows.get(obj_id, current_step)))
                should_be_active = step_for_presence >= 0 and any(
                    start <= step_for_presence <= end for start, end in valid_periods)

                if not should_be_active and obj_id in self._static_agents:
                    self._static_agents[obj_id].destroy()
                    del self._static_agents[obj_id]
                    logger.debug(f"Removed agent {obj_id} at step {current_step}")

                if should_be_active and obj_id not in self.all_agents:

                    if current_step > 0:
                        self.spawn_agent(obj_id, self.current_agent_data[obj_id],
                                         traj_step=rows.get(obj_id))
                        logger.debug(f"Spawned agent {obj_id} at step {current_step}")

    def _drop_nuplan_predropped(self, nuplan_batch, current_step):
        """Remove cars dropped by the IDM handoff / late-admission gate the same way as spawn drops.

        The batch already emits no pose for them, so they are out of simulation and scoring.
        Adding them to _drop_spawn here makes the renderer (SUPPRESSED_ACTORS), the source rows
        (-1) and the rollout record see the same decision. If a proxy already exists, the
        _prune_unmaterialized_nuplan_proxies call right after removes it.
        """
        for token, reason in getattr(nuplan_batch, 'predropped_vehicles', {}).items():
            if token not in getattr(self, '_spawn_overlap_dropped', set()):
                self._drop_spawn(token, current_step, current_step, f'predrop:{reason}')

    def _prune_unmaterialized_nuplan_proxies(self, nuplan_batch, current_step):
        """Remove refused vehicle proxies, not source-replayed people/bicycles."""
        for store in (self._dynamic_agents, self._static_agents):
            for obj_id in [
                    token for token in store
                    if token != 'ego'
                    and not self._is_source_replayed_token(token)
                    and not nuplan_batch.can_materialize(token)]:
                agent = store[obj_id]
                remove_from_batch = getattr(agent.policy, 'remove_from_batch', None)
                if remove_from_batch is not None:
                    remove_from_batch()
                agent.destroy()
                del store[obj_id]
                logger.debug("Deferred nuPlan IDM proxy %s at step %s",
                             obj_id, current_step)

    def _sync_proxy_to_nuplan_pose(self, nuplan_batch, obj_id, current_step):
        """Seed a newly visible proxy at nuPlan's snapped pose, not its raw source pose.

        The stock builder collision-checks ``box_on_baseline`` and the late-admission extension
        checks the corresponding projected footprint.  ``spawn_agent`` normally initializes a
        Odyssey object from the unsnapped source row.  Leaving that pose visible for one
        frame creates a representation-only collision even though the nuPlan agent was admitted
        safely.  Align only IDM-owned vehicles; official open-loop objects keep their logged
        pose exactly as nuPlan returns it.
        """
        if obj_id == 'ego' or nuplan_batch.source_mode_for(obj_id) != 'idm':
            return
        agent = self.all_agents.get(obj_id)
        pose = nuplan_batch.pose_for(current_step, obj_id)
        if agent is None or pose is None:
            return
        x, y, heading, speed = pose
        agent.set_position(np.array([x, y], dtype=np.float64))
        agent.set_heading_theta(float(heading))
        agent.set_velocity(
            float(speed) * np.array([np.cos(heading), np.sin(heading)], dtype=np.float64))
        agent.set_angular_velocity(0.0)
        # reset() refreshes last_position/heading/velocity after the correction so the first
        # controller step does not infer a source-to-rail teleport.
        agent.reset()
            
    def after_step(self, *args, **kwargs):
        for v in self._dynamic_agents.values():
            # TODO: set action to None for now.
            return_info = v.after_step()
            if return_info is not None:
                for key, value in return_info.items():
                    if key in self._trajectory_buffer[v.id]["state"]:
                        self._trajectory_buffer[v.id]["state"][key][self.engine.episode_step] = value
                        self._trajectory_buffer[v.id]["state"]["valid"][self.engine.episode_step] = 1

        if self.engine.global_config.visualize_BEV:
            self._BEV_vis.draw(self.engine.episode_step)
        
        if self.engine.episode_step == self.engine.managers['scenario_manager'].current_scene["log_length"] - 1:
            return self._trajectory_buffer
        else:
            return None
    
    @property
    def source_end_liveness_stats(self):
        """Scene-local source-end retention counters for rollout diagnostics."""
        return {
            'stall_retirements': int(getattr(self, '_source_end_stall_retirements', 0)),
            'retained_agent_ticks': int(
                getattr(self, '_source_end_retained_agent_ticks', 0)),
        }

    @property
    def parallel_runtime_stats(self):
        runtime = getattr(self, '_parallel_runtime', {})
        ego_s = float(runtime.get('ego_planner_s', 0.0))
        idm_s = float(runtime.get('idm_s', 0.0))
        critical_s = float(runtime.get('critical_path_s', 0.0))
        return {
            'steps': int(runtime.get('steps', 0)),
            'ego_planner_s': ego_s,
            'idm_s': idm_s,
            'critical_path_s': critical_s,
            'overlap_s': max(ego_s + idm_s - critical_s, 0.0),
        }

    def output_gif(self):
        if self.engine.global_config.visualize_BEV:
            self._BEV_vis.output_gif()

    def destroy(self):
        if getattr(self, '_ego_planner_executor', None) is not None:
            self._ego_planner_executor.shutdown(wait=True, cancel_futures=True)
            self._ego_planner_executor = None
        super().destroy()
        
    @property
    def current_agent_data(self):
        return self.engine.current_scene[SD.OBJECT_TRACKS]

    @property
    def sdc_track_id(self):
        return str(self.engine.current_scene[SD.SDC_ID])

    @property
    def ego_agent(self):
        return self._dynamic_agents['ego']

    @property
    def alive_agents(self):
        dynamic_agents = self._dynamic_agents.copy()
        static_agents = self._static_agents.copy()
        return dynamic_agents.update(static_agents)

    @property
    def get_dynamic_agents(self):
        return self._dynamic_agents

    @property
    def all_agents(self):
        """
        Return a merged dictionary of all agents, including both dynamic and static agents.
        """
        all_agents = self._dynamic_agents.copy()
        all_agents.update(self._static_agents)
        return all_agents
