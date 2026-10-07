# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
import json
import os
from typing import Dict, List
import uuid
import pickle
import numpy as np
from pathlib import Path
from pyquaternion import Quaternion
import cv2
import scipy
import copy

from odyssey.manager.base_manager import BaseManager
from odyssey.scenario.scenarios.scenario_description import ScenarioDescription as SD
from odyssey.utils.agent_utils import Video_visualizer, SceneDataSaver
from odyssey.utils.driving_command_table import build_driving_command_table
from odyssey.utils.nuplan_map_utils import route_roadblock_ids

import logging
logger = logging.getLogger(__name__)

FPS = 2
FRAME_INTERVAL = 1
FPS_KEYFRAME = FPS / FRAME_INTERVAL
ODYSSEY_ROOT = os.getenv('ODYSSEY_ROOT', os.path.abspath('.'))

# Closed-loop driving_command modes (global_config.driving_command_mode):
#   log          (default) replay the log's per-frame GT-derived command.
#   log_progress  the log's own stored label, indexed by WHERE THE EGO IS along the logged drive
#                 rather than by the clock: project the live ego onto the logged track and read
#                 that frame's label. No map query and no re-derived rule at run time. See
#                 DataManager._progress_driving_command and
#                 odyssey.utils.driving_command_table.
#   log_hold      scan the whole scenario log; if any frame is left/right, hold it for every frame
#                 (scene-constant). See DataManager._apply_log_hold.


def _apply_render_camera_calibrations(frame_data, render_results):
    """Export opt-in rendered calibration; absent metadata preserves legacy records.

    OpenScene sensor2lidar_rotation is camera-to-lidar R: projection uses
    R.T @ (point_lidar - translation). A camera-to-ego matrix is not that
    transform unless lidar2ego is identity.
    """
    calibrations = render_results.get("camera_calibrations")
    if calibrations is None:
        return
    if not isinstance(calibrations, dict):
        raise ValueError("camera_calibrations must be a camera-name mapping")
    if not calibrations:
        return

    def rigid_matrix(value, name):
        matrix = np.asarray(value, dtype=np.float64)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5, rtol=0)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-5, rtol=0)):
            raise ValueError(name + " must be a finite rigid 4x4 transform")
        return matrix

    lidar_to_ego = rigid_matrix(frame_data["lidar2ego"], "lidar2ego")
    updates = {}
    for name, calibration in calibrations.items():
        if name not in frame_data["cams"]:
            raise ValueError("rendered calibration camera is absent from frame: " + str(name))
        if calibration.get("projection_semantics") != "training_undistorted":
            raise ValueError("unsupported rendered camera projection semantics")
        camera_to_ego = rigid_matrix(calibration["camera_to_ego"], "camera_to_ego")
        intrinsic = np.asarray(calibration["cam_intrinsic"])
        distortion = np.asarray(calibration["distortion"])
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError("rendered intrinsic must be finite 3x3")
        if (distortion.ndim != 1 or distortion.size == 0
                or not np.isfinite(distortion).all() or np.any(distortion != 0)):
            raise ValueError("training_undistorted requires a zero distortion vector")
        camera_to_lidar = np.linalg.solve(lidar_to_ego, camera_to_ego)
        updates[name] = dict(
            sensor2lidar_rotation=camera_to_lidar[:3, :3].copy(),
            sensor2lidar_translation=camera_to_lidar[:3, 3].copy(),
            cam_intrinsic=intrinsic.copy(), distortion=distortion.copy(),
            camera_to_ego=camera_to_ego.copy(),
            sensor2ego_rotation=Quaternion(matrix=camera_to_ego[:3, :3]).elements.copy(),
            sensor2ego_translation=camera_to_ego[:3, 3].copy(),
            projection_semantics="training_undistorted")
    for name, values in updates.items():
        frame_data["cams"][name].update(values)


def _sim_dt(scene):
    """This scene's outer step length (s), resolved once by ScenarioManager via resolve_cadence.

    scene["sample_rate"] * 0.05 is the right formula (sample_rate counts 0.05 s periods), but
    deriving the same number in several places eventually diverges, so it is read here instead.
    Fails if missing rather than guessing.
    """
    c = scene.get("cadence") if hasattr(scene, "get") else None
    if c is None:
        raise RuntimeError("scene['cadence'] is missing. sim_dt is not re-derived here.")
    return float(c.sim_dt)


class DataManager(BaseManager):
    PRIORITY = 8

    def __init__(self):
        super().__init__()
        self.output_dir = Path(self.engine.global_config.get('data_output_dir', f'{ODYSSEY_ROOT}/data_output'))
        self.episode_data = {}
        self.episode_data_processed = {}
        self.seq_index = []
        self._video_visualizer = None
        self._data_saver = None
        # per-scene cache for the closed-loop driving_command mode log_hold (held command)
        self._log_hold_cache = {}
        # log_progress: per-scene label table, and the ego's progress along the logged drive at the
        # PREVIOUS query. The progress is episode state, not scene state -- reset() clears it so a
        # re-run of the same scene starts at the beginning instead of resuming the last ego's place.
        self._command_table_cache = {}
        self._command_progress_s = None
        # per-scene cache of the ego route's nuPlan roadblock ids (see _attach_route_info)
        self._route_roadblock_ids_cache = {}

    def _get_current_frame_data(self):
        """
        getting current frame data
        Returns:
            frame_data: dict of navsim metadata.
        """

        agent_manager = self.engine.agent_manager
        current_scene = self.engine.current_scene
        ego_vehicle = self.engine.agent_manager.ego_agent
        current_scene_id = current_scene[SD.ID]
        if len(current_scene_id.split('-')[-1]) == 3:
            suffix = current_scene_id.split('-')[-1]
        else:
            suffix = '000'

        # get original frame data
        openscene_info_dict = current_scene["metadata"]["openscene_data_infos_dict"]
        frame_idx = self.engine.episode_step
        # Fine (e.g. 0.1s) rollout: the openscene frame TEMPLATES are not upsampled, so a sub-step
        # reuses its containing original frame's template (camera rig / tokens; the ego pose,
        # timestamp and rendered images are overwritten per sub-step below). S is READ from the
        # cadence ScenarioManager resolved for this scene -- the one place the step is derived --
        # rather than re-derived here: log_length // #templates equals S only if log_length ==
        # #templates, a condition nothing declares or checks.
        _n_tmpl = len(openscene_info_dict)
        _S = int(current_scene["cadence"].upsample_n)
        # Held at the last template past the end of the recording, not left to run off it.
        # A slow ego needs MORE steps than the log to cover the same route (num_future carries 2x
        # the GT drive time for exactly that), but the openscene templates were only recorded for
        # the log's own length -- 40 for a _w40 scene. Without the clamp base_idx walks past the dict, the
        # lookup loop below never matches, frame_data stays None and the next attribute write
        # raises. What the template actually carries -- camera rig, lidar2ego, token -- is fixed
        # vehicle calibration that does not change with time, so holding the last one is sound;
        # the ego pose, timestamp and rendered images are all overwritten per sub-step anyway.
        # The 3DGS renderer itself has no such limit: the background model ignores timestamp
        # entirely (vanilla_gaussian_splatting.get_global_gaussians takes **kwargs and drops it),
        # so novel views keep synthesising as long as the ego stays inside the reconstruction.
        base_idx = min(frame_idx // _S, _n_tmpl - 1) if _n_tmpl else 0
        frame_data = None
        for i, (_, original_frame_data) in enumerate(openscene_info_dict.items()):
            if base_idx == i:
                frame_data = copy.deepcopy(original_frame_data)
                break
        
        # Token related fields
        if frame_data is not None:
            original_token = frame_data['token']
            # generate new token: original_token-reproduction_idx
            new_token = f"{original_token}-{suffix}" # 2cbf505c735c5c34-000
            parts = current_scene_id.split('-')
            if len(parts[-1]) == 3:
                new_scene_token = '-'.join(parts[-2:]) # for synthetic data, e.g. bb4f37403cea5b0e-001
            else:
                new_scene_token = parts[-1] # for original data, e.g. bb4f37403cea5b0e
            # Outputs are filed under the scene's published name (odyssey_sceneNNN) when the
            # launcher provides one; the asset id stays the key for loading the reconstruction.
            new_scene_token = os.environ.get('ODYSSEY_SCENE_NAME') or new_scene_token

        if 'synthetic_scene_info' in current_scene[SD.METADATA]:
            synthetic_frame = current_scene[SD.METADATA]['synthetic_scene_info']['frames'][base_idx]
            new_token = synthetic_frame['token']
            new_scene_token = current_scene[SD.METADATA]['synthetic_scene_info']['scene_metadata']['scene_token']

        # time_stamp related
        base_timestamp = current_scene.get("base_timestamp", 0)
        time_stamp = base_timestamp + int(frame_idx * _sim_dt(current_scene) * 1e6)

        render_results = self.engine.managers['render_manager'].rendering_results[frame_idx]
        ego2global = render_results['ego2global']
        ego2global_translation = ego2global[:3, 3]
        ego2global_rotation = Quaternion(matrix=ego2global[:3, :3]).elements

        frame_data['ego2global_translation'] = ego2global_translation
        frame_data['ego2global_rotation'] = ego2global_rotation
        frame_data['ego2global'] = ego2global
        frame_data['lidar2global'] = ego2global @ frame_data['lidar2ego']

        # loc: [0:3], quat: [3:7], accel: [7:10], velocity: [10:13], rotation_rate: [13:16]
        acc_global = ego_vehicle.rear_vehicle.current_acceleration
        velo_global = ego_vehicle.current_velocity
        ego_heading = ego_vehicle.rear_vehicle.current_heading
        ego2global_rotation_2d = np.array([
            [np.cos(ego_heading), -np.sin(ego_heading)],
            [np.sin(ego_heading), np.cos(ego_heading)]
        ])
        global2ego_rotation_2d = ego2global_rotation_2d.T

        acc_ego = acc_global @ global2ego_rotation_2d.T
        velo_ego = velo_global @ global2ego_rotation_2d.T

        ego_dynamic_state = [
            velo_ego[0],  # velocity_x
            velo_ego[1],  # velocity_y
            acc_ego[0],  # acceleration_x
            acc_ego[1]   # acceleration_y
        ]

        can_bus = frame_data['can_bus']
        can_bus[0:3] = ego2global_translation
        can_bus[3:7] = ego2global_rotation
        can_bus[7:9] = acc_ego
        can_bus[10:12] = velo_ego
        can_bus[15] = ego_vehicle.rear_vehicle.current_angular_velocity


        # update all related token and name fields
        _route_tokens = current_scene[SD.METADATA].get('nuplan_lidar_pc_tokens')
        _route_sidecar_token = str(_route_tokens[0]) if _route_tokens is not None and len(_route_tokens) else ''
        frame_data.update({
            'token': new_token,  # new frame token
            'frame_idx': frame_idx,  # new frame_idx
            'timestamp': time_stamp, # new timestamp
            'log_name': current_scene_id,  # new log name
            'log_token': new_scene_token,  # new log token
            'scene_name': current_scene_id,  # new scene name
            'scene_token': new_scene_token,  # new scene token
            # Stable source identity for sidecars baked from the nuPlan window. OmniRe scenes use
            # a checkpoint/asset id as their runtime scene_name, so matching a sidecar by
            # that display name is impossible. The first source lidar token is already stored in
            # the scenario PKL and is the exact sidecar filename; carrying it into every rollout
            # frame keeps lookup identity-based (never nearest-position based).
            'route_sidecar_token': _route_sidecar_token,
            'lidar_path': None,
            'ego_dynamic_state': ego_dynamic_state,
            'sample_prev': None,
            'sample_next': None,
            'can_bus': can_bus,
        })

        # Closed-loop driving_command source (default 'log' = replay the log value). log_progress
        # reads the label where the ego is along the logged drive; log_hold holds a scenario-wide
        # turn. Must run after the simulated ego pose above is in place.
        _cmd_mode = self.engine.global_config.get('driving_command_mode', 'log')
        if _cmd_mode == 'log_progress':
            self._progress_driving_command(frame_data, ego_vehicle, base_idx)
        elif _cmd_mode == 'log_hold':
            self._apply_log_hold(frame_data)

        # Route roadblock ids + map name for planners that condition on the routed lane centerline
        # (e.g. a route-centerline input). The nuPlan map itself is NOT sent over the IPC -- only the ids
        # and the map version, so the consumer can rebuild the centerline with its own map_api.
        self._attach_route_info(frame_data, ego_vehicle)

        _apply_render_camera_calibrations(frame_data, render_results)

        # update cams
        for cam_name, cam_data in frame_data['cams'].items():
            if 'synthetic_scene_info' in current_scene[SD.METADATA]:
                synthetic_frame = current_scene[SD.METADATA]['synthetic_scene_info']['frames'][base_idx]
                data_path = synthetic_frame['camera_dict'][cam_name.lower()]['data_path']
            else:
                original_path = cam_data['data_path']
                filename = original_path.split('/')[-1].split('.')[0]
                # Image names must differ per step. Otherwise each step's freshly rendered image
                # overwrites one file, and the model and the video see only a few frames.
                #
                # Always append the step number rather than guessing whether filenames collide: 10 Hz
                # reconstructed scenes have _S == 1, but cams are inherited from the 2 Hz records they
                # contain, so filenames repeat every 5 frames. The step number is always unique.
                _sub = f"-{frame_idx:04d}"
                data_path = f"{new_scene_token}/{cam_name}/{filename}-{suffix}{_sub}.jpg"

            if isinstance(data_path, Path):
                data_path = data_path.as_posix()
            cam_data['data_path'] = data_path
            # get image from render_results
            # ODYSSEY_RENDER_CAMS may have narrowed the render to the cameras the planner reads
            # (mtgs.py::_select_cams), while frame_data['cams'] still carries the OpenScene
            # template's full 8. A view that was not rendered has no image to save and no
            # consumer -- blank its data_path so the frame does not advertise a file that was
            # never written, and move on. Empty string rather than None: ipc_common wraps this
            # field in np.array(), and None would make an object-dtype array. navsim's
            # Cameras.from_camera_dict only reads data_path for cameras in the planner's own
            # sensor_config and hands the rest an empty Camera(), so a blank never gets opened.
            # Unfiltered rollouts render every camera, so this skips nothing there.
            if cam_name not in render_results['cameras']:
                cam_data['data_path'] = ""
                continue
            img = render_results['cameras'][cam_name]['image']
            output_path = self.output_dir / 'sensor_blobs' / data_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if os.environ.get('ODYSSEY_RUNTIME_PROFILE'):
                from odyssey_runtime.session import get_session
                get_session().encode_camera(frame_idx, cam_name, img, output_path)
            elif not cv2.imwrite(output_path.as_posix(), img):
                logger.error("problem in saving")
            # save_pre_restore: the same view before the restorer, under the same relative path.
            pre = render_results['cameras'][cam_name].get('image_pre_restore')
            if pre is not None:
                pre_path = self.output_dir / 'sensor_blobs_pre_restore' / data_path
                pre_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(pre_path.as_posix(), pre):
                    logger.error("problem in saving the pre-restore image")

        # update anns
        frame_data['anns'] = self._get_annotations(
            agent_manager, 
            agent_manager.ego_agent,
            frame_data
        )

        # The replay row of every actor and signal (AgentManager.source_rows). A signal with
        # row -1 has no phase the simulator shows (convert_to_traffic_lights reads it UNKNOWN).
        if self.engine.global_config.get('export_source_rows', False):
            frame_data['source_rows'] = agent_manager.source_rows(frame_idx)
            frame_data['signal_rows'] = {
                str(m['traffic_light_lane']): agent_manager.traffic_light_source_row(
                    str(m['traffic_light_lane']), frame_idx)
                for m in current_scene['dynamic_map_states'].values()
                if m['type'] == 'TRAFFIC_LIGHT'}

        return frame_data

    def _progress_driving_command(self, frame_data, ego_vehicle, base_idx):
        """Overwrite frame_data['driving_command'] with the log's label for the point the ego REACHED.

        The log stores a driving_command per frame, derived by OpenScene at that frame's GT pose. In
        closed loop the ego is rarely where the log was at the same tick -- it is slower or faster,
        and a lane-width to the side -- so replaying that label BY THE CLOCK (mode `log`) hands the
        planner the command for a piece of road it has not got to yet. This mode keeps the label and
        changes only the index: project the live ego onto the logged track, and read the label of the
        frame it has reached.

        Why this rather than recomputing at the live pose: recomputing needs a route, and its
        answer is only as good as that route -- a mis-baked route silently yields a confidently
        wrong command, and a wrong SIDE drives into oncoming traffic. Here there is no route and no
        rule at run time: the
        GT track is in the scene and cannot be wrong, so the command's correctness is decoupled from
        the route machinery entirely. (It is also not new information: driving_command is a GT-derived
        label, and mode `log` already replays it. Only the index changes.)

        Progress, not nearest. The search is restricted to a window around the previous query's
        progress (driving_command_table.SEARCH_BACK_M / SEARCH_AHEAD_M), so a drive that passes the
        same place twice cannot snap onto the wrong pass. An ego that leaves the logged drive
        entirely stops advancing and keeps the last label it reached, which is the honest answer --
        the log has nothing to say about road the log never drove.

        `unknown` is FILLED, not passed through: navtrain carries no `unknown`, so a planner
        conditioned on that slot gets a one-hot it never saw in training. The replacement is
        OpenScene's own 20 m / +-2 m rule applied to the LOGGED TRACK rather than to a lane
        centerline -- no route, no map query, nothing that can be mis-baked. Every frame says which
        it got in `driving_command_source`: `log_label`, or `gt_lookahead` / `gt_lookahead_held`
        where the track supplied it (driving_command_table._fill_unknown).

        UNGUARDED ON PURPOSE: a mode that silently falls back to the log is indistinguishable from
        one that never ran, and establishing which cost a frame-by-frame audit of 264 runs once
        already.
        """
        scene_id = self.engine.current_scene[SD.ID]
        table = self._command_table_cache.get(scene_id)
        if table is None:
            infos = list(
                self.engine.current_scene["metadata"]["openscene_data_infos_dict"].values())
            table = build_driving_command_table(infos)
            self._command_table_cache[scene_id] = table
            n_unknown = int((np.argmax(table.onehot, axis=1) == 3).sum())
            if n_unknown:
                n_left = int((np.argmax(table.filled, axis=1) == 3).sum())
                logger.info(f"[log_progress] scene {scene_id}: {n_unknown}/{len(infos)} logged "
                            f"labels are `unknown`; {n_unknown - n_left} filled from the GT track, "
                            f"{n_left} left standing (track shorter than the lookahead)")

        ego_xy = np.asarray(frame_data['ego2global'])[:2, 3]
        # First query of the episode: anchor on the frame the clock points at, so a scene whose ego
        # starts mid-track does not have to walk there from zero.
        anchor = (self._command_progress_s if self._command_progress_s is not None
                  else float(table.cum[min(int(base_idx), len(table.cum) - 1)]))
        k = table.locate(ego_xy, anchor, base_idx)
        self._command_progress_s = float(table.cum[k])

        onehot = table.filled[k]
        prev = frame_data.get('driving_command')
        dtype = getattr(prev, 'dtype', None)
        frame_data['driving_command'] = onehot.astype(dtype) if dtype is not None else onehot.copy()
        frame_data['driving_command_source'] = table.source[k]

    def _apply_log_hold(self, frame_data):
        """Hold a scenario-wide turn (log_hold): scan the whole log's per-frame driving_command; if
        any frame is left/right, set every frame's command to it (majority when both appear, else
        straight). Scene-constant, cached per scene. Guarded: any failure keeps the log value."""
        try:
            scene_id = self.engine.current_scene[SD.ID]
            held = self._log_hold_cache.get(scene_id)
            if held is None:
                infos = self.engine.current_scene["metadata"]["openscene_data_infos_dict"]
                nl = nr = 0
                for info in infos.values():
                    dc = np.asarray(info.get("driving_command", []))
                    if dc.size >= 3:
                        a = int(np.argmax(dc))
                        if a == 0:
                            nl += 1
                        elif a == 2:
                            nr += 1
                idx = (0 if nl >= nr else 2) if (nl and nr) else (0 if nl else (2 if nr else 1))
                held = np.zeros(4, dtype=np.int64)
                held[idx] = 1
                self._log_hold_cache[scene_id] = held
            prev = frame_data.get('driving_command')
            dtype = getattr(prev, 'dtype', None)
            frame_data['driving_command'] = held.astype(dtype) if dtype is not None else held.copy()
        except Exception as e:
            logger.warning(f"[driving_command] log_hold failed, keeping log value: {e}")

    def _attach_route_info(self, frame_data, ego_vehicle):
        """Attach the ego route's nuPlan roadblock ids + map name to the frame dict.

        Consumed by planners that condition on the routed lane centerline (e.g. a PLUTO-style
        reference line). We deliberately ship only the ROADBLOCK IDS
        and the map version rather than a baked polyline: the consumer then rebuilds the centerline
        through the very same nuPlan helpers the model was TRAINED with
        (_load_route_dicts -> _route_roadblock_correction -> _get_starting_lane ->
        _get_discrete_centerline), so there is no train/inference geometry mismatch. Rebuilding it
        per step also means _get_starting_lane re-solves against the live (drifted) closed-loop ego,
        which is the behaviour we want -- a polyline frozen at reset would not.

        The route is fixed for the scene (set once at reset), so the id list is cached per scene;
        the same ids are written to every frame. Ordering follows checkpoint_lanes (start->dest),
        which is what _get_discrete_centerline's Dijkstra roadblock_window assumes.

        Guarded: any failure leaves the keys absent, and the bridge decides whether that is fatal
        (it is, when the checkpoint was trained with use_route_centerline). Silence here must never
        crash a run that does not use the feature.
        """
        try:
            scene = self.engine.current_scene
            frame_data['route_roadblock_ids'] = list(self._route_roadblock_ids(scene, ego_vehicle))
            frame_data['map_name'] = scene.get("map")
        except Exception as e:
            logger.warning(f"[route_centerline] could not attach route info: {e}")

    def _route_roadblock_ids(self, scene, ego_vehicle):
        """The scene's route as nuPlan roadblock ids, in the GLOBAL frame. Cached per scene.

        ONE definition of "the ego's route", shared by everything that needs it: the frame field
        planners condition on, and the list log_progress fills `unknown` labels against. It is the
        scorer's own helper, asked with the scorer's own inputs -- the logged ego track and the
        converter's local->global offset.

        It does not walk ``navigation.checkpoint_lanes``, which is picked out on the simulator's
        SCENE-LOCAL map (see route_roadblock_ids for why that frame mix-up matters); pdm_policy and
        MetricManager use the same helper, so one concept is decided in one place.
        """
        scene_id = scene[SD.ID]
        ids = self._route_roadblock_ids_cache.get(scene_id)
        if ids is None:
            positions = np.asarray(ego_vehicle.object_track[SD.POSITION], dtype=np.float64)
            # local -> global. The converter states this as initial_ego_center =
            # -old_origin_in_current_coordinate; the scene metadata carries the origin itself.
            origin = np.asarray(scene["metadata"]["old_origin_in_current_coordinate"],
                                dtype=np.float64)[:2]
            ids = [str(r) for r in
                   route_roadblock_ids(positions, -origin, self._scene_map_api(scene), logger)]
            self._route_roadblock_ids_cache[scene_id] = ids
            logger.info(f"[route] scene {scene_id}: {len(ids)} route roadblocks")
        return ids

    def _scene_map_api(self, scene):
        """nuPlan map api for this scene's location (get_maps_api is lru-cached upstream)."""
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
        from odyssey.utils.nuplan_map_utils import resolve_map_location
        map_root = self.engine.global_config['nuplan_map_root']
        return get_maps_api(map_root, "nuplan-maps-v1.0", resolve_map_location(scene, map_root))

    def _get_annotations(self, agent_manager, ego_vehicle, original_frame_data):
        """
        getting annotations for all agents in the scene
        Args:
            agent_manager: agent manager
            ego_vehicle: ego vehicle
            original_frame_data: original frame data for getting instance_tokens and track_tokens
        Returns:
            dict: dict with gt_boxes, gt_names, gt_velocity_3d etc.
        """

        if len(agent_manager.all_agents) <= 1:  # only ego vehicle
            return {
                'gt_boxes': np.zeros((0, 7), dtype=np.float32),
                'gt_names': np.array([], dtype=str),
                'gt_velocity_3d': np.zeros((0, 3), dtype=np.float32),
                'instance_tokens': [],
                'track_tokens': [],
                'original_track_tokens': []
            }

        gt_boxes = []
        gt_names = []
        gt_velocity_3d = []
        track_tokens = []
        # getting tokens from original data
        original_anns = original_frame_data.get('anns', {})
        original_track_tokens = original_anns.get('track_tokens', [])
        
        # getting ego position and heading for coordinate transformation
        ego_pos = ego_vehicle.rear_vehicle.current_position
        ego_heading = ego_vehicle.rear_vehicle.current_heading

        for agent_id, agent in agent_manager.all_agents.items():
            if agent.id == "ego":
                continue
            # calculating relative position for coordinate transformation
            rel_pos = agent.current_position - ego_pos
            
            # calculating relative heading for coordinate transformation
            rel_heading = agent.current_heading - ego_heading
            
            # coordinate transformation (global -> ego)
            cos_heading = np.cos(-ego_heading)
            sin_heading = np.sin(-ego_heading)
            x_ego = rel_pos[0] * cos_heading - rel_pos[1] * sin_heading
            y_ego = rel_pos[0] * sin_heading + rel_pos[1] * cos_heading
            
            # building box info (x,y,z,l,w,h,yaw)
            box = [
                x_ego,                  # x in ego frame
                y_ego,                  # y in ego frame
                0.0,                    # z 
                agent.length,           # length
                agent.width,            # width
                agent.height,           # height
                rel_heading            # relative heading
            ]
            
            # getting velocity for coordinate transformation
            if hasattr(agent, 'current_velocity'):
                vel_x = agent.current_velocity[0] * cos_heading - agent.current_velocity[1] * sin_heading
                vel_y = agent.current_velocity[0] * sin_heading + agent.current_velocity[1] * cos_heading
                velocity_3d = [vel_x, vel_y, 0.0]  # z direction velocity set to 0
            else:
                velocity_3d = [0.0, 0.0, 0.0]
            
            # get agent type
            agent_type = agent_manager.current_agent_data[agent.id]['type']
            agent_type = agent_type.lower()
            if agent_type == "traffic_barrier":
                agent_type = "barrier"
            elif agent_type == "traffic_object":
                agent_type = "generic_object"

            gt_boxes.append(box)
            gt_names.append(agent_type)
            gt_velocity_3d.append(velocity_3d)
            track_tokens.append(agent.id)
            
        gt_boxes = np.array(gt_boxes)
        # convert heading to [-pi, pi] range
        gt_boxes[:, -1] = gt_boxes[:, -1] % (2 * np.pi)
        gt_boxes[:, -1][gt_boxes[:, -1] > np.pi] -= 2 * np.pi

        return {
            'gt_boxes': gt_boxes,           # Ground truth boxes (x,y,z,l,w,h,yaw)
            'gt_names': np.array(gt_names),           # Class names
            'gt_velocity_3d': np.array(gt_velocity_3d),  # 3D velocity
            'instance_tokens': track_tokens,  # keep original instance tokens
            'track_tokens': track_tokens,      # current track tokens, aligning with gt_boxes
            'original_track_tokens': original_track_tokens       # keep original track tokens
        }

    def _frame_supply_cap(self) -> int:
        """How many steps this rollout may be fed frames for.

        Was `log_length` -- the recording's own length. Closed loop the ego is routinely SLOWER
        than the log it is replayed against, so it needs more steps than the recording took to
        cover the same route; num_future carries 2x the GT drive time to give it those steps, and
        done_function() ends the run the moment the GT goal is actually reached. Capping the
        supply at log_length made that budget unusable: the sim kept stepping while this method
        silently stopped writing, the planner had nothing to answer, and the plan wait ran into
        its full timeout.

        Everything downstream of here is fine past log_length -- the 3DGS background ignores
        timestamp, _get_current_frame_data holds the last openscene template, and rigid_object
        freezes out-of-range actors near the ego (ODYSSEY_ACTOR_EGO_RADIUS) -- so the honest cap is
        the step budget itself, derived exactly as base_env derives `truncateds`.
        """
        cfg = self.engine.global_config
        try:
            horizon = int(cfg['num_history']) + int(cfg['num_future']) - 1
            mult = float(cfg.get('gt_budget_multiplier', 1.0) or 1.0)
            budget = int(horizon * mult)
        except (AttributeError, KeyError, TypeError, ValueError):
            budget = 0
        supply = int(self.engine.current_scene['log_length'])
        return max(supply, budget) if budget > 0 else supply

    def save_current_frame_data(self):
        if self.engine.episode_step < self._frame_supply_cap():
            frame_data = self._get_current_frame_data()
            if frame_data:
                token = frame_data['token']
                self.episode_data[token] = frame_data
                if self.engine.global_config.get('record_frame_index', False):
                    self._append_frame_index(frame_data)
                if self.engine.global_config.use_planner_actions:
                    ego_client = self.engine.managers['agent_manager'].ego_agent.client
                    ego_client.process_frame(frame_data, self.engine.episode_step)

    def _append_frame_index(self, frame_data):
        """One measured line per step: which frame this was and which images it wrote.

        The per-step record a consumer needs (token, timestamp, ego pose, camera file) is
        otherwise only in the PLANNER's frames (planner_client.process_frame, kept with --record),
        so a rollout with no planner -- the privileged GT-replay arms -- leaves nothing that
        says what each rendered image is. nvidia_viz/synth_vis_data.py papers over that for the
        viewer by holding the 0.5 s scored pose for 5 frames, and its own docstring says the
        real fix belongs here. This is that: measured values, every step, planner or not.

        meta_datas/<log>.pkl is not it either -- save_data() writes that from reset(), which a
        single-scenario rollout never reaches.

        Append-only JSONL so a crashed rollout still leaves every step it did finish.
        """
        try:
            ego2global = np.asarray(frame_data['ego2global'], dtype=float)
            record = {
                'step': int(self.engine.episode_step),
                'token': frame_data['token'],
                'timestamp': int(frame_data['timestamp']),
                'log_name': frame_data.get('log_name'),
                'ego2global_translation': [float(v) for v in ego2global[:3, 3]],
                'ego2global_rotation': [float(v) for v in frame_data['ego2global_rotation']],
                'cams': {name: cam.get('data_path', '') for name, cam in frame_data['cams'].items()},
            }
            path = self.output_dir / 'frame_index.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception:
            # A side output must never take the rollout down with it.
            logger.exception('frame_index: could not append step %s', self.engine.episode_step)

    def after_step(self):
        return {}

    def reset(self):
        self.save_data()
        self.episode_data = {}
        self.episode_data_processed = {}
        self.seq_index = []
        # log_progress anchors each query on the previous one; a new episode starts from the clock
        # again rather than resuming the last ego's place along the drive. The TABLE is scene state
        # and stays cached -- only the progress is per-episode.
        self._command_progress_s = None
        if self._video_visualizer is None:
            if self.engine.global_config.visualize_video:
                self._video_visualizer = Video_visualizer()
            else:
                self._video_visualizer = None
        if self._data_saver is None:
            if self.engine.global_config.save_data:
                self._data_saver = SceneDataSaver()
            else:
                self._data_saver = None
        self.original_frame_data = copy.deepcopy(list(self.engine.current_scene["metadata"]["openscene_data_infos_dict"].values()))
        return {}

    def after_reset(self):
        return {}

    def save_data(self):
        if len(self.episode_data) == 0:
            return

        data_infos = list(self.episode_data.values())
        data_infos[0]['sample_prev'] = None
        data_infos[-1]['sample_next'] = None

        for i in range(len(data_infos)):
            data_infos[i]['sample_prev'] = data_infos[i-1]['token'] if i > 0 else None
            data_infos[i]['sample_next'] = data_infos[i+1]['token'] if i < len(data_infos) - 1 else None

        filename = data_infos[0]['log_name'] + '.pkl'
        filepath = self.output_dir / 'meta_datas' / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data_infos, f)
        if self._video_visualizer is not None:
            self._video_visualizer.generate_video()
        if self._data_saver is not None:
            self._data_saver.save_scene_data(data_infos[0]['scene_token'], self.original_frame_data)
