from enum import IntEnum
from typing import Any, Dict, List, Tuple, Optional
import cv2
import functools
import os
from pathlib import Path
import numpy as np
from PIL import Image
import numpy.typing as npt

import torch

from shapely import affinity
from shapely.geometry import Polygon, LineString

from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from navsim.agents.safedrive.safedrive_config import SafeDrive_Config
from navsim.agents.safedrive.score_module.token_id import token_to_id  # pair-NC GT: token -> stable int id
from navsim.agents.sdroute.route_target import build_sdroute_target
from navsim.common.dataclasses import AgentInput, Scene, Annotations
from navsim.common.enums import BoundingBoxIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

import copy


def normalize_angle(angle):
    return torch.arctan2(torch.sin(angle), torch.cos(angle))



# ---------------------------------------------------------------------------------------------
# Camera preprocessing. One function for both moments it can run: while building the cache
# (cache_camera_images=True) and inside the dataloader worker at collate time when the cache
# holds CameraPaths (cache_camera_images=False). Pixels are identical either way.
# ---------------------------------------------------------------------------------------------
CAMERA_CROP_OFFSET = (28, 0)          # rows cut from top and bottom of the 1920x1080 image
CAMERA_RESIZE_SHAPE = (512, 256)      # (W, H) after resize
CAMERA_IMG_SIZE = (1920, 1080)


@functools.lru_cache(maxsize=64)
def _undistort_maps(intrinsics_bytes: bytes, k_dtype: str, distortion_bytes: bytes, d_dtype: str):
    """initUndistortRectifyMap costs ~45 ms; calibration repeats per camera, so cache by value.
    The arrays are rebuilt with their original dtype so the maps match a direct call bit for bit."""
    K = np.frombuffer(intrinsics_bytes, dtype=k_dtype).reshape(3, 3)
    D = np.frombuffer(distortion_bytes, dtype=d_dtype)
    return cv2.initUndistortRectifyMap(K, D, None, K, CAMERA_IMG_SIZE, cv2.CV_32FC1)


def _compute_post_transform(img_shape, crop_offset, resize_factor=0.25):
    A = np.eye(2)
    b = np.array(img_shape) / 2
    b = A @ (-b) + b

    ida_rot = A
    ida_tran = -np.array([crop_offset[1], crop_offset[0]])  # Note: H, W -> Y, X
    ida_tran = A @ ida_tran + b

    post_rot = np.eye(3)
    post_tran = np.zeros(3)
    post_rot[:2, :2] = ida_rot * resize_factor
    post_tran[:2] = ida_tran * resize_factor
    return post_rot, post_tran


def preprocess_camera_image(image: np.ndarray, intrinsics: np.ndarray, distortion: np.ndarray) -> torch.Tensor:
    """RGB HxWx3 uint8 (as navsim loads it) -> undistort, crop, resize -> uint8 3x256x512 tensor."""
    K = np.ascontiguousarray(intrinsics)
    D = np.ascontiguousarray(distortion)
    map1, map2 = _undistort_maps(K.tobytes(), K.dtype.str, D.tobytes(), D.dtype.str)
    undist = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
    cropped = undist[CAMERA_CROP_OFFSET[0]:-CAMERA_CROP_OFFSET[0]]
    resized = cv2.resize(cropped, CAMERA_RESIZE_SHAPE)
    # uint8 CHW, not ToTensor()'s float32: the image is ~75% of a cached sample and float32
    # barely gzips. The model divides by 255 on the way in (same values).
    return torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1)


def camera_post_transform():
    """post_rot / post_tran of the fixed crop+resize; independent of the image content."""
    cropped_shape = (CAMERA_IMG_SIZE[1] - 2 * CAMERA_CROP_OFFSET[0], CAMERA_IMG_SIZE[0])
    return _compute_post_transform(cropped_shape, CAMERA_CROP_OFFSET)


class CameraPaths:
    """What the feature cache holds instead of the image tensor when cache_camera_images=False.

    paths: [frame][cam] relative to <OPENSCENE_DATA_ROOT>/sensor_blobs/<split>, same convention as
    the cached lidar .pcd path. intrinsics (F, N, 3, 3) and distortion (F, N, 5) are what
    preprocess_camera_image needs. custom_collate_fn turns a batch of these into the
    (B, F, N, 3, 256, 512) uint8 tensor inside the dataloader worker; the online evaluation
    path calls to_tensor() directly.
    """

    __slots__ = ("paths", "intrinsics", "distortion")

    def __init__(self, paths, intrinsics, distortion):
        self.paths = paths
        self.intrinsics = np.asarray(intrinsics)      # (F, N, 3, 3), dtype as navsim gives it
        self.distortion = np.asarray(distortion)      # (F, N, 5)

    @staticmethod
    def _sensor_blobs_root() -> Path:
        return Path(os.environ.get("OPENSCENE_DATA_ROOT", "dataset")) / "sensor_blobs"

    def to_tensor(self) -> torch.Tensor:
        # Runs inside a dataloader worker. OpenCV sizes its thread pool to every core, so many workers
        # oversubscribe the CPU; use one thread per worker.
        cv2.setNumThreads(1)
        root = self._sensor_blobs_root()
        frames = []
        for f, cams in enumerate(self.paths):
            imgs = []
            for n, rel in enumerate(cams):
                path = root / "trainval" / rel
                if not path.exists():
                    path = root / "test" / rel
                image = np.array(Image.open(path))          # RGB uint8, exactly as navsim's Camera
                imgs.append(preprocess_camera_image(image, self.intrinsics[f, n], self.distortion[f, n]))
            frames.append(torch.stack(imgs, dim=0))
        return torch.stack(frames, dim=0)                       # (F, N, 3, 256, 512) uint8


class SafeDrive_FeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder for GRAD."""

    def __init__(self, config: SafeDrive_Config):
        """
        Initializes feature builder.
        :param config: global config dataclass of GRAD
        """
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features = {}

        features['camera_feature'], features['matrices'], features['lidar2img'] = self._get_camera_feature_bevdepth(agent_input)
        if self._config.use_lidar:
            features["lidar_feature"] = self._get_lidar_feature_SECOND(agent_input)

        status_features = []
        ego_poses = []
        for frame_id in range(4-self._config.num_input_frames,4,1):
            status_feature = torch.concatenate(
                [
                    torch.tensor(agent_input.ego_statuses[frame_id].driving_command, dtype=torch.float32),
                    torch.tensor(agent_input.ego_statuses[frame_id].ego_velocity, dtype=torch.float32),
                    torch.tensor(agent_input.ego_statuses[frame_id].ego_acceleration, dtype=torch.float32),
                ],
            )
            status_features.append(status_feature)
            ego_poses.append(torch.tensor(agent_input.ego_statuses[frame_id].ego_pose, dtype=torch.float32))
        features["status_feature"] = torch.stack(status_features, dim=0)
        features["ego_poses"] = torch.stack(ego_poses, dim=0)

        return features

    def _get_camera_feature_bevdepth(self, agent_input: AgentInput) -> torch.Tensor:
        """Stack the camera images (or their paths) and the matrices the BEV encoder projects with."""
        def compute_lidar2img_matrix(intrin, rot, tran, post_rot, post_tran):
            # lidar -> cam (rotation/translation)
            rot_inv = np.linalg.inv(rot)
            tran_inv = -rot_inv @ tran
            lidar2cam = np.eye(4)
            lidar2cam[:3, :3] = rot_inv
            lidar2cam[:3, 3] = tran_inv

            # cam -> image
            cam2img = np.eye(4)
            cam2img[:3, :3] = intrin

            # Final projection
            lidar2img = cam2img @ lidar2cam

            post = np.eye(4)
            post[:2, :2] = post_rot[:2, :2]
            post[:2, 3]  = post_tran[:2]
            return post @ lidar2img

        post_rot, post_tran = camera_post_transform()
        cache_images = self._config.cache_camera_images

        tensor_images = []
        path_frames, intrin_frames, dist_frames = [], [], []
        matrics = []
        lidar2imgs_list = []
        for frame_id in range(4-self._config.num_input_frames,4,1):
            cameras = agent_input.cameras[frame_id]
            #                 cameras.cam_l1, cameras.cam_r1,
            #                 cameras.cam_l2, cameras.cam_b0, cameras.cam_r2]
            cam_list = [cameras.cam_l0, cameras.cam_f0, cameras.cam_r0]
            images, rots, trans, intrins, lidar2imgs, post_rots, post_trans = [], [], [], [], [], [], []
            paths, dists = [], []
            for cam in cam_list:
                intrin = cam.intrinsics[:3, :3]
                rot = cam.sensor2lidar_rotation[:3, :3]
                tran = cam.sensor2lidar_translation[:3]
                if cache_images:
                    images.append(preprocess_camera_image(cam.image, cam.intrinsics, cam.distortion).unsqueeze(0))
                else:
                    paths.append(str(cam.camera_path))
                    dists.append(np.asarray(cam.distortion))
                intrins.append(intrin)
                rots.append(rot)
                trans.append(tran)
                lidar2img = compute_lidar2img_matrix(intrin, rot, tran, post_rot, post_tran)
                lidar2imgs.append(torch.tensor(lidar2img).float())
                post_rots.append(post_rot)
                post_trans.append(post_tran)
            lidar2imgs = torch.stack(lidar2imgs, dim=0)  # (N_cam, 4, 4)
            lidar2imgs_list.append(lidar2imgs)  # (N_cam, 4, 4)
            if cache_images:
                tensor_images.append(torch.cat(images, dim=0))  # (N_cam, 3, 256, 512) uint8
            else:
                path_frames.append(paths)
                intrin_frames.append(np.stack([np.asarray(cam.intrinsics) for cam in cam_list]))
                dist_frames.append(np.stack(dists))
            matrics .append([
                torch.tensor(np.stack(rots)).float(),
                torch.tensor(np.stack(trans)).float(),
                torch.tensor(np.stack(intrins)).float(),
                torch.tensor(np.stack(post_rots)).float(),
                torch.tensor(np.stack(post_trans)).float(),
                torch.eye(3).repeat(1, 1, 1).float()
            ])
        if cache_images:
            camera_feature = torch.stack(tensor_images, dim=0)   # (F, N_cam, 3, 256, 512) uint8
        else:
            camera_feature = CameraPaths(path_frames, np.stack(intrin_frames), np.stack(dist_frames))
        lidar2imgs = torch.stack(lidar2imgs_list, dim=0)

        return camera_feature, matrics, lidar2imgs

    def _get_lidar_feature_SECOND(self, agent_input: AgentInput) -> torch.Tensor:
        feature_list = []
        for frame_id in range(4-self._config.num_input_frames,4,1):
            lidar_pc_path = agent_input.lidars[frame_id].lidar_path
            feature_list.append(lidar_pc_path)
        features = feature_list
        return features


class SafeDrive_TargetBuilder(AbstractTargetBuilder):
    """Output target builder for GRAD."""

    def __init__(self, config: SafeDrive_Config):
        """
        Initializes target builder.
        :param config: global config dataclass of GRAD
        """
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        trajectory = torch.tensor(
            scene.get_future_trajectory(num_trajectory_frames=self._config.trajectory_sampling.num_poses).poses
        )
        frame_idx = scene.scene_metadata.num_history_frames - 1
        annotations = scene.frames[frame_idx].annotations
        ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)

        agent_states, agent_labels, agent_tokens = self._compute_agent_targets_per_class(annotations)

        bev_semantic_map = self._compute_bev_semantic_map(annotations, scene.map_api, ego_pose)
        target_dict = {
            "trajectory": trajectory,
            "agent_states": agent_states,
            "agent_labels": agent_labels,
            "agent_tokens": agent_tokens,
            "bev_semantic_map": bev_semantic_map,
            "scene_frame_token": scene.frames[frame_idx].token,
        }

        # The SD route goes in the target dict because only this builder gets a Scene (and so the
        # map). Caches built without it are completed by run_sdroute_backfill.py.
        if getattr(self._config, "use_sdroute", False):
            route, route_mask = build_sdroute_target(scene, frame_idx)
            target_dict["route_centerline"] = route
            target_dict["route_centerline_mask"] = route_mask

        # pair-NC GT needs a stable int id per GT agent to map the collision tokens
        # -> agent rows in the loss. agent_tokens is a padded list of str ("" for pad -> id 0).
        if getattr(self._config, "use_target_scores", False):
            target_dict["agent_token_ids"] = torch.tensor(
                [token_to_id(t) for t in agent_tokens], dtype=torch.long)

        # per-class BEV segmentation targets
        if self._config.bev_seg_multi_class:
            bev_semantic_map_multi_class = self._compute_bev_semantic_map_multi_class(annotations, scene.map_api, ego_pose, num_classes=7, exclusive=False, return_uint8=False)
            target_dict["bev_semantic_map_multi_class"] = bev_semantic_map_multi_class

        # the same targets for the future frames
        current_frame_valid_tokens_ = annotations.track_tokens   # for no future car
        roi_mask = self.boxes_in_bev_roi(annotations.boxes, self._config)
        current_frame_valid_tokens = [t for t, keep in zip(current_frame_valid_tokens_, roi_mask) if keep]

        all_future_frame_bev_semantic_map = []
        all_future_frame_bev_semantic_map_multi = []
        future_start_idx = frame_idx + 1        # 4
        future_end_idx = len(scene.frames) -2   # 12

        current_ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)
        current_frame_index = frame_idx
        current_future_trajectory = scene.get_future_trajectory()

        for future_idx in range(future_start_idx, future_end_idx):
            frame_interval = future_idx - current_frame_index - 1
            ref_frame_offset = current_future_trajectory.poses[frame_interval]

            # Default future annotation
            future_frame_bev = scene.frames[future_idx]
            future_annotations_bev = future_frame_bev.annotations

            # NO future car
            mask = np.isin(future_annotations_bev.track_tokens, list(current_frame_valid_tokens))
            track_tokens_np = np.asarray(future_annotations_bev.track_tokens)
            filt_track_tokens = track_tokens_np[mask].tolist()
            filt_names = [n for n, m in zip(future_annotations_bev.names, mask) if m]
            filt_boxes = future_annotations_bev.boxes[mask]

            future_boxes_in_current_frame = (
                self.transform_boxes_from_future_to_current_ego_frame(
                    filt_boxes, ref_frame_offset
                )
            )

            future_anno_in_cur_frame = copy.copy(future_annotations_bev)
            future_anno_in_cur_frame.boxes = future_boxes_in_current_frame
            future_anno_in_cur_frame.track_tokens = filt_track_tokens
            future_anno_in_cur_frame.names = filt_names

            future_bev_semantic_map = self._compute_bev_semantic_map(
                future_anno_in_cur_frame, scene.map_api, current_ego_pose  # (H, W)
            )
            all_future_frame_bev_semantic_map.append(future_bev_semantic_map)

            if self._config.bev_seg_multi_class:
                future_bev_multi = self._compute_bev_semantic_map_multi_class(
                    future_anno_in_cur_frame, scene.map_api, current_ego_pose,
                    num_classes=7, exclusive=False, return_uint8=False
                )
                all_future_frame_bev_semantic_map_multi.append(future_bev_multi)

        future_bev_stack = torch.stack(all_future_frame_bev_semantic_map, dim=0)    # (T, H, W)
        target_dict["all_future_frame_bev_semantic_map"] = future_bev_stack

        if self._config.bev_seg_multi_class:
            future_bev_multi_stack = torch.stack(all_future_frame_bev_semantic_map_multi, dim=0)
            target_dict["all_future_frame_bev_semantic_map_multi_class"] = future_bev_multi_stack

        # Future agent targets
        all_agent_traj = []
        all_agent_motion_masks = []
        all_future_ego_states = []
        future_start_idx = frame_idx + 1

        for future_idx in range(trajectory.shape[0]):
            frame_offset = trajectory[future_idx]
            future_frame = scene.frames[future_start_idx + future_idx]
            future_annos = future_frame.annotations

            # transform boxes and velocity
            boxes_t = torch.tensor(future_annos.boxes)
            vel_t   = torch.tensor(future_annos.velocity_3d) if hasattr(future_annos, "velocity_3d") and future_annos.velocity_3d is not None else None

            future_boxes_t, future_vel_t = self._transform_boxes_and_velocity_to_current_ego_frame(
                boxes_t, vel_t, frame_offset
            )

            future_anno = copy.deepcopy(future_annos)
            future_anno.boxes = future_boxes_t
            if future_vel_t is not None:
                future_anno.velocity_3d = future_vel_t.cpu().numpy()

            # Get future agent targets and semantic map
            # future_agent_states, _, future_agent_tokens, = self._compute_agent_targets(future_anno)
            future_agent_states, _, future_agent_tokens, = self._compute_agent_targets_per_class(future_anno)

            # Align future agents with current ones
            future_agent_states, future_agent_tokens, future_agent_masks = self._align_future_agent_states(
                future_agent_states, future_agent_tokens, agent_tokens
            )

            # Get future ego pose
            future_status_features = torch.cat(
                [
                    torch.tensor(future_frame.ego_status.driving_command, dtype=torch.float32),
                    torch.tensor(future_frame.ego_status.ego_velocity, dtype=torch.float32),
                    torch.tensor(future_frame.ego_status.ego_acceleration, dtype=torch.float32),
                ],
                dim=0,
            )
            all_agent_traj.append(future_agent_states[:,:7])
            all_agent_motion_masks.append(future_agent_masks)
            all_future_ego_states.append(future_status_features)

        all_agent_traj_stack = torch.stack(all_agent_traj, dim=0).transpose(0,1)  # (T, N, 3)
        all_agent_traj_masks_stack = torch.stack(all_agent_motion_masks, dim=0).bool().transpose(0,1)

        all_agent_traj_stack = torch.cat([agent_states[:,None,:7], all_agent_traj_stack], dim=1)

        all_agent_traj_masks_stack = torch.cat([agent_labels[:,None], all_agent_traj_masks_stack], dim=1)
        target_dict['motion_traj'] = all_agent_traj_stack
        target_dict['motion_mask'] = all_agent_traj_masks_stack  # (N, T)

        return target_dict

    def _transform_boxes_and_velocity_to_current_ego_frame(
        self,
        boxes: torch.Tensor,                 # (N,7): [x,y,z,l,w,h,yaw]
        velocity_3d: Optional[torch.Tensor],  # (N,3) or (N,2): [vx, vy, (vz)]
        points_rel: torch.Tensor             # (3,): [dx, dy, dtheta]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Same, for boxes plus their velocity vectors."""
        dx, dy, dtheta = points_rel

        cos_t = torch.cos(dtheta)
        sin_t = torch.sin(dtheta)
        R = torch.stack([
            torch.stack([cos_t, -sin_t]),
            torch.stack([sin_t,  cos_t]),
        ])  # (2,2)

        XY_f = torch.stack([boxes[:, BoundingBoxIndex._X], boxes[:, BoundingBoxIndex._Y]], dim=-1)  # (N,2)
        XY_c = XY_f @ R.T + torch.tensor([dx, dy], device=boxes.device)
        yaw_c = boxes[:, BoundingBoxIndex._HEADING] + dtheta

        boxes_c = boxes.clone()
        boxes_c[:, BoundingBoxIndex._X]       = XY_c[:, 0]
        boxes_c[:, BoundingBoxIndex._Y]       = XY_c[:, 1]
        boxes_c[:, BoundingBoxIndex._HEADING] = yaw_c

        vel_c = None
        if velocity_3d is not None:
            if velocity_3d.shape[1] >= 2:
                vxy_f = velocity_3d[:, :2]           # (N,2) [vx,vy] in future ego
                vxy_c = vxy_f @ R.T                  # rotate to current ego
                if velocity_3d.shape[1] == 2:
                    vel_c = vxy_c
                else:
                    vz = velocity_3d[:, 2:3]         # keep vz as-is
                    vel_c = torch.cat([vxy_c, vz], dim=1)
            else:
                vel_c = velocity_3d.clone()

        return boxes_c, vel_c

    def _compute_agent_targets(self, annotations: Annotations) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts 2D agent bounding boxes in ego coordinates
        :param annotations: annotation dataclass
        :return: tuple of bounding box values and labels (binary)
        """

        max_agents = self._config.num_bounding_boxes
        agent_states_list: List[npt.NDArray[np.float32]] = []
        agent_tokens_list: List[str] = []

        def _xy_in_lidar(x: float, y: float, config: SafeDrive_Config) -> bool:
            return (config.lidar_min_x <= x <= config.lidar_max_x) and (config.lidar_min_y <= y <= config.lidar_max_y)

        for box, name, token, vel in zip(annotations.boxes, annotations.names, annotations.track_tokens, annotations.velocity_3d):
            box_x, box_y, box_heading, box_length, box_width = (
                box[BoundingBoxIndex.X],
                box[BoundingBoxIndex.Y],
                box[BoundingBoxIndex.HEADING],
                box[BoundingBoxIndex.LENGTH],
                box[BoundingBoxIndex.WIDTH],
            )

            if name == "vehicle" and _xy_in_lidar(box_x, box_y, self._config):
                agent_states_list.append(np.array([box_x, box_y, box_heading, box_length, box_width, vel[0], vel[1]], dtype=np.float32))
                agent_tokens_list.append(token)

        agents_states_arr = np.array(agent_states_list)

        # filter num_instances nearest
        agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()+2), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)
        selected_tokens = []

        if len(agents_states_arr) > 0:
            distances = np.linalg.norm(agents_states_arr[..., BoundingBox2DIndex.POINT], axis=-1)
            argsort = np.argsort(distances)[:max_agents]

            # filter detections
            agents_states_arr = agents_states_arr[argsort]
            agent_states[: len(agents_states_arr)] = agents_states_arr
            agent_labels[: len(agents_states_arr)] = True
            selected_tokens = [agent_tokens_list[i] for i in argsort]

        # Fill remaining with empty strings if fewer than max_agents
        while len(selected_tokens) < max_agents:
            selected_tokens.append("")

        return torch.tensor(agent_states), torch.tensor(agent_labels), selected_tokens

    def _compute_bev_semantic_map(
        self, annotations: Annotations, map_api: AbstractMap, ego_pose: StateSE2
    ) -> torch.Tensor:
        """
        Creates sematic map in BEV
        :param annotations: annotation dataclass
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :return: 2D torch tensor of semantic labels
        """

        bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)
        for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
            if entity_type == "polygon":
                entity_mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
            elif entity_type == "linestring":
                entity_mask = self._compute_map_linestring_mask(map_api, ego_pose, layers)
            else:
                entity_mask = self._compute_box_mask(annotations, layers)
            bev_semantic_map[entity_mask] = label

        return torch.Tensor(bev_semantic_map)

    def _compute_map_polygon_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary mask given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """

        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(map_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        map_polygon_mask = np.rot90(map_polygon_mask)[::-1]
        return map_polygon_mask > 0

    def _compute_map_linestring_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of linestring given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_linestring_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(map_object.baseline_path.linestring, ego_pose)
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(map_linestring_mask, [points], isClosed=False, color=255, thickness=2)
        # OpenCV has origin on top-left corner
        map_linestring_mask = np.rot90(map_linestring_mask)[::-1]
        return map_linestring_mask > 0

    def _compute_box_mask(self, annotations: Annotations, layers: TrackedObjectType) -> npt.NDArray[np.bool_]:
        """
        Compute binary of bounding boxes in BEV space
        :param annotations: annotation dataclass
        :param layers: bounding box labels to include
        :return: binary mask as numpy array
        """
        box_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for name_value, box_value in zip(annotations.names, annotations.boxes):
            agent_type = tracked_object_types[name_value]
            if agent_type in layers:
                x, y, heading = box_value[0], box_value[1], box_value[-1]
                box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
                agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
                exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(box_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0

    def _compute_agent_targets_per_class(self, annotations: Annotations):
        """Agent targets split per class, each padded to its own box budget."""
        veh_limit = self._config.num_vehicle_bounding_boxes
        ped_limit = self._config.num_pedestrian_bounding_boxes
        with_vel = True

        def _xy_in_lidar(x: float, y: float, cfg: SafeDrive_Config) -> bool:
            return (cfg.lidar_min_x <= x <= cfg.lidar_max_x) and (cfg.lidar_min_y <= y <= cfg.lidar_max_y)

        veh_list, ped_list = [], []
        veh_tok,  ped_tok  = [], []

        for box, name, token, vel in zip(annotations.boxes, annotations.names, annotations.track_tokens, annotations.velocity_3d):
            x, y, yaw, L, W = box[BoundingBoxIndex.X], box[BoundingBoxIndex.Y], box[BoundingBoxIndex.HEADING], box[BoundingBoxIndex.LENGTH], box[BoundingBoxIndex.WIDTH]
            if not _xy_in_lidar(x, y, self._config):
                continue
            if name == "vehicle":
                arr = [x, y, yaw, L, W] + ([vel[0], vel[1]] if with_vel else [])
                veh_list.append(np.array(arr, dtype=np.float32))
                veh_tok.append(token)
            elif name == "pedestrian":
                arr = [x, y, yaw, L, W] + ([vel[0], vel[1]] if with_vel else [])
                ped_list.append(np.array(arr, dtype=np.float32))
                ped_tok.append(token)

        def _select_nearest(arr_list, tok_list, k):
            if len(arr_list) == 0 or k <= 0:
                return [], []
            arr = np.stack(arr_list)  # (M, D)
            dists = np.linalg.norm(arr[:, :2], axis=1)
            take = np.argsort(dists)[:k]
            return [arr_list[i] for i in take], [tok_list[i] for i in take]

        veh_list, veh_tok = _select_nearest(veh_list, veh_tok, veh_limit)
        ped_list, ped_tok = _select_nearest(ped_list, ped_tok, ped_limit)

        total = veh_limit + ped_limit
        D = (5 + 2) if with_vel else 5
        states = np.zeros((total, D), dtype=np.float32)
        labels = np.full((total,), -1, dtype=np.int64)
        tokens = [""] * total

        idx = 0
        for a, t in zip(veh_list, veh_tok):
            states[idx] = a; labels[idx] = 0; tokens[idx] = t; idx += 1
        for a, t in zip(ped_list, ped_tok):
            states[idx] = a; labels[idx] = 1; tokens[idx] = t; idx += 1

        return torch.tensor(states), torch.tensor(labels, dtype=torch.long), tokens

    def _compute_bev_semantic_map_multi_class(
        self,
        annotations: Annotations,
        map_api: AbstractMap,
        ego_pose: StateSE2,
        num_classes: int = 7,            # 0..6
        exclusive: bool = False,  # force one-hot, resolving overlaps
        priority: Optional[List[int]] = None,  # overwrite priority when exclusive, ascending by default
        return_uint8: bool = False,  # uint8 (0/1) instead of bool
    ) -> torch.Tensor:
        """
        Per-class binary BEV masks.
        - channels: [0 bg, 1 road, 2 walkway, 3 centerline, 4 static, 5 vehicle, 6 pedestrian]
        - exclusive=False: multi-hot, overlaps set both channels; bg is the complement
        - exclusive=True: one-hot by priority; bg keeps the pixels nothing wrote to
        Returns (num_classes, H, W).
        """
        H, W = self._config.bev_semantic_frame

        cls_masks = {c: np.zeros((H, W), dtype=np.bool_) for c in range(1, num_classes)}

        for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
            if label == 0 or label >= num_classes:
                continue
            if entity_type == "polygon":
                mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
            elif entity_type == "linestring":
                mask = self._compute_map_linestring_mask(map_api, ego_pose, layers)
            else:
                mask = self._compute_box_mask(annotations, layers)
            cls_masks[label] |= mask

        others_union = np.zeros((H, W), dtype=np.bool_)
        for c in range(1, num_classes):
            others_union |= cls_masks.get(c, np.zeros((H, W), dtype=np.bool_))
        bg_mask = ~others_union

        if not exclusive:
            out = np.zeros((num_classes, H, W), dtype=np.bool_)
            out[0] = bg_mask
            for c in range(1, num_classes):
                out[c] = cls_masks.get(c, np.zeros((H, W), dtype=np.bool_))
        else:
            prio = priority if priority is not None else list(range(1, num_classes))
            label_map = np.zeros((H, W), dtype=np.int16)  # 0 = background
            for c in prio:
                m = cls_masks.get(c, None)
                if m is None:
                    continue
                label_map[m] = c
            out = np.zeros((num_classes, H, W), dtype=np.bool_)
            for c in range(num_classes):
                out[c] = (label_map == c)

        if return_uint8:
            return torch.from_numpy(out.astype(np.uint8))
        else:
            return torch.from_numpy(out)

    def boxes_in_bev_roi(self, boxes: np.ndarray, cfg: SafeDrive_Config) -> np.ndarray:
        """
        Args
            boxes: (N, 7)  [x, y, z, l, w, h, yaw]  or (N, 4+) with x,y at 0,1
        Returns
            mask: (N,) bool  – True if box center inside lidar/BEV bounds
        """
        x = boxes[:, BoundingBoxIndex.X]
        y = boxes[:, BoundingBoxIndex.Y]
        return (
            (0 <= x) & (x <= cfg.lidar_max_x) &
            (cfg.lidar_min_y <= y) & (y <= cfg.lidar_max_y)
    )

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        """
        Transform shapely geometry in local coordinates of origin.
        :param geometry: shapely geometry
        :param origin: pose dataclass
        :return: shapely geometry
        """

        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])

        return rotated_geometry

    def _coords_to_pixel(self, coords):
        """
        Transform local coordinates in pixel indices of BEV map
        :param coords: _description_
        :return: _description_
        """

        # NOTE: remove half in backward direction
        pixel_center = np.array([[0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center

        return coords_idcs.astype(np.int32)

    def transform_boxes_from_future_to_current_ego_frame(self, boxes_future: np.ndarray, points_rel: np.ndarray) -> np.ndarray:
        """Bring future-frame boxes into the current ego frame."""
        dx, dy, dtheta = points_rel  # dtheta: rotation of the future ego pose relative to the current one
        cos_theta = np.cos(dtheta)
        sin_theta = np.sin(dtheta)
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                    [sin_theta, cos_theta]])

        x_future = boxes_future[:, BoundingBoxIndex._X]
        y_future = boxes_future[:, BoundingBoxIndex._Y]
        heading_future = boxes_future[:, BoundingBoxIndex._HEADING]

        coords_future = np.stack((x_future, y_future), axis=-1)
        coords_current = coords_future @ rotation_matrix.T + np.array([dx, dy])
        heading_current = heading_future + dtheta

        boxes_current = boxes_future.copy()
        boxes_current[:, BoundingBoxIndex._X] = coords_current[:, 0]
        boxes_current[:, BoundingBoxIndex._Y] = coords_current[:, 1]
        boxes_current[:, BoundingBoxIndex._HEADING] = heading_current

        return boxes_current

    def _align_future_agent_states(self, source_agent: torch.Tensor, source_tokens: List[str], ref_tokens: List[str]) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """Reorder a future frame's agents onto the current frame's token order."""
        token_to_idx = {token: i for i, token in enumerate(source_tokens)}
        aligned_states = []
        aligned_tokens = []
        aligned_labels = []
        for token in ref_tokens:
            if token in token_to_idx and token != "":  # Allow empty token for padding
                aligned_states.append(source_agent[token_to_idx[token]])
                aligned_tokens.append(token)
                aligned_labels.append(True)
            else:
                # No match found, pad with zero tensor
                aligned_states.append(torch.zeros_like(source_agent[0]))
                aligned_tokens.append("")  # Empty token
                aligned_labels.append(False)

        return (
            torch.stack(aligned_states, dim=0),
            aligned_tokens,
            torch.tensor(aligned_labels, dtype=torch.bool)
        )


class BoundingBox2DIndex(IntEnum):
    """Intenum for bounding boxes in GRAD."""

    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)
