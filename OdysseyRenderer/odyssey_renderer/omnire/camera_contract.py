"""Validated camera/exposure mathematics without native engine imports.

Exposure is applied AFTER full Gaussian+sky composition and BEFORE uint8,
BGR conversion, or image remapping. Camera residuals are used once, at the
selected saved LiDAR tick; this module never advances ego/controller state.
"""
from collections.abc import Mapping
import numpy as np
import torch

CAMERA_NAMES = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0")


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _timestamps(value, name):
    result = _numpy(value)
    if result.dtype != np.int64 or result.ndim != 1 or not len(result) or np.any(np.diff(result) <= 0):
        raise ValueError(name + " must be nonempty strictly increasing int64 timestamps")
    return result


def _camera_identity(names, ids=None):
    names = list(names)
    if len(names) != 8 or len(set(names)) != 8 or set(names) != set(CAMERA_NAMES):
        raise ValueError("saved cameras must contain exactly the eight canonical names")
    expected = [CAMERA_NAMES.index(name) for name in names]
    if ids is not None and list(ids) != expected:
        raise ValueError("camera IDs disagree with canonical camera names")
    return names, expected


def validate_exposure(config, training_timestamps_us, camera_names):
    """Validate actual saved matrices and select requested camera rows."""
    if not isinstance(config, Mapping) or any(config.get(key) != value for key, value in (
        ("schema_version", 1), ("mode", "camera_frame"),
        ("camera_structure", "affine"), ("frame_structure", "affine"))):
        raise ValueError("unsupported saved camera-frame affine exposure schema")
    times = _timestamps(training_timestamps_us, "training timestamps")
    saved = _timestamps(config["training_timestamps_us"], "exposure timestamps")
    if not np.array_equal(times, saved):
        raise ValueError("exposure and scene timestamps disagree")
    names, ids = _camera_identity(config["camera_names"], config.get("camera_ids"))
    requested = list(camera_names)
    if not requested or len(set(requested)) != len(requested) or any(name not in names for name in requested):
        raise ValueError("invalid requested exposure cameras")
    base, residual = config["exposure"], config["residual"]
    for value, shape in ((base, (8, 3, 4)), (residual, (8, len(times), 3, 4))):
        if (not torch.is_tensor(value) or value.dtype != torch.float32 or tuple(value.shape) != shape
                or not bool(torch.isfinite(value).all())):
            raise ValueError("exposure must contain finite saved float32 camera/frame affine tensors")
    indices = [names.index(name) for name in requested]
    return {"camera_names": requested, "exposure": base[indices], "residual": residual[indices],
            "training_timestamps_us": times.copy()}


def apply_exposure(rgb, validated, frame, camera_name):
    """Apply saved base+residual right-multiplication; no identity assumption."""
    if isinstance(frame, (bool, np.bool_)) or int(frame) != frame or not 0 <= frame < validated["residual"].shape[1]:
        raise ValueError("exposure frame outside saved domain")
    if camera_name not in validated["camera_names"]:
        raise ValueError("exposure camera outside selected domain")
    if not torch.is_tensor(rgb) or rgb.ndim < 1 or rgb.shape[-1] != 3 or not rgb.is_floating_point():
        raise ValueError("exposure input must be floating Torch RGB with last dimension three")
    index = validated["camera_names"].index(camera_name)
    matrix = (validated["exposure"][index] + validated["residual"][index, int(frame)]).to(
        device=rgb.device, dtype=rgb.dtype)
    return (rgb.matmul(matrix[:3, :3]) + matrix[:3, 3]).clamp(0, 1)


def _se3(value, shape, name):
    result = np.asarray(_numpy(value), dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(name + " must be finite SE3 matrices with expected shape")
    rotation = result[..., :3, :3]
    if (not np.allclose(result[..., 3, :], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1., atol=1e-6, rtol=0)):
        raise ValueError(name + " contains invalid rotation/homogeneous row")
    return result


def validate_time_calibration(metadata, expected_timestamps=None, expected_anchor=None,
                              expected_camera_names=None, expected_map_name=None):
    """Normalize portable calibration and reject mismatched scene identity."""
    required = {"schema_version": 1, "projection_mode": "training_undistorted",
        "near_plane": .1, "rasterize_mode": "classic", "rigid_camera_time": True,
        "camera_pose_policy": "logged_residual_at_tick"}
    if not isinstance(metadata, Mapping) or any(metadata.get(k) != v for k, v in required.items()):
        raise ValueError("unsupported render/time calibration contract")
    names, ids = _camera_identity(metadata["camera_names"], metadata.get("camera_ids"))
    if expected_camera_names is not None:
        requested = list(expected_camera_names)
        if not requested or len(set(requested)) != len(requested) or any(n not in names for n in requested):
            raise ValueError("requested cameras not represented by saved calibration")
    times = _timestamps(metadata["training_timestamps_us"], "training timestamps")
    if expected_timestamps is not None and not np.array_equal(times, _timestamps(expected_timestamps, "scene timestamps")):
        raise ValueError("training timestamps disagree with scene")
    images = _numpy(metadata["image_timestamps_us"])
    if images.dtype != np.int64 or images.shape != (len(times), 8):
        raise ValueError("image timestamps must be int64 [frames,8]")
    if np.any(np.abs(images - times[:, None]) > 50000):
        raise ValueError("camera time offset exceeds 0.05s")
    anchor = np.asarray(_numpy(metadata.get("recon2world_translation", metadata.get("anchor"))), dtype=np.float64)
    if anchor.shape != (3,) or not np.isfinite(anchor).all():
        raise ValueError("reconstruction anchor must be finite xyz")
    if "anchor" in metadata and not np.allclose(anchor, _numpy(metadata["anchor"]), atol=1e-6, rtol=0):
        raise ValueError("conflicting anchor aliases")
    if expected_anchor is not None:
        expected_anchor = _numpy(expected_anchor)
        if expected_anchor.shape != (3,) or not np.isfinite(expected_anchor).all() or not np.allclose(
                anchor, expected_anchor, atol=1e-6, rtol=0):
            raise ValueError("reconstruction anchor disagrees with scene")
    if expected_map_name is not None and metadata.get("map_name") != expected_map_name:
        raise ValueError("map identity disagrees with scene")
    residual = _se3(metadata["camera_to_ego"], (len(times), 8, 4, 4), "camera_to_ego")
    result = dict(metadata, camera_names=names, camera_ids=ids, training_timestamps_us=times.copy(),
        image_timestamps_us=images.copy(), camera_to_ego=residual.copy(),
        recon2world_translation=anchor.copy(), anchor=anchor.copy())
    have_ego, have_camera = "ego_to_global" in metadata, "camera_to_global" in metadata
    if have_ego != have_camera:
        raise ValueError("global calibration verification requires both ego and camera pose tables")
    if have_ego:
        ego = _se3(metadata["ego_to_global"], (len(times), 4, 4), "ego_to_global")
        cameras = _se3(metadata["camera_to_global"], (len(times), 8, 4, 4), "camera_to_global")
        if not np.allclose(ego[:, None] @ residual, cameras, atol=1e-6, rtol=0):
            raise ValueError("global camera recomposition disagrees")
        result.update(ego_to_global=ego.copy(), camera_to_global=cameras.copy())
    # Validate without replacing raw saved exposure schema in the returned metadata.
    validate_exposure(metadata["exposure"], times, names)
    return result


def sample_rigid_camera_pose(lidar_timestamps_us, image_timestamps_us, frame, camera_index,
                             translations, quaternions, visible, active=None):
    """Neighbor exposure interpolation as in the training code, current absence and boundaries preserved.

    Tensors are [frames,actors,3/4] and visibility is [frames,actors]. Active can
    be the same shape or a static [actors] mask. Returned poses are cloned and
    normalized; the present mask is authoritative, including absent actors.
    """
    times = _timestamps(lidar_timestamps_us, "LiDAR timestamps")
    images = _numpy(image_timestamps_us)
    if images.dtype != np.int64 or images.ndim != 2 or images.shape[0] != len(times):
        raise ValueError("image timestamps must be int64 [frames,cameras]")
    if (isinstance(frame, (bool, np.bool_)) or int(frame) != frame or not 0 <= frame < len(times)
            or isinstance(camera_index, (bool, np.bool_)) or int(camera_index) != camera_index
            or not 0 <= camera_index < images.shape[1]):
        raise ValueError("rigid frame/camera outside saved domain")
    frame, camera_index = int(frame), int(camera_index)
    if (not torch.is_tensor(translations) or translations.ndim != 3
            or translations.shape[0] != len(times) or translations.shape[2] != 3
            or not translations.is_floating_point()):
        raise ValueError("rigid translations must be floating [frames,actors,3]")
    shape = translations.shape[:2]
    if (not torch.is_tensor(quaternions) or tuple(quaternions.shape) != (*shape, 4)
            or quaternions.dtype != translations.dtype or quaternions.device != translations.device):
        raise ValueError("rigid quaternion shape/dtype/device mismatch")
    if not torch.is_tensor(visible) or tuple(visible.shape) != tuple(shape) or visible.dtype != torch.bool:
        raise ValueError("rigid visibility must be bool [frames,actors]")
    if visible.device != translations.device:
        raise ValueError("rigid visibility device mismatch")
    if active is None:
        active = torch.ones_like(visible)
    elif torch.is_tensor(active) and tuple(active.shape) == (shape[1],):
        active = active[None].expand(shape)
    if (not torch.is_tensor(active) or tuple(active.shape) != tuple(shape)
            or active.dtype != torch.bool or active.device != visible.device):
        raise ValueError("rigid active mask shape/dtype/device mismatch")
    delta = int(images[frame, camera_index]) - int(times[frame])
    if abs(delta) > 50000:
        raise ValueError("camera time offset exceeds 0.05s")
    current = visible[frame] & active[frame]
    translation = translations[frame].clone()
    quaternion = torch.nn.functional.normalize(quaternions[frame], dim=-1)
    if (not bool(torch.isfinite(translation[current]).all())
            or not bool(torch.isfinite(quaternions[frame][current]).all())
            or bool((quaternions[frame][current].norm(dim=-1) == 0).any())):
        raise ValueError("invalid current rigid pose")
    neighbor = frame + (1 if delta > 0 else -1 if delta < 0 else 0)
    diagnostics = {"offset_us": delta, "neighbor_frame": neighbor, "alpha": 0.,
                   "interpolated_actor_count": 0, "fallback": None}
    if delta == 0:
        diagnostics["fallback"] = "zero_offset"
        return translation, quaternion, current.clone(), diagnostics
    if not 0 <= neighbor < len(times):
        diagnostics["fallback"] = "neighbor_out_of_range"
        return translation, quaternion, current.clone(), diagnostics
    alpha = float(delta) / float(int(times[neighbor]) - int(times[frame]))
    if not 0 <= alpha <= 1:
        raise ValueError("camera offset exceeds neighbor tick")
    valid = current & visible[neighbor] & active[neighbor]
    if bool(valid.any()):
        other_trans = translations[neighbor][valid]
        other_quat = quaternions[neighbor][valid]
        if (not bool(torch.isfinite(other_trans).all()) or not bool(torch.isfinite(other_quat).all())
                or bool((other_quat.norm(dim=-1) == 0).any())):
            raise ValueError("invalid neighboring rigid pose")
        weight = translation.new_tensor(alpha)
        translation[valid] = translation[valid] + weight * (other_trans - translation[valid])
        a = quaternion[valid]
        b = torch.nn.functional.normalize(other_quat, dim=-1)
        b = torch.where((a * b).sum(-1, keepdim=True) < 0, -b, b)
        quaternion[valid] = torch.nn.functional.normalize((1 - weight) * a + weight * b, dim=-1)
    diagnostics.update(alpha=alpha, interpolated_actor_count=int(valid.sum().item()))
    if not bool(valid.all()):
        diagnostics["fallback"] = "current_or_neighbor_inactive"
    return translation, quaternion, current.clone(), diagnostics
