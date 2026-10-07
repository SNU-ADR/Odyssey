# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
import logging
import os

import numpy as np
import torch
from torch.nn import Parameter
try:
    from gsplat.cuda._wrapper import spherical_harmonics
except ImportError:
    print("Please install gsplat>=1.0.0")

from odyssey_renderer.mtgs.utils.gaussian_utils import quat_mult, quat_to_rotmat, interpolate_quats, IDFT
from odyssey_renderer.mtgs.gaussian_model.vanilla_gaussian_splatting import VanillaPortableGaussianModel
logger = logging.getLogger(__name__)

# Ego position for ODYSSEY_ACTOR_EGO_RADIUS, published by MTGSRender.update_world once per step. Module
# state rather than another argument: the pose is decided three frames below the renderer and the
# value is constant for the whole frame.
_EGO_XY = None
_HOLD_STATS = {"restored": 0, "dropped": 0}


def set_ego_xy(xy):
    """xy: (2,) ego position in the reconstruction frame, or None to leave presence to the log."""
    global _EGO_XY
    _EGO_XY = xy


def actor_mask_stats():
    return dict(_HOLD_STATS)


class RigidPortableSubModel(VanillaPortableGaussianModel):
    """Portable Gaussian Splatting model

    Args:
        asset: portable Gaussian Model with config include
               - type
               - sh_degree
               - scale_dim
               - fourier_features_dim
               - fourier_features_scale
               - fourier_in_space
               - log_timestamps
    """

    # MTGS hides an actor outside its frames by parking it at z=100000; anything above this is that
    # sentinel rather than a real pose.
    ABSENT_Z = 1000.0

    MODEL_TYPE = "rigid"

    def __init__(self, **kwargs):
        self.log_replay = kwargs.get("log_replay", False)
        super().__init__(**kwargs)

    def _update_default_config(self, config):
        config = super()._update_default_config(config)
        config["fourier_features_dim"] = config.get("fourier_features_dim", None)
        config["fourier_features_scale"] = config.get("fourier_features_scale", 1.0)
        config["fourier_in_space"] = config.get("fourier_in_space", 'temporal')
        config["log_timestamps"] = config.get("log_timestamps", None)
        return config

    def load_state_dict(self, dict: dict):
        super().load_state_dict(dict)
        self.exist_log = ('instance_trans' in dict.keys())
        if self.exist_log:
            self.log_trans = Parameter(dict['instance_trans'].squeeze())
            self.log_quats = Parameter(dict['instance_quats'].squeeze())
            self.static_in_log = (self.log_trans.dim() == 1)
            if not self.static_in_log:
                assert getattr(self.config, "log_timestamps", None) is not None
                self.log_start_time = self.config.log_timestamps.min().item()
                self.log_timestamps = self.config.log_timestamps.squeeze() - self.log_start_time

            
            log_type = "static" if self.static_in_log else "dynamic"
            logger.debug(f"Log pose for `{self.model_name_abbr}` loaded. LOG TYPE: `{log_type}`")
            if self.log_replay:
                logger.debug(f"Log replay for `{self.model_name_abbr}` enabled.")

    def set_static_force(self):
        self.static_in_log = True
        in_frame_mask = self.log_trans[:, 2] < self.ABSENT_Z
        self.log_trans = Parameter(self.log_trans[in_frame_mask][0])
        self.log_quats = Parameter(self.log_quats[in_frame_mask][0])

    def get_means(self, global_quat, global_trans):
        local_means = self.gauss_params['means']
        rot_cur_frame = quat_to_rotmat(global_quat[None, ...])[0, ...]
        self.global_means = local_means @ rot_cur_frame.T + global_trans
        return self.global_means

    def get_quats(self, global_quat, global_trans=None):
        local_quats = self.quats / self.quats.norm(dim=-1, keepdim=True)
        global_quats = quat_mult(global_quat[None, ...], local_quats)
        return global_quats

    def get_fourier_features(self, x):
        scaled_x = x * self.config.fourier_features_scale
        input_is_normalized = (self.config.fourier_in_space == 'temporal')
        idft_base = IDFT(scaled_x, self.config.fourier_features_dim, input_is_normalized).to(self.device)
        return torch.sum(self.features_dc * idft_base[..., None], dim=1, keepdim=False)
    
    def get_true_features_dc(self, timestamp=None, cam_obj_yaw=None):
        if self.config.fourier_features_dim is None:
            return self.features_dc
        normalized_x = timestamp if self.config.fourier_in_space == 'temporal' else cam_obj_yaw
        assert normalized_x is not None
        return self.get_fourier_features(normalized_x)

    def get_gaussian_rgbs(self, camera_to_worlds, timestamp, device=None):
        device = device if device is not None else self.device
        assert device != torch.device("cpu"), "`sphereical_harmonics` in `gsplat` only supports CUDA"
        true_features_dc = self.get_true_features_dc(timestamp, None)
        colors = torch.cat((true_features_dc[:, None, :], self.features_rest), dim=1).to(device)
        if self.sh_degree > 0:
            viewdirs = self.global_means.detach().to(device) - camera_to_worlds[..., :3, 3].to(device)  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            rgbs = spherical_harmonics(self.sh_degree, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])

        return rgbs

    def get_opacity(self):
        return torch.sigmoid(self.gauss_params['opacities']).squeeze(-1)

    # Frames where the actor was not in the scene are marked with a z sentinel (~1e5).
    # calibrate_agent_state uses the same threshold for the agents it claims.
    ABSENT_Z = 1000.0

    def _present_cache(self):
        """(present mask, filtered ts on device, filtered ts on host, unique?) built once.

        Presence and timestamps never change after load, so the mask, the gather and the numpy
        mirror are all computed on first use. This is what lets the index search below run on the
        host: without the cache, `bool(present.any())` alone would sync every actor every frame.
        """
        cached = getattr(self, "_present_cache_v", None)
        if cached is None:
            present = self.log_trans[:, 2] < self.ABSENT_Z
            ts = self.log_timestamps[present]
            host = ts.detach().cpu().numpy()
            cached = (present, ts, host, bool(np.unique(host).size == host.size))
            self._present_cache_v = cached
        return cached

    def _pose_indices_host(self, host_ts, unique, rel):
        """(prev, next) as ints, or None when the device path must decide.

        argmin's choice among equal minima is not something to rely on, so a tie sends the caller
        back to the tensor path: duplicate timestamps, or a query outside the range where one of the
        masked arrays is all +inf.
        """
        if not unique or host_ts.size == 0:
            return None
        r = host_ts.dtype.type(rel)
        diffs = r - host_ts
        if diffs[diffs >= 0].size == 0 or diffs[diffs <= 0].size == 0:
            return None
        return (int(np.argmin(np.where(diffs >= 0, diffs, np.inf))),
                int(np.argmin(np.where(diffs <= 0, -diffs, np.inf))))

    def _revive_near_ego(self, trans, quats, host_ts, rel, timestamp):
        """Hold an actor at its nearest real pose while that pose is beside the ego.

        Only reached where the log has nothing to interpolate. The pose is FROZEN, not
        extrapolated: a stationary car in the right place is a bounded, visible error, while
        invented motion would put a fabricated trajectory into an evaluation.
        """
        radius = os.environ.get("ODYSSEY_ACTOR_EGO_RADIUS", "")
        if not radius or _EGO_XY is None or host_ts.size == 0:
            _HOLD_STATS["dropped"] += 1
            return None, None, timestamp
        k = int(np.argmin(np.abs(host_ts - host_ts.dtype.type(rel))))
        pose = trans[k]
        ego = _EGO_XY.to(device=pose.device, dtype=pose.dtype)
        if float(torch.linalg.norm(pose[:2] - ego)) > float(radius):
            _HOLD_STATS["dropped"] += 1
            return None, None, timestamp
        _HOLD_STATS["restored"] += 1
        return quats[k], pose, timestamp

    def _get_log_pose_from_timestamp(self, timestamp):
        self.log_timestamps = self.log_timestamps.to(self.device)
        relative_timestamp = timestamp - self.log_start_time

        # Absent frames sit at the z sentinel; interpolating through one drags the actor toward it,
        # so they are not candidates at all. Cached -- presence cannot change after load.
        present, ts, host_ts, unique = self._present_cache()
        if host_ts.size == 0:
            return None, None, timestamp
        trans = self.log_trans[present]
        quats = self.log_quats[present]

        # Outside the node's own range there is nothing to interpolate between, and t would
        # extrapolate. A block built from several traversals is full of nodes whose timestamps are
        # hours from this rollout, and every one of them lands here.
        if relative_timestamp < host_ts[0] or relative_timestamp > host_ts[-1]:
            return self._revive_near_ego(trans, quats, host_ts, relative_timestamp, timestamp)

        idx = self._pose_indices_host(host_ts, unique, relative_timestamp)
        if idx is not None:
            prev_frame, next_frame = idx
        else:
            diffs = relative_timestamp - ts
            prev_frame = torch.argmin(torch.where(diffs >= 0, diffs, float('inf')))
            next_frame = torch.argmin(torch.where(diffs <= 0, -diffs, float('inf')))

        if next_frame == prev_frame:
            # Timestamp exactly matches a frame, no interpolation needed
            return quats[next_frame], trans[next_frame], timestamp

        # prev and next can straddle a stretch the actor was absent for; bridging it would draw the
        # car gliding across road it was never on.
        gap = ts[next_frame] - ts[prev_frame]
        if len(host_ts) > 1:
            step = float(np.median(np.diff(host_ts)))
            if float(gap) > 2.5 * step:
                return self._revive_near_ego(trans, quats, host_ts, relative_timestamp, timestamp)

        t = (relative_timestamp - ts[prev_frame]) / gap
        # torch.lerp requires weight and the interpolated tensors to share a dtype. t's dtype
        # follows whatever relative_timestamp/ts are (float64 for a scenario whose timestamps
        # come from a plain numpy/python computation, e.g. omnire_timestamps_us-derived), while
        # trans/quats are float32 -- cast to match rather than assume either side's dtype.
        t = t.to(trans.dtype) if torch.is_tensor(t) else trans.new_tensor(t)
        quat_interp = interpolate_quats(quats[prev_frame], quats[next_frame], t).squeeze()
        trans_interp = torch.lerp(trans[prev_frame], trans[next_frame], t)

        return quat_interp, trans_interp, timestamp

    def _decide_global_pose(self, quat=None, trans=None, timestamp=None):
        if self.static_in_log:
            return self.log_quats, self.log_trans, timestamp        
        if quat is not None and trans is not None:
            assert quat.shape == (4,) and trans.shape == (3,)
            return quat.float(), trans.float(), timestamp
        return self._get_log_pose_from_timestamp(timestamp)

    def get_global_gaussians(self, quat=None, trans=None, timestamp=None, **kwargs):
        quat, trans, timestamp = self._decide_global_pose(
                                        quat=quat,
                                        trans=trans,
                                        timestamp=timestamp,
                                    )
        if quat is None or trans is None:
            # No pose the log actually held at this time -- update_world skips a node whose
            # gaussians come back None, which is what "absent" should look like.
            return None

        return {
            "means": self.get_means(global_trans=trans, global_quat=quat),
            "scales": self.get_scales(),
            "quats": self.get_quats(global_trans=trans, global_quat=quat),
            "opacities": self.get_opacity(),
        }
