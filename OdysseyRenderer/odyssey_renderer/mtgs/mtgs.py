# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
from __future__ import annotations
import logging
from functools import cached_property
from typing import Tuple, Union
from typing_extensions import Literal
from pathlib import Path

import numpy as np
import cv2
from pyquaternion import Quaternion
import os
import torch

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

try:
    from gsplat.cuda._wrapper import spherical_harmonics
except ImportError:
    spherical_harmonics = None

from odyssey.scenario.scenarios.scenario_description import ScenarioDescription as SD
from odyssey.engine.engine_utils import get_global_config
from odyssey.utils.geometry_utils import Sim2

from odyssey_renderer.base_renderer import BaseRenderer, RenderState
from odyssey_renderer.mtgs.restorer_hook import get_restorer
from odyssey_renderer.mtgs.utils.gaussian_utils import matrix_to_quaternion, quat_to_rotmat, quat_to_angle
from odyssey_renderer.mtgs.utils.portable_utils import convert_to_attribute_dict, AttrDict, alias_numpy2_pickle_modules
from odyssey_renderer.mtgs.utils.image_utils import rgb_float_to_bgr_uint8
from odyssey_renderer.mtgs.gaussian_model import rigid_object
from odyssey_renderer.mtgs.gaussian_model.vanilla_gaussian_splatting import VanillaPortableGaussianModel as VanillaModel
from odyssey_renderer.mtgs.gaussian_model.rigid_object import RigidPortableSubModel as RigidModel
from odyssey_renderer.mtgs.gaussian_model.rigid_object_mirrored import (
    MirroredRigidPortableSubModel as MirroredModel, flip_spherical_harmonics,
)

logger = logging.getLogger(__name__)


def save_pre_restore_enabled():
    """ODYSSEY_SAVE_PRE_RESTORE=1 (runner config save_pre_restore): keep each camera image as
    rendered, before the restorer, under the key 'image_pre_restore' for the data manager."""
    return os.environ.get("ODYSSEY_SAVE_PRE_RESTORE", "0") in ("1", "true", "True")

MODEL_MAPPING = {
    "VanillaGaussianSplattingModel": VanillaModel,
    "SkyboxGaussianSplattingModel": VanillaModel,
    "RigidSubModel": RigidModel,
    "MirroredRigidSubModel": MirroredModel,
    "DeformableSubModel": None,
}


# --------------------------------------------------------------------------- #
# ODYSSEY_RENDER_CAMS: render only the cameras the planner actually reads.
#
# The scenario carries all 8 surround views, but no planner we run reads all 8 -- LTF asks
# for 3 (navsim transfuser_agent.py sets cam_l1/l2/r1/r2/b0 False) and DrivoR for 4
# (drivoR.yaml gives cam_f0/l0/r0/b0 a history and the rest an empty list). Rendering the
# rest is pure waste, and with an image restorer it is expensive waste: the restorer restores
# every rendered view, and its cost is linear in the batch.
#
# Comma-separated camera names, e.g. "cam_f0,cam_l0,cam_r0". Empty (the default) keeps all
# 8, so this is inert until a rollout opts in.
#
# ORDER IS THE SCENARIO'S, NOT THE ENV VAR'S. sensor_mapping numbers the cameras by their
# enumeration order below, and the planner-side feature builders index views positionally
# (DrivoR's image encoder treats index 0 as the front camera). Reordering here would
# silently feed the model a permuted view stack, so this filters `sensors` in place and
# never reorders it. Unknown names are a hard error rather than a silent no-op -- a typo'd
# camera would otherwise quietly render the full 8 and look like the flag did nothing.
def _select_cams(sensors):
    want = [c.strip() for c in os.environ.get("ODYSSEY_RENDER_CAMS", "").split(",") if c.strip()]
    if not want:
        return sensors
    # Case-insensitive on purpose. The scenario names its cameras in upper case (CAM_F0),
    # while every caller and doc writes the flag in lower case (cam_f0) -- navsim's own
    # sensor configs use the lower form, which is where the lists were copied from. Matching
    # exactly would reject the documented spelling outright, so compare case-folded and keep
    # the scenario's own keys.
    keep = {c.upper() for c in want}
    unknown = [c for c in want if c.upper() not in {k.upper() for k in sensors}]
    if unknown:
        raise ValueError(
            f"ODYSSEY_RENDER_CAMS names cameras this scenario does not have: {unknown}. "
            f"Available: {sorted(sensors.keys())}")
    return {name: info for name, info in sensors.items() if name.upper() in keep}

# --------------------------------------------------------------------------- #
# Keep the render on the GPU through undistort (ODYSSEY_RENDER_GPU_UNDISTORT=1)
#
# Default OFF. The default path (rgb_float_to_bgr_uint8 below + cv2.remap) is untouched and
# stays bit-identical to what it is today (render/mtgs/utils/test_image_utils.py pins that).
# This flag adds a second, GPU-resident path: undistort becomes a grid_sample against the
# SAME maps OpenCV would have used, and there is exactly one device->host copy, as uint8
# (50 MB instead of 199 MB for 8 views at 1080p -- measured 28.7 ms vs 104.7 ms). It cannot
# be combined with an image restorer, which takes host-side uint8 images.
#
# EXACTNESS. Not bit-identical to the cv2.remap path, and the reason is the opposite of what
# you would guess -- measured, not assumed:
#
#     grid_sample vs exact float bilinear :  max 0.004   <- this path is exact
#     cv2.remap   vs exact float bilinear :  max 5.98     <- the CURRENT/default path is not
#
# cv2.remap quantises its interpolation weights to 5 bits (INTER_BITS=32 steps) whatever the
# input depth, so at a high-contrast edge it can be ~6/255 off true bilinear. grid_sample
# computes the weights in fp32. So the GPU path is STRICTLY MORE ACCURATE, and the pixels
# still change relative to today's baseline -- which is why this is flagged and judged by
# frame-locked pixel comparison, not an equality assert.
#
# The quantise/resample ORDER is preserved (quantise to uint8 levels first, then resample,
# then round) so the flag does not silently move the pipeline to float-domain undistort.
# Depth is deliberately left on the original CPU path: it is not restored, not consumed by
# the planner's image encoder, and porting it would widen the blast radius for nothing.
_GPU_UNDISTORT = None


def _gpu_undistort_enabled() -> bool:
    global _GPU_UNDISTORT
    if _GPU_UNDISTORT is None:
        _GPU_UNDISTORT = os.environ.get("ODYSSEY_RENDER_GPU_UNDISTORT", "0") in ("1", "true", "True")
    return _GPU_UNDISTORT


def _remap_grid(map_inverse_distorts, height, width, device):
    """cv2 absolute-pixel remap maps -> one (C,H,W,2) grid_sample grid, built once.

    cv2.remap reads source pixel (map1[y,x], map2[y,x]) with integer coordinates at pixel
    CENTRES. grid_sample(align_corners=False) reads normalised g where the source pixel
    index is ((g+1)*W - 1)/2, so g = (2*x + 1)/W - 1 inverts it exactly. Out-of-range reads
    give 0 on both sides (BORDER_CONSTANT vs padding_mode="zeros").
    """
    grids = []
    for map1, map2 in map_inverse_distorts:
        gx = torch.from_numpy((2.0 * map1.astype(np.float32) + 1.0) / width - 1.0)
        gy = torch.from_numpy((2.0 * map2.astype(np.float32) + 1.0) / height - 1.0)
        grids.append(torch.stack([gx, gy], dim=-1))
    return torch.stack(grids, dim=0).to(device=device, dtype=torch.float32).contiguous()


def auto_submodel(atom_asset):
    if not isinstance(atom_asset, AttrDict):
        atom_asset = convert_to_attribute_dict(atom_asset)
    original_model_type = atom_asset.config.type
    return MODEL_MAPPING[original_model_type]


class MTGSRenderEngine(BaseRenderer):
    def __init__(
        self,
        *args,
        enable_collider: bool = False,
        render_depth: bool = False,
        rasterize_mode: Literal["classic", "antialiased"] = "antialiased",
        radius_clip: float = 0.,
        background_color: Union[Literal["random", "black", "white"], Tuple] = "black",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.enable_collider = enable_collider
        self.render_depth = render_depth
        self.rasterize_mode = rasterize_mode
        self.radius_clip = radius_clip
        self.bg_color = self._init_background_color(background_color)
        self.asset_manager = MTGSAssetManager(self.device)
        self.sensor_caches = None
        # P1's grid_sample grid is derived from map_inverse_distorts, so it must die with them.
        self._remap_grid_cache = None
        # Per-asset render caches. All three are keyed by nothing but the loaded asset, so
        # set_asset()/reset() must clear them or a scene switch renders the previous scene's
        # background. See _clear_asset_caches().
        self._static_gs_cache = {}      # submodel name -> get_global_gaussians() output
        self._sh_color_cache = {}       # submodel name -> assembled (N,16,3) SH coefficients
        self._static_names = None       # submodel names whose gaussians never depend on time

    def _init_background_color(self, background_color):
        if isinstance(background_color, str):
            if background_color == "random":
                background = torch.rand(3, device=self.device)
            elif background_color == "white":
                background = torch.ones(3, device=self.device)
            elif background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                raise ValueError(f"Unknown background color {background_color}")
            return background
        elif isinstance(background_color, tuple):
            assert len(background_color) == 3
            for color in background_color:
                assert 0 <= color <= 255
            return torch.tensor(background_color, device=self.device).float() / 255.

    @property
    def background_color(self) -> torch.Tensor:
        return self.bg_color.to(self.device)

    def _prepare_metas(self):
        self.model_types = {}
        self.node_types = {}
        self.submodel_names = {}
        self.timestamp = None
        self.sensor_mapping = {}

    def _init_gaussian_models(self):
        if hasattr(self, "gaussian_models"):
            del self.gaussian_models
            torch.cuda.empty_cache()
        self.gaussian_models = torch.nn.ParameterDict()

    @cached_property
    def background_asset(self):
        return self.gaussian_models["background"]

    @property
    def num_submodels(self):
        return len(self.node_types.keys())

    @property
    def num_cameras(self):
        return len(self.sensor_mapping.keys())

    def reset(self, current_scene_id, **kwargs):
        current_scene: SD = self.engine.managers['scenario_manager'].current_scene
        # The scene's reconstruction is given by path (scene_checkpoint: <scene>/scene.ckpt of the
        # published release); nothing is looked up by name.
        checkpoint = get_global_config().get('scene_checkpoint')
        if not checkpoint:
            raise SystemExit('[asset] scene_checkpoint is not set: pass the scene\'s Gaussian checkpoint '
                             '(<scene>/scene.ckpt)')
        checkpoint = Path(checkpoint)
        _check_asset_identity(current_scene, checkpoint)
        reset_asset = self.asset_manager.reset(checkpoint)
        if reset_asset:
            self._prepare_metas()
            self._init_gaussian_models()
            self.set_asset(self.asset_manager.background_asset)
            self.digitaltwin_agent2states = self.calibrate_agent_state()
            self.sensor_caches = None
            self._clear_asset_caches()

    def _clear_asset_caches(self):
        """Drop every cache whose contents belong to the previously loaded asset.

        A rollout walks many scenes in one process, so this is the failure mode that matters:
        a stale entry does not raise, it renders the previous scene's background into the new
        one. Called from reset() alongside sensor_caches, which has the same lifetime.
        """
        self._static_gs_cache = {}
        self._sh_color_cache = {}
        self._static_names = None
        self._remap_grid_cache = None
        torch.cuda.empty_cache()

    def _is_static_model(self, model) -> bool:
        """True when the model's global gaussians do not depend on the timestamp.

        Only VanillaModel (background / skybox) qualifies: its get_global_gaussians() ignores
        every kwarg and returns exp(scales) / normalised quats / sigmoid(opacities) over
        parameters that never change. Rigid submodels interpolate a log pose, so they are out.
        Checked by type rather than by name so an asset that names its background something
        else is still handled -- and so a future dynamic model is excluded by default.
        """
        # A subclass may declare that its output cannot change between frames.
        # Without this, an OmniRe node that is genuinely static (the road) drops
        # out of the memo and re-runs exp/clamp/norm/sigmoid over every one of its
        # Gaussians on every frame -- 10.9 M of them in a large asset.
        return type(model) is VanillaModel or getattr(model, "STATIC_IN_WORLD", False)

    def _static_model_names(self):
        if self._static_names is None:
            # Built from submodel_names rather than by iterating gaussian_models: that is a
            # ParameterDict holding modules, and set_asset is the only thing that fills it.
            self._static_names = {
                name for name in set(self.submodel_names.values())
                if self._is_static_model(self.gaussian_models[name])
            }
        return self._static_names

    @torch.no_grad()
    def _global_gaussians_cached(self, model_name, gaussian_model, quat, trans, timestamp):
        """get_global_gaussians(), memoised for models whose output cannot change.

        The background and skybox are the bulk of the scene (order 1.3 M gaussians against
        ~350 k for all actors) and update_world's only guard is the per-timestamp early
        return, which never fires in a rollout because time advances every step. So exp() /
        norm() / sigmoid() over those 1.3 M ran once per frame to produce the same numbers.

        Two things are deliberately NOT changed here:
          * the concat ORDER in update_world -- gsplat's alpha compositing keeps the
            floating-point accumulation order of the input arrays, so reordering them shifts
            pixels (measured: permuting 1.4 M gaussians moves 149 px by +-1). The cache
            returns values into the same position in the same loop.
          * grad. no_grad wraps this because gauss_params are nn.Parameters with
            requires_grad=True; without it the cached tensors pin an autograd graph for the
            renderer's lifetime, which defeats the point of caching.
        """
        if model_name not in self._static_model_names():
            return gaussian_model.get_global_gaussians(quat=quat, trans=trans, timestamp=timestamp)
        hit = self._static_gs_cache.get(model_name)
        if hit is None:
            hit = gaussian_model.get_global_gaussians(quat=quat, trans=trans, timestamp=timestamp)
            self._static_gs_cache[model_name] = hit
        return hit

    def calibrate_agent_state(self):
        """Get the agent states in the reconstruction."""
        digitaltwin_agent2states = {}
        current_scene: SD = self.engine.managers['scenario_manager'].current_scene
        digitaltwin_ego2globals = torch.tensor(
            np.array(current_scene[SD.METADATA]['digitaltwin_ego2globals']), dtype=torch.float64, device=self.device
        )
        digitaltwin_ego2globals_trans = digitaltwin_ego2globals[:, :3, 3]
        digitaltwin_ego2globals_quat = matrix_to_quaternion(digitaltwin_ego2globals[:, :3, :3])
        digitaltwin_agent2states['ego'] = {
            'translation': digitaltwin_ego2globals_trans,
            'rotation': digitaltwin_ego2globals_quat,
        }

        for asset_token, model_name in self.submodel_names.items():
            if self.node_types[asset_token] != 'na':
                continue

            if self.gaussian_models[model_name].static_in_log:
                continue

            in_frame_mask = self.gaussian_models[model_name].log_trans[:, 2] < 1000

            digitaltwin_agent2states[asset_token] = {
                'translation': self.gaussian_models[model_name].log_trans[in_frame_mask].double(),
                'rotation': self.gaussian_models[model_name].log_quats[in_frame_mask].double(),
            }
        return digitaltwin_agent2states

    def get_submodel_name(self, token):
        node_type = self.node_types[token]
        if len(node_type) > 3:
            return token
        model_type = self.model_types[token]
        return f"{node_type}_{model_type}_{token}"

    def set_asset(self, asset):
        self.recon2global_translation = asset['background']['config']['recon2world_translation']

        for asset_name in asset.keys():
            logger.debug(f"Loading asset <{asset_name}>...")
            sub_asset = convert_to_attribute_dict(asset[asset_name])
            asset_class = auto_submodel(sub_asset)
            asset_token = asset_name.split("_")[-1]
            # "ground" joins these as scenery: node_type != "na" keeps it out of
            # the digital-twin agent tables, which read log_trans/static_in_log
            # attributes only actor submodels have.
            if asset_token in ["background", "skybox", "ground"]:
                self.node_types[asset_token] = asset_token
            else:
                self.node_types[asset_token] = "na"

            self.model_types[asset_token] = asset_class.MODEL_TYPE
            model_name = self.get_submodel_name(asset_token)
            self.submodel_names[asset_token] = model_name
            self.gaussian_models[model_name] = asset_class(
                asset=sub_asset,
                model_name=model_name
            )

    @staticmethod
    def get_transform_matrix(rotation, translation):
        if not isinstance(rotation, np.ndarray):
            rotation = rotation.cpu().numpy()
        if not isinstance(translation, np.ndarray):
            translation = translation.cpu().numpy()
        if rotation.ndim == 1:
            # is quaternion
            rotation = Quaternion(rotation).rotation_matrix
        if translation.ndim == 1:
            # change to column vector
            translation = np.expand_dims(translation, axis=-1)
        if rotation.ndim == 2:
            rotation = np.expand_dims(rotation, axis=0)
            translation = np.expand_dims(translation, axis=0)

        transform = np.tile(np.eye(4), (rotation.shape[0], 1, 1))
        transform[:, :3, :3] = rotation
        transform[:, :3, 3:4] = translation
        return torch.from_numpy(transform).float()

    def set_sensors(self, sensors, ego2global):
        if self.sensor_caches is not None:
            camera_to_egos, intrinsics, map_inverse_distorts, hw_dict = self.sensor_caches
            camera_to_worlds = torch.einsum("ij,bjk->bik", ego2global, camera_to_egos)
            return camera_to_worlds, intrinsics, map_inverse_distorts, hw_dict

        sensors = _select_cams(sensors)
        self.sensor_mapping = {}
        sensor2egos = []
        intrinsics = []
        map_inverse_distorts = []
        height, width = None, None
        for idx, (cam_name, cam_info) in enumerate(sensors.items()):
            self.sensor_mapping[cam_name] = idx
            sensor2ego = self.get_transform_matrix(
                rotation=cam_info["sensor2ego_rotation"], 
                translation=cam_info["sensor2ego_translation"]
            ).squeeze().to(self.device)             # 4 x 4

            intrinsic = np.array(cam_info["intrinsic"])
            distortion = np.array(cam_info["distortion"])
            new_intrinsic, roi = cv2.getOptimalNewCameraMatrix(
                intrinsic, distortion, (cam_info["width"], cam_info["height"]), 1
            )
            map_inverse_distort = cv2.initInverseRectificationMap(
                intrinsic, distortion, None, new_intrinsic, (cam_info["width"], cam_info["height"]), cv2.CV_32FC1
            )

            intrinsic_torch = torch.from_numpy(new_intrinsic).float().to(self.device).squeeze()
            sensor2egos.append(sensor2ego)
            intrinsics.append(intrinsic_torch)
            map_inverse_distorts.append(map_inverse_distort)

            assert (height is None) or (height == cam_info["height"]), "inconsistent height among RGB cameras"
            assert (width is None) or (width == cam_info["width"]), "inconsistent width among RGB cameras"
            height, width = cam_info["height"], cam_info["width"]

        sensor2egos = torch.stack(sensor2egos, dim=0)      # C x 4 x 4
        intrinsics = torch.stack(intrinsics, dim=0)     # C x 3 x 3
        height = height if isinstance(height, int) else height.item()
        width = width if isinstance(width, int) else width.item()
        self.sensor_caches = (sensor2egos, intrinsics, map_inverse_distorts, {"height": height, "width": width})

        camera_to_worlds = torch.einsum("ij,bjk->bik", ego2global, sensor2egos)
        return camera_to_worlds, intrinsics, map_inverse_distorts, {"height": height, "width": width}

    def update_world(self, timestamp, agent_states):
        if self.timestamp == timestamp:
            return
        self.timestamp = timestamp
        if hasattr(self, "collected_gaussians"):
            del self.collected_gaussians
        self.collected_gaussians = {}

        # See rigid_object.set_ego_xy -- ODYSSEY_ACTOR_EGO_RADIUS reads this to decide whether an actor
        # the log has dropped is close enough to the ego to be worth restoring.
        ego_state = agent_states.get('ego')
        rigid_object.set_ego_xy(
            torch.as_tensor(ego_state[:2], dtype=torch.float32) if ego_state is not None else None
        )

        gs_dict = {
            "means": [],
            "scales": [],
            "quats": [],
            "opacities": [],
        }

        # Which nodes actually contributed geometry this frame. update_gaussian_rgbs has to
        # walk the same list: a node that was skipped never ran get_means, so it has no
        # global_means to colour, and its colours would not line up with the means anyway.
        #
        # Nodes DO skip now: a chunk-bank checkpoint's deformable node returns None outside its
        # own config.log_timestamps (blended_deformable.py). A skipped node is an actor the
        # simulator is moving and the camera cannot see -- the planner has no way to avoid it,
        # so the scorer must not charge it. MetricManager reads this through
        # `rendered_tokens()`; see there for why the set, not the actor list, is the invariant.
        self.active_tokens = []

        for asset_token in self.node_types.keys():
            model_name = self.submodel_names[asset_token]
            gaussian_model = self.gaussian_models[model_name]
            if asset_token in agent_states.keys():
                quat, trans = self.get_agent_pose(asset_token, agent_states[asset_token])
            else:
                quat, trans = None, None            # use original log

            gs = self._global_gaussians_cached(
                model_name, gaussian_model,
                quat=quat,
                trans=trans,
                timestamp=timestamp,
            )
            if gs is None:
                continue
            self.active_tokens.append(asset_token)
            for k in gs_dict.keys():
                gs_dict[k].append(gs[k].to(self.device))

        for key, value in gs_dict.items():
            self.collected_gaussians[key] = torch.cat(value, dim=0)

    @torch.no_grad()
    def _sh_colors_cached(self, model_name, gaussian_model):
        """The pose- and camera-independent front half of get_gaussian_rgbs(), memoised.

        get_gaussian_rgbs() is called once PER CAMERA, but its first steps -- cat(features_dc,
        features_rest), and for mirrored actors repeat(2,...) + flip_spherical_harmonics --
        look at neither the pose nor the camera. With 8 cameras that assembly ran 8x per model
        per frame for nothing. The background is the expensive one: its coefficient tensor is
        (N,16,3) float32, about 252 MB at 1.3 M gaussians, rebuilt eight times a frame.

        The view-dependent part (spherical_harmonics over viewdirs) is NOT cached and still
        runs per camera -- only the assembly is reused.

        Returns None when caching would not be provably safe, and the caller falls back to
        the original whole-of-get_gaussian_rgbs path:
          * spherical_harmonics unavailable (gsplat import failed) -- the fallback path has
            its own sigmoid branch for sh_degree == 0 and we do not duplicate it here;
          * sh_degree == 0, where the colours are sigmoid(colors[:,0,:]) and there is nothing
            worth holding;
          * fourier_features_dim is set, where get_true_features_dc actually consumes the
            timestamp so the assembly is not static;
          * any model type other than the two we transcribed the assembly from.

        MEMORY. This trades ~252 MB of VRAM for the background plus roughly half that across
        the actors. If a rollout is already tight on VRAM, ODYSSEY_SH_COLOR_CACHE=actors keeps the
        actor caches and skips the background one (which is the worst ratio of bytes held to
        time saved, being a single large cat).
        """
        if spherical_harmonics is None:
            return None
        if getattr(gaussian_model, "sh_degree", 0) <= 0:
            return None
        if getattr(gaussian_model.config, "fourier_features_dim", None) is not None:
            return None

        scope = os.environ.get("ODYSSEY_SH_COLOR_CACHE", "all")
        if scope not in ("all", "actors", "off"):
            raise ValueError(f"ODYSSEY_SH_COLOR_CACHE must be all|actors|off, got {scope!r}")
        if scope == "off":
            return None
        is_static = model_name in self._static_model_names()
        if scope == "actors" and is_static:
            return None

        cached = self._sh_color_cache.get(model_name)
        if cached is not None:
            return cached

        if type(gaussian_model) is MirroredModel:
            # Transcribed from MirroredRigidPortableSubModel.get_gaussian_rgbs: cat ->
            # unsqueeze/repeat(2,1,1,1) -> flip the mirrored half -> view. Identical ops in
            # identical order, so the result is the same tensor it would have built inline.
            true_features_dc = gaussian_model.get_true_features_dc(self.timestamp, None)
            colors = torch.cat(
                (true_features_dc[:, None, :], gaussian_model.features_rest), dim=1
            ).to(self.device)
            colors = colors.unsqueeze(0).repeat(2, 1, 1, 1)
            colors[1, ...] = flip_spherical_harmonics(colors[1, ...])
            colors = colors.view(-1, 16, 3)
        elif is_static:
            # VanillaPortableGaussianModel.get_gaussian_rgbs: a single cat, no mirroring.
            # That method asserts features_dc.dim() == 2 first; rather than duplicate the
            # assert, hand the odd case back to the model so it raises its own message.
            if gaussian_model.features_dc.dim() != 2:
                return None
            colors = torch.cat(
                (gaussian_model.features_dc[:, None, :], gaussian_model.features_rest), dim=1
            ).to(self.device)
        else:
            # RigidPortableSubModel and anything else: not transcribed, so not cached.
            return None

        self._sh_color_cache[model_name] = colors
        return colors

    @staticmethod
    def _rgbs_from_colors(gaussian_model, colors, camera_to_world):
        """The view-dependent half of get_gaussian_rgbs(), verbatim.

        Only `colors` came from the cache; the arithmetic and the function called are the
        same ones the model would have run. Mirrored models evaluate against global_means
        (set by get_means during update_world); vanilla models against their own means.
        """
        device = colors.device
        means = (gaussian_model.global_means if type(gaussian_model) is MirroredModel
                 else gaussian_model.means)
        viewdirs = means.detach().to(device) - camera_to_world[..., :3, 3].to(device)
        viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
        rgbs = spherical_harmonics(gaussian_model.sh_degree, viewdirs, colors)
        return torch.clamp(rgbs + 0.5, 0.0, 1.0)

    def update_gaussian_rgbs(self, camera_to_worlds):
        rgb_list = []
        for asset_token in getattr(self, 'active_tokens', list(self.node_types.keys())):
            model_name = self.submodel_names[asset_token]
            gaussian_model = self.gaussian_models[model_name]
            colors = self._sh_colors_cached(model_name, gaussian_model)
            rgbs = []
            for i in range(camera_to_worlds.shape[0]):
                if colors is None:
                    per_cam = gaussian_model.get_gaussian_rgbs(
                        camera_to_worlds=camera_to_worlds[i:i+1],
                        timestamp=self.timestamp,
                        device=self.device
                    )
                else:
                    per_cam = self._rgbs_from_colors(
                        gaussian_model, colors, camera_to_worlds[i:i+1])
                rgbs.append(per_cam.unsqueeze(0))
            rgbs = torch.cat(rgbs, dim=0)       # C x ni x 3
            rgb_list.append(rgbs)
            if torch.isnan(rgbs).any() or torch.isinf(rgbs).any():
                print(f"NaN or inf in rgbs for model {model_name}")

        self.collected_gaussians['rgbs'] = torch.cat(rgb_list, dim=1)   # C x N x 3

    @torch.no_grad()
    def render(self, render_state: RenderState):
        timestamp = render_state[RenderState.TIMESTAMP]
        agent_states = render_state[RenderState.AGENT_STATE]
        for agent_state in agent_states.values():
            agent_state[:2] -= self.recon2global_translation[:2]

        self.update_world(
            timestamp=timestamp,
            agent_states=agent_states,
        )
        ego2global_raw = self.get_agent_pose('ego', agent_states['ego'], return_matrix=True)
        ego2global = ego2global_raw.to(device=self.device, dtype=torch.float32)

        if render_state.get(RenderState.SKIP_CAMERAS):
            return self._pose_only_render(ego2global_raw, render_state)

        cameras = render_state[RenderState.CAMERAS]
        camera_to_worlds, INTRINSICS, map_inverse_distorts, SHAPE = self.set_sensors(
            sensors=cameras,
            ego2global=ego2global,
        )
        self.update_gaussian_rgbs(camera_to_worlds)
        
        # shift the camera to center of scene looking at center
        R = camera_to_worlds[:, :3, :3]     # C x 3 x 3
        T = camera_to_worlds[:, :3, 3:4]    # C x 3 x 1
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.transpose(1,2)            # C x 3 x 3
        T_inv = -R_inv @ T
        viewmat = self.get_transform_matrix(
                rotation=R_inv, 
                translation=T_inv
            ).to(self.device)               # C x 4 x 4

        if self.render_depth:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        render, alpha, _ = rasterization(
            means=self.collected_gaussians['means'],
            quats=self.collected_gaussians['quats'],
            scales=self.collected_gaussians['scales'],
            opacities=self.collected_gaussians['opacities'],
            colors=self.collected_gaussians['rgbs'],
            viewmats=viewmat,       # [C, 4, 4]
            Ks=INTRINSICS.cuda(),   # [C, 3, 3]
            width=SHAPE['width'],
            height=SHAPE['height'],
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            radius_clip=self.radius_clip,
        )
        alpha = alpha[:, ...]
        rgb_raw = self.composite_background(render[:, ..., :3], alpha,
                                            camera_to_worlds, INTRINSICS, SHAPE)

        return_dict = self._postprocess_render(
            rgb_raw, render, alpha, map_inverse_distorts, SHAPE, render_mode)

        ego2global_nuplan = ego2global_raw.cpu().numpy()
        ego2global_nuplan[:3, 3] = ego2global_nuplan[:3, 3] + self.recon2global_translation[:3]

        meta_data_dict = {
            'ego2global': ego2global_nuplan,
            'render_state': render_state
        }
        return_dict.update(meta_data_dict)
        return return_dict

    def _pose_only_render(self, ego2global_raw, render_state):
        """A skipped step (render_every): the ego pose, no images, no restorer call.

        `cameras` is empty rather than absent so the data manager takes its existing branch for
        a camera that was not rendered -- it blanks that camera's data_path and writes nothing.
        No `camera_calibrations` either: the exporter leaves the frame's own calibration alone
        when the key is missing, which is what a frame with no image should carry.
        """
        ego2global = ego2global_raw.cpu().numpy()
        ego2global[:3, 3] = ego2global[:3, 3] + self.recon2global_translation[:3]
        return {'cameras': {}, 'ego2global': ego2global, 'render_state': render_state}

    def _postprocess_render(self, rgb_raw, render, alpha, map_inverse_distorts,
                            SHAPE, render_mode):
        """Shared output conversion/remapping and one batched restorer call."""
        # Image restoration of the 3DGS renders (the Fixer), applied online during the rollout
        # and batched across all cameras in one call. No-op when no restorer is installed (see
        # restorer_hook.py). Restores the final undistorted image the planner sees.
        _restorer = get_restorer()
        _gpu_path = _gpu_undistort_enabled()
        if _gpu_path and _restorer is not None:
            # The restorer takes host-side uint8 images; handing it the GPU-resident ones would
            # copy them down and back, which is worse than not setting the flag at all.
            raise RuntimeError(
                "ODYSSEY_RENDER_GPU_UNDISTORT=1 cannot be combined with an image restorer "
                "(the restorer takes host-side uint8 images).")

        if _gpu_path:
            # Undistort -> quantise -> ONE uint8 D2H. See the note above the
            # _remap_grid helper for why this matches the cv2 path to within a rounding step.
            import torch.nn.functional as F
            rgb = torch.clamp(rgb_raw, 0.0, 1.0)
            C = rgb.shape[0]
            if getattr(self, "_remap_grid_cache", None) is None:
                self._remap_grid_cache = _remap_grid(
                    map_inverse_distorts, SHAPE['height'], SHAPE['width'], rgb.device)

            # Quantise BEFORE resampling, exactly as the CPU path does (rgb_float_to_bgr_uint8's
            # .to(torch.uint8) is a truncation, hence floor and not round). Values stay in
            # 0..255 float so the bilinear filter runs on the same numbers cv2.remap would see.
            x = (rgb * 255.0).floor_().clamp_(0.0, 255.0)         # (C,H,W,3) RGB
            x = x.permute(0, 3, 1, 2).contiguous()                 # (C,3,H,W)
            x = F.grid_sample(x, self._remap_grid_cache, mode="bilinear",
                              padding_mode="zeros", align_corners=False)
            x = x.round_().clamp_(0.0, 255.0)                      # cv2 rounds its uint8 output

            arr = x.to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()  # RGB
            rgb_list = [cv2.cvtColor(arr[i], cv2.COLOR_RGB2BGR) for i in range(C)]

            sensor_dict = {}
            for cam_name, cam_idx in self.sensor_mapping.items():
                sensor_dict[cam_name] = {"image": rgb_list[cam_idx]}
            if render_mode == "RGB+ED":
                # Depth keeps the original CPU route: not restored, not read by the planner's
                # image encoder, so there is nothing to gain and a conversion to get wrong.
                depth_im = render[:, ..., 3:4]
                depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max())
                depth_np = depth_im.cpu().numpy()
                for cam_name, cam_idx in self.sensor_mapping.items():
                    sensor_dict[cam_name]["depth"] = cv2.remap(
                        depth_np[cam_idx],
                        map_inverse_distorts[cam_idx][0],
                        map_inverse_distorts[cam_idx][1], cv2.INTER_LINEAR)
        else:
            # Default path -- byte-for-byte identical to what this did before the port.
            # clamp / scale / uint8 / channel-swap all finish on the GPU and come down once
            # as uint8: a quarter of the transfer and no per-camera cv2.cvtColor loop.
            # Measured 8 cameras at 1080p: 88.2 ms -> about 20 ms.
            # (render/mtgs/utils/test_image_utils.py pins the bit-identity.)
            rgb = rgb_float_to_bgr_uint8(rgb_raw)

            outputs = {"image": rgb}
            if render_mode == "RGB+ED":
                depth_im = render[:, ..., 3:4]
                depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max())
                outputs["depth"] = depth_im.cpu().numpy()

            sensor_dict = {}
            for cam_name, cam_idx in self.sensor_mapping.items():
                sensor_dict[cam_name] = {}
                for image_key in outputs.keys():
                    sensor_dict[cam_name][image_key] = cv2.remap(
                        outputs[image_key][cam_idx],
                        map_inverse_distorts[cam_idx][0],
                        map_inverse_distorts[cam_idx][1], cv2.INTER_LINEAR)

            if _restorer is not None:
                _cam_names = list(sensor_dict.keys())
                if save_pre_restore_enabled():
                    # The restorer below rebinds 'image' to new arrays; this keeps the rendered one.
                    for _c in _cam_names:
                        sensor_dict[_c]['image_pre_restore'] = sensor_dict[_c]['image']
                _restored = _restorer.restore_batch(
                    [sensor_dict[c]['image'] for c in _cam_names])
                for _c, _r in zip(_cam_names, _restored):
                    sensor_dict[_c]['image'] = _r

        return_dict = {
            'cameras': sensor_dict, 
            'lidars': {}    # TODO: support LiDAR rendering
        }

        return return_dict

    def rendered_tokens(self):
        """Set of actor tokens actually drawn this frame; None if nothing has been drawn yet.

        Returns the active_tokens update_world fills every frame. None and the empty set differ --
        None means "the renderer has nothing to report yet", the empty set means "no actor was
        drawn". The scorer must tell them apart so that runs without rendering are not filtered
        by mistake.
        """
        toks = getattr(self, "active_tokens", None)
        return None if toks is None else set(toks)

    def composite_background(self, rgb, alpha, camera_to_worlds, intrinsics, shape):
        """Fill what the gaussians did not cover. Overridable, because an asset may carry
        its own sky.

        The stock asset paints a flat `background_color` behind a `skybox` gaussian node.
        An OmniRe chunk-bank checkpoint has no skybox node at all and stores the original
        reconstruction's sky as a cubemap instead, so it overrides this to look that up
        (odyssey_renderer/omnire/bundled_switch.py). Extracted verbatim -- with no
        override this is the expression that was inline here, so every existing asset
        renders bit-identically.
        """
        return rgb + (1 - alpha) * self.background_color

    def get_agent_pose(self, name, agent_state, return_matrix=False):
        if name not in self.digitaltwin_agent2states.keys():
            return None if return_matrix else (None, None)

        digitaltwin_agent_state = self.digitaltwin_agent2states[name]

        agent_state_xy = torch.tensor(agent_state[:2], device=self.device, dtype=torch.float64)
        agent_state_heading = torch.tensor(agent_state[-1], device=self.device, dtype=torch.float64)

        nearsest_idx = torch.argmin(torch.norm(digitaltwin_agent_state['translation'][:, :2] - agent_state_xy, dim=-1))
        log_trans = digitaltwin_agent_state['translation'][nearsest_idx]
        log_quat = digitaltwin_agent_state['rotation'][nearsest_idx]

        log_rot_matrix = quat_to_rotmat(log_quat)
        log_rot_yaw = quat_to_angle(log_quat, focus="yaw")["yaw"]
        heading_diff = agent_state_heading - log_rot_yaw
        cos_vals = torch.cos(heading_diff)
        sin_vals = torch.sin(heading_diff)
        rot_matrix_diff = torch.zeros(3, 3, device=self.device, dtype=torch.float64)
        rot_matrix_diff[0, 0] = cos_vals
        rot_matrix_diff[0, 1] = -sin_vals
        rot_matrix_diff[1, 0] = sin_vals
        rot_matrix_diff[1, 1] = cos_vals
        rot_matrix_diff[2, 2] = 1.0

        new_trans = log_trans.clone()
        new_trans[:2] = agent_state_xy
        new_rot_matrix = rot_matrix_diff @ log_rot_matrix

        if return_matrix:
            agent2global = torch.zeros(4, 4, device=self.device, dtype=torch.float64)
            agent2global[:3, :3] = new_rot_matrix
            agent2global[:3, 3] = new_trans
            agent2global[3, 3] = 1.0
            return agent2global
        else:
            quat = matrix_to_quaternion(new_rot_matrix)
            trans = new_trans
            return quat, trans


def _check_asset_identity(scene, path):
    """Check that the asset on disk is the one this scenario was baked from.

    Assets are replaced in place at the same path. Scenario actor positions are the training poses
    of the ckpt at bake time, so reusing an old pkl after an asset is re-delivered makes **the
    scored positions and the drawn positions silently diverge**. Compare against the sha256 the
    builder recorded and exit on a mismatch.

    Older scenarios without this field have nothing to compare, so only a one-line warning is
    logged -- those pkls were baked before the field existed, and failing here would break
    existing experiments.
    """
    import hashlib
    metadata = scene[SD.METADATA]
    identity = metadata.get('omnire_asset_identity')
    pose_source = metadata.get('actor_pose_source')
    asset_id = path.name
    if identity is None:
        logger.warning(f"[asset-identity] {asset_id}: scenario carries no asset identity -- "
                       f"cannot verify (rebuild the scenario to record it)")
        return
    if pose_source == 'scenario':
        raise SystemExit(
            f"[asset-identity] {asset_id}: this scenario takes actor positions from the nuPlan "
            f"log (actor_pose_source=scenario) but runs with a render asset. The image would show "
            f"reconstructed poses while scoring uses log poses, so the two diverge. Rebake it "
            f"with --actor-pose checkpoint.")
    if not path.exists():
        raise SystemExit(f"[asset-identity] asset ckpt is missing -- {path}")
    size = path.stat().st_size
    if identity.get('asset_bytes') not in (None, size):
        raise SystemExit(f"[asset-identity] {asset_id}: byte size differs "
                         f"(scenario {identity['asset_bytes']} vs disk {size}) -- the asset was replaced")
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 22), b''):
            h.update(block)
    digest = h.hexdigest()
    if identity.get('asset_sha256') and digest != identity['asset_sha256']:
        raise SystemExit(
            f"[asset-identity] {asset_id}: the asset on disk differs from the one this scenario "
            f"was baked from.\n"
            f"  scenario: {identity['asset_sha256']}\n  disk    : {digest}\n"
            f"  Actor positions are the ckpt poses at bake time -- rebake the scenario.")
    logger.info(f"[asset-identity] {asset_id}: sha256 match ({digest[:12]}…), "
                f"actor pose source {pose_source}")


class MTGSAssetManager:
    """Holds the loaded background asset; a reset to the same checkpoint keeps it (and its caches)."""

    def __init__(self, device: torch.device):
        self.current_checkpoint = None
        self.device = device

    def reset(self, checkpoint: Path):
        checkpoint = Path(checkpoint)
        if self.current_checkpoint == checkpoint:
            return False

        if getattr(self, "background_asset", None) is not None:
            try:
                if hasattr(self.background_asset, "to"):
                    self.background_asset.to("cpu")
            except Exception:
                pass
            del self.background_asset
            self.background_asset = None
            torch.cuda.empty_cache()

        self.current_checkpoint = checkpoint
        self.load_asset()
        return True

    def load_asset(self):
        path = self.current_checkpoint
        logger.info(f"Loading the scene checkpoint {path} ...")
        assert path.exists(), f"asset {path} not exist!"
        alias_numpy2_pickle_modules()
        self.background_asset = torch.load(path, map_location=self.device, weights_only=False)

