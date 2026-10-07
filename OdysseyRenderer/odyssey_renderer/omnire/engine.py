"""Self-contained appearance and actor contract over the shared Gaussian kernel."""
import json
import logging
import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


from odyssey_renderer.base_renderer import RenderState
from odyssey_renderer.ego_lift import LogPathLift, attitude, colmap_ego
from odyssey_renderer.lidar_ground import GroundField, RoadSurface
from odyssey_renderer.mtgs import mtgs
from odyssey_renderer.mtgs.utils.gaussian_utils import matrix_to_quaternion, quat_mult, quat_to_angle
from odyssey_renderer.mtgs.mtgs import MTGSRenderEngine
from odyssey_renderer.omnire import restorer_client
from odyssey_renderer.omnire.actor_contract import validate_actor_manifest
from odyssey_renderer.omnire import registry
from odyssey_renderer.omnire.gaussian_assembly import GaussianAssembly
from odyssey_renderer.omnire.traffic_light import OmniReTrafficLightSubModel
from odyssey_renderer.omnire.camera_contract import (
    validate_time_calibration, validate_exposure, apply_exposure,
    sample_rigid_camera_pose,
)

logger = logging.getLogger(__name__)

# Tokens that are scenery, not actors: no log horizon, no per-frame visibility,
# never counted as an agent. Defined once because the same set is consulted in
# two places and a literal in each would drift.
SCENERY_TOKENS = ("background", "skybox", "ground")


def _quat_wxyz(R):
    """Batched 3x3 -> unit quaternion (w, x, y, z), the convention gsplat's quat_to_rotmat reads."""
    m = np.asarray(R, dtype=np.float64)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = np.zeros((len(m), 4))
    pick = [trace > 0]
    for k in range(3):
        diagonal = m[:, k, k] >= np.maximum(m[:, (k + 1) % 3, (k + 1) % 3], m[:, (k + 2) % 3, (k + 2) % 3])
        pick.append(~np.any(pick, axis=0) & diagonal)
    scale = np.sqrt(np.maximum(1.0 + trace, 1e-12)) * 2
    q[pick[0]] = np.stack([0.25 * scale, (m[:, 2, 1] - m[:, 1, 2]) / scale,
                           (m[:, 0, 2] - m[:, 2, 0]) / scale,
                           (m[:, 1, 0] - m[:, 0, 1]) / scale], axis=-1)[pick[0]]
    for k in range(3):
        j, l = (k + 1) % 3, (k + 2) % 3
        scale = np.sqrt(np.maximum(1.0 + m[:, k, k] - m[:, j, j] - m[:, l, l], 1e-12)) * 2
        out = np.zeros((len(m), 4))
        out[:, 0] = (m[:, l, j] - m[:, j, l]) / scale
        out[:, 1 + k] = 0.25 * scale
        out[:, 1 + j] = (m[:, j, k] + m[:, k, j]) / scale
        out[:, 1 + l] = (m[:, k, l] + m[:, l, k]) / scale
        q[pick[1 + k]] = out[pick[1 + k]]
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


class OmniReRenderEngine(MTGSRenderEngine):
    """Canonical calibrated appearance, explicit actor population, and static Ground caches."""

    # Keep lightweight/test instances that bypass __init__ on the legacy path.
    render_scale = 1.0

    def __init__(self, *args, actor_pose_source="checkpoint", horizon_extension_factor=1.0,
                 lift="tick", road_surface=None, render_scale=1.0, **kwargs):
        # How many times the stored training rows to accept steps for. Closed loop runs up to 2x
        # the GT drive duration, so 1.0 (stored range only) would fail rendering right at the end
        # of GT. Steps beyond the range are clamped to the last stored row. Log-replay actors are
        # hidden there, but live IDM vehicles are still drawn.
        factor = float(horizon_extension_factor)
        if not np.isfinite(factor) or factor < 1.0:
            raise ValueError("horizon_extension_factor must be a finite number >= 1.0")
        self._horizon_factor = factor
        self._beyond_horizon = False
        requested = os.environ.get("ODYSSEY_OMNIRE_ACTOR_POSE_SOURCE", actor_pose_source)
        if requested not in ("checkpoint", "scenario"):
            raise ValueError("actor_pose_source must be checkpoint or scenario")
        incumbent = registry.register_model("OmniReTrafficLightSubModel", OmniReTrafficLightSubModel)
        if incumbent is not OmniReTrafficLightSubModel:
            raise ValueError("OmniReTrafficLightSubModel registry collision")
        # Which bodies read z, roll and pitch off the road under their own (x, y), instead of
        # their stock source (see render/ego_lift.py). The stock source is the logged pose of
        # the same tick for the ego, and the checkpoint's learned pose for an actor.
        #   tick  neither -- every existing run
        #   ego   the camera only
        #   all   the camera and the rigid actors; each actor keeps its learned height above
        #         the road as one constant, so only the wobble against the road is removed
        if lift not in ("tick", "ego", "all"):
            raise ValueError("lift must be tick, ego or all, got %r" % (lift,))
        self.lift = lift
        # Where the lifted bodies read the road from. Unset means the logged trajectory, which
        # only knows the ground it drove over; a baked lidar field knows the whole surface (see
        # render/lidar_ground.py). It changes the ANSWER, not who asks -- `lift` still decides
        # that -- so `lift=tick` ignores this exactly as it ignores the path lift.
        #
        # It is for the ACTORS only. The ego always reads the reconstructed ego path (the COLMAP
        # ego behind the checkpoint's cameras, see _colmap_ego): projecting onto that path keeps it
        # in the frame the scene was trained in, and a closed-loop ego that wanders off the road
        # (a kerb, a driveway) would otherwise pick up whatever the field says is there.
        # `lift=ego` has no actor to hand the field to, so asking for both is refused rather than
        # ignored.
        self._road_surface = os.environ.get("ODYSSEY_OMNIRE_ROAD_SURFACE", road_surface) or None
        if self._road_surface and lift == "ego":
            raise ValueError("road_surface only moves the actors, and lift=ego lifts none; "
                             "use lift=all")
        # Rasterisation resolution scale. The Fixer was trained on native 1024x576 renders
        # (render scale 0.53333), while the rollout
        # renders at 1080p and the sidecar shrinks to 576x1024 and scales back up by 1.875. That
        # upsample smears thin structures such as lane lines and road arrows. Downscaling here
        # makes the sidecar resize an identity and gives the planner the same pixel grid as the
        # inspection viewer.
        #
        # 1.0 is the standard behaviour. **Keep 1.0 for rollouts that drive a planner.**
        #
        # Planner camera preprocessing assumes 1920x1080 pixel for pixel:
        #   SafeDrive  safedrive_features.py builds its undistort maps with
        #              CAMERA_IMG_SIZE=(1920,1080), crops image[28:-28], and
        #              camera_post_transform() derives post_rot/post_tran from that constant --
        #              a smaller image breaks the geometry.
        #   transfuser family (DiffusionDrive/GTRS/navsim/DriveVLA/GoalFlow)
        #              crops image[28:-28, 416:-416]. Removing 416 from each side of a 1024 width
        #              leaves 192 -- the side cameras all but vanish and the 4:1 aspect breaks.
        #   GoalFlow   shrinks the stitched 1024x4096 to 512x2048. It is the most demanding;
        #              at 1024x576 there is no headroom left for that downsample.
        #
        # Lowering the resolution requires proportional planner crop constants and undistort
        # maps / post_transform computed from the actual size. Until then this value is **for
        # render comparison experiments only** (rendering without a planner).
        self.render_scale = float(render_scale)
        if not (0.0 < self.render_scale <= 1.0):
            raise ValueError("render_scale must be in (0, 1], got %r" % (render_scale,))
        self._path_lift = None
        self._ego_lift = None
        self._camera_residual = None
        self._rig = None
        self._actor_height = None
        self._actor_road = None
        self._static_actors = None
        self._calibration = None
        self._rigid_tables = {}
        self.original_sky = None
        registry.register_all()
        restorer_client.install_from_env()
        super().__init__(*args, **kwargs)
        self.actor_pose_source = requested
        self.render_simulated_vehicles = requested == "scenario"
        self._vehicle_tokens = set()
        self._render_agent_states = {}
        self._render_agent_rows = {}
        self._held_at_log_pose = frozenset()
        self._suppressed_actors = frozenset()
        self._vehicle_pose_refs = {}
        self._vehicle_dims = {}
        self._ground_tree = None

    def set_asset(self, asset):
        self._actor_manifest = validate_actor_manifest(asset)
        if "_r5_chunk_bank" in asset:
            raise ValueError("scene checkpoint requires one canonical unbaked bank")
        cfg = asset.get("background", {}).get("config", {})
        if cfg.get("c018_render", {}).get("rigid_camera_time_baked"):
            raise ValueError("F0-baked assets are not canonical banks")
        metadata = cfg.get("omnire_calibration")
        if metadata is None:
            raise ValueError("scene checkpoint requires background.config.omnire_calibration")
        self._calibration = validate_time_calibration(metadata,
            expected_anchor=cfg.get("recon2world_translation"))
        if self._calibration.get("ego_to_global") is None:
            raise ValueError("scene checkpoint requires saved ego_to_global for at-tick calibration")
        if self.rasterize_mode != self._calibration["rasterize_mode"]:
            raise ValueError("rasterize_mode must match saved metadata")
        sky = cfg.get("omnire_environment")
        if not torch.is_tensor(sky) or sky.ndim != 4 or sky.shape[0] != 6 or sky.shape[-1] != 3:
            raise ValueError("scene checkpoint requires a raw [6,H,W,3] sky cubemap")
        if not bool(torch.isfinite(sky).all()):
            raise ValueError("nonfinite sky")
        self._exposure = validate_exposure(self._calibration["exposure"],
            self._calibration["training_timestamps_us"], self._calibration["camera_names"])
        tables, visibility, ground_sources = {}, {}, {}
        source_times = self._calibration["training_timestamps_us"]
        for name, node in asset.items():
            kind = node.get("config", {}).get("type")
            if kind in ("VanillaGaussianSplattingModel", "SkyboxGaussianSplattingModel",
                        "OmniReTrafficLightSubModel", "OmniReGroundSubModel"):
                continue
            state, config = node["state_dict"], node["config"]
            if "instance_valid" not in state:
                raise ValueError("actor requires exact instance_valid: " + name)
            valid = torch.as_tensor(state["instance_valid"], dtype=torch.bool).detach().cpu().reshape(-1)
            if len(valid) != len(source_times):
                raise ValueError("actor visibility horizon mismatch: " + name)
            visibility[name.split("_")[-1]] = valid.numpy()
            if "instance_trans" in state:
                trans = torch.as_tensor(state["instance_trans"]).detach().cpu().reshape(-1, 3)
                if len(trans) != len(source_times):
                    raise ValueError("actor pose horizon mismatch: " + name)
                ground_sources[name.split("_")[-1]] = (trans.numpy(), valid.numpy())
            if kind not in ("RigidSubModel", "MirroredRigidSubModel"):
                continue
            times = torch.as_tensor(config["log_timestamps"]).detach().cpu().numpy().reshape(-1)
            if not np.array_equal(times, source_times):
                raise ValueError("rigid timestamps differ from calibration: " + name)
            trans = torch.as_tensor(state["instance_trans"]).detach().cpu().reshape(-1, 1, 3)
            quats = torch.as_tensor(state["instance_quats"]).detach().cpu().reshape(-1, 1, 4)
            visible = valid.reshape(-1, 1)
            if len(trans) != len(source_times) or len(quats) != len(source_times) or len(visible) != len(source_times):
                raise ValueError("rigid pose horizon mismatch: " + name)
            tables[name.split("_")[-1]] = (trans, quats, visible)
        super().set_asset(asset)
        for model in self.gaussian_models.values():
            if isinstance(model, OmniReTrafficLightSubModel):
                model.to(self.device)
        self._rigid_tables = tables
        self._rigid_indices = {token: i for i, token in enumerate(tables)}
        self._rigid_valid_rows = {
            token: torch.nonzero(table[2][:, 0], as_tuple=True)[0]
            for token, table in tables.items()
        }
        self._packed_rigid = tuple(torch.cat([table[i] for table in tables.values()], dim=1)
                                   for i in range(3)) if tables else None
        self._actor_visibility = visibility
        self._ground_pose_sources = ground_sources
        self.original_sky = sky.to(self.device).contiguous()
        self.sensor_caches = None
        self._camera_cache = {}
        self._clear_asset_caches()

    def _clear_asset_caches(self):
        super()._clear_asset_caches()
        self._rigid_attribute_cache = {}
        self._gaussian_assembly = GaussianAssembly()

    @torch.no_grad()
    def _rigid_attributes_cached(self, name, model):
        # Only these concrete inference models have pose/time-independent attributes.
        # Keep all scripted quaternion/rotation calls in their original camera order:
        # their cold and optimized executions need not round identically.
        if type(model) not in (mtgs.RigidModel, mtgs.MirroredModel):
            return model.get_scales(), model.get_opacity()
        cached = self._rigid_attribute_cache.get(name)
        if cached is None:
            cached = (model.get_scales(), model.get_opacity())
            self._rigid_attribute_cache[name] = cached
        return cached

    def calibrate_agent_state(self):
        # TL is time-varying appearance on fixed heads, not a traffic agent.
        matrices = torch.as_tensor(self._calibration["ego_to_global"], dtype=torch.float64, device=self.device).clone()
        matrices[:, :3, 3] -= torch.as_tensor(self.recon2global_translation, dtype=torch.float64, device=self.device)
        self._logged_ego = matrices
        # tick keeps the logged (GT) pose of the same tick and the saved per-row camera residual --
        # every existing run. A lift reads the reconstructed ego path and the residual against it.
        self._camera_residual = self._calibration["camera_to_ego"]
        self._rig = None
        self._path_lift = None
        if self.lift != "tick":
            ego_path, self._rig = self._colmap_ego()
            self._camera_residual = self._rig
            self._path_lift = LogPathLift(ego_path)
        self._ego_lift = self._path_lift
        if self._path_lift is not None and self._road_surface:
            self._path_lift = RoadSurface(GroundField.load(self._road_surface), self._path_lift,
                                          self.recon2global_translation)
            logger.info("road surface %s: covers %.1f%% of the logged path, datum %+.3f m "
                        "(p90 spread %.3f m)", self._road_surface, 100 * self._path_lift.cover,
                        self._path_lift.offset, self._path_lift.spread)
        self._actor_height = self._measure_actor_height() if self.lift == "all" else None
        self._actor_road = (self._smooth_actor_road()
                            if self._actor_height is not None and isinstance(self._path_lift, RoadSurface)
                            else None)
        self._static_actors = (self._left_where_the_checkpoint_put_them()
                               if self._actor_height is not None else None)
        return {"ego": {"translation": matrices[:, :3, 3],
                        "rotation": matrix_to_quaternion(matrices[:, :3, :3])}}

    def _colmap_ego(self):
        """-> (ego poses [rows, 4, 4], camera residual [rows, cameras, 4, 4]), reconstruction frame.

        The checkpoint's ego_to_global is the logged (GT) ego. Its z can be metres away from the
        ego the reconstruction was trained on, and the difference varies along the drive. The
        saved camera_to_ego absorbs that row by row, so it is right only at the row it was saved
        for. A closed-loop ego that falls behind or runs ahead of the log would be lifted at one
        place and handed the residual of another, which can put the camera under the road.

        The reconstructed ego is the checkpoint's own camera (camera_to_global, CAM_F0) with the
        scenario's CAM_F0 sensor2ego taken off. Against it every camera's residual is the rig
        itself, the same at every row (CAM_F0 about 1.5 m above the ego), so it does not matter
        which row it came from.
        """
        cameras = self.engine.managers["scenario_manager"].current_scene["cameras"]
        camera_to_global = np.array(self._calibration["camera_to_global"], dtype=np.float64)   # [rows, cams, 4, 4]
        camera_to_global[..., :3, 3] -= np.asarray(self.recon2global_translation, dtype=np.float64)
        return colmap_ego(camera_to_global, self._calibration["camera_names"], cameras)

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        scene = self.engine.managers["scenario_manager"].current_scene
        metadata = scene["metadata"]
        self._vehicle_tokens = {
            token for token, track in scene["object_track"].items()
            if token != "ego" and track["type"] == "VEHICLE"
        }
        self._vehicle_heights = {}
        ground_xy, ground_z = [], []
        for token, (poses, valid) in self._ground_pose_sources.items():
            track = scene["object_track"].get(token)
            if track is None:
                continue
            heights = np.asarray(track["state"]["height"], dtype=float).reshape(-1)
            measured = heights[np.isfinite(heights) & (heights > 0)]
            if len(measured) == 0:
                continue
            height = float(np.median(measured))
            if token in self._vehicle_tokens:
                self._vehicle_heights[token] = height
            mask = valid & np.isfinite(poses).all(axis=1)
            if mask.any():
                ground_xy.append(poses[mask, :2])
                ground_z.append(poses[mask, 2] - height / 2)
        if self.actor_pose_source == "scenario":
            if not ground_xy:
                raise ValueError("reactive rendering needs optimized actor ground samples")
            self._ground_tree = cKDTree(np.concatenate(ground_xy))
            self._ground_z = np.concatenate(ground_z)
        else:
            self._ground_tree = None
        validate_time_calibration(self._calibration,
            expected_timestamps=metadata["omnire_timestamps_us"],
            expected_anchor=metadata["omnire_recon2world_translation"])
        self._simulation_base = int(scene["base_timestamp"])
        self._simulation_dt_us = float(self.engine.sim_dt) * 1e6
        if abs(self._simulation_base-int(self._calibration["training_timestamps_us"][0])) > 1:
            raise ValueError("scene simulation base differs from calibration")
        return result

    def _measure_actor_height(self):
        """Each rigid actor's body origin above the road, as ONE constant per actor.

        The learned pose already carries a height that is right on average -- what it does
        not carry is a stable relation to the road: on a descending road an actor can swing by
        up to a metre around its own median. So the median
        over that actor's own visible frames is kept (no class or box-height guess, and no
        per-frame correction that would cancel the wobble the reconstruction may be right
        about) and the per-frame height comes from the road.

        With a lidar road the learned z is not used at all: the box bottom goes on the road,
        i.e. the origin sits half the box height above it. The logged path cannot do this --
        it reports the ego body origin, not the road -- which is why it keeps the median. The
        median assumes learned-z minus road is one number per actor, which fails for an actor
        that, for example, parks and then climbs a ramp. The box half-height is the checkpoint's
        own geometry: the lowest opaque Gaussian sits at about -height/2, so the origin is the
        box centre.
        """
        if self._path_lift is None or self._packed_rigid is None:
            return None
        if isinstance(self._path_lift, RoadSurface):
            tracks = self.engine.managers["scenario_manager"].current_scene["object_track"]
            heights = []
            for token in self._rigid_indices:                 # the packed column order
                track = tracks.get(token)
                if track is None:
                    raise ValueError("actor has no scenario track to read its box height: " + token)
                h = np.asarray(track["state"]["height"], dtype=float).reshape(-1)
                h = h[np.isfinite(h) & (h > 0)]
                if not len(h):
                    raise ValueError("actor has no measured box height: " + token)
                # ground() answers in the logged path's datum (road + offset); take it back off
                heights.append(float(np.median(h)) / 2 - self._path_lift.offset)
            return np.asarray(heights)
        trans, quats, visible = (t.cpu().numpy() for t in self._packed_rigid)
        heights = np.zeros(trans.shape[1])
        for actor in range(trans.shape[1]):
            seen = np.flatnonzero(visible[:, actor].reshape(-1))
            if not len(seen):
                continue
            xy, q = trans[seen, actor, :2], quats[seen, actor]
            yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                             1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            road, _ = self._path_lift.ground(xy, yaw)
            heights[actor] = np.median(trans[seen, actor, 2] - road)
        return heights

    def _left_where_the_checkpoint_put_them(self):
        """-> bool [actors] marking actors the renderer leaves untouched, or None if there are none.

        Actors excluded when baking the road surface are not moved by the renderer either. The
        decision is not remade here; it reads the value the baker left next to the surface
        (drop_static in <npz>.json, the xy-extent threshold of an actor's own trajectory) -- if
        the two decisions disagreed, the mismatch would show up directly in the image.

        Parked cars must not be put on the road because the rigid node is not the whole car.
        Most of a parked car is baked into the background Gaussians and the node holds only part
        of the body, so moving just the node onto the road misaligns it with the rest in the
        background and slices the car horizontally (the surface baked without static actors can
        rise most of a metre where a parked car stood). Moving cars do not have this problem --
        with no stationary interval to bake into the background, the node is the whole car.
        """
        span = 0.0
        if self._road_surface:
            diag = os.path.splitext(self._road_surface)[0] + ".json"
            if os.path.exists(diag):
                with open(diag) as f:
                    span = float(json.load(f).get("drop_static") or 0.0)
        if not span or self._packed_rigid is None:
            return None
        trans, _, visible = (t.cpu().numpy() for t in self._packed_rigid)
        keep = np.zeros(trans.shape[1], dtype=bool)
        for actor in range(trans.shape[1]):
            seen = np.flatnonzero(visible[:, actor].reshape(-1))
            if len(seen):
                keep[actor] = float(np.hypot(*np.ptp(trans[seen, actor, :2], axis=0))) < span
        logger.info("%d/%d static actors stay where the checkpoint put them (drop_static %.1f m)",
                    int(keep.sum()), len(keep), span)
        return keep if keep.any() else None

    # Wheel contact points as fractions of the box (+-0.3 length, +-0.4 width: about the
    # wheelbase and track of a 4.6 x 1.9 m car), how long a smoothing window is, and the
    # steepest tilt a car is given. Real road grades are a few degrees, plus a few degrees of
    # crossfall.
    _WHEELS = ((0.3, 0.4), (0.3, -0.4), (-0.3, 0.4), (-0.3, -0.4))
    _SMOOTH_ROWS = 5.0            # gaussian sigma in checkpoint rows (10 Hz -> 0.5 s)
    _MAX_TILT = np.tan(np.radians(8.0))

    def _smooth_actor_road(self):
        """Per actor and per checkpoint row: the road under the car, as a car would feel it.

        Reading the road at ONE point under the body centre hands every bump of the field to
        the car: a moving actor can jump tens of centimetres and many degrees of pitch in one
        frame, as if it were driving off-road. A car has four wheels and a
        suspension, so this reads the road at the four wheel contacts, fits the plane through
        them (height at the centre and the world grade), and then smooths that along the
        actor's own track with a 0.5 s gaussian -- run by run, never across a gap in its
        visibility. This cuts the per-frame height and pitch jitter by about an order of
        magnitude. The tilt is capped at _MAX_TILT.

        The actors replay the checkpoint, so their whole track is known up front and this is
        computed once. -> (height [rows, actors], grade [rows, actors, 2]), NaN where hidden.
        """
        trans, quats, visible = (t.cpu().numpy() for t in self._packed_rigid)
        tracks = self.engine.managers["scenario_manager"].current_scene["object_track"]
        rows, actors = trans.shape[:2]
        height = np.full((rows, actors), np.nan)
        grade = np.full((rows, actors, 2), np.nan)
        wheels = np.asarray(self._WHEELS)
        for token, actor in self._rigid_indices.items():
            seen = np.flatnonzero(visible[:, actor].reshape(-1))
            if not len(seen):
                continue
            state = tracks[token]["state"]                   # _measure_actor_height checked it exists
            size = []
            for key in ("length", "width"):
                v = np.asarray(state[key], dtype=float).reshape(-1)
                v = v[np.isfinite(v) & (v > 0)]
                if not len(v):
                    raise ValueError("actor has no measured box %s: %s" % (key, token))
                size.append(float(np.median(v)))
            xy, q = trans[seen, actor, :2], quats[seen, actor]
            yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                             1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
            fwd, left = wheels[:, 0] * size[0], wheels[:, 1] * size[1]
            offset = np.stack([fwd * c - left * s, fwd * s + left * c], axis=-1)   # [n, 4, 2]
            z, _ = self._path_lift.ground((xy[:, None, :] + offset).reshape(-1, 2),
                                          np.repeat(yaw, len(wheels)))
            z = z.reshape(len(seen), len(wheels))
            # plane z = h + gx dx + gy dy through the four wheels, in world axes
            design = np.concatenate([np.ones((*offset.shape[:2], 1)), offset], axis=-1)
            coef = np.linalg.solve(design.transpose(0, 2, 1) @ design,
                                   (design.transpose(0, 2, 1) @ z[..., None]))[..., 0]
            breaks = np.flatnonzero(np.diff(seen) > 1) + 1
            for run in np.split(np.arange(len(seen)), breaks):
                coef[run] = gaussian_filter1d(coef[run], self._SMOOTH_ROWS, axis=0, mode="nearest")
            g = coef[:, 1:]
            steep = np.linalg.norm(g, axis=-1, keepdims=True)
            g = g * np.minimum(1.0, self._MAX_TILT / np.maximum(steep, 1e-12))
            height[seen, actor], grade[seen, actor] = coef[:, 0], g
        return height, grade

    def _stand_actors_on_road(self, pos, rotation, frame):
        """Put the sampled rigid actors on the road under their own (x, y). CPU, before upload.

        Only z and the tilt move: (x, y) and yaw stay exactly as the checkpoint interpolated
        them for this camera's exposure time, so the correction cannot smear an actor along
        the road or turn it. With a smoothed table (_smooth_actor_road) the road comes from the
        row `frame` and is carried to the interpolated (x, y) along that row's smoothed grade
        -- the camera exposure is within 50 ms of the row, so that is at most about a metre.
        """
        q = rotation.numpy()
        yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                         1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        if self._actor_road is not None:
            height, grade = self._actor_road[0][frame], self._actor_road[1][frame]
            step = pos.numpy()[:, :2] - self._packed_rigid[0][frame, :, :2].cpu().numpy()
            road = height + (grade * step).sum(-1)
            normal = np.concatenate([-grade, np.ones((len(grade), 1))], axis=-1)
            normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
            hidden = ~np.isfinite(road)                       # absent at this row: leave untouched
            if hidden.any():
                road = np.where(hidden, pos.numpy()[:, 2] - self._actor_height, road)
                normal[hidden] = (0.0, 0.0, 1.0)
        else:
            road, normal = self._path_lift.ground(pos.numpy()[:, :2], yaw)
        pos = pos.clone()
        z = road + self._actor_height
        rot = _quat_wxyz(attitude(yaw, normal))
        keep = self._static_actors
        if keep is not None:                       # keep the checkpoint's pose (see method above)
            z = np.where(keep, pos[:, 2].numpy(), z)
            rot = np.where(keep[:, None], rotation.numpy(), rot)
        pos[:, 2] = torch.as_tensor(z, dtype=pos.dtype)
        return pos, torch.as_tensor(rot, dtype=rotation.dtype)

    def _tick_index(self, timestamp):
        """timestamp -> (requested physics tick, saved source row).

        Inside the saved horizon the two are equal. Ticks up to ``len(rows) * horizon_extension_factor``
        are accepted and map to the LAST saved row -- camera calibration, exposure, logged ego attitude,
        sky and traffic lights hold that row. Log-replayed actors are hidden; live IDM vehicles
        remain visible through their simulator states.
        Off-grid ticks and ticks past the extended horizon still fail.
        """
        times = self._calibration["training_timestamps_us"]
        value = float(timestamp)
        if not np.isfinite(value):
            raise ValueError("timestamp must be finite")
        exact = np.flatnonzero(times == value)
        if len(exact):
            return int(exact[0]), int(exact[0])
        base = getattr(self, "_simulation_base", int(times[0]))
        dt = getattr(self, "_simulation_dt_us", float(np.median(np.diff(times))))
        index = int(round((value-base)/dt))
        limit = int(round(len(times) * self._horizon_factor))
        if value < base or not 0 <= index < limit or abs(value-(base+index*dt)) > 1.:
            raise ValueError("timestamp is outside the allowed horizon "
                             f"({len(times)} saved rows x {self._horizon_factor}) or off the physics tick grid")
        return index, min(index, len(times) - 1)

    def _resolve_frame(self, timestamp):
        return self._tick_index(timestamp)[1]

    def _ego_at_tick(self, ego_state, frame):
        if self._ego_lift is not None:
            pose = self._ego_lift.lift(float(ego_state[0]), float(ego_state[1]), float(ego_state[-1]))
            return torch.as_tensor(pose, dtype=self._logged_ego.dtype, device=self._logged_ego.device)
        result = self._logged_ego[frame].clone()
        yaw = torch.atan2(result[1, 0], result[0, 0])
        delta = torch.as_tensor(ego_state[-1], dtype=result.dtype, device=result.device) - yaw
        c, s = torch.cos(delta), torch.sin(delta)
        rot = torch.eye(3, dtype=result.dtype, device=result.device)
        rot[0, 0], rot[0, 1], rot[1, 0], rot[1, 1] = c, -s, s, c
        result[:3, :3] = rot @ result[:3, :3]
        result[:2, 3] = torch.as_tensor(ego_state[:2], dtype=result.dtype, device=result.device)
        return result

    def _camera_inputs(self, sensors, ego, frame):
        selected = mtgs._select_cams(sensors)
        if not selected:
            raise ValueError("scene checkpoint requires at least one camera")
        names = self._calibration["camera_names"]
        self.sensor_mapping = {name: i for i, name in enumerate(selected)}
        matrices, intrinsics, maps, indices = [], [], [], []
        shape = None
        for name, camera in selected.items():
            if name not in names:
                raise ValueError("camera absent from the calibration: " + name)
            index = names.index(name)
            h, w = int(camera["height"]), int(camera["width"])
            intrinsic = np.asarray(camera["intrinsic"], dtype=np.float64)
            if self.render_scale != 1.0:
                # Exactly the viewer's convention (omnire_render_only.base_view): size is
                # round(original * s), K is K * s and then K[2,2] = 1. The viewer applies no
                # half-pixel correction either -- doing so would offset the two by half a pixel.
                h = max(32, int(round(h * self.render_scale)))
                w = max(32, int(round(w * self.render_scale)))
                intrinsic = intrinsic * self.render_scale
                intrinsic[2, 2] = 1.0
            if shape is not None and shape != {"height": h, "width": w}:
                raise ValueError("batched output requires equal camera dimensions")
            shape = {"height": h, "width": w}
            residual = torch.as_tensor(self._camera_residual[frame, index], dtype=ego.dtype, device=self.device)
            matrices.append((ego @ residual).float())
            key = (name, h, w, tuple(intrinsic.reshape(-1)))
            cache = self._camera_cache.get(key)
            if cache is None:
                y, x = np.mgrid[:h, :w].astype(np.float32)
                cache = (torch.as_tensor(intrinsic, dtype=torch.float32, device=self.device), (x, y))
                self._camera_cache[key] = cache
            intrinsics.append(cache[0])
            maps.append(cache[1])
            indices.append(index)
        return torch.stack(matrices), torch.stack(intrinsics), maps, shape, indices

    def _simulated_vehicle_pose(self, token, state, keep_learned_z=False):
        """Place the vehicle over the nearest optimized ground sample in this scene.

        `keep_learned_z` keeps the height the checkpoint learned at the nearest row instead -- for
        a parked car that the anchor surface left out (see _left_where_the_checkpoint_put_them).
        """
        cached = self._vehicle_pose_refs.get(token)
        if cached is not None:
            return cached
        rows = self._rigid_valid_rows[token]
        if len(rows) == 0:
            return None
        translations, quaternions, _ = self._rigid_tables[token]
        xy = torch.as_tensor(state[:2], dtype=translations.dtype, device=translations.device)
        distances = ((translations[rows, 0, :2] - xy) ** 2).sum(dim=1)
        row = int(rows[torch.argmin(distances)])
        anchor = translations[row, 0].to(self.device).clone()
        learned_quat = quaternions[row, 0].to(self.device)
        if not keep_learned_z:
            if self._ground_tree is None or token not in self._vehicle_heights:
                raise ValueError("reactive vehicle has no ground/height data: " + token)
            _, ground_index = self._ground_tree.query(np.asarray(state[:2], dtype=float))
            anchor[2] = float(self._ground_z[ground_index]) + self._vehicle_heights[token] / 2
        self._vehicle_pose_refs[token] = (anchor, learned_quat)
        return anchor, learned_quat

    def _stand_simulated_vehicle_on_road(self, index, token, state):
        """-> (pos [3], wxyz [4]) numpy: a vehicle the simulator drives, stood on the road under it now.

        The reactive counterpart of _stand_actors_on_road. A simulator-driven vehicle is not where
        the checkpoint put it, so there is no learned row to read a height from; the road surface
        answers at its current (x, y) every frame instead. Same rule as a replayed actor: the four
        wheel contacts give a plane, its tilt is capped at _MAX_TILT, the box bottom goes on the
        road (origin = road + height/2) and the attitude comes from (yaw, road normal). What is
        missing is the 0.5 s smoothing along the track -- the future of this track is not known --
        so this relies on the surface being smooth, which the anchor surface is (a 2.5 m gaussian).

        Without this a reactive vehicle takes its height from the single learned actor bottom
        nearest to it (_simulated_vehicle_pose, every frame): a nearest-neighbour lookup, so the
        height steps as the nearest sample changes, and the samples include the parked cars'
        bottoms that the anchor surface leaves out because they sit up to a metre below the road.
        """
        key = ("road_pose", token)
        cached = self._vehicle_pose_refs.get(key)
        if cached is not None:
            return cached
        dims = self._vehicle_dims.get(token)
        if dims is None:
            state_rows = self.engine.managers["scenario_manager"].current_scene["object_track"][token]["state"]
            dims = []
            for key in ("length", "width"):
                v = np.asarray(state_rows[key], dtype=float).reshape(-1)
                v = v[np.isfinite(v) & (v > 0)]
                if not len(v):
                    raise ValueError("reactive vehicle has no measured box %s: %s" % (key, token))
                dims.append(float(np.median(v)))
            self._vehicle_dims[token] = dims
        x, y, yaw = float(state[0]), float(state[1]), float(state[-1])
        wheels = np.asarray(self._WHEELS)
        fwd, left = wheels[:, 0] * dims[0], wheels[:, 1] * dims[1]
        c, s = np.cos(yaw), np.sin(yaw)
        offset = np.stack([fwd * c - left * s, fwd * s + left * c], axis=-1)            # [4, 2]
        z, _ = self._path_lift.ground(np.array([x, y]) + offset, np.full(4, yaw))
        design = np.concatenate([np.ones((4, 1)), offset], axis=-1)
        h, gx, gy = np.linalg.lstsq(design, z, rcond=None)[0]
        grade = np.array([gx, gy]) * min(1.0, self._MAX_TILT / max(float(np.hypot(gx, gy)), 1e-12))
        normal = np.array([-grade[0], -grade[1], 1.0])
        normal /= np.linalg.norm(normal)
        result = (np.array([x, y, h + float(self._actor_height[index])]),
                  _quat_wxyz(attitude([yaw], normal[None]))[0])
        self._vehicle_pose_refs[key] = result
        return result

    def _collect_camera(self, frame, camera_index, camera_to_world, common, assembler=None):
        timestamp = int(self._calibration["training_timestamps_us"][frame])
        collected = {key: [] for key in ("means", "scales", "quats", "opacities", "rgbs")}
        reusable = {key: [] for key in collected}
        actors = set()
        if self._packed_rigid is not None:
            pos, rotation, present, _ = sample_rigid_camera_pose(
                self._calibration["training_timestamps_us"], self._calibration["image_timestamps_us"],
                frame, camera_index, *self._packed_rigid)
            if self._actor_height is not None:
                pos, rotation = self._stand_actors_on_road(pos, rotation, frame)
            # One CPU interpolation batch and two small H2D copies per camera;
            # no per-actor GPU scalar validation/synchronization.
            pos, rotation = pos.to(self.device), rotation.to(self.device)
        for token, name in self.submodel_names.items():
            model = self.gaussian_models[name]
            if token in self._suppressed_actors:
                continue        # held out of the world: not simulated or scored, so not drawn
            simulated_vehicle = (self.actor_pose_source == "scenario"
                                 and token in self._vehicle_tokens)
            if simulated_vehicle and token not in self._render_agent_states:
                continue
            # Log-replay actors are evaluated at **their own replay row**, not the sim frame. When
            # the replay clock moves an actor off the log clock it is still drawn where the
            # simulator placed it, and pedestrian deformation (walking pose) is evaluated at that
            # row too, so it stays inside its own domain. On the log clock the row equals the sim
            # frame, so everything below gives exactly the sim-frame values.
            actor_frame, actor_stamp = frame, timestamp
            row = self._render_agent_rows.get(token)
            if row is not None and not simulated_vehicle:
                actor_frame = int(np.clip(row, 0, len(self._calibration["training_timestamps_us"]) - 1))
                actor_stamp = int(self._calibration["training_timestamps_us"][actor_frame])
            if (self._beyond_horizon and not simulated_vehicle and row is None
                    and token not in SCENERY_TOKENS
                    and not isinstance(model, OmniReTrafficLightSubModel)):
                continue                      # beyond the stored range: actor's log has ended
            if (not simulated_vehicle and token in self._actor_visibility
                    and not self._actor_visibility[token][actor_frame]):
                continue
            if token in self._rigid_tables:
                index = self._rigid_indices[token]
                if not simulated_vehicle and not bool(present[index]) and actor_frame == frame:
                    continue
                actor_pos, actor_quat = pos[index], rotation[index]
                if actor_frame != frame:
                    # The batch sampler draws every rigid actor of a frame at once, corrected to
                    # the camera exposure time. An actor whose row has moved is not in that batch,
                    # so read its own row's pose directly -- dropping the exposure correction (a
                    # few ms within the frame) in favour of the right row.
                    trans, quats, visible = self._rigid_tables[token]
                    if not bool(visible[actor_frame]):
                        continue
                    actor_pos = trans[actor_frame, 0].to(self.device)
                    actor_quat = quats[actor_frame, 0].to(self.device)
                on_road = False
                if simulated_vehicle:
                    state = self._render_agent_states[token]
                    # With a road surface, a moving vehicle stands on the road under its current
                    # position, and a parked vehicle excluded when baking the road keeps the height
                    # the checkpoint learned -- the same rule as the log-replay path. The latter
                    # holds only while the simulator pins it at its log pose: IDM also drives off
                    # vehicles that stood in a lane, and one carrying its parked height would float.
                    parked = (self._static_actors is not None and bool(self._static_actors[index])
                              and token in self._held_at_log_pose)
                    on_road = (self._actor_height is not None and not parked
                               and isinstance(self._path_lift, RoadSurface))
                    if on_road:
                        p_np, q_np = self._stand_simulated_vehicle_on_road(index, token, state)
                        actor_pos = torch.as_tensor(p_np, dtype=actor_pos.dtype, device=actor_pos.device)
                        actor_quat = torch.as_tensor(q_np, dtype=actor_quat.dtype, device=actor_quat.device)
                    else:
                        learned_pose = self._simulated_vehicle_pose(token, state, keep_learned_z=parked)
                        if learned_pose is None:
                            continue
                        actor_pos, actor_quat = learned_pose
                if simulated_vehicle and not on_road:
                    actor_pos = actor_pos.clone()
                    actor_pos[:2] = torch.as_tensor(state[:2], dtype=actor_pos.dtype,
                                                   device=actor_pos.device)
                    old_yaw = quat_to_angle(actor_quat)["yaw"]
                    delta = torch.as_tensor(state[-1], dtype=actor_quat.dtype,
                                            device=actor_quat.device) - old_yaw
                    half = delta / 2
                    turn = torch.stack((torch.cos(half), torch.zeros_like(half),
                                        torch.zeros_like(half), torch.sin(half)))
                    actor_quat = quat_mult(turn, actor_quat)
                # Avoid the native static-in-log early return: this explicit camera
                # pose is authoritative for both static and moving rigid actors.
                means = model.get_means(global_quat=actor_quat, global_trans=actor_pos)
                if type(model) in (mtgs.RigidModel, mtgs.MirroredModel):
                    scales, opacities = self._rigid_attributes_cached(name, model)
                    quats = model.get_quats(global_quat=actor_quat, global_trans=actor_pos)
                else:
                    # Preserve evaluation order for unknown model implementations.
                    scales = model.get_scales()
                    quats = model.get_quats(global_quat=actor_quat, global_trans=actor_pos)
                    opacities = model.get_opacity()
                gs = dict(means=means, scales=scales, quats=quats, opacities=opacities)
            else:
                if token not in common:
                    common[token] = self._global_gaussians_cached(
                        name, model, None, None, actor_stamp)
                gs = common[token]
                if gs is None:
                    continue
            if int(gs["means"].shape[0]) == 0:
                continue
            colors = None if isinstance(model, OmniReTrafficLightSubModel) else self._sh_colors_cached(name, model)
            # Colour is evaluated at the same row. Pedestrians take colour from the deformation
            # part get_global_gaussians chose, so using the sim frame here alone would give pose
            # and colour from different times.
            rgb = (model.get_gaussian_rgbs(camera_to_worlds=camera_to_world[None], timestamp=actor_stamp, device=self.device)
                   if colors is None else self._rgbs_from_colors(model, colors, camera_to_world[None]))
            static = gs is self._static_gs_cache.get(name)
            rigid = self._rigid_attribute_cache.get(name)
            for key in ("means", "scales", "quats", "opacities"):
                collected[key].append(gs[key].to(self.device))
                reusable[key].append(static or (rigid is not None and
                    ((key == "scales" and gs[key] is rigid[0]) or
                     (key == "opacities" and gs[key] is rigid[1]))))
            collected["rgbs"].append(rgb.to(self.device))
            reusable["rgbs"].append(False)
            if token not in SCENERY_TOKENS and not isinstance(model, OmniReTrafficLightSubModel):
                actors.add(token)
        if not collected["means"]:
            raise ValueError("bank produced no Gaussian geometry")
        if assembler is None:
            return {key: torch.cat(values, dim=0) for key, values in collected.items()}, actors
        return {key: assembler.assemble(key, values, reusable[key])
                for key, values in collected.items()}, actors

    def set_traffic_light_source_frames(self, mapping, *, active_head_ids=None, allow_unobserved=None):
        nodes = [m for m in self.gaussian_models.values() if isinstance(m, OmniReTrafficLightSubModel)]
        if len(nodes) != 1:
            raise ValueError("traffic-light override requires one combined head group")
        if allow_unobserved:
            nodes[0].set_source_frames(mapping, active_head_ids=active_head_ids, allow_unobserved=allow_unobserved)
        else:
            nodes[0].set_source_frames(mapping, active_head_ids=active_head_ids)

    def _apply_traffic_light_source_frames(self, render_state):
        mapping = render_state.get(RenderState.TL_SOURCE_FRAMES)
        if mapping is not None:
            # Strict mode is deliberate.  Passing active_head_ids would make the traffic-light
            # node ignore in-window representative overrides and silently return to natural replay.
            allowed = render_state.get(RenderState.TL_ALLOW_UNOBSERVED)
            if allowed:                  # label-allowed unobserved representative frames (ALLOW_UNOBSERVED_SCENES only)
                self.set_traffic_light_source_frames(mapping, allow_unobserved=allowed)
            else:
                self.set_traffic_light_source_frames(mapping)

    @torch.no_grad()
    def render(self, render_state):
        tick, frame = self._tick_index(render_state[RenderState.TIMESTAMP])
        self._beyond_horizon = tick >= len(self._calibration["training_timestamps_us"])
        self.timestamp = int(self._calibration["training_timestamps_us"][frame])
        self._apply_traffic_light_source_frames(render_state)
        states = {token: np.array(value, dtype=float, copy=True)
                  for token, value in render_state[RenderState.AGENT_STATE].items()}
        for value in states.values():
            value[:2] -= np.asarray(self.recon2global_translation)[:2]
        self._render_agent_states = states
        # Row each actor is replaying. On the log clock it equals the sim frame, so none of the
        # checks below change.
        self._render_agent_rows = {
            token: int(row)
            for token, row in (render_state.get(RenderState.AGENT_SOURCE_ROW) or {}).items()}
        self._suppressed_actors = frozenset(
            render_state.get(RenderState.SUPPRESSED_ACTORS, ()))
        self._held_at_log_pose = frozenset(
            render_state.get(RenderState.HELD_AT_LOG_POSE, ()))
        self._vehicle_pose_refs = {}
        ego = self._ego_at_tick(states["ego"], frame)
        if render_state.get(RenderState.SKIP_CAMERAS):
            return self._pose_only_render(ego, render_state)
        cameras, intrinsics, maps, shape, indices = self._camera_inputs(render_state[RenderState.CAMERAS], ego, frame)
        renders, alphas, rgbs, common = [], [], [], {}
        self.active_tokens = []
        active = set()
        mode = "RGB+ED" if self.render_depth else "RGB"
        for name, output_index in self.sensor_mapping.items():
            camera = cameras[output_index]
            gaussians, tokens = self._collect_camera(
                frame, indices[output_index], camera, common, assembler=self._gaussian_assembly)
            active.update(tokens)
            view = torch.eye(4, device=self.device, dtype=camera.dtype)
            view[:3, :3] = camera[:3, :3].T
            view[:3, 3] = -view[:3, :3] @ camera[:3, 3]
            rendered, alpha, _ = mtgs.rasterization(
                means=gaussians["means"], quats=gaussians["quats"], scales=gaussians["scales"],
                opacities=gaussians["opacities"], colors=gaussians["rgbs"][None],
                viewmats=view[None], Ks=intrinsics[output_index:output_index+1],
                width=shape["width"], height=shape["height"], tile_size=16, packed=False,
                near_plane=self._calibration["near_plane"], far_plane=1e10, render_mode=mode,
                sparse_grad=False, absgrad=True, rasterize_mode=self.rasterize_mode, radius_clip=self.radius_clip)
            rgb = self.composite_background(rendered[..., :3], alpha, camera[None],
                intrinsics[output_index:output_index+1], shape)
            rgb = apply_exposure(rgb, self._exposure, frame, name)
            renders.append(rendered); alphas.append(alpha); rgbs.append(rgb)
            del gaussians
        self.active_tokens = sorted(active)
        result = self._postprocess_render(torch.cat(rgbs), torch.cat(renders), torch.cat(alphas), maps, shape, mode)
        ego_global = ego.cpu().numpy().copy()
        ego_global[:3, 3] += np.asarray(self.recon2global_translation)
        # The calibration handed on (planner extrinsics, viewer projections) is the RIG: each
        # camera against the reconstructed ego (_colmap_ego), with or without a lift. The image
        # always comes from the reconstruction's camera -- tick renders logged ego @ saved
        # residual, which is exactly camera_to_global at that row -- but the saved camera_to_ego
        # is relative to the logged ego, whose height is off from the reconstructed road by
        # metres per row. A
        # BEV model reading that as its extrinsics looks for the ground in the wrong place. The
        # rig is ~1.5 m and constant, which is what the model was trained on. DataManager can
        # replace the factory calibration for this opt-in renderer without changing other
        # renderers' camera dictionaries.
        if self._rig is None:
            self._rig = self._colmap_ego()[1]
        camera_calibrations = {
            name: dict(camera_to_ego=np.array(self._rig[frame, indices[index]], copy=True),
                       cam_intrinsic=intrinsics[index].detach().cpu().numpy().copy(),
                       distortion=np.zeros(5, dtype=np.float64),
                       projection_semantics="training_undistorted")
            for name, index in self.sensor_mapping.items()
        }
        result.update(ego2global=ego_global, render_state=render_state,
                      camera_calibrations=camera_calibrations,
                      omnire=dict(source_frame=frame, source_timestamp_us=self.timestamp,
                          requested_tick=tick, beyond_saved_horizon=bool(self._beyond_horizon),
                          camera_pose_policy="logged_residual_at_tick",
                          actor_pose_source=self.actor_pose_source,
                          actor_manifest_schema_version=self._actor_manifest["schema_version"],
                          actor_population_authority=self._actor_manifest["population_authority"]))
        return result

    def composite_background(self, rgb, alpha, camera_to_worlds, intrinsics, shape):
        """Fill uncovered pixels from the live bank's sky cubemap.

        Matches the original reconstruction's EnvLight: a linear cubemap lookup with no
        clamping and no semantic filling, and never a blend of two sources' skies.
        """
        if self.original_sky is None:
            # No bank: the stock flat fill is right (the asset has its own skybox node),
            # and importing nvdiffrast for it would be wrong.
            return super().composite_background(rgb, alpha, camera_to_worlds, intrinsics, shape)

        import nvdiffrast.torch as dr

        height, width = shape["height"], shape["width"]
        device = rgb.device
        intrinsics = intrinsics.to(device=device, dtype=torch.float32)
        camera_to_worlds = camera_to_worlds.to(device=device, dtype=torch.float32)

        # Ray direction through every pixel centre, camera frame then world frame.
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32) + .5,
            torch.arange(width, device=device, dtype=torch.float32) + .5,
            indexing="ij")
        pixels = torch.stack((x, y, torch.ones_like(x)), -1)
        dirs = torch.einsum("cij,hwj->chwi", torch.linalg.inv(intrinsics), pixels)
        dirs = torch.einsum("cij,chwj->chwi", camera_to_worlds[:, :3, :3], dirs)
        # The cubemap was baked in OpenGL axes (y up, -z forward); ours is z up.
        opengl = torch.stack((dirs[..., 0], dirs[..., 2], -dirs[..., 1]), -1).contiguous()

        # One texture call for every camera: flatten the camera axis into the row axis,
        # which the cube lookup does not care about, then restore the caller's shape.
        sky = dr.texture(self.original_sky[None].to(device),
                         opengl.reshape(1, -1, width, 3),
                         filter_mode="linear", boundary_mode="cube").reshape_as(rgb)
        return rgb + (1 - alpha) * sky
