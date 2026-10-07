"""Shared skeleton for the third-party navsim planner backends.

The OdysseyZoo models (LTF, DrivoR, DiffusionDrive, SafeDrive, ReCogDrive) implement the same navsim
``AbstractAgent`` interface::

    agent.initialize()                 # load the checkpoint
    agent.get_sensor_config()          # which cameras/lidar per history frame
    agent.get_feature_builders()       # frame -> feature tensors
    agent.forward(features)            # -> dict with a trajectory

so one skeleton covers them; a subclass only supplies the repo layout, the hydra
config name, and any per-model quirk (see each module's docstring for which).

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
No extra trajectory-selection machinery (score EMA, temporal-consistency reselect,
route-centerline injection, aux/spatial-waypoint heads) is layered on here. These backends run their model's own stock selection and
emit exactly what it predicted. That keeps a third-party number comparable to the
model's published numbers; layering our tuning on top would measure neither.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from .. import ipc_common as ipc
except ImportError:                      # run as a plain script, not a package
    import ipc_common as ipc             # type: ignore[no-redef]


class NavsimPlanner:
    """One closed-loop planner step: raw frame dicts -> (N,3) ego-local @0.5 s.

    Subclasses set ``REPO_ENV``/``DEFAULT_CFG`` and may override ``_build_agent``,
    ``_build_features`` and ``_extract_trajectory``.
    """

    #: env var naming this model's repo checkout, and the fallback dir under models/
    REPO_ENV: str = ""
    REPO_DIRNAME: str = ""
    #: hydra agent config name (a .yaml under navsim/planning/script/config/common/agent)
    DEFAULT_CFG: str = ""
    #: human-readable tag used in log lines
    TAG: str = "planner"

    #: Seconds between the poses this model outputs. Nearly every navsim planner emits the
    #: standard 8 x 0.5 s, which ipc.upsample_trajectory resamples to the 0.1 s grid the simulator
    #: assumes. A model that ALREADY emits 0.1 s (e.g. a 40-pose vocabulary) must
    #: declare 0.1 here, or the upsampler stretches a 4 s plan into 20 s and the ego crawls.
    PLAN_DT: float = ipc.PLAN_DT

    #: True when the model opens its cameras as files under ``sensor_blobs_root`` instead of
    #: taking decoded images (ReCogDrive). The session then writes each step's JPEGs, the
    #: bytes it sends, before the request, also without --record.
    CAMERA_FILES: bool = False

    #: Where an adapter that feeds the SD route itself puts it: "features" or "targets".
    #: None means the adapter does not; the profile's navigation.sd_route then decides, and
    #: this class feeds ``route_centerline`` (1,P,5) and ``route_centerline_mask`` (1,P) there
    #: (the OdysseyZoo convention; P and the horizon come from the model config's
    #: ``route_cl_num_points`` / ``route_cl_horizon``, default 120 points over 120 m).
    SD_ROUTE: Optional[str] = None

    def __init__(self, cfg_name: str, ckpt_path: str, repo: str,
                 sensor_blobs_root: str, device: str = "cuda",
                 overrides: Optional[List[str]] = None,
                 history_stride: int = 1,
                 navigation: Optional[dict] = None,
                 route_file: str = "",
                 cameras: Optional[Dict[str, List[int]]] = None,
                 history_frames: Optional[int] = None):
        self.navigation = dict(navigation or {})
        self.route_file = str(route_file or "")
        # The profile's cameras ({"cam_f0": [history indices], ...}) and the number of history
        # frames the model receives. When given, build() narrows the model's native sensor
        # request to exactly these (see narrow_sensor_config); None leaves it untouched.
        self.cameras = {k.lower(): list(v) for k, v in cameras.items()} if cameras else None
        self.history_frames = int(history_frames) if history_frames else ipc.NUM_HISTORY_FRAMES
        self.dropped_sensors: List[str] = []
        requested = self.navigation.get("sd_route", "none")
        self._generic_route = self.SD_ROUTE is None and requested in ("features", "targets")
        self._route_builder = None
        self.cfg_name = cfg_name or self.DEFAULT_CFG
        self.ckpt_path = ckpt_path
        self.repo = str(repo)
        self.sensor_blobs_root = Path(sensor_blobs_root)
        self.device = device
        self.overrides = list(overrides) if overrides else []
        self.history_stride = max(1, int(history_stride or 1))

        self.agent = None
        self._feature_builders = None
        self._sensor_config = None
        # Per-instance runtime adapters; ordinary file-backed inference is unchanged.
        self.input_builder = None
        self.history_sampler = None
        self.model_observer = None
        self.feature_validator = None

    # ------------------------------------------------------------------ build
    def build(self) -> None:
        """Put the repo on sys.path, hydra-compose the agent, load the checkpoint."""
        if not self.ckpt_path or not os.path.isfile(self.ckpt_path):
            raise SystemExit(
                f"[{self.TAG}] checkpoint not found: {self.ckpt_path!r}\n"
                f"  download the published weights (huggingface.co/ADRLAB/odyssey-models) and point "
                f"ODYSSEY_MODELS_ROOT or the agent config's model.checkpoint at them")
        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)
        # These repos resolve asset paths (backbone weights, vocabularies) relative to CWD.
        # IPC paths are absolute, so
        # the chdir is safe for the server loop.
        os.chdir(self.repo)

        self._build_agent()
        self.agent = self.agent.eval().to(self.device)
        self._feature_builders = self.agent.get_feature_builders()
        self._sensor_config = self.agent.get_sensor_config()
        if self.cameras is not None:
            self._sensor_config, self.dropped_sensors = narrow_sensor_config(
                self._sensor_config, self.cameras, self.history_frames)
            if self.dropped_sensors:
                print(f"[{self.TAG}] sensors narrowed to the profile's cameras; not requested: "
                      f"{' '.join(self.dropped_sensors)}", flush=True)
        print(f"[{self.TAG}] loaded cfg={self.cfg_name} ckpt={os.path.basename(self.ckpt_path)} "
              f"repo={self.repo}", flush=True)
        if self._generic_route:
            try:
                from ..route_sidecar import StaticRouteCenterline
            except ImportError:          # run as a plain script, not a package
                from route_sidecar import StaticRouteCenterline  # type: ignore[no-redef]
            # strict: a missing route stops the run; a route model must not plan route-blind.
            self._route_builder = StaticRouteCenterline(
                getattr(self.agent, "_config", None), sidecar_dir=self.route_file, strict=True)
            print(f"[{self.TAG}] SD route fed via {self.navigation['sd_route']} (profile navigation)",
                  flush=True)

    @property
    def sd_route_input(self) -> str:
        """Where this planner receives the SD route: "features", "targets" or "none"."""
        return self.SD_ROUTE or (self.navigation.get("sd_route", "none") if self._generic_route else "none")

    def _build_agent(self) -> None:
        """Hydra-instantiate the agent yaml and run ``initialize()``.

        The default covers every agent whose yaml takes ``checkpoint_path`` as a field.
        """
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate

        agent_cfg_dir = os.path.join(
            self.repo, "navsim/planning/script/config/common/agent")
        with initialize_config_dir(version_base=None, config_dir=agent_cfg_dir):
            cfg = compose(config_name=self.cfg_name, overrides=self.overrides)
        if self.overrides:
            print(f"[{self.TAG}] config overrides: {self.overrides}", flush=True)
        self.agent = instantiate(cfg, checkpoint_path=self.ckpt_path)
        self.agent.initialize()

    # ------------------------------------------------------------------ infer
    def infer(self, frames: List[dict]) -> np.ndarray:
        """frames = parsed .pkl dicts oldest..newest -> (N,3) ego-local @0.5 s."""
        import torch

        scene_list = (self.history_sampler(frames) if self.history_sampler is not None
                      else ipc.pad_history(frames, history_stride=self.history_stride))
        # Lateral accel, derived (see ipc.inject_ay): the plant leaves ACCELERATION_Y at zero even
        # though a yawing bicycle has a_y = v_x*omega. These planners read ego_acceleration in
        # their status_feature, so that zero reaches them and the same switch has to apply here. Deep-copied so the on-disk frames and the simulator's own
        # state keep ay = 0.
        if ipc.inject_ay_enabled():
            import copy
            scene_list = ipc.inject_ay([copy.deepcopy(f) for f in scene_list])
            if not getattr(self, "_inject_ay_logged", False):
                print(f"[{self.TAG}] INJECT_AY on: ay <- v_x*omega "
                      f"(newest frame ay={scene_list[-1]['ego_dynamic_state'][3]:+.4f})", flush=True)
                self._inject_ay_logged = True

        agent_input = self._build_agent_input(scene_list)
        features = self._build_features(agent_input, scene_list)
        features = {k: self._batch(v) for k, v in features.items()}
        if self.feature_validator is not None:
            self.feature_validator(features)

        with torch.no_grad():
            out = self._forward(features)
        if self.model_observer is not None:
            self.model_observer(features, getattr(self, '_last_route', None), out)
        return self._extract_trajectory(out)

    def _batch(self, value):
        """Add the batch dim and move to the device: tensors directly, and tensors nested in
        lists/tuples one by one (SafeDrive's ``matrices`` is a list of lists of tensors; left
        unbatched, the frame axis would land in the batch slot). Anything else is unchanged."""
        import torch

        if torch.is_tensor(value):
            return value.unsqueeze(0).to(self.device)
        if isinstance(value, (list, tuple)):
            return type(value)(self._batch(x) for x in value)
        return value

    def _build_agent_input(self, scene_list: List[dict]):
        """Frame dicts -> navsim ``AgentInput``.

        These agents' feature builders read ``cam.image`` directly, so images are decoded
        here by navsim's stock loader -- i.e. no ``just_img_path``.
        """
        from navsim.common.dataclasses import AgentInput

        if self.input_builder is not None:
            return self.input_builder(scene_list, self._sensor_config, self.sensor_blobs_root)
        ipc.patch_lidar_for_camera_only()
        return AgentInput.from_scene_dict_list(
            scene_list,
            self.sensor_blobs_root,
            num_history_frames=ipc.NUM_HISTORY_FRAMES,
            sensor_config=self._sensor_config,
        )

    def _build_features(self, agent_input, scene_list: List[dict]) -> Dict:
        features: Dict = {}
        for b in self._feature_builders:
            features.update(b.compute_features(agent_input))
        if self._generic_route:
            # The route follows the NEWEST raw frame (ego pose + scene token).
            self._newest_frame = scene_list[-1]
            if self.navigation["sd_route"] == "features":
                route, mask = self._route_builder.build(self._newest_frame)
                # infer() adds the batch dim to every tensor in this dict.
                features["route_centerline"] = route[0]
                features["route_centerline_mask"] = mask[0]
                self._last_route = (route, mask)
        return features

    def _forward(self, features: Dict) -> Dict:
        if self._generic_route and self.navigation["sd_route"] == "targets":
            route, mask = self._route_builder.build(self._newest_frame)
            targets = {"route_centerline": route.to(self.device),
                       "route_centerline_mask": mask.to(self.device)}
            self._last_route = targets
            return self.agent.forward(features, targets)
        return self.agent.forward(features)

    def _extract_trajectory(self, out: Dict) -> np.ndarray:
        """Pull the (N,3) ego-local trajectory out of the model's output dict."""
        import torch

        traj = out["trajectory"]
        if isinstance(traj, torch.Tensor):
            traj = traj.detach().cpu().numpy()
        traj = np.asarray(traj, dtype=np.float64)
        if traj.ndim == 3:              # (B,N,3) -> drop the batch dim we added
            traj = traj[0]
        if traj.ndim != 2 or traj.shape[-1] < 3:
            raise RuntimeError(
                f"[{self.TAG}] unexpected trajectory shape {traj.shape}; expected (N,3)")
        return traj[:, :3]

    @property
    def uses_driving_command(self) -> bool:
        """Does the loaded model actually READ the navigation command?

        Asked of the model's own composed config, not of a list of tag names kept here:
        ``drop_driving_command`` slices the command(4) off status_feature at ingest AND
        narrows _status_encoding to match (8 -> 4), so a True value means the four channels
        reach no weight. The command-free SD-route arms set it (their yaml: "the route is
        the only routing signal reaching the planner") and their build() already refuses to
        run without it; plain transfuser_agent leaves it unset and does condition on it.

        The profile's navigation.driving_command is checked against this at startup.
        """
        cfg = getattr(self.agent, "_config", None)
        return not bool(getattr(cfg, "drop_driving_command", False))


RIG_CAMERAS = ("cam_f0", "cam_l0", "cam_r0", "cam_l1", "cam_r1", "cam_l2", "cam_r2", "cam_b0")


def narrow_sensor_config(native, cameras: Dict[str, List[int]], history_frames: int):
    """Restrict the model's native ``SensorConfig`` to the cameras the profile declares.

    A native config may request more than the model reads (DiffusionDrive asks for all eight
    views plus lidar, then stitches three). Only the declared cameras are rendered, and a
    requested-but-unrendered camera makes navsim's loader open a blank path, so the request
    is narrowed to what is declared. The other direction is refused: a declared camera (or
    history index) the model does not request would be rendered for nothing, and the profile
    would state an input that reaches no weight.

    Returns (config, dropped) where ``dropped`` lists the narrowed fields as ``name[indices]``.
    Fields equal to the declaration are left untouched, so a model whose native request already
    matches keeps its exact config object values.
    """
    import copy
    import dataclasses

    def indices(value):
        if value is True:
            return list(range(history_frames))
        return [] if not value else [int(i) for i in value]

    changes, dropped = {}, []
    for name in RIG_CAMERAS:
        native_idx = indices(getattr(native, name))
        declared = [int(i) for i in cameras.get(name, [])]
        extra = sorted(set(declared) - set(native_idx))
        if extra:
            raise ValueError(
                f"{name.upper()}: the profile declares history indices {declared}, but the model "
                f"requests {native_idx}; it would never read indices {extra}")
        if declared != native_idx:
            changes[name] = declared if declared else False
            dropped.append(f"{name}{native_idx}")
    if indices(getattr(native, "lidar_pc", False)):
        changes["lidar_pc"] = False
        dropped.append(f"lidar_pc{indices(getattr(native, 'lidar_pc'))}")
    if not changes:
        return native, dropped
    if dataclasses.is_dataclass(native):
        return dataclasses.replace(native, **changes), dropped
    narrowed = copy.copy(native)
    for key, value in changes.items():
        setattr(narrowed, key, value)
    return narrowed, dropped


def route_arm_enabled(config, flag):
    """OdysseyZoo's single route arm is the former KV arm; preserve legacy guards."""
    if hasattr(config, flag):
        return bool(getattr(config, flag))
    return flag == "use_sdroute_kv" and bool(getattr(config, "use_sdroute", False))


def route_arm_value(config, name):
    """Removed knobs have fixed values in OdysseyZoo's published route architecture."""
    if hasattr(config, name):
        return getattr(config, name)
    if getattr(config, "use_sdroute", False) and not hasattr(config, "use_sdroute_kv"):
        return {"sdroute_horizon_m": 120, "sdroute_seg_index_mode": "learned"}.get(name)
    return None
