"""Native planner feature builders with explicit in-memory camera loading."""

from collections import deque
import io
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
from .profile import ModelProfile


def seed_planner(seed):
    """Initialize explicit per-process model randomness before native imports/build."""
    if seed is None:
        return
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_jpeg(image):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("camera must be uint8 HWC BGR")
    ok, encoded = cv2.imencode(".jpg", image)  # same default parameters as cv2.imwrite
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return encoded.tobytes()


def wire_plan(native, plan_dt):
    from odyssey_bridge.ipc_common import upsample_trajectory

    return upsample_trajectory(native, plan_dt=plan_dt).astype(np.float32)


def sensor_summary(sensor):
    """The native SensorConfig as {field: bool | [indices]} of plain types."""
    out = {}
    for name in ("cam_f0", "cam_l0", "cam_r0", "cam_l1", "cam_r1", "cam_l2", "cam_r2",
                 "cam_b0", "lidar_pc"):
        value = getattr(sensor, name, False)
        out[name] = value if isinstance(value, bool) else [int(i) for i in value]
    return out


def device_name():
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # the name is informational only
        return "unknown"


def reset_episode(model, profile):
    name = profile.data.get("reset_method")
    if name:
        method = getattr(model, name, None)
        if not callable(method):
            raise ValueError("declared recurrent reset method is not callable")
        method()


class MemoryInputs:
    def __init__(self, profile):
        self.profile = profile
        self.images = {name: {} for name in profile.cameras}

    @property
    def retained_images(self):
        return sum(len(v) for v in self.images.values())

    def reset(self):
        for bank in self.images.values():
            bank.clear()

    def ingest(self, step, blobs):
        if set(blobs) != set(self.images):
            raise ValueError("camera set differs from profile")
        for name, encoded in blobs.items():
            with Image.open(io.BytesIO(encoded)) as im:
                image = np.array(im)  # exact native Cameras.from_camera_dict decode
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError("decoded camera must be RGB uint8 HWC")
            bank = self.images[name]
            bank[step] = image
            oldest = max(0, step + self.profile._ticks(min(self.profile.cameras[name])))
            for key in list(bank):
                if key < oldest:
                    del bank[key]

    def build(self, scene_list, sensor_config, sensor_root):
        from navsim.common.dataclasses import AgentInput, SensorConfig

        # The native metadata/coordinate conversion runs unchanged. Loading is disabled
        # explicitly, then requested cameras are filled; no global IO monkeypatch.
        # Use the same camera-only parser compatibility as the file adapter.
        # Native forks differ in whether Path(None) is guarded before LiDAR is
        # disabled; this does not request or fabricate a point cloud.
        from odyssey_bridge.ipc_common import patch_lidar_for_camera_only
        patch_lidar_for_camera_only()
        out = AgentInput.from_scene_dict_list(
            scene_list,
            Path(sensor_root),
            len(scene_list),
            SensorConfig.build_all_sensors(False),
        )
        for i, frame in enumerate(scene_list):
            for name in self.profile.cameras:
                if name.lower() not in sensor_config.get_sensors_at_iteration(i):
                    continue
                image = self.images[name].get(int(frame["frame_idx"]))
                if image is None:
                    raise ValueError(f'missing camera {name} at {frame["frame_idx"]}')
                ci = frame["cams"][name]
                # Fill the native empty Camera rather than assume every fork
                # accepts LTF's optional camera_path constructor argument.
                camera = getattr(out.cameras[i], name.lower())
                camera.image = image
                camera.sensor2lidar_rotation = ci["sensor2lidar_rotation"]
                camera.sensor2lidar_translation = ci["sensor2lidar_translation"]
                camera.intrinsics = ci["cam_intrinsic"]
                camera.distortion = ci["distortion"]
                if hasattr(camera, "camera_path"):
                    camera.camera_path = ci["data_path"]
        return out


class PlannerWorker:
    def __init__(self, config, buffer):
        self.profile = ModelProfile(config["profile"])
        seed_planner(self.profile.data.get("seed"))
        from .adapters import load_adapter
        self.buffer = buffer
        self.config = config
        cls = load_adapter(config.get("adapter", ""))
        kw = dict(
            cfg_name=config["cfg"],
            ckpt_path=config["checkpoint"],
            repo=config["repo"],
            sensor_blobs_root=config["sensor_root"],
            device=config.get("device", "cuda"),
            history_stride=config.get("history_stride", 5),
            overrides=config.get("overrides", []),
            navigation=self.profile.navigation,
            cameras={name: self.profile.camera_indices(name) for name in self.profile.cameras},
            history_frames=len(self.profile.times),
        )
        if config.get("route_file"):
            kw["route_file"] = config["route_file"]
        self.planner = cls(**kw)
        built = time.perf_counter()
        self.planner.build()
        build_seconds = time.perf_counter() - built
        self.profile.validate_sensor_config(self.planner._sensor_config)
        self.profile.validate_navigation(self.planner)
        self.planner.feature_validator = self.profile.validate_features
        if self.planner.PLAN_DT != self.profile.data["output"]["plan_dt"]:
            raise ValueError(
                f"native planner PLAN_DT differs from profile: the adapter emits poses every "
                f"{self.planner.PLAN_DT} s, output.plan_dt declares {self.profile.data['output']['plan_dt']} s"
            )
        reset_name = self.profile.data.get("reset_method")
        self.reset_model = (
            getattr(self.planner.agent, reset_name, None) if reset_name else None
        )
        if self.profile.data.get("stateful") and not callable(self.reset_model):
            raise ValueError("declared recurrent reset method is not callable")
        self.inputs = MemoryInputs(self.profile)
        if config.get("input_mode", "memory") == "memory":
            self.planner.input_builder = self.inputs.build
            self.planner.history_sampler = lambda fs: self.profile.sample_history(
                fs, int(fs[-1]["frame_idx"])
            )
        self.history = deque(maxlen=self.profile.history_capacity)
        self.episode = None
        self.last_step = -1
        self.audit = None
        if config.get("audit_dir"):
            from .audit import ModelAudit

            self.audit = ModelAudit(config["audit_dir"])
            self.planner.model_observer = self.audit.observe
        self.metadata = {
            "planner_id": self.profile.data["planner_id"],
            "plan_dt": self.planner.PLAN_DT,
            "cameras": list(self.profile.cameras),
            "history_capacity": self.profile.history_capacity,
            # Plain types only: the metadata is pickled back to the simulator and shown by
            # `odyssey_runtime check`.
            "adapter": f"{cls.__module__}:{cls.__name__}",
            "sensor_config": sensor_summary(self.planner._sensor_config),
            "dropped_sensors": list(getattr(self.planner, "dropped_sensors", [])),
            "camera_files": bool(getattr(self.planner, "CAMERA_FILES", False)),
            "navigation": {
                "driving_command": bool(self.planner.uses_driving_command),
                "sd_route": str(self.planner.sd_route_input),
            },
            "overrides": list(config.get("overrides", [])),
            "cfg": config["cfg"],
            "repo": config["repo"],
            "checkpoint": config["checkpoint"],
            "build_seconds": round(build_seconds, 2),
            "device_name": device_name(),
        }

    def handle(self, message):
        from odyssey_bridge.route_sidecar import RouteExhausted

        step = int(message["step"])
        episode = message["episode"]
        frame = message["frame"]
        if episode != self.episode or step == 0:
            if step != 0:
                raise ValueError("new episode must start at step zero")
            self.history.clear()
            self.inputs.reset()
            self.last_step = -1
            self.episode = episode
            reset_episode(self.planner.agent, self.profile)
        if step != self.last_step + 1 or int(frame["frame_idx"]) != step:
            raise ValueError("nonconsecutive planner observation")
        self.last_step = step
        offset = 0
        blobs = {}
        for name, size in message["images"]:
            if size < 1 or offset + size > message["payload_size"]:
                raise ValueError("invalid image payload bounds")
            blobs[name] = self.buffer[offset : offset + size]
            offset += size
        if offset != message["payload_size"]:
            raise ValueError("unexpected bytes after camera batch")
        if self.config.get("input_mode", "memory") == "memory":
            self.inputs.ingest(step, blobs)
        self.history.append(frame)
        if self.audit:
            self.audit.step = step
        start = time.perf_counter()
        try:
            plan = self.planner.infer(list(self.history))
        except RouteExhausted:
            return {"route_exhausted": True, "step": step, "episode": episode}
        declared = list(self.profile.data["output"]["shape"])
        if list(plan.shape) != declared:
            raise ValueError(
                f"planner output shape {list(plan.shape)} differs from the declared output.shape {declared}"
            )
        if not np.isfinite(plan).all():
            raise ValueError("planner output contains non-finite values")
        wire = wire_plan(plan, self.planner.PLAN_DT)
        return {
            "step": step,
            "episode": episode,
            "plan": wire,
            "native": plan,
            "planner_ms": (time.perf_counter() - start) * 1000,
            "retained_images": self.inputs.retained_images,
        }
