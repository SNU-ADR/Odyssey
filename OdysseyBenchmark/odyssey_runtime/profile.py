"""Declared sensor/time contract, checked against the model before observation delivery."""

import json
import math
from pathlib import Path

CAMERAS = (
    "CAM_F0",
    "CAM_L0",
    "CAM_R0",
    "CAM_L1",
    "CAM_R1",
    "CAM_L2",
    "CAM_R2",
    "CAM_B0",
)


class ModelProfile:
    def __init__(self, data):
        known = {
            "version",
            "planner_id",
            "sim_dt",
            "history_times_s",
            "camera_times_s",
            "planning_interval_s",
            "render_resolution",
            "preprocessing",
            "stateful",
            "reset_method",
            "output",
            "feature_shapes",
            "shared_capacity_mb",
            "record_images",
            "record_legacy_ipc",
            "audit",
            "model_overrides",
            "seed",
            "navigation",
        }
        if set(data) - known:
            raise ValueError(f"unknown profile settings: {sorted(set(data) - known)}")
        seed = data.get("seed")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("seed must be an integer in [0, 2**32)")
        overrides = data.get("model_overrides", [])
        if not isinstance(overrides, list) or any(
            not isinstance(x, str) or not x for x in overrides
        ):
            raise ValueError(
                "model_overrides must be a list of native Hydra override strings"
            )
        self.data = dict(data)
        self.dt = float(data["sim_dt"])
        if data.get("version") != 1 or not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("profile version must be 1 and sim_dt positive")
        if (
            data.get("render_resolution") != "native"
            or data.get("preprocessing") != "model_native"
        ):
            raise ValueError(
                "this backend requires native render resolution and model_native preprocessing"
            )
        if float(data["planning_interval_s"]) != self.dt:
            raise ValueError(
                "this backend preserves one planner update per simulation tick"
            )
        self.times = tuple(float(t) for t in data["history_times_s"])
        if (
            not self.times
            or self.times[-1] != 0
            or tuple(sorted(set(self.times))) != self.times
        ):
            raise ValueError(
                "history must increase strictly from the past through zero"
            )
        self.offsets = tuple(self._ticks(t) for t in self.times)
        self.cameras = {
            k: tuple(float(t) for t in v) for k, v in data["camera_times_s"].items()
        }
        if not self.cameras or any(k not in CAMERAS for k in self.cameras):
            unknown = sorted(k for k in self.cameras if k not in CAMERAS)
            raise ValueError(
                f"profile must name supported cameras: {unknown or 'none declared'}; "
                f"the rig has {', '.join(CAMERAS)}"
            )
        for k, v in self.cameras.items():
            if (
                not v
                or tuple(sorted(set(v))) != v
                or any(t not in self.times for t in v)
            ):
                raise ValueError(
                    f"{k} camera times must be a nonempty ordered subset of history"
                )
        nav = data.get("navigation")
        if (
            not isinstance(nav, dict)
            or set(nav) != {"driving_command", "sd_route"}
            or not isinstance(nav["driving_command"], bool)
            or nav["sd_route"] not in ("none", "features", "targets")
        ):
            raise ValueError(
                'navigation must declare {"driving_command": true|false, '
                '"sd_route": "none"|"features"|"targets"}'
            )
        self.navigation = dict(nav)
        if data.get("stateful") and not data.get("reset_method"):
            raise ValueError("stateful adapters require an explicit reset_method")
        output = data["output"]
        if (
            output.get("coordinates") != "ego_future"
            or list(output["shape"])[1:] != [3]
            or output["shape"][0] <= 0
        ):
            raise ValueError("output must declare positive (N,3) future ego poses")
        if not math.isfinite(float(output["plan_dt"])) or float(output["plan_dt"]) <= 0:
            raise ValueError("output plan_dt must be positive")

    def _ticks(self, seconds):
        x = seconds / self.dt
        if not math.isfinite(x) or seconds > 0 or abs(x - round(x)) > 1e-8:
            raise ValueError("history times must be past integral simulation ticks")
        return int(round(x))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    @property
    def history_capacity(self):
        return 1 - self.offsets[0]

    def camera_indices(self, name):
        return [self.times.index(t) for t in self.cameras.get(name, ())]

    def sample_history(self, frames, step):
        by_step = {int(f["frame_idx"]): f for f in frames}
        selected = []
        for off in self.offsets:
            wanted = max(0, step + off)
            if wanted not in by_step:
                raise ValueError(f"missing observation at step {wanted}")
            selected.append(by_step[wanted])
        return selected

    def validate_sensor_config(self, sensor):
        for name in CAMERAS:
            value = getattr(sensor, name.lower())
            actual = (
                list(range(len(self.times)))
                if value is True
                else ([] if value is False else list(value))
            )
            if actual != self.camera_indices(name):
                raise ValueError(
                    f"{name}: native sensor history {actual} differs from profile {self.camera_indices(name)}"
                )
        if getattr(sensor, "lidar_pc", False):
            raise ValueError("camera-only runtime cannot supply lidar")

    def validate_navigation(self, planner):
        """The model must read exactly the navigation inputs the profile declares."""
        declared = self.navigation
        reads_command = bool(planner.uses_driving_command)
        if reads_command != declared["driving_command"]:
            raise ValueError(
                f"model {'reads' if reads_command else 'drops'} driving_command; "
                f"profile declares driving_command={str(declared['driving_command']).lower()}"
            )
        if planner.sd_route_input != declared["sd_route"]:
            raise ValueError(
                f"adapter feeds the SD route via {planner.sd_route_input!r}; "
                f"profile declares sd_route={declared['sd_route']!r}"
            )
        flag = getattr(getattr(planner.agent, "_config", None), "use_sdroute", None)
        if flag is not None and bool(flag) != (declared["sd_route"] != "none"):
            raise ValueError(
                f"model config use_sdroute={bool(flag)} contradicts sd_route={declared['sd_route']!r}"
            )

    def validate_features(self, features):
        for key, shape in self.data.get("feature_shapes", {}).items():
            if key not in features:
                raise ValueError(
                    f"{key}: declared in feature_shapes, but the model has no such feature "
                    f"(it has {sorted(features)})"
                )
            actual = list(getattr(features[key], "shape", ()))
            if actual != list(shape):
                raise ValueError(
                    f"{key}: model feature shape {actual} differs from profile {list(shape)}"
                )
