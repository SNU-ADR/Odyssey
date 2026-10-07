"""One causal observation/plan session owned by the simulation process."""

import atexit
import io
import json
import os
from pathlib import Path
import pickle
import time
import uuid
import numpy as np
from .profile import ModelProfile
from .planner import encode_jpeg
from .recording import Recorder
from .transport import SharedWorker

_SESSION = None

#: Frame fields the planner does not receive (NAVSIM's agent input has none of them): the other
#: road users' boxes and tracks, every gt_* field (logged boxes, the ego's logged past and future),
#: the log's mission route and signal states, and the replay's source rows.
PRIVILEGED_FIELDS = frozenset({"anns", "roadblock_ids", "route_roadblock_ids", "traffic_lights",
                               "source_rows", "signal_rows"})


def planner_view(frame):
    """The frame as the planner receives it: without the privileged fields."""
    return {k: v for k, v in frame.items() if k not in PRIVILEGED_FIELDS and not k.startswith("gt_")}


def get_session():
    global _SESSION
    path = os.environ.get("ODYSSEY_RUNTIME_PROFILE")
    if not path:
        return None
    if _SESSION is None:
        _SESSION = RuntimeSession(
            ModelProfile.load(path), Path(os.environ["ODYSSEY_RUNTIME_OUTPUT"])
        )
        atexit.register(close_session)
    return _SESSION


def close_session():
    global _SESSION
    value, _SESSION = _SESSION, None
    if value is not None:
        value.close()


class RuntimeSession:
    def __init__(self, profile, output):
        self.profile = profile
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.recorder = Recorder()
        self.worker = None
        self.images = {}
        self.image_paths = {}
        self.image_step = None
        self.response = None
        self.episode = None
        self.closed = False
        self.previous_start = None
        self.timings = (self.output / "timings.jsonl").open("a", buffering=1)
        (self.output / "profile.json").write_text(json.dumps(profile.data, indent=2))

    def encode_camera(self, step, name, image, path):
        if self.image_step != step:
            self.images.clear()
            self.image_paths.clear()
            self.image_step = step
        if name not in self.profile.cameras:
            raise ValueError(f"unrequested camera {name}")
        encoded = encode_jpeg(image)
        self.images[name] = encoded
        self.image_paths[name] = path
        if self.profile.data.get("record_images", True):
            self.recorder.submit_bytes(path, encoded)

    def _start_worker(self):
        root = Path(os.environ["ODYSSEY_ROOT"])
        sub = self.output.parent
        profile = self.profile.data
        if os.environ.get("ODYSSEY_PLANNER") != profile["planner_id"]:
            raise ValueError("selected planner differs from runtime profile")
        cfg = dict(
            profile=profile,
            cfg=os.environ["ODYSSEY_PLANNER_CFG"],
            checkpoint=os.environ["ODYSSEY_PLANNER_CKPT"],
            repo=os.environ["ODYSSEY_PLANNER_REPO"],
            sensor_root=str(sub / "odyssey_output/sensor_blobs"),
            route_file=os.environ.get("ODYSSEY_ROUTE_FILE", ""),
            history_stride=int(os.environ.get("ODYSSEY_HISTORY_STRIDE", "5")),
            overrides=profile.get("model_overrides", []),
            adapter=os.environ.get("ODYSSEY_PLANNER_ADAPTER", ""),
        )
        if profile.get("audit"):
            cfg["audit_dir"] = str(self.output / "model_audit")
        env = dict(os.environ)
        for k in ("LD_LIBRARY_PATH", "PYTHONHOME"):
            env.pop(k, None)
        # Model side-car packages reach the planner process only, never the simulator.
        extra = os.environ.get("ODYSSEY_PLANNER_PYTHONPATH", "")
        if extra:
            env["PYTHONPATH"] = extra + os.pathsep + env.get("PYTHONPATH", "")
        self.worker = SharedWorker(
            os.environ["ODYSSEY_PLANNER_PY"],
            "odyssey_runtime.planner:PlannerWorker",
            cfg,
            int(profile.get("shared_capacity_mb", 64)) * 1024**2,
            self.output / "planner_worker.log",
            env=env,
            timeout=float(os.environ.get("ODYSSEY_PLAN_WAIT_TIMEOUT_S", "300")),
        )

    def observe(self, frame, step, sim_dt):
        start = time.perf_counter()
        if abs(sim_dt - self.profile.dt) > 1e-9:
            raise ValueError("world cadence differs from model profile")
        if self.image_step != step or set(self.images) != set(self.profile.cameras):
            raise ValueError("camera batch missing or from another step")
        if step == 0:
            self.episode = uuid.uuid4().hex
            self.previous_start = None
        if self.worker is None:
            self._start_worker()
        # Preserve native camera metadata and all numeric dtypes through private pickle RPC.
        frame = planner_view(frame)
        blobs = [self.images[k] for k in self.profile.cameras]
        request = {
            "episode": self.episode,
            "step": step,
            "frame": frame,
            "images": [(k, len(self.images[k])) for k in self.profile.cameras],
        }
        # Adapters that read cameras by path (ReCogDrive, CAMERA_FILES) open this step's JPEG
        # files: write them even without --record (the bytes sent below), and finish the queued
        # writes first, or a slow filesystem hands them a missing or partial file.
        camera_files = getattr(self.worker, "ready", {}).get("camera_files", False)
        if camera_files and not self.profile.data.get("record_images", True):
            for name in self.profile.cameras:
                self.recorder.submit_bytes(self.image_paths[name], self.images[name])
        self.recorder.flush()
        rpc = time.perf_counter()
        response = self.worker.request(request, b"".join(blobs))
        if response["step"] != step or response["episode"] != self.episode:
            raise RuntimeError("planner returned stale episode/step")
        self.response = response
        if self.profile.data.get("record_legacy_ipc", True):
            from odyssey_bridge.ipc_common import scene_prefix

            sub = self.output.parent
            self.recorder.submit_bytes(
                sub / "frames" / f'{frame["log_token"]}_{step}.pkl',
                pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL),
            )
            stem = sub / "plan_traj" / f'{scene_prefix(frame["log_token"])}_{step+1}'
            if response.get("route_exhausted"):
                self.recorder.submit_bytes(str(stem) + ".route_exhausted", b"")
            else:
                stream = io.BytesIO()
                np.save(stream, response["plan"])
                self.recorder.submit_bytes(str(stem) + ".npy", stream.getvalue())
        row = {
            "step": step,
            "planner_rpc_ms": (time.perf_counter() - rpc) * 1000,
            "planner_ms": response.get("planner_ms"),
            "jpeg_bytes": sum(map(len, blobs)),
            "retained_images": response.get("retained_images"),
            "observation_interval_ms": (
                None
                if self.previous_start is None
                else (start - self.previous_start) * 1000
            ),
        }
        self.previous_start = start
        self.timings.write(json.dumps(row) + "\n")
        self.images.clear()
        self.image_paths.clear()

    def plan(self, step):
        if (
            self.response is None
            or self.response["step"] != step
            or self.response["episode"] != self.episode
        ):
            raise ValueError("no planner response for this step/episode")
        return None if self.response.get("route_exhausted") else self.response["plan"]

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.recorder.close()
        finally:
            if self.worker:
                self.worker.close()
            self.timings.close()
