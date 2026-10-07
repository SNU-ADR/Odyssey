"""The Fixer model over shared memory, with identical restore()."""

import os
from pathlib import Path
import sys
import time
import numpy as np
from .transport import SharedWorker


def validate_batch(images):
    if not images:
        raise ValueError("empty restorer batch")
    batch = np.stack(images)
    if batch.dtype != np.uint8 or batch.ndim != 4 or batch.shape[-1] != 3:
        raise ValueError("restorer requires same-sized uint8 HWC cameras")
    return batch


class FixerWorker:
    def __init__(self, config, buffer):
        sys.path.insert(0, config["root"])
        self.buffer = buffer
        from fixer_server import Restorer

        self.model = Restorer(config["checkpoint"], str(Path(config["root"]) / "src"))
        self.metadata = {"checkpoint": str(Path(config["checkpoint"]).resolve()),
                         "cudnn_benchmark_disabled": self.model.cudnn_benchmark_disabled}

    def handle(self, message):
        shape = tuple(message["shape"])
        size = int(np.prod(shape))
        if (
            len(shape) != 4
            or shape[-1] != 3
            or any(x <= 0 for x in shape)
            or size != message["payload_size"]
        ):
            raise ValueError("invalid restorer shape or payload")
        batch = np.ndarray(shape, dtype=np.uint8, buffer=self.buffer)
        start = time.perf_counter()
        out = self.model.restore(batch, bgr=True)
        if out.shape != shape or out.dtype != np.uint8:
            raise ValueError("restorer output shape/dtype changed")
        self.buffer[:size] = out.tobytes()
        return {"shape": shape, "server_ms": (time.perf_counter() - start) * 1000}


class SharedRestorer:
    def __init__(self, preset, capacity_mb=64):
        from odyssey_renderer.omnire.restorer_client import (
            _preset_paths,
            _env,
            _required,
        )

        root, checkpoint = _preset_paths(preset)
        self.config = {"root": root, "checkpoint": checkpoint}
        self.python = _required("ODYSSEY_FIXER_PY", _env("ODYSSEY_FIXER_PY", None))
        self.env = dict(os.environ)
        for k in (
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
            "PYTHONHOME",
            "TORCH_EXTENSIONS_DIR",
            "TORCH_CUDA_ARCH_LIST",
            "CUDA_HOME",
        ):
            self.env.pop(k, None)
        self.env.update(
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            CUDA_VISIBLE_DEVICES=_required("ODYSSEY_FIXER_GPU", _env("ODYSSEY_FIXER_GPU", None)),
            PYTHONPATH=str(Path(root) / "shims"),
            FIXER_MODELS_DIR=str(Path(root) / "models"),
            HF_HUB_OFFLINE="1",
            PYTHONNOUSERSITE="1",
            PYTHONUNBUFFERED="1",
        )
        self.log_path = (
            Path(os.environ.get("ODYSSEY_RUNTIME_OUTPUT", root))
            / f"fixer_worker_{os.getpid()}.log"
        )
        self.capacity = int(capacity_mb * 1024 * 1024)
        self.worker = None
        self.identifier = preset
        self.last_server_ms = None

    def restore_batch(self, images):
        batch = validate_batch(images)
        if self.worker is None:
            self.worker = SharedWorker(
                self.python,
                "odyssey_runtime.restorer:FixerWorker",
                self.config,
                self.capacity,
                self.log_path,
                env=self.env,
                timeout=float(os.environ.get("ODYSSEY_FIXER_START_TIMEOUT_S", "900")),
            )
            if self.worker.ready["checkpoint"] != str(Path(self.config["checkpoint"]).resolve()):
                self.close()
                raise RuntimeError("Fixer checkpoint identity mismatch")
        reply = self.worker.request({"shape": batch.shape}, batch.tobytes())
        if tuple(reply["shape"]) != batch.shape:
            raise RuntimeError("restorer returned wrong shape")
        self.last_server_ms = reply["server_ms"]
        out = np.ndarray(batch.shape, dtype=np.uint8, buffer=self.worker.buffer).copy()
        return list(out)

    def close(self):
        if self.worker:
            self.worker.close()
            self.worker = None
