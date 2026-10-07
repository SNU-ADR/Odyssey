"""Optional matched-input evidence; disabled in timing measurements."""

import hashlib
import json
from pathlib import Path
import numpy as np


def arrays(value, prefix=""):
    if isinstance(value, dict):
        for key, v in value.items():
            yield from arrays(v, prefix + str(key) + "/")
    elif hasattr(value, "detach"):
        yield prefix.rstrip("/"), value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        yield prefix.rstrip("/"), value


class ModelAudit:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.step = 0

    def observe(self, features, targets, output):
        data = dict(
            arrays({"features": features, "targets": targets or {}, "output": output})
        )
        np.savez(self.root / f"{self.step:06d}.npz", **data)
        hashes = {
            k: {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "sha256": hashlib.sha256(v.tobytes()).hexdigest(),
            }
            for k, v in data.items()
        }
        (self.root / f"{self.step:06d}.json").write_text(json.dumps(hashes, indent=2))
