"""A numpy 1 interpreter loads checkpoints saved with numpy 2 (some scene checkpoints are)."""
import io
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from odyssey_renderer.mtgs.utils.portable_utils import alias_numpy2_pickle_modules  # noqa: E402


def numpy2_bytes(value):
    """torch.save bytes whose reconstruct function points under numpy._core, as numpy 2 writes them."""
    buf = io.BytesIO()
    torch.save({"recon2world_translation": value}, buf, _use_new_zipfile_serialization=False)
    raw = buf.getvalue()
    assert b"numpy.core.multiarray" in raw
    return raw.replace(b"numpy.core.multiarray", b"numpy._core.multiarray")


@pytest.mark.skipif(np.__version__ >= "2", reason="numpy 2 reads numpy._core natively")
def test_a_numpy2_checkpoint_loads_with_the_same_values():
    value = np.array([331261.05403610866, 4690963.884356629, -2.3595586421898447])
    raw = numpy2_bytes(value)
    for k in [m for m in sys.modules if m == "numpy._core" or m.startswith("numpy._core.")]:
        del sys.modules[k]
    with pytest.raises(ModuleNotFoundError, match="numpy._core"):
        torch.load(io.BytesIO(raw), weights_only=False)
    alias_numpy2_pickle_modules()
    out = torch.load(io.BytesIO(raw), weights_only=False)["recon2world_translation"]
    assert out.dtype == value.dtype and np.array_equal(out, value)
