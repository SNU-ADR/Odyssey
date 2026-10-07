"""Pin the GPU post-processing to be bit-identical to the original CPU path.

Reference path (original mtgs.py):
    rgb = torch.clamp(x, 0, 1).cpu().numpy() * 255
    rgb = rgb.astype(np.uint8)
    rgb = [cv2.cvtColor(rgb[i], cv2.COLOR_RGB2BGR) for i in range(rgb.shape[0])]

    python test_image_utils.py
"""
import cv2
import numpy as np
import pytest
import torch

from odyssey_renderer.mtgs.utils.image_utils import rgb_float_to_bgr_uint8


def _reference(x):
    rgb = torch.clamp(x, 0.0, 1.0).detach().cpu().numpy() * 255
    rgb = rgb.astype(np.uint8)
    return [cv2.cvtColor(rgb[i], cv2.COLOR_RGB2BGR) for i in range(rgb.shape[0])]


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_bit_identical_to_cpu_reference(device):
    torch.manual_seed(0)
    x = torch.rand(3, 64, 48, 3, device=device)
    got = rgb_float_to_bgr_uint8(x)
    want = _reference(x)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g.dtype == np.uint8 and g.shape == w.shape
        assert np.array_equal(g, w)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_out_of_range_values_are_clamped_identically(device):
    x = torch.tensor([[[[-0.5, 0.0, 0.5], [1.0, 1.5, 2.0]]]], device=device)
    assert np.array_equal(rgb_float_to_bgr_uint8(x)[0], _reference(x)[0])


def test_exact_boundary_values_round_the_same_way():
    """Values that are exactly an integer, or just below one, after ×255 expose truncation differences."""
    vals = [0.0, 1.0 / 255, 0.5, 127.5 / 255, 128.0 / 255, 254.0 / 255, 1.0]
    x = torch.tensor(vals, dtype=torch.float32).reshape(1, 1, len(vals), 1).repeat(1, 1, 1, 3)
    assert np.array_equal(rgb_float_to_bgr_uint8(x)[0], _reference(x)[0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
