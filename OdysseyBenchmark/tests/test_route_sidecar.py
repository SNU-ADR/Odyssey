"""StaticRouteCenterline: an explicit navigation pose samples the route the same way as the true
pose, and a scene's route file is used as it is."""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'odyssey_bridge'))
from route_sidecar import StaticRouteCenterline  # noqa: E402


class _Cfg:
    route_cl_num_points = 120
    route_cl_horizon = 120.0


def _quat(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


@pytest.fixture
def sidecar_dir(tmp_path):
    xy = np.stack([np.arange(0.0, 400.0, 1.0), np.zeros(400)], axis=1)   # straight east, 400 m
    np.savez(tmp_path / "abcdef0123456789.npz", route_xy=xy, route_s=xy[:, 0].copy(),
             scene_name="scene_test", verdict="ok", src_tokens=np.array(["t"]),
             map_location="us-test")
    return str(tmp_path)


def _frame(x=50.0, y=0.0, yaw=0.0, token="abcdef0123456789-000"):
    return {"scene_name": "scene_test", "token": token,
            "ego2global_translation": [x, y, 0.0], "ego2global_rotation": _quat(yaw)}


def _build(sidecar_dir, frame, env, monkeypatch):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    b = StaticRouteCenterline(_Cfg(), sidecar_dir=sidecar_dir, strict=True)
    rc, mask = b.build(frame)
    return b, rc[0].numpy(), mask[0].numpy()


def test_explicit_nav_pose_still_wins(sidecar_dir, monkeypatch):
    fr = _frame()
    _, clean, m0 = _build(sidecar_dir, fr, {}, monkeypatch)
    b = StaticRouteCenterline(_Cfg(), sidecar_dir=sidecar_dir, strict=True)
    rc, mask = b.build(fr, nav_pose=(50.0, 0.0, 0.0))
    both = m0 & mask[0].numpy()
    assert np.allclose(rc[0].numpy()[both], clean[both], atol=1e-6)


def test_a_route_file_is_used_directly(tmp_path, monkeypatch):
    """The file is the scene's route whatever tokens it was built from (hand-edited routes of
    odyssey_scene039/080/093 come from a neighbouring window of the same drive)."""
    xy = np.stack([np.arange(0.0, 400.0, 1.0), np.zeros(400)], axis=1)
    path = tmp_path / "route.npz"
    np.savez(path, route_xy=xy, route_s=xy[:, 0].copy(), scene_name="another label", verdict="ok",
             src_tokens=np.array(["t"]), tokens=np.array(["abcdef0123456789"]), map_location="us-test")
    for token in ("abcdef0123456789", "0000000000000000"):
        frame = dict(_frame(), scene_name="odyssey_scene001", route_sidecar_token=token)
        b, rc, mask = _build(str(path), frame, {}, monkeypatch)
        assert mask.sum() > 0 and b.map_location(frame) == "us-test"
