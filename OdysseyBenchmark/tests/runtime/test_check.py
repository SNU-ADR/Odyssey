"""`odyssey_runtime check` helpers, the synthetic probe input, and argument validation."""
import argparse
import os

import numpy as np
import pytest

from odyssey_runtime.agent_config import resolve_interpreter
from odyssey_runtime.check import default_gpu, hint_for
from odyssey_runtime.launch import gpus_outside_visible, gpu_visibility_error, max_steps_arg
from odyssey_runtime.probe import frame, route_polyline, write_route, STRAIGHT
from odyssey_runtime.profile import ModelProfile
from tests.runtime.test_profile import spec


@pytest.mark.parametrize("text, expect", [
    ("ValueError: native planner PLAN_DT differs from profile: the adapter emits poses every 0.5 s, "
     "output.plan_dt declares 0.1 s", "output.plan_dt"),
    ("ValueError: CAM_F0: native sensor history [3] differs from profile [2]", "camera_times_s does not match"),
    ("CAM_F0: the profile declares history indices [1], but the model requests [3]; it would never read indices [1]",
     "history_times_s as long as"),
    ("status: declared in feature_shapes, but the model has no such feature", "not one of the model's features"),
    ("RuntimeError: No CUDA GPUs are available", "--device cpu"),
])
def test_hints_match_the_error_they_explain(text, expect):
    assert expect in hint_for(text)


def test_check_defaults_to_the_first_visible_gpu():
    assert default_gpu({}) == "0"
    assert default_gpu({"CUDA_VISIBLE_DEVICES": ""}) == "0"
    assert default_gpu({"CUDA_VISIBLE_DEVICES": "2,3"}) == "2"


def test_gpus_outside_an_exported_cuda_visible_devices_are_refused():
    assert gpus_outside_visible(["0", "1"], {}) == []
    assert gpus_outside_visible(["0"], {"CUDA_VISIBLE_DEVICES": ""}) == []
    assert gpus_outside_visible(["0", "2"], {"CUDA_VISIBLE_DEVICES": "2,3"}) == ["0"]
    assert gpus_outside_visible(["2", "3.2"], {"CUDA_VISIBLE_DEVICES": "2,3"}) == []
    assert gpus_outside_visible(["0"], {"CUDA_VISIBLE_DEVICES": "GPU-1234"}) == []      # UUIDs: not checked
    assert "use 2,3" in gpu_visibility_error(["0"], {"CUDA_VISIBLE_DEVICES": "2,3"})


@pytest.mark.parametrize("text, ok", [("40", True), ("20", True), ("42", False), ("15", False)])
def test_max_steps_is_validated_when_parsing(text, ok):
    if ok:
        assert max_steps_arg(text) == int(text)
    else:
        with pytest.raises(argparse.ArgumentTypeError, match="multiple of 5"):
            max_steps_arg(text)


def test_model_python_resolves_like_the_other_paths(tmp_path):
    assert resolve_interpreter("envs/m/bin/python", str(tmp_path), {}) == str(tmp_path / "envs/m/bin/python")
    assert resolve_interpreter("/abs/python", str(tmp_path), {}) == "/abs/python"
    assert resolve_interpreter("${PY}", str(tmp_path), {"PY": "/env/python"}) == "/env/python"
    assert resolve_interpreter("sh", str(tmp_path), dict(os.environ)).endswith("/sh")


def test_feature_errors_name_the_actual_shape_and_keys():
    import torch

    p = ModelProfile(dict(spec(), feature_shapes={"camera_feature": [1, 3, 4, 4]}))
    with pytest.raises(ValueError, match=r"\[1, 3, 2, 2\] differs from profile \[1, 3, 4, 4\]"):
        p.validate_features({"camera_feature": torch.zeros(1, 3, 2, 2)})
    with pytest.raises(ValueError, match=r"no such feature \(it has \['image'\]\)"):
        p.validate_features({"image": torch.zeros(1)})


def test_probe_routes_differ_and_load_like_a_scene_route(tmp_path):
    from odyssey_bridge.route_sidecar import StaticRouteCenterline

    straight, left = route_polyline(False), route_polyline(True)
    assert np.allclose(np.diff(straight, axis=0)[:, 1], 0)
    assert left[-1, 1] - left[0, 1] > 200          # ends far to the left of the ego
    routes = []
    for turn in (False, True):
        path = write_route(tmp_path / f"route_{turn}.npz", turn=turn)
        route, mask = StaticRouteCenterline(None, sidecar_dir=path, strict=True).build(frame(STRAIGHT, {"CAM_F0": b""}))
        route, mask = np.asarray(route), np.asarray(mask)
        assert route.shape == (1, 120, 5) and mask.shape == (1, 120) and mask.sum() > 100
        routes.append(route)
    assert np.abs(routes[0][0, :, 1]).max() < 1e-6                      # straight ahead
    assert routes[1][0, 60, 1] > 10                                     # 60 m on, well to the left (+y)


def test_probe_frame_carries_the_rig_and_no_annotations():
    f = frame(STRAIGHT, {"CAM_F0": b"", "CAM_L0": b""})
    assert set(f["cams"]) == {"CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0"}
    assert f["cams"]["CAM_F0"]["data_path"] and not f["cams"]["CAM_B0"]["data_path"]
    assert not any(k.startswith("gt_") or k == "anns" for k in f)
    assert list(f["driving_command"]) == STRAIGHT and f["frame_idx"] == 0
