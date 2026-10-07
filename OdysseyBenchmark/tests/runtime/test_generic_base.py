"""The generic adapter narrows the model's sensor request to the profile and batches nested features."""
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Union

import pytest
import torch

from odyssey_bridge.planners.base import NavsimPlanner, narrow_sensor_config


@dataclass
class Sensors:
    cam_f0: Union[bool, List[int]]
    cam_l0: Union[bool, List[int]]
    cam_l1: Union[bool, List[int]]
    cam_l2: Union[bool, List[int]]
    cam_r0: Union[bool, List[int]]
    cam_r1: Union[bool, List[int]]
    cam_r2: Union[bool, List[int]]
    cam_b0: Union[bool, List[int]]
    lidar_pc: Union[bool, List[int]]


def all_sensors(value):
    return Sensors(*([value] * 9))


THREE = {"cam_f0": [3], "cam_l0": [3], "cam_r0": [3]}


def test_native_request_wider_than_the_profile_is_narrowed():
    narrowed, dropped = narrow_sensor_config(all_sensors([3]), THREE, 4)
    assert (narrowed.cam_f0, narrowed.cam_l0, narrowed.cam_r0) == ([3], [3], [3])
    assert all(getattr(narrowed, c) is False for c in ("cam_l1", "cam_l2", "cam_r1", "cam_r2", "cam_b0"))
    assert narrowed.lidar_pc is False
    assert dropped == ["cam_l1[3]", "cam_r1[3]", "cam_l2[3]", "cam_r2[3]", "cam_b0[3]", "lidar_pc[3]"]


def test_matching_request_is_returned_untouched():
    native = Sensors(cam_f0=[3], cam_l0=[3], cam_r0=[3], cam_l1=[], cam_l2=[], cam_r1=[], cam_r2=[],
                     cam_b0=[3], lidar_pc=[])
    narrowed, dropped = narrow_sensor_config(native, dict(THREE, cam_b0=[3]), 4)
    assert narrowed is native and dropped == []


def test_true_means_every_history_frame():
    native = Sensors(cam_f0=True, cam_l0=False, cam_r0=False, cam_l1=False, cam_l2=False, cam_r1=False,
                     cam_r2=False, cam_b0=False, lidar_pc=False)
    narrowed, dropped = narrow_sensor_config(native, {"cam_f0": [0, 1, 2, 3]}, 4)
    assert narrowed is native and dropped == []
    narrowed, dropped = narrow_sensor_config(native, {"cam_f0": [3]}, 4)
    assert narrowed.cam_f0 == [3] and dropped == ["cam_f0[0, 1, 2, 3]"]


def test_profile_may_not_declare_what_the_model_does_not_request():
    with pytest.raises(ValueError, match="CAM_B0"):
        narrow_sensor_config(all_sensors([3]), dict(THREE, cam_b0=[2, 3]), 4)
    with pytest.raises(ValueError, match="CAM_F0"):
        narrow_sensor_config(all_sensors([3]), {"cam_f0": [0, 3]}, 4)


def test_plain_objects_are_copied_not_mutated():
    native = SimpleNamespace(cam_f0=[3], cam_l0=[3], cam_r0=[3], cam_l1=[3], cam_l2=[], cam_r1=[],
                             cam_r2=[], cam_b0=[], lidar_pc=True)
    narrowed, dropped = narrow_sensor_config(native, THREE, 4)
    assert narrowed is not native and native.cam_l1 == [3] and native.lidar_pc is True
    assert narrowed.cam_l1 is False and narrowed.lidar_pc is False
    assert dropped == ["cam_l1[3]", "lidar_pc[0, 1, 2, 3]"]


def planner():
    return NavsimPlanner("cfg", "", ".", ".", device="cpu",
                         navigation={"driving_command": True, "sd_route": "none"})


def test_batching_adds_one_batch_dim_to_tensors_at_any_nesting():
    p = planner()
    out = p._batch({"a": 1})                       # dicts are left alone
    assert out == {"a": 1}
    assert p._batch(torch.zeros(3, 4)).shape == (1, 3, 4)
    nested = p._batch([[torch.zeros(2, 4, 4), torch.ones(2)], [torch.zeros(2, 4, 4), 7]])
    assert nested[0][0].shape == (1, 2, 4, 4) and nested[0][1].shape == (1, 2)
    assert nested[1][1] == 7 and isinstance(nested, list) and isinstance(nested[0], list)
    assert isinstance(p._batch((torch.zeros(1),)), tuple)


def test_cameras_keyword_is_optional_and_lower_cased():
    p = NavsimPlanner("cfg", "", ".", ".", device="cpu", cameras={"CAM_F0": [3]}, history_frames=4)
    assert p.cameras == {"cam_f0": [3]} and p.history_frames == 4
    assert planner().cameras is None
