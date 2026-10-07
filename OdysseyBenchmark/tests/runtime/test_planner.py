import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import cv2
import numpy as np
from PIL import Image
import pytest
from odyssey_runtime.profile import ModelProfile, CAMERAS
from odyssey_runtime.planner import MemoryInputs, encode_jpeg, wire_plan
from tests.runtime.test_profile import spec


@pytest.fixture
def native():
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "OdysseyZoo/models/navsim"))
    from navsim.common.dataclasses import AgentInput, SensorConfig

    return AgentInput, SensorConfig


def frame(step):
    return dict(
        frame_idx=step,
        ego2global_translation=[1.0 + step, 2.0, 0.0],
        ego2global_rotation=[1.0, 0.0, 0.0, 0.0],
        ego_dynamic_state=[2.0, 0.0, 0.1, 0.0],
        driving_command=np.array([0, 1, 0, 0]),
        lidar_path=None,
        cams={
            k: dict(
                data_path=f"{step}/{k}.jpg",
                sensor2lidar_rotation=np.eye(3),
                sensor2lidar_translation=np.zeros(3),
                cam_intrinsic=np.eye(3),
                distortion=np.zeros(5),
            )
            for k in CAMERAS
        },
    )


def test_in_memory_jpeg_is_byte_exact_and_pillow_pixels_match(tmp_path):
    a = np.random.RandomState(4).randint(0, 256, (100, 900, 3), dtype=np.uint8)
    path = tmp_path / "image.jpg"
    assert cv2.imwrite(str(path), a)
    encoded = encode_jpeg(a)
    assert encoded == path.read_bytes()
    assert np.array_equal(
        np.array(Image.open(io.BytesIO(encoded))), np.array(Image.open(path))
    )


def test_native_agent_input_and_feature_builder_are_exact(tmp_path, native):
    AgentInput, SensorConfig = native
    p = ModelProfile(spec())
    memory = MemoryInputs(p)
    frames = [frame(i) for i in range(16)]
    sensor = SensorConfig.build_all_sensors(False)
    sensor.cam_f0 = sensor.cam_l0 = sensor.cam_r0 = [3]
    for f in frames:
        blobs = {}
        for k in p.cameras:
            a = np.full((100, 900, 3), f["frame_idx"] + CAMERAS.index(k), np.uint8)
            encoded = encode_jpeg(a)
            blobs[k] = encoded
            path = tmp_path / f["cams"][k]["data_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(encoded)
        memory.ingest(f["frame_idx"], blobs)
    selected = p.sample_history(frames, 15)
    expected = AgentInput.from_scene_dict_list(selected, tmp_path, 4, sensor)
    actual = memory.build(selected, sensor, tmp_path)
    for a, b in zip(actual.ego_statuses, expected.ego_statuses):
        for key in ("ego_pose", "ego_velocity", "ego_acceleration", "driving_command"):
            assert np.array_equal(getattr(a, key), getattr(b, key))
    from navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder

    builder = TransfuserFeatureBuilder(SimpleNamespace(latent=True))
    x, y = builder.compute_features(actual), builder.compute_features(expected)
    assert x.keys() == y.keys()
    for key in x:
        assert np.array_equal(x[key].numpy(), y[key].numpy())
    assert (
        memory.retained_images == 3
    )  # LTF reads only current images, despite state history


def test_temporal_images_are_retained_until_last_consumer(native, tmp_path):
    _, SensorConfig = native
    d = spec()
    d["camera_times_s"] = {"CAM_F0": [-1.5, -0.5, 0]}
    p = ModelProfile(d)
    m = MemoryInputs(p)
    fs = [frame(i) for i in range(21)]
    for i in range(21):
        m.ingest(i, {"CAM_F0": encode_jpeg(np.full((8, 8, 3), i, np.uint8))})
    sensor = SensorConfig.build_all_sensors(False)
    sensor.cam_f0 = [0, 2, 3]
    a = m.build(p.sample_history(fs, 20), sensor, tmp_path)
    assert [a.cameras[i].cam_f0.image[0, 0, 0] for i in [0, 2, 3]] == [5, 15, 20]
    assert m.retained_images == 16
    m.reset()
    assert m.retained_images == 0
    with pytest.raises(ValueError, match="missing"):
        m.build(p.sample_history(fs, 20), sensor, tmp_path)


def test_float32_wire_is_the_plan_upsampled_to_the_simulator_grid():
    a = np.array([[1, 0.2, 0.1], [2, 0.5, 0.3]], dtype=np.float64)
    out = wire_plan(a, 0.5)
    assert out.dtype == np.float32 and out.shape == (10, 3)
    # 0.1 s rows, future only: the 0.5 s poses land on rows 4 and 9, the rows between are linear.
    assert np.allclose(out[[4, 9]], a)
    assert np.allclose(out[0], [0.2, 0.04, 0.02])


def test_camera_only_native_parser_accepts_absent_lidar_without_mutation(native, tmp_path, monkeypatch):
    AgentInput, SensorConfig = native
    import navsim.common.dataclasses as dc
    from odyssey_bridge import ipc_common
    # Exercise first-use compatibility even when earlier tests patched this module.
    monkeypatch.setattr(dc, "Path", Path)
    monkeypatch.setattr(ipc_common, "_LIDAR_PATCHED", {})
    original = AgentInput.from_scene_dict_list
    def strict_parser(frames, root, count, sensors):
        for i, item in enumerate(frames):
            assert "lidar_pc" not in sensors.get_sensors_at_iteration(i)
            dc.Path(item["lidar_path"])  # DiffusionDrive/SafeDrive do this unconditionally.
        return original(frames, root, count, sensors)
    monkeypatch.setattr(AgentInput, "from_scene_dict_list", strict_parser)
    profile = ModelProfile(spec())
    memory = MemoryInputs(profile)
    scenes = [frame(i) for i in range(4)]
    sensors = SensorConfig.build_all_sensors(False)
    result = memory.build(scenes, sensors, tmp_path)
    assert len(result.ego_statuses) == 4
    assert all(item["lidar_path"] is None for item in scenes)
    assert all(lidar.lidar_pc is None for lidar in result.lidars)


def test_native_camera_without_optional_path_field(native, tmp_path, monkeypatch):
    _, SensorConfig = native
    import navsim.common.dataclasses as dc
    class CameraWithoutPath:
        def __init__(self, image=None, sensor2lidar_rotation=None,
                     sensor2lidar_translation=None, intrinsics=None, distortion=None):
            self.image = image
            self.sensor2lidar_rotation = sensor2lidar_rotation
            self.sensor2lidar_translation = sensor2lidar_translation
            self.intrinsics = intrinsics
            self.distortion = distortion
    monkeypatch.setattr(dc, "Camera", CameraWithoutPath)
    profile = ModelProfile(spec())
    memory = MemoryInputs(profile)
    scenes = [frame(i) for i in range(4)]
    image = np.full((8, 8, 3), 17, np.uint8)
    memory.ingest(3, {name: encode_jpeg(image) for name in profile.cameras})
    sensors = SensorConfig.build_all_sensors(False)
    sensors.cam_f0 = sensors.cam_l0 = sensors.cam_r0 = [3]
    result = memory.build(scenes, sensors, tmp_path)
    for name in profile.cameras:
        camera = getattr(result.cameras[-1], name.lower())
        assert isinstance(camera, CameraWithoutPath)
        assert not hasattr(camera, "camera_path")
        assert np.array_equal(camera.image, image)
        assert np.array_equal(camera.intrinsics, np.eye(3))
