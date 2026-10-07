import copy
from types import SimpleNamespace
import pytest
from odyssey_runtime.profile import ModelProfile


def spec():
    return dict(
        version=1,
        planner_id="ltf_sdroute_kv",
        sim_dt=0.1,
        history_times_s=[-1.5, -1.0, -0.5, 0.0],
        camera_times_s={"CAM_F0": [0.0], "CAM_L0": [0.0], "CAM_R0": [0.0]},
        planning_interval_s=0.1,
        render_resolution="native",
        preprocessing="model_native",
        stateful=False,
        reset_method=None,
        output=dict(plan_dt=0.5, shape=[8, 3], coordinates="ego_future"),
        navigation=dict(driving_command=False, sd_route="targets"),
    )


def test_history_is_sampled_by_actual_step_and_padding():
    p = ModelProfile(spec())
    frames = [dict(frame_idx=i) for i in range(21)]
    assert [x["frame_idx"] for x in p.sample_history(frames, 20)] == [5, 10, 15, 20]
    assert [x["frame_idx"] for x in p.sample_history(frames[:3], 2)] == [0, 0, 0, 2]
    assert p.history_capacity == 16


def test_missing_observation_is_not_silently_substituted():
    p = ModelProfile(spec())
    with pytest.raises(ValueError, match="missing"):
        p.sample_history([dict(frame_idx=i) for i in range(21) if i != 10], 20)


@pytest.mark.parametrize(
    "change",
    [
        {"history_times_s": [-0.15, 0]},
        {"planning_interval_s": 0.2},
        {"history_times_s": [0, -0.5]},
        {"render_resolution": [1024, 256]},
        {"stateful": True},
        {"sim_dt": 0},
        {"camera_times_s": {"CAM_F0": [-0.2]}},
    ],
)
def test_unsupported_or_ambiguous_contract_fails(change):
    d = spec()
    d.update(change)
    with pytest.raises(ValueError):
        ModelProfile(d)


def test_native_sensor_config_must_match_each_camera_time():
    p = ModelProfile(spec())
    sensor = SimpleNamespace(
        cam_f0=[3],
        cam_l0=[3],
        cam_r0=[3],
        cam_l1=False,
        cam_l2=False,
        cam_r1=False,
        cam_r2=False,
        cam_b0=False,
        lidar_pc=False,
    )
    p.validate_sensor_config(sensor)
    sensor.cam_f0 = [1, 2, 3]
    with pytest.raises(ValueError, match="CAM_F0"):
        p.validate_sensor_config(sensor)


def test_camera_histories_can_differ_and_stateful_reset_is_explicit():
    d = spec()
    d.update(stateful=True, reset_method="reset_episode")
    d["camera_times_s"]["CAM_F0"] = [-1.5, -0.5, 0]
    p = ModelProfile(d)
    assert p.camera_indices("CAM_F0") == [0, 2, 3]
    assert p.camera_indices("CAM_L0") == [3]


@pytest.mark.parametrize(
    "extra", [{"image_size": [128, 256]}, {"model_overrides": "image_size=256"}]
)
def test_profile_rejects_unknown_or_malformed_model_settings(extra):
    data = spec()
    data.update(extra)
    with pytest.raises(ValueError):
        ModelProfile(data)
