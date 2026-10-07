import os
from pathlib import Path
import numpy as np
import pytest
from odyssey_runtime.session import RuntimeSession, get_session, close_session
from odyssey_runtime.profile import ModelProfile
from tests.runtime.test_profile import spec


def test_default_path_does_not_create_runtime(monkeypatch):
    monkeypatch.delenv("ODYSSEY_RUNTIME_PROFILE", raising=False)
    assert get_session() is None


def test_images_must_cover_profile_and_have_same_step(tmp_path):
    s = RuntimeSession(ModelProfile(spec()), tmp_path)
    try:
        s.encode_camera(0, "CAM_F0", np.zeros((8, 8, 3), np.uint8), tmp_path / "f.jpg")
        with pytest.raises(ValueError, match="camera"):
            s.observe({"frame_idx": 0}, 0, 0.1)
        with pytest.raises(ValueError, match="step"):
            s.plan(0)
    finally:
        s.close()


def test_runtime_rejects_wrong_world_cadence_before_model_start(tmp_path):
    s = RuntimeSession(ModelProfile(spec()), tmp_path)
    try:
        with pytest.raises(ValueError, match="cadence"):
            s.observe({}, 0, 0.5)
    finally:
        s.close()


def test_profile_validates_final_feature_shapes():
    d = spec()
    d["feature_shapes"] = {"camera_feature": [1, 3, 256, 1024]}
    p = ModelProfile(d)
    p.validate_features({"camera_feature": np.zeros((1, 3, 256, 1024), np.float32)})
    with pytest.raises(ValueError, match="camera_feature"):
        p.validate_features({"camera_feature": np.zeros((1, 3, 128, 512), np.float32)})


def test_stateful_profile_requires_call_on_reset_without_skipping():
    from odyssey_runtime.planner import reset_episode

    class Model:
        def __init__(self):
            self.n = 9

        def reset_cache(self):
            self.n = 0

    d = spec()
    d.update(stateful=True, reset_method="reset_cache")
    m = Model()
    reset_episode(m, ModelProfile(d))
    assert m.n == 0
    with pytest.raises(ValueError):
        reset_episode(object(), ModelProfile(d))


def test_session_delivers_current_plan_and_rejects_stale_reply(tmp_path):
    class Worker:
        closed = False
        stale = False

        def request(self, message, payload):
            assert payload.startswith(b"\xff\xd8")
            return dict(
                step=message["step"] - int(self.stale),
                episode=message["episode"],
                plan=np.ones((40, 3), np.float32),
            )

        def close(self):
            self.closed = True

    s = RuntimeSession(ModelProfile(spec()), tmp_path / "runtime")
    s.worker = Worker()
    try:
        for step in range(2):
            for camera in s.profile.cameras:
                s.encode_camera(
                    step,
                    camera,
                    np.zeros((8, 8, 3), np.uint8),
                    tmp_path / f"{step}_{camera}.jpg",
                )
            frame = dict(frame_idx=step, log_token="scene")
            if step == 0:
                s.observe(frame, step, 0.1)
                assert s.plan(step).dtype == np.float32
                np.testing.assert_array_equal(s.plan(step), np.ones((40, 3)))
                s.worker.stale = True
            else:
                with pytest.raises(RuntimeError, match="stale"):
                    s.observe(frame, step, 0.1)
                with pytest.raises(ValueError, match="no planner response"):
                    s.plan(step)
    finally:
        s.close()
    assert s.worker.closed
    assert (tmp_path / "plan_traj/scene_1.npy").exists()


@pytest.mark.parametrize("camera_files", [True, False])
def test_camera_files_exist_at_request_time_without_record(tmp_path, camera_files):
    """A CAMERA_FILES planner (ReCogDrive) opens the step's JPEGs: they are complete when the
    request arrives, also without --record; other planners get no files."""
    d = spec()
    d.update(record_images=False, record_legacy_ipc=False)
    s = RuntimeSession(ModelProfile(d), tmp_path / "runtime")
    paths = {c: tmp_path / "sensor_blobs" / f"{c}.jpg" for c in s.profile.cameras}

    class Worker:
        ready = {"camera_files": camera_files}

        def request(self, message, payload):
            if camera_files:
                assert b"".join(paths[c].read_bytes() for c in s.profile.cameras) == payload
            else:
                assert not any(p.exists() for p in paths.values())
            return dict(step=message["step"], episode=message["episode"],
                        plan=np.ones((40, 3), np.float32))

        def close(self):
            pass

    s.worker = Worker()
    try:
        for camera in s.profile.cameras:
            s.encode_camera(0, camera, np.zeros((8, 8, 3), np.uint8), paths[camera])
        s.observe(dict(frame_idx=0, log_token="scene"), 0, 0.1)
        assert s.plan(0).shape == (40, 3)
    finally:
        s.close()


def test_final_recording_error_closes_worker(tmp_path):
    class Worker:
        closed = False

        def close(self):
            self.closed = True

    s = RuntimeSession(ModelProfile(spec()), tmp_path / "runtime")
    s.worker = Worker()
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    s.recorder.submit_bytes(blocker / "frame.jpg", b"data")
    with pytest.raises(FileExistsError):
        s.close()
    assert s.worker.closed and s.timings.closed


def test_profile_model_overrides_reach_native_worker(tmp_path, monkeypatch):
    import odyssey_runtime.session as session

    captured = {}

    class Worker:
        def __init__(self, python, factory, config, *args, **kwargs):
            captured.update(config)

        def close(self):
            pass

    monkeypatch.setattr(session, "SharedWorker", Worker)
    for key, value in dict(
        ODYSSEY_ROOT=str(tmp_path),
        ODYSSEY_PLANNER="ltf_sdroute_kv",
        ODYSSEY_PLANNER_CFG="model_config",
        ODYSSEY_PLANNER_CKPT="model.ckpt",
        ODYSSEY_PLANNER_REPO=str(tmp_path),
        ODYSSEY_PLANNER_PY="python",
    ).items():
        monkeypatch.setenv(key, value)
    d = spec()
    d["model_overrides"] = ["config.example=1"]
    s = RuntimeSession(ModelProfile(d), tmp_path / "runtime")
    try:
        s._start_worker()
        assert captured["overrides"] == ["config.example=1"]
    finally:
        s.close()


def test_failed_report_and_final_recording_error_prevent_success_flag(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from odyssey.runner.executor import run_simulation
    import odyssey_runtime.session as session

    monkeypatch.setenv("ODYSSEY_RUNTIME_PROFILE", "enabled")
    env = SimpleNamespace(
        config=SimpleNamespace(output_dir=tmp_path),
        run=lambda: [SimpleNamespace(succeeded=False)],
    )
    monkeypatch.setattr(session, "close_session", lambda: None)
    run_simulation(env)
    assert not (tmp_path / "simulation_completed.flag").exists()
    env.run = lambda: [SimpleNamespace(succeeded=True)]

    def fail():
        raise OSError("recorder failed")

    monkeypatch.setattr(session, "close_session", fail)
    with pytest.raises(OSError, match="recorder failed"):
        run_simulation(env)
    assert not (tmp_path / "simulation_completed.flag").exists()
