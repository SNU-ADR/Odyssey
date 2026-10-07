from collections import deque
from types import SimpleNamespace

import numpy as np

import odyssey.envs.base_env as base_env_module
from odyssey.envs.base_env import BaseEnv


def _audit_env(monkeypatch):
    cadence = SimpleNamespace(sim_dt=0.1, score_stride_steps=5)
    scene = {"cadence": cadence}
    ego = SimpleNamespace(current_position=np.zeros(2), current_velocity=np.zeros(2))
    engine = SimpleNamespace(
        episode_step=0,
        global_config={
            "rollout_dt": 0.1,
            "hopeless_stall_early_stop_enabled": True,
            "hopeless_stall_min_elapsed_s": 100.0,
            "hopeless_stall_window_s": 60.0,
            "hopeless_stall_max_travel_m": 1.0,
            "hopeless_stall_max_speed_mps": 0.2,
        },
        managers={
            "scenario_manager": SimpleNamespace(current_scene=scene),
            "agent_manager": SimpleNamespace(ego_agent=ego),
        },
    )
    monkeypatch.setattr(base_env_module, "get_engine", lambda: engine)
    env = BaseEnv.__new__(BaseEnv)
    env._stall_scene = None
    env._stall_last_step = None
    env._stall_samples = deque()
    env._stall_cumulative_distance_m = 0.0
    return env, engine, ego


def test_hopeless_stall_requires_full_minimum_and_window(monkeypatch):
    env, engine, _ego = _audit_env(monkeypatch)
    result = None
    for step in range(1001):
        engine.episode_step = step
        result = env._hopeless_stall_state()
        if step < 1000:
            assert result is None
    assert result["term_reason"] == "hopeless_stall"
    assert result["stall_window_s"] == 60.0
    assert result["stall_travel_m"] == 0.0


def test_hopeless_stall_does_not_end_a_slowly_progressing_ego(monkeypatch):
    env, engine, ego = _audit_env(monkeypatch)
    for step in range(1001):
        engine.episode_step = step
        ego.current_position = np.array([step * 0.002, 0.0])  # 1.2 m per 60 s
        assert env._hopeless_stall_state() is None
