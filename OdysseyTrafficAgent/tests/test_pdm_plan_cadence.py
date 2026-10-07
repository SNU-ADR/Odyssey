from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from odyssey.components.agents.policy.pdm_policy import PDMPolicy


class _Converter:
    def __init__(self):
        self.ego_state = object()
        self.observation = object()
        self.output_trajectory = object()

    def convert_to_current_ego_state(self, _step):
        return self.ego_state

    def convert_to_detections_tracks_from_agent_input(self, _step):
        return self.observation

    def convert_to_trajectory(self, _trajectory, wp_dt):
        assert wp_dt == 0.1
        return self.output_trajectory


def _policy(monkeypatch, step, trajectory):
    monkeypatch.setattr(
        "odyssey.components.agents.policy.base_policy.get_engine",
        lambda: SimpleNamespace(episode_step=step),
    )
    policy = object.__new__(PDMPolicy)
    policy.agent = SimpleNamespace(trajectory=trajectory)
    policy.current_scene = object()
    policy.converter = _Converter()
    policy.ego_states_list = []
    policy.observations_list = []
    policy._plan_stride = 2
    policy._gt_warmup_steps = 0
    policy._future_sampling = SimpleNamespace(interval_length=0.1)
    policy._logger = Mock()
    return policy


def test_pdm_skipped_tick_reuses_plan_and_keeps_history(monkeypatch):
    trajectory = object()
    policy = _policy(monkeypatch, step=1, trajectory=trajectory)
    policy._get_planner_inputs = Mock(side_effect=AssertionError("must not replan"))

    assert policy.act() is trajectory
    assert policy.ego_states_list == [policy.converter.ego_state]
    assert policy.observations_list == [policy.converter.observation]
    policy._get_planner_inputs.assert_not_called()


def test_pdm_plans_on_stride_boundary(monkeypatch):
    policy = _policy(monkeypatch, step=2, trajectory=object())
    map_api = object()
    initialization = {
        "map_api": map_api,
        "route_roadblock_dict_ids": ["roadblock"],
    }
    policy._get_planner_inputs = Mock(return_value=(object(), initialization))
    policy._pdm_initialized_for = (id(map_api), ("roadblock",))
    policy._pdm_closed = Mock()
    policy._pdm_closed.compute_planner_trajectory.return_value = object()

    assert policy.act() is policy.converter.output_trajectory
    policy._pdm_closed.compute_planner_trajectory.assert_called_once()


def test_pdm_replays_gt_while_collecting_reactive_warmup_history(monkeypatch):
    policy = _policy(monkeypatch, step=2, trajectory=None)
    policy._gt_warmup_steps = 15
    policy.current_scene = {"cadence": SimpleNamespace(sim_dt=0.1)}
    positions = np.column_stack((np.arange(24, dtype=float), np.zeros(24)))
    policy.agent.object_track = {
        "position": positions,
        "velocity": np.ones((24, 2), dtype=float),
        "heading": np.zeros(24, dtype=float),
        "angular_velocity": np.zeros(24, dtype=float),
    }
    policy._get_planner_inputs = Mock(side_effect=AssertionError("must not run PDM in warm-up"))

    trajectory = policy.act()

    assert np.array_equal(trajectory.waypoints, positions[2:11])
    assert trajectory.wp_dt == 0.1
    assert policy.ego_states_list == [policy.converter.ego_state]
    assert policy.observations_list == [policy.converter.observation]
    policy._get_planner_inputs.assert_not_called()
