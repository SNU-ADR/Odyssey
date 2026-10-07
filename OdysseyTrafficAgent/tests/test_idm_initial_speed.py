"""Unit tests for source-row longitudinal IDM velocity seeding."""
from types import SimpleNamespace

import numpy as np
import pytest

from odyssey.components.agents.policy.nuplan_idm_policy import NuPlanIDMBatch


def _batch(positions, velocities, valid, row, heading=0.0):
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch.scene = {"object_track": {"car": {"state": {
        "position": np.asarray(positions, dtype=float),
        "velocity": np.asarray(velocities, dtype=float),
        "valid": np.asarray(valid, dtype=bool),
    }}}}
    batch.converter = SimpleNamespace(object_source_row=lambda _token, _step: row)
    batch._dt = 0.1
    batch._initial_speed_sanity_max_mps = 40.0
    batch._idm_initial_speed_records = {}
    agent = SimpleNamespace(
        _state=SimpleNamespace(velocity=3.0),
        _requires_state_update=False,
        to_se2=lambda: SimpleNamespace(heading=heading),
    )
    return batch, agent


def test_zero_raw_velocity_recovers_causal_pose_speed():
    batch, agent = _batch(
        [[0, 0, 0], [0.5, 0, 0]], [[0, 0], [0, 0]], [1, 1], row=1
    )
    record = batch._recover_idm_initial_speed("car", agent, simulation_step=99)
    assert record["source_kind"] == "backward_pose_difference"
    assert record["derived_mps"] == pytest.approx(5.0)
    assert record["applied_mps"] == pytest.approx(5.0)
    assert agent._state.velocity == pytest.approx(5.0)


def test_normal_raw_velocity_has_priority_over_pose_difference():
    batch, agent = _batch(
        [[0, 0, 0], [0.5, 0, 0]], [[0, 0], [7, 0]], [1, 1], row=1
    )
    record = batch._recover_idm_initial_speed("car", agent, simulation_step=1)
    assert record["source_kind"] == "raw_velocity"
    assert record["raw_mps"] == pytest.approx(7.0)
    assert np.isnan(record["derived_mps"])
    assert agent._state.velocity == pytest.approx(7.0)


def test_first_valid_row_uses_next_pose_only_when_no_prior_pose_exists():
    batch, agent = _batch(
        [[0, 0, 0], [0.4, 0, 0]], [[0, 0], [0, 0]], [1, 1], row=0
    )
    record = batch._recover_idm_initial_speed("car", agent, simulation_step=0)
    assert record["source_kind"] == "forward_pose_difference"
    assert record["derived_mps"] == pytest.approx(4.0)


@pytest.mark.parametrize("positions", [
    [[0, 0, 0], [0, 0.5, 0]],  # lateral jitter only
    [[0, 0, 0], [-0.5, 0, 0]],  # backwards relative to heading
])
def test_nonforward_motion_is_not_seeded(positions):
    batch, agent = _batch(positions, [[0, 0], [0, 0]], [1, 1], row=1)
    record = batch._recover_idm_initial_speed("car", agent, simulation_step=1)
    assert record["applied_mps"] == 0.0
    assert record["clamp_reason"] == "nonforward_motion"
    assert agent._state.velocity == 0.0


def test_source_speed_is_always_seeded_without_a_mode_flag():
    batch, agent = _batch(
        [[0, 0, 0], [0.5, 0, 0]], [[0, 0], [0, 0]], [1, 1], row=1,
    )
    record = batch._recover_idm_initial_speed("car", agent, simulation_step=1)
    assert record["derived_mps"] == pytest.approx(5.0)
    assert record["applied_mps"] == pytest.approx(5.0)
    assert agent._state.velocity == pytest.approx(5.0)
