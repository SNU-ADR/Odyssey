import pytest

from odyssey.components.agents.policy.build_policy import build_policy
from odyssey.components.agents.policy.nuplan_idm_policy import NuPlanIDMPolicy


def test_reactive_background_uses_nuplan_idm_policy():
    policy = build_policy(
        "background-vehicle",
        {"agent_policy": "nuplan_idm_policy"},
    )

    assert policy is NuPlanIDMPolicy


def test_nuplan_idm_cannot_be_selected_as_ego_policy():
    with pytest.raises(ValueError, match="background traffic only"):
        build_policy("ego", {"ego_policy": "nuplan_idm_policy"})


@pytest.mark.parametrize("object_id, config_key", [
    ("background-vehicle", "agent_policy"),
    ("ego", "ego_policy"),
])
def test_removed_odyssey_idm_fails_with_migration_message(object_id, config_key):
    with pytest.raises(ValueError, match="agent_policy=nuplan_idm_policy"):
        build_policy(object_id, {config_key: "idm_policy"})
