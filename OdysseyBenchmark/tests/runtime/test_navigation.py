"""Navigation inputs are declared in the profile and checked against, or fed to, the model."""
from types import SimpleNamespace

import pytest
import torch

from odyssey_runtime.profile import ModelProfile
from odyssey_bridge.planners.base import NavsimPlanner
from tests.runtime.test_profile import spec


def profile(**navigation):
    data = spec()
    data["navigation"] = navigation
    return ModelProfile(data)


@pytest.mark.parametrize("navigation", [None, {}, {"driving_command": "no", "sd_route": "none"},
                                        {"driving_command": True, "sd_route": "map"},
                                        {"driving_command": True, "sd_route": "none", "hd_map": True}])
def test_profile_requires_a_valid_navigation_declaration(navigation):
    data = spec()
    if navigation is None:
        del data["navigation"]
    else:
        data["navigation"] = navigation
    with pytest.raises(ValueError, match="navigation"):
        ModelProfile(data)


def fake_planner(sd_route_input="none", reads_command=True, use_sdroute=None):
    config = SimpleNamespace() if use_sdroute is None else SimpleNamespace(use_sdroute=use_sdroute)
    return SimpleNamespace(uses_driving_command=reads_command, sd_route_input=sd_route_input,
                           agent=SimpleNamespace(_config=config))


def test_declaration_must_match_the_model():
    p = profile(driving_command=False, sd_route="targets")
    p.validate_navigation(fake_planner("targets", reads_command=False, use_sdroute=True))
    with pytest.raises(ValueError, match="driving_command"):
        p.validate_navigation(fake_planner("targets", reads_command=True))
    with pytest.raises(ValueError, match="SD route"):
        p.validate_navigation(fake_planner("features", reads_command=False))
    with pytest.raises(ValueError, match="use_sdroute"):
        p.validate_navigation(fake_planner("targets", reads_command=False, use_sdroute=False))


class Builder:
    def compute_features(self, agent_input):
        return {"camera_feature": torch.zeros(3, 4)}


class Route:
    def build(self, frame):
        assert frame["frame_idx"] == 7                    # the newest frame
        return torch.ones(1, 120, 5), torch.ones(1, 120, dtype=torch.bool)


class Agent:
    def forward(self, features, targets=None):
        self.seen = (features, targets)
        return {"trajectory": torch.zeros(8, 3)}


def generic(where, adapter_route=None):
    cls = type("Generic", (NavsimPlanner,), {"SD_ROUTE": adapter_route})
    planner = cls("cfg", "", ".", ".", device="cpu",
                  navigation={"driving_command": False, "sd_route": where})
    planner._feature_builders, planner._route_builder, planner.agent = [Builder()], Route(), Agent()
    return planner


def test_base_feeds_the_declared_route_into_features():
    planner = generic("features")
    features = planner._build_features(None, [{"frame_idx": 6}, {"frame_idx": 7}])
    assert features["route_centerline"].shape == (120, 5)       # infer() adds the batch dim
    assert features["route_centerline_mask"].shape == (120,)
    planner._forward(features)
    assert planner.agent.seen[1] is None
    assert planner.sd_route_input == "features"


def test_base_feeds_the_declared_route_into_targets():
    planner = generic("targets")
    features = planner._build_features(None, [{"frame_idx": 7}])
    assert "route_centerline" not in features
    planner._forward(features)
    assert planner.agent.seen[1]["route_centerline"].shape == (1, 120, 5)
    assert planner.sd_route_input == "targets"


def test_adapter_that_feeds_its_own_route_is_not_doubled():
    planner = generic("targets", adapter_route="features")
    assert planner.sd_route_input == "features" and not planner._generic_route
