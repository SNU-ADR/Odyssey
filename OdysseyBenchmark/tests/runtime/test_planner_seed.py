import random
import numpy as np
import pytest
import torch
from odyssey_runtime import planner
from odyssey_runtime.profile import ModelProfile
from tests.runtime.test_profile import spec

def test_profile_accepts_explicit_seed_and_rejects_invalid_values():
    profile = spec()
    profile["seed"] = 2026
    assert ModelProfile(profile).data["seed"] == 2026
    for value in (-1, 2**32, True, 1.5, "2026"):
        profile["seed"] = value
        with pytest.raises(ValueError, match="seed"):
            ModelProfile(profile)

def test_seed_controls_python_numpy_and_torch():
    assert hasattr(planner, "seed_planner"), "planner seed initialization missing"
    def draw():
        return random.random(), np.random.random(), torch.rand(4)
    planner.seed_planner(2026)
    a = draw()
    planner.seed_planner(2026)
    b = draw()
    assert a[:2] == b[:2]
    assert torch.equal(a[2], b[2])

def test_absent_seed_does_not_reset_state():
    assert hasattr(planner, "seed_planner"), "planner seed initialization missing"
    random.seed(44)
    expected = [random.random(), random.random()]
    random.seed(44)
    assert random.random() == expected[0]
    planner.seed_planner(None)
    assert random.random() == expected[1]
