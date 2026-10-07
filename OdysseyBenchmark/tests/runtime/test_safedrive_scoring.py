"""SafeDrive's shipped configs rank plans by the retuned test-time scoring, set on the built model."""
import pytest

from odyssey_bridge.planners.safedrive import (
    TUNED_BASELINE, TUNED_SDROUTE, SafeDriveSDRouteTunedPlanner, SafeDriveTunedPlanner,
    apply_test_scoring)


class Model:
    """The attributes SafeDrive_Model.__init__ hardcodes: imitation-only selection."""

    def __init__(self):
        self.scoring_test = False
        self.pair_NC_scoring = False
        self.twdac_scoring = False
        self.TwDAC_bevseg_pred = False
        self.TwDAC_bbox_margin = (1.0, 1.0)
        self.imi_test_weight = 1.0


class Agent:
    def __init__(self, model):
        self._safedrive_model = model


@pytest.mark.parametrize("weights", [TUNED_BASELINE, TUNED_SDROUTE])
def test_every_weight_lands_on_the_model_and_turns_scoring_on(weights):
    model = Model()
    apply_test_scoring(Agent(model), weights, "t")
    assert all(getattr(model, k) == v for k, v in weights.items())
    assert model.scoring_test is True and list(weights)[-1] == "scoring_test"
    assert len(weights) == 17 and "no_EP_TC_sum_scoring" not in weights


def test_a_model_without_the_scoring_switch_is_refused():
    with pytest.raises(SystemExit, match="imitation argmax"):
        apply_test_scoring(object(), TUNED_SDROUTE, "t")
    with pytest.raises(SystemExit, match="imitation argmax"):
        apply_test_scoring(Agent(object()), TUNED_SDROUTE, "t")


def test_each_arm_carries_its_own_tuned_weights():
    assert SafeDriveTunedPlanner.TEST_SCORING is TUNED_BASELINE
    assert SafeDriveSDRouteTunedPlanner.TEST_SCORING is TUNED_SDROUTE
    assert TUNED_BASELINE["pair_NC_scoring"] is False and TUNED_SDROUTE["pair_NC_scoring"] is True
    assert TUNED_SDROUTE["DAC_test_weight"] == 0.0
