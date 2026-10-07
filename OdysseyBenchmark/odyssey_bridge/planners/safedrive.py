"""SafeDrive with the retuned test-time scoring of the paper's sheets.

Repo   OdysseyZoo/models/SafeDrive
Cfg    safedrive_baseline_agent.yaml / safedrive_sdroute_agent.yaml
Ckpt   ckpts/safedrive_baseline.ckpt / ckpts/safedrive_sdroute.ckpt

SafeDrive drives the candidate that a weighted sum of its safety heads ranks first only when the
model attribute ``scoring_test`` is True (SafeDrive_Model._score_at_test_time); otherwise it drives
the imitation argmax and the NC/DAC/EP/TTC/PwNC/TwDAC/pdm/DDC/TLC/LK heads decide nothing. The
model hardcodes ``scoring_test``, ``pair_NC_scoring``, ``twdac_scoring``, ``TwDAC_bevseg_pred`` and
``TwDAC_bbox_margin`` in ``__init__`` and copies the ``*_test_weight`` values from the config there,
so a Hydra override cannot switch scoring on (SafeDrive docs/train_eval.md). These adapters set all
of them on ``agent._safedrive_model`` after the build -- the same attributes, values and order as
the runs behind the sheets safedrive_scored_tuned and safedrive_sd_scored_tuned.

``no_EP_TC_sum_scoring`` stays at the model's False (``W * log(EP + TTC)``), as in those runs.
The SafeDrive ablations (OdysseyBenchmark/agents/ablations/) are imitation-selected and use the generic adapter.
"""
from __future__ import annotations

from typing import Dict

from .base import NavsimPlanner

#: Test-time weights retuned on navtest open loop (navtest PDMS: baseline 90.80 vs 90.06 with the
#: authors' weights, sdroute 90.58 vs 89.15). Every key is set, so none is left at the model's
#: imitation-only default; ``scoring_test`` comes last because it turns the rest on.
TUNED_BASELINE: Dict[str, object] = {
    "imi_test_weight": 0.4733,
    "NC_test_weight": 42.28,
    "DAC_test_weight": 54.86,
    "EP_test_weight": 0.3794,
    "TTC_test_weight": 2.392,
    "W_test_weight": 24.79,
    "pdm_score_test_weight": 0.3251,
    "DDC_test_weight": 1.87,
    "TLC_test_weight": 0.5554,
    "LK_test_weight": 0.625,
    "PwNC_test_weight": 0.0,          # with pair_NC_scoring False
    "TwDAC_test_weight": 16.09,
    "TwDAC_bbox_margin": (1.4, 1.6),
    "pair_NC_scoring": False,
    "twdac_scoring": True,
    "TwDAC_bevseg_pred": True,
    "scoring_test": True,
}
TUNED_SDROUTE: Dict[str, object] = {
    "imi_test_weight": 5.47,
    "NC_test_weight": 8.0,
    "DAC_test_weight": 0.0,           # the tuning switched the drivable-area term off
    "EP_test_weight": 11.34,
    "TTC_test_weight": 61.1,
    "W_test_weight": 130.8,
    "pdm_score_test_weight": 0.125,
    "DDC_test_weight": 0.8,
    "TLC_test_weight": 67.07,
    "LK_test_weight": 1.0,
    "PwNC_test_weight": 0.8452,
    "TwDAC_test_weight": 32.44,
    "TwDAC_bbox_margin": (1.4, 1.6),
    "pair_NC_scoring": True,
    "twdac_scoring": True,
    "TwDAC_bevseg_pred": True,
    "scoring_test": True,
}


def apply_test_scoring(agent, weights: Dict[str, object], tag: str) -> None:
    """Set the test-time scoring attributes on the built model and read them back."""
    model = getattr(agent, "_safedrive_model", None)
    # The checkpoint keys are agent._safedrive_model.*, so a missing attribute means a different
    # model, and without it this run would quietly drive the imitation argmax.
    if model is None or not hasattr(model, "scoring_test"):
        raise SystemExit(f"[{tag}] agent has no _safedrive_model.scoring_test; cannot apply the "
                         f"test-time scoring, and without it SafeDrive drives the imitation argmax")
    for key, value in weights.items():
        setattr(model, key, value)
    wrong = {k: getattr(model, k, None) for k, v in weights.items() if getattr(model, k, None) != v}
    if wrong:
        raise SystemExit(f"[{tag}] test-time scoring did not take: {wrong}")
    print(f"[{tag}] test-time scoring on: " + " ".join(
        f"{k.replace('_test_weight', '')}={v}" for k, v in weights.items() if k.endswith("_test_weight")),
        flush=True)


class SafeDriveTunedPlanner(NavsimPlanner):
    """safedrive_baseline: the sheet safedrive_scored_tuned."""

    TAG = "safedrive_scored_tuned"
    TEST_SCORING = TUNED_BASELINE

    def build(self) -> None:
        super().build()
        apply_test_scoring(self.agent, self.TEST_SCORING, self.TAG)


class SafeDriveSDRouteTunedPlanner(SafeDriveTunedPlanner):
    """safedrive_sdroute: the sheet safedrive_sd_scored_tuned."""

    TAG = "safedrive_sd_scored_tuned"
    TEST_SCORING = TUNED_SDROUTE
