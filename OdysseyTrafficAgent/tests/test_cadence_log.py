"""The startup CADENCE check reads sub-steps with the controller's rule (when unset: outer step / 0.1 s)."""
import logging

from omegaconf import OmegaConf

from odyssey.runner.run_simulation import _log_cadence_check


def check(caplog, **cfg):
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _log_cadence_check(OmegaConf.create(dict(ego_controller="two_stage_controller", **cfg)), {})
    return caplog.text


def test_an_unset_substep_count_is_derived_at_half_a_second(caplog):
    text = check(caplog, rollout_dt=0.5, two_stage_substeps=None)
    assert "5 sub-steps of 0.1000s" in text and "MISMATCH" not in text


def test_an_explicit_wrong_substep_count_is_still_flagged(caplog):
    assert "MISMATCH" in check(caplog, rollout_dt=0.5, two_stage_substeps=1)
