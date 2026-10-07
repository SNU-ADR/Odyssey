"""The spawn-overlap gate is actually on in the shipped config.

The gate is split across two keys: `..._gate` turns the check on, and `..._action` decides what
to do when it fires. With action set to drop but gate left false, nothing happens, yet a skim of
the config reads as "set to drop" -- it once stayed mismatched like that. Pinning both values
keeps that combination from silently coming back.
"""
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
CFG = ROOT / "OdysseyTrafficAgent/odyssey/configs/default_runner.yaml"


def _cfg():
    return yaml.safe_load(CFG.read_text(encoding="utf-8"))


def test_the_gate_is_on_by_default():
    assert _cfg()["trajectory_spawn_ego_overlap_gate"] is True


def test_an_overlapping_spawn_is_dropped_by_default():
    assert _cfg()["trajectory_spawn_ego_overlap_action"] == "drop"


def test_both_keys_are_declared_so_hydra_can_override_them():
    """In struct mode an undeclared key cannot be overridden with `=`. Both values must be switchable
    per run (that is how the gate-OFF baseline is reproduced), so both must be declared."""
    cfg = _cfg()
    for key in ("trajectory_spawn_ego_overlap_gate",
                "trajectory_spawn_ego_overlap_action"):
        assert key in cfg
