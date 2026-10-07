"""R vehicles use sector opening plus their own source duration, without GT replay."""
from types import SimpleNamespace

import numpy as np

from odyssey.manager import base_manager as base_manager_module
from odyssey.manager.agent_manager import BaseAgentManager
from odyssey.scenario.hybrid_replay import SectorReplay
from odyssey.scenario.scenarios.scenario_description import ScenarioDescription as SD


def _clock():
    route = np.stack([np.arange(100, dtype=float), np.zeros(100)], axis=1)
    valid_rows = np.arange(5, 51)
    positions = np.stack([np.full(len(valid_rows), 45.0),
                          np.ones(len(valid_rows))], axis=1)
    return SectorReplay(route, {"car": (valid_rows, positions)},
                        signals={"light": [45.0, 0.0]}, sector_len_s=40.0, src_dt=1.0)


def test_sector_gate_opens_own_5_to_50_track_at_35_to_80():
    clock = _clock()
    assert clock.sector["car"] == 1
    assert clock.step(34, (39.0, 0.0))["car"] == -1
    assert clock.actor_lifecycle_row("car", 34) == -1

    # R starts at this actor's first valid source row; NR and TLC keep the
    # ordinary sector-wide source row rather than being rewritten per actor.
    assert clock.step(35, (40.0, 0.0))["car"] == 40
    assert clock.actor_lifecycle_row("car", 35) == 5
    assert clock.signal_row("light", 35) == 40
    clock.step(80, (40.0, 0.0))
    assert clock.actor_lifecycle_row("car", 80) == 50
    clock.step(81, (40.0, 0.0))
    assert clock.actor_lifecycle_row("car", 81) == -1


def test_manager_uses_actor_row_for_r_presence_and_idm_seed(monkeypatch):
    clock = _clock()
    clock.step(35, (40.0, 0.0))
    scene = {SD.OBJECT_TRACKS: {"car": {"type": "VEHICLE", "metadata": {}}}}
    engine = SimpleNamespace(
        global_config={"agent_policy": "nuplan_idm_policy"},
        current_scene=scene,
        _nuplan_idm_batch=None,
    )
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)
    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._idm_spawn_clock = clock
    manager._idm_lifecycle_mode = "sector"
    manager._idm_lifecycle_clock = None
    manager._idm_source_step = 35
    manager._static_pinned_vehicle_ids = frozenset()
    manager._spawn_overlap_dropped = set()

    assert manager._presence_step("car", 35, {}) == 5
    assert manager.object_source_row("car", 35) == 5
    assert manager._presence_step("car", 80, {}) == 50
    assert manager.object_source_row("car", 81) == -1

    # A permanently pinned static vehicle does not use the R sector clock.
    manager._static_pinned_vehicle_ids = frozenset({"car"})
    assert manager.object_source_row("car", 35) == 35
