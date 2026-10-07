"""A sector knob set to 0 must not be replaced by its default.

`cfg.get(k) or default` swallows 0 -- 0.0 or 15.0 is 15.0. sector_lead_m=0 means "appear when
the boundary is reached", the baseline for lead spawning, so it is a value that is really used.
A run submitted that way once silently ran with 15 and scored exactly like 15.
"""
import pathlib
import re

SRC = (pathlib.Path(__file__).resolve().parents[1]
       / "odyssey/manager/agent_manager.py").read_text(encoding="utf-8")
BLOCK = SRC[SRC.index("clock = SectorReplay("):SRC.index("src_dt=self.engine.sim_dt)")]


def test_no_falsy_default_in_the_sector_knobs():
    assert " or hybrid_replay.SECTOR_" not in BLOCK, BLOCK


def test_zero_survives_the_knob_reader():
    ns = {}
    exec(re.search(r"def _knob\(key, default\):\n(?:.*\n)+?\s+return .*\n", SRC).group(0)
         .replace("cfg.get", "CFG.get"), {"CFG": {"k": 0.0}}, ns)
    assert ns["_knob"]("k", 15.0) == 0.0          # value given
    assert ns["_knob"]("missing", 15.0) == 15.0   # value not given
