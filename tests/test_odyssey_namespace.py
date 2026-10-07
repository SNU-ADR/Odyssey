"""Public Odyssey namespace and launch vocabulary."""
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_odyssey_types_are_exposed_under_new_namespace():
    module = importlib.import_module("odyssey.utils.type")
    assert module.OdysseyObjectType.is_vehicle(module.OdysseyObjectType.VEHICLE)


def test_launch_uses_odyssey_output_and_environment(tmp_path, monkeypatch):
    from odyssey_runtime.launch import build_run
    monkeypatch.setenv("ODYSSEY_PLANNER_PY", "/envs/model/bin/python")
    spec = build_run(ROOT, ROOT / "OdysseyBenchmark/agents/ltf_sdroute.yaml",
                     "odyssey_scene001", "nr", 40, "0", tmp_path / "run")
    assert spec["env"]["ODYSSEY_ROOT"] == str(ROOT)
    assert any("output_dir=" + str(tmp_path / "run/odyssey_output") == arg
               for arg in spec["command"])
    assert "/odyssey/runner/run_simulation.py" in spec["command"][1]


def test_launch_refuses_variables_outside_the_benchmark(monkeypatch, tmp_path):
    """A research toggle left in the shell (route noise, termination radius, extra planner
    overrides) would change what is measured without appearing in the result."""
    from odyssey_runtime.launch import EnvironmentRefused, build_run
    monkeypatch.setenv("ODYSSEY_PLANNER_PY", "/envs/model/bin/python")
    for name in ("ODYSSEY_ROUTE_NOISE", "ODYSSEY_SD_GOAL_END_DIST_M", "ODYSSEY_PLANNER_EXTRA_OVERRIDES"):
        monkeypatch.setenv(name, "1")
        with pytest.raises(EnvironmentRefused, match=name):
            build_run(ROOT, ROOT / "OdysseyBenchmark/agents/ltf_sdroute.yaml", "odyssey_scene001", "nr", 40, "0", tmp_path / "run")
        monkeypatch.delenv(name)
    # install locations, infrastructure timing and values the launcher sets itself are accepted
    monkeypatch.setenv("ODYSSEY_PLAN_WAIT_TIMEOUT_S", "600")
    monkeypatch.setenv("ODYSSEY_FIXER_START_TIMEOUT_S", "1200")
    monkeypatch.setenv("ODYSSEY_INJECT_AY", "0")
    spec = build_run(ROOT, ROOT / "OdysseyBenchmark/agents/ltf_sdroute.yaml", "odyssey_scene001", "nr", 40, "0", tmp_path / "run")
    assert spec["env"]["ODYSSEY_INJECT_AY"] == "1"


def test_launch_requires_odysseyzoo(monkeypatch, tmp_path):
    from odyssey_runtime.launch import build_run
    monkeypatch.setenv("ODYSSEY_PLANNER_PY", "/envs/model/bin/python")
    monkeypatch.setenv("ODYSSEY_ZOO_ROOT", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="OdysseyZoo"):
        build_run(ROOT, ROOT / "OdysseyBenchmark/agents/ltf_sdroute.yaml", "odyssey_scene001", "nr", 40, "0", tmp_path / "run")
