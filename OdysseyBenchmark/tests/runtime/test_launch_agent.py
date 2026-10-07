"""The launcher composes a run from an agent config: model selection, interpreter, adapter."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from odyssey_runtime.launch import build_run, dry_view

ROOT = Path(__file__).resolve().parents[3]
MODELS = {"ltf": "navsim", "diffusiondrive": "DiffusionDrive", "drivor": "DrivoR", "safedrive": "SafeDrive"}


SCENES_ROOT = __import__("os").environ.get("ODYSSEY_SCENES_ROOT", "")


def fake_zoo(tmp_path, monkeypatch):
    if not (Path(SCENES_ROOT) / "odyssey_scene001/scene.ckpt").is_file():
        pytest.skip("odyssey-scenes release not installed")
    zoo = tmp_path / "zoo"
    for folder in MODELS.values():
        (zoo / "models" / folder).mkdir(parents=True)
    (zoo / "sdroute").mkdir()
    (zoo / "sdroute/graph.py").write_text("")
    monkeypatch.setenv("ODYSSEY_ZOO_ROOT", str(zoo))
    monkeypatch.setenv("ODYSSEY_MODELS_ROOT", str(zoo))          # weights beside the code, as in the zoo
    monkeypatch.setenv("ODYSSEY_PLANNER_PY", "/envs/model/bin/python")
    return zoo


#: SafeDrive ranks plans by its test-time scoring, set by its own adapter; the rest run the generic one.
ADAPTERS = {("safedrive", "baseline"): "odyssey_bridge.planners.safedrive:SafeDriveTunedPlanner",
            ("safedrive", "sdroute"): "odyssey_bridge.planners.safedrive:SafeDriveSDRouteTunedPlanner"}


@pytest.mark.parametrize("model,arm", [(m, a) for m in MODELS for a in ("sdroute", "baseline")])
def test_agent_config_selects_model_interpreter_and_generic_adapter(tmp_path, monkeypatch, model, arm):
    zoo = fake_zoo(tmp_path, monkeypatch)
    spec = build_run(ROOT, ROOT / f"OdysseyBenchmark/agents/{model}_{arm}.yaml", "odyssey_scene001", "nr", 40, "0", tmp_path / "out")
    env, folder = spec["env"], MODELS[model]
    assert env["ODYSSEY_PLANNER"] == f"{model}_{arm}" == spec["profile"]["planner_id"]
    assert env["ODYSSEY_PLANNER_PY"] == "/envs/model/bin/python"
    assert env["ODYSSEY_PLANNER_REPO"] == str(zoo / "models" / folder)
    assert env["ODYSSEY_PLANNER_CKPT"] == str(zoo / "models" / folder / "ckpts" / f"{model}_{arm}.ckpt")
    assert env["ODYSSEY_PLANNER_CFG"] == f"{model}_{arm}_agent"
    assert env["ODYSSEY_PLANNER_ADAPTER"] == ADAPTERS.get((model, arm), "")
    assert "ODYSSEY_PLANNER_PYTHONPATH" not in env
    assert spec["profile"]["model_overrides"] == (["batch_size=1", "scheduler_args.num_epochs=1"] if model == "drivor" else [])
    assert spec["profile"]["audit"] is False and spec["profile"]["record_images"] is False
    assert any(a == f"job_name=odyssey_scene001_nr_{model}_{arm}" for a in spec["command"])
    assert spec["agent"].model.name == f"{model}_{arm}"
    view = dry_view(spec)
    assert view["agent"] == {"path": str(ROOT / f"OdysseyBenchmark/agents/{model}_{arm}.yaml"), "name": f"{model}_{arm}", "adapter": ADAPTERS.get((model, arm), "")}
    json.dumps(view)


def test_cli_deployment_arguments_override_the_config(tmp_path, monkeypatch):
    fake_zoo(tmp_path, monkeypatch)
    spec = build_run(ROOT, ROOT / "OdysseyBenchmark/agents/ltf_sdroute.yaml", "odyssey_scene001", "nr", 40, "0", tmp_path / "out",
                     repo="/models/native", checkpoint="/weights/model.ckpt", agent_config="native_agent",
                     seed=3, audit=True)
    env = spec["env"]
    assert (env["ODYSSEY_PLANNER_REPO"], env["ODYSSEY_PLANNER_CKPT"], env["ODYSSEY_PLANNER_CFG"]) == \
        ("/models/native", "/weights/model.ckpt", "native_agent")
    assert spec["profile"]["seed"] == 3 and spec["profile"]["audit"] is True



def test_a_timetable_set_scene_is_driven_with_its_own_published_label(tmp_path, monkeypatch):
    fake_zoo(tmp_path, monkeypatch)
    spec = build_run(ROOT, ROOT / "OdysseyBenchmark/agents/ltf_sdroute.yaml", "odyssey_scene052", "r", 40, "0", tmp_path / "out")
    options = dict(a.split("=", 1) for a in spec["command"] if "=" in a)
    assert options["tlc_timetable_set"] == "tlc_timetable" and options["tl_set"] == "tlc_timetable"
    assert options["tl_control_path"] == str(Path(SCENES_ROOT) / "odyssey_scene052" / "tl_control.json")


def test_the_restorer_is_the_torch_fixer_and_is_recorded(tmp_path, monkeypatch):
    fake_zoo(tmp_path, monkeypatch)
    spec = build_run(ROOT, ROOT / "OdysseyBenchmark/agents/ltf_sdroute.yaml", "odyssey_scene001", "nr", 40, "0", tmp_path / "a")
    assert spec["env"]["ODYSSEY_RESTORER"] == "fixer_h1b16" == dry_view(spec)["restorer"]


def test_environ_argument_replaces_the_process_environment(tmp_path, monkeypatch):
    zoo = fake_zoo(tmp_path, monkeypatch)
    environ = {"ODYSSEY_ZOO_ROOT": str(zoo), "ODYSSEY_MODELS_ROOT": str(zoo), "ODYSSEY_PLANNER_PY": "/other/python",
               "NUPLAN_MAPS_ROOT": str(tmp_path / "maps/final_gpu2"), "PATH": "/usr/bin", "ODYSSEY_SCENES_ROOT": SCENES_ROOT}
    spec = build_run(ROOT, ROOT / "OdysseyBenchmark/agents/ltf_sdroute.yaml", "odyssey_scene001", "nr", 40, "2", tmp_path / "out",
                     environ=environ)
    assert spec["env"]["ODYSSEY_PLANNER_PY"] == "/other/python"
    assert spec["env"]["NUPLAN_MAPS_ROOT"] == str(tmp_path / "maps/final_gpu2")
    assert any(a == f"nuplan_map_root={tmp_path / 'maps/final_gpu2'}" for a in spec["command"])
    assert spec["env"]["CUDA_VISIBLE_DEVICES"] == "2" and "HOME" not in spec["env"]


def test_extra_pythonpath_reaches_only_the_planner_process(tmp_path, monkeypatch):
    import yaml
    from odyssey_runtime import agent_config
    data = yaml.safe_load((ROOT / "OdysseyBenchmark/agents/ltf_sdroute.yaml").read_text())
    data["model"]["extra_pythonpath"] = ["side"]
    (tmp_path / "side").mkdir()
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump(data))
    env = {"ODYSSEY_PLANNER_PY": sys.executable, "ODYSSEY_ZOO_ROOT": str(tmp_path), "ODYSSEY_MODELS_ROOT": str(tmp_path)}
    assert agent_config.load(path, environ=env).planner_env()["ODYSSEY_PLANNER_PYTHONPATH"] == str(tmp_path / "side")

    from odyssey_runtime import session
    captured = {}

    class Worker:
        def __init__(self, python, factory, config, capacity, log, env=None, timeout=300):
            captured.update(config=config, env=env)

        def close(self):
            pass

    monkeypatch.setattr(session, "SharedWorker", Worker)
    from tests.runtime.test_profile import spec as profile_spec
    from odyssey_runtime.profile import ModelProfile
    for key, value in dict(ODYSSEY_ROOT=str(ROOT), ODYSSEY_PLANNER="ltf_sdroute_kv", ODYSSEY_PLANNER_CFG="c",
                           ODYSSEY_PLANNER_CKPT="k", ODYSSEY_PLANNER_REPO="r", ODYSSEY_PLANNER_PY=sys.executable,
                           ODYSSEY_PLANNER_ADAPTER="my.py:X", ODYSSEY_PLANNER_PYTHONPATH="/side",
                           PYTHONPATH="/base").items():
        monkeypatch.setenv(key, value)
    s = session.RuntimeSession(ModelProfile(profile_spec()), tmp_path / "runtime")
    try:
        s._start_worker()
    finally:
        s.close()
    assert captured["config"]["adapter"] == "my.py:X"
    assert captured["env"]["PYTHONPATH"].split(":")[:2] == ["/side", "/base"]


def test_run_refuses_a_config_whose_paths_do_not_exist(tmp_path, monkeypatch):
    fake_zoo(tmp_path, monkeypatch)
    cmd = [sys.executable, "-m", "odyssey_runtime", "run", "--agent", str(ROOT / "OdysseyBenchmark/agents/ltf_sdroute.yaml"),
           "--scene", "odyssey_scene001", "--react", "nr", "--max-steps", "40", "--output", str(tmp_path / "run")]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2, result.stderr
    assert "model.python is not an executable file" in result.stderr
    assert "model.checkpoint is not a file" in result.stderr
    assert not (tmp_path / "run").exists()
