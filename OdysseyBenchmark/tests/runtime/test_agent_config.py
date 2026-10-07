"""One YAML file ports a model: the loader resolves it into the runtime profile and model selection."""
import os
from pathlib import Path
import sys

import pytest
import yaml

from odyssey_runtime import agent_config
from odyssey_runtime.agent_config import HIDDEN, load
from tests.runtime.test_profile import spec

ROOT = Path(__file__).resolve().parents[3]


def minimal(**model):
    data = dict(
        model=dict(dict(name="my_model", python=sys.executable, repo="repo", agent_config="my_agent",
                        checkpoint="weights/my.ckpt"), **model),
        history_times_s=[-1.5, -1.0, -0.5, 0.0],
        camera_times_s={"CAM_F0": [0.0], "CAM_L0": [0.0], "CAM_R0": [0.0]},
        navigation={"driving_command": False, "sd_route": "targets"},
        output={"plan_dt": 0.5, "shape": [8, 3]},
    )
    return data


def write(tmp_path, data, name="agent.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return path


def test_effective_profile_is_the_runtime_contract_plus_fixed_fields(tmp_path):
    cfg = load(write(tmp_path, minimal()), environ={})
    expected = dict(spec(), planner_id="my_model", model_overrides=[], **HIDDEN)
    assert cfg.profile.data == expected
    assert cfg.model.name == "my_model" and cfg.model.adapter == ""
    assert cfg.model.repo == str(tmp_path / "repo")
    assert cfg.model.checkpoint == str(tmp_path / "weights/my.ckpt")
    assert cfg.model.agent_yaml == str(tmp_path / "repo/navsim/planning/script/config/common/agent/my_agent.yaml")


def test_paths_expand_variables_and_home(tmp_path, monkeypatch):
    data = minimal(python="${MY_PY}", repo="$ZOO/models/navsim", checkpoint="~/w.ckpt",
                   adapter="${ZOO}/adapters/a.py:A", extra_pythonpath=["$ZOO/side"])
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    env = {"MY_PY": "/envs/py", "ZOO": "/zoo", "HOME": str(tmp_path / "home")}
    cfg = load(write(tmp_path, data), environ=env)
    assert cfg.model.python == "/envs/py"
    assert cfg.model.repo == "/zoo/models/navsim"
    assert cfg.model.checkpoint == str(tmp_path / "home/w.ckpt")
    assert cfg.model.adapter == "/zoo/adapters/a.py:A"
    assert cfg.model.extra_pythonpath == ("/zoo/side",)


def test_unset_variable_names_the_field(tmp_path):
    with pytest.raises(ValueError, match=r"model\.checkpoint: environment variable WEIGHTS"):
        load(write(tmp_path, minimal(checkpoint="${WEIGHTS}/m.ckpt")), environ={})


def test_overrides_are_passed_through_unexpanded(tmp_path):
    cfg = load(write(tmp_path, minimal(overrides=["batch_size=1", "++a.b=${c}"])), environ={})
    assert cfg.model.overrides == ("batch_size=1", "++a.b=${c}")
    assert cfg.profile.data["model_overrides"] == ["batch_size=1", "++a.b=${c}"]


@pytest.mark.parametrize("change, message", [
    (lambda d: d.update(sim_dt=0.2), "unknown settings"),
    (lambda d: d["model"].update(gpu=1), "unknown model settings"),
    (lambda d: d["model"].pop("checkpoint"), "model requires"),
    (lambda d: d.pop("model"), "'model' mapping"),
    (lambda d: d["model"].update(name="a/b"), "plain label"),
    (lambda d: d["model"].update(overrides="x=1"), "list of Hydra"),
    (lambda d: d["model"].update(adapter="no_colon"), "model.adapter"),
    (lambda d: d["model"].update(adapter="a.py:not an identifier"), "model.adapter"),
    (lambda d: d.pop("output"), "'output'"),
    (lambda d: d.pop("navigation"), "navigation"),
    (lambda d: d.update(camera_times_s={"CAM_F0": [0.3]}), "camera"),
    (lambda d: d.update(output={"plan_dt": 0.5, "shape": [8, 2]}), "output"),
])
def test_invalid_configs_are_refused_with_the_file_named(tmp_path, change, message):
    data = minimal()
    change(data)
    with pytest.raises(ValueError, match=message) as error:
        load(write(tmp_path, data), environ={})
    assert "agent.yaml" in str(error.value)


def test_non_mapping_or_broken_yaml(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="mapping"):
        load(path, environ={})
    path.write_text("model: [unterminated\n")
    with pytest.raises(ValueError, match="YAML"):
        load(path, environ={})


def test_adapter_file_relative_to_config_and_module_form(tmp_path):
    cfg = load(write(tmp_path, minimal(adapter="adapters/mine.py:Mine")), environ={})
    assert cfg.model.adapter == f"{tmp_path / 'adapters/mine.py'}:Mine"
    cfg = load(write(tmp_path, minimal(adapter="pkg.sub.module:Cls")), environ={})
    assert cfg.model.adapter == "pkg.sub.module:Cls"


def test_cli_overrides_and_seed(tmp_path):
    cfg = load(write(tmp_path, minimal()), environ={})
    same = cfg.with_model(repo=None, checkpoint=None, agent_config=None).with_seed(None)
    assert same is cfg
    changed = cfg.with_model(repo="other", checkpoint="/w/x.ckpt", agent_config="their_agent").with_seed(7)
    assert changed.model.repo == os.path.abspath("other")
    assert changed.model.checkpoint == "/w/x.ckpt" and changed.model.agent_config == "their_agent"
    assert changed.profile.data["seed"] == 7 and "seed" not in cfg.profile.data


def test_verify_paths_lists_every_missing_piece(tmp_path):
    cfg = load(write(tmp_path, minimal(python="/no/such/python", adapter="a.py:A",
                                       extra_pythonpath=["side"])), environ={})
    problems = "\n".join(cfg.verify_paths())
    for piece in ("model.python", "model.repo", "model.checkpoint", "model.adapter", "model.extra_pythonpath"):
        assert piece in problems
    (tmp_path / "repo/navsim/planning/script/config/common/agent").mkdir(parents=True)
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights/my.ckpt").write_bytes(b"x")
    (tmp_path / "a.py").write_text("")
    (tmp_path / "side").mkdir()
    cfg = load(write(tmp_path, minimal(adapter="a.py:A", extra_pythonpath=["side"])), environ={})
    assert cfg.verify_paths() == ["model.agent_config 'my_agent' has no yaml at "
                                  + str(tmp_path / "repo/navsim/planning/script/config/common/agent/my_agent.yaml")]
    (tmp_path / "repo/navsim/planning/script/config/common/agent/my_agent.yaml").write_text("")
    assert cfg.verify_paths() == []


def test_planner_env_selects_the_model(tmp_path):
    cfg = load(write(tmp_path, minimal(extra_pythonpath=["side", "more"])), environ={})
    env = cfg.planner_env()
    assert env["ODYSSEY_PLANNER"] == "my_model" and env["ODYSSEY_PLANNER_PY"] == sys.executable
    assert env["ODYSSEY_PLANNER_REPO"] == str(tmp_path / "repo")
    assert env["ODYSSEY_PLANNER_CKPT"] == str(tmp_path / "weights/my.ckpt")
    assert env["ODYSSEY_PLANNER_CFG"] == "my_agent" and env["ODYSSEY_PLANNER_ADAPTER"] == ""
    assert env["ODYSSEY_PLANNER_PYTHONPATH"] == os.pathsep.join([str(tmp_path / "side"), str(tmp_path / "more")])
    assert "ODYSSEY_PLANNER_PYTHONPATH" not in load(write(tmp_path, minimal()), environ={}).planner_env()


@pytest.mark.parametrize("name", sorted(p.name for p in (ROOT / "OdysseyBenchmark/agents").glob("*.yaml")))
def test_shipped_configs_load_against_a_fake_zoo(tmp_path, name):
    env = {"ODYSSEY_PLANNER_PY": sys.executable, "RECOGDRIVE_PLANNER_PY": "/envs/recogdrive/bin/python",
           "ODYSSEY_ZOO_ROOT": str(tmp_path / "zoo"), "ODYSSEY_MODELS_ROOT": str(tmp_path / "models")}
    cfg = load(ROOT / "OdysseyBenchmark/agents" / name, environ=env)
    assert cfg.model.python == env["RECOGDRIVE_PLANNER_PY" if "recogdrive" in name else "ODYSSEY_PLANNER_PY"]
    assert cfg.model.name == name[:-5]
    assert cfg.model.repo.startswith(str(tmp_path / "zoo/models/"))
    # Weights come from the published release (ADRLAB/odyssey-models), which mirrors the zoo layout.
    assert cfg.model.checkpoint == cfg.model.repo.replace(str(tmp_path / "zoo"), str(tmp_path / "models")) \
        + f"/ckpts/{name[:-5]}.ckpt"
    assert cfg.profile.data["planner_id"] == cfg.model.name
    # ReCogDrive and SafeDrive (test-time scoring) have their own adapters; the generic one runs the rest.
    safedrive = {"safedrive_baseline": "odyssey_bridge.planners.safedrive:SafeDriveTunedPlanner",
                 "safedrive_sdroute": "odyssey_bridge.planners.safedrive:SafeDriveSDRouteTunedPlanner"}
    assert "recogdrive" in name or cfg.model.adapter == safedrive.get(name[:-5], "")
    generic = cfg.profile.navigation["sd_route"] != "none"
    assert ("sdroute" in name) == generic
    assert cfg.profile.navigation["driving_command"] == (not generic)


def test_is_agent_config_by_extension():
    assert agent_config.is_agent_config("x/agent.yaml") and agent_config.is_agent_config(Path("a.yml"))
    assert not agent_config.is_agent_config("runtime/profile.json")
