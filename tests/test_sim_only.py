"""Contracts for the simulation-only distribution."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_runner_reports_do_not_import_training_framework():
    code = ('import sys; import odyssey.runner.utils; '
            'assert not any(n.startswith("pytorch_lightning") for n in sys.modules)')
    subprocess.run([sys.executable, '-c', code], check=True)


def test_renderer_has_no_historical_parent():
    from odyssey_renderer.omnire.engine import OmniReRenderEngine
    names = [c.__module__ for c in OmniReRenderEngine.__mro__[1:]]
    assert not any(n.endswith(('.v7_engine', '.v8_engine', '.engine')) for n in names)


AGENT = ROOT / 'OdysseyBenchmark/agents/ltf_sdroute.yaml'


@pytest.fixture(autouse=True)
def planner_interpreter(monkeypatch):
    monkeypatch.setenv('ODYSSEY_PLANNER_PY', '/envs/model/bin/python')   # the shipped configs expand it


def test_default_run_disables_analysis_recordings():
    from odyssey_runtime.launch import build_run
    profile = AGENT
    out = ROOT / 'artifacts/test_launch'
    spec = build_run(ROOT, profile, 'odyssey_scene001', 'nr', 40, '0', out)
    assert spec['profile']['record_images'] is False
    assert spec['profile']['record_legacy_ipc'] is False
    assert spec['command'][1].endswith('run_simulation.py')
    assert 'renderer=omnire' in spec['command']
    assert 'num_future=25' in spec['command']
    assert 'renderer.road_surface=' + str(Path(os.environ['ODYSSEY_SCENES_ROOT']).resolve() / 'odyssey_scene001/road_surface.npz') in spec['command']
    # one flat result folder, no resume bookkeeping, no per-step frame index unless recording
    assert f"data_output_dir={out / 'odyssey_output'}" in spec['command']
    assert 'record_frame_index=false' in spec['command'] and 'runner_report_file=runner_report.json' in spec['command']
    assert not any('completed_scenarios' in a or 'openscene_format' in a for a in spec['command'])
    assert spec['env']['ODYSSEY_PLANNER'] == 'ltf_sdroute'
    assert spec['env']['ODYSSEY_OMNIRE_ACTOR_POSE_SOURCE'] == 'scenario'


def test_reactive_run_uses_idm():
    from odyssey_runtime.launch import build_run
    spec = build_run(ROOT, AGENT, 'odyssey_scene001', 'r',
                     40, '0', ROOT / 'artifacts/test_launch')
    assert 'agent_policy=nuplan_idm_policy' in spec['command']
    assert 'agent_replay=sector_lead' not in spec['command']


def test_model_deployment_can_be_supplied_without_changing_adapter_code():
    from odyssey_runtime.launch import build_run
    spec = build_run(ROOT, AGENT, 'odyssey_scene001', 'nr',
                     40, '0', ROOT / 'artifacts/test_launch',
                     repo='/models/native', checkpoint='/weights/model.ckpt', agent_config='native_agent')
    assert spec['env']['ODYSSEY_PLANNER_ADAPTER'] == ''          # the generic adapter runs it
    assert spec['env']['ODYSSEY_PLANNER_REPO'] == '/models/native'
    assert spec['env']['ODYSSEY_PLANNER_CKPT'] == '/weights/model.ckpt'
    assert spec['env']['ODYSSEY_PLANNER_CFG'] == 'native_agent'


def test_route_builder_comes_from_odysseyzoo_without_host_environment(monkeypatch):
    from odyssey_runtime.launch import build_run
    monkeypatch.delenv('ODYSSEY_ZOO_ROOT', raising=False)
    monkeypatch.delenv('ODYSSEY_MODELS_ROOT', raising=False)
    spec = build_run(ROOT, AGENT, 'odyssey_scene001', 'nr',
                     40, '0', ROOT/'artifacts/test_launch')
    assert spec['env']['ODYSSEY_ZOO_ROOT'] == str(ROOT/'OdysseyZoo')
    assert spec['env']['ODYSSEY_PLANNER_CKPT'] == str(ROOT/'OdysseyZoo/models/navsim/ckpts/ltf_sdroute.ckpt')


def test_relative_model_paths_resolve_before_worker_changes_directory(monkeypatch, tmp_path):
    from odyssey_runtime.launch import build_run
    monkeypatch.chdir(tmp_path)
    spec = build_run(ROOT, AGENT, 'odyssey_scene001', 'nr',
                     40, '0', ROOT/'artifacts/test_launch',
                     repo='native', checkpoint='weights/model.ckpt', agent_config='native_agent')
    assert spec['env']['ODYSSEY_PLANNER_REPO'] == str(tmp_path/'native')
    assert spec['env']['ODYSSEY_PLANNER_CKPT'] == str(tmp_path/'weights/model.ckpt')
