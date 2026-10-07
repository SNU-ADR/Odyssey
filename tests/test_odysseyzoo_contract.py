from types import SimpleNamespace
import importlib
import os
from pathlib import Path
import pytest


def scene001_available():
    root = os.environ.get('ODYSSEY_SCENES_ROOT')
    return bool(root) and (Path(root) / 'odyssey_scene001/scene.ckpt').is_file()


def contract():
    from odyssey_bridge.planners import base
    assert hasattr(base, 'route_arm_enabled'), 'OdysseyZoo route config is unsupported'
    return base


def test_new_route_flag_is_kv_only_and_legacy_false_is_not_overridden():
    c = contract()
    assert c.route_arm_enabled(SimpleNamespace(use_sdroute=True), 'use_sdroute_kv')
    assert not c.route_arm_enabled(SimpleNamespace(use_sdroute=True), 'use_sdroute_status')
    assert not c.route_arm_enabled(SimpleNamespace(use_sdroute=True, use_sdroute_kv=False), 'use_sdroute_kv')
    assert c.route_arm_enabled(SimpleNamespace(use_sdroute_kv=True), 'use_sdroute_kv')


def test_fixed_new_route_settings_do_not_silently_accept_old_experiment_configs():
    c = contract()
    cfg = SimpleNamespace(use_sdroute=True)
    assert c.route_arm_value(cfg, 'sdroute_horizon_m') == 120
    assert c.route_arm_value(cfg, 'sdroute_seg_index_mode') == 'learned'
    assert c.route_arm_value(SimpleNamespace(use_sdroute_kv=True), 'sdroute_horizon_m') is None
    assert c.route_arm_value(SimpleNamespace(use_sdroute=True, sdroute_horizon_m=80), 'sdroute_horizon_m') == 80


def test_scoring_loads_route_builder_from_odyssey_zoo_root(tmp_path, monkeypatch):
    from odyssey_benchmark import sdroute_sdf
    graph = tmp_path / 'sdroute/graph.py'
    graph.parent.mkdir()
    graph.write_text('VALUE = 17')
    monkeypatch.setenv('ODYSSEY_ZOO_ROOT', str(tmp_path))
    sdroute_sdf._module.cache_clear()
    try:
        assert sdroute_sdf._module('graph').VALUE == 17
    finally:
        sdroute_sdf._module.cache_clear()


def test_installer_accepts_explicit_odysseyzoo_manifest(tmp_path, monkeypatch):
    import hashlib
    import json
    import shutil
    from pathlib import Path
    from tests.runtime.test_native_patch_install import installer, ROOT
    source = ROOT / 'OdysseyZoo/models/SafeDrive'
    if not source.is_dir():
        pytest.skip('external OdysseyZoo snapshot unavailable')
    manifests = [(p, json.loads(p.read_text()))
                 for p in sorted((ROOT/'OdysseyBenchmark/patches').glob('safedrive-odysseyzoo-v*.json'))]
    first = 'navsim/agents/safedrive/safedrive_model.py'
    current = hashlib.sha256((source/first).read_bytes()).hexdigest()
    matches = [(p, m) for p, m in manifests if current in
               (m['files'][first]['before'], m['files'][first]['after'])]
    assert len(matches) == 1, 'external SafeDrive source has no unique pinned manifest'
    manifest_path, manifest = matches[0]
    for name in manifest['files']:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if (source / name).is_file():
            shutil.copyfile(source / name, target)
    import subprocess
    first = next(iter(manifest["files"]))
    if hashlib.sha256((tmp_path / first).read_bytes()).hexdigest() == manifest["files"][first]["after"]:
        subprocess.run(["git", "apply", "-R", str(manifest_path.with_suffix(".patch"))],
                       cwd=tmp_path, check=True)
    mod = installer()
    monkeypatch.chdir(ROOT)
    manifest_path = manifest_path.relative_to(ROOT)
    got = mod.apply(tmp_path, manifest_path=manifest_path)
    assert mod.apply(tmp_path, manifest_path=manifest_path) == got
    assert got['id'] == manifest['id']
    for name, identity in got['files'].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == identity['after']


@pytest.mark.parametrize('model,folder,config', [
    ('ltf','navsim','ltf_sdroute_agent'), ('drivor','DrivoR','drivor_sdroute_agent'),
    ('diffusiondrive','DiffusionDrive','diffusiondrive_sdroute_agent'),
    ('safedrive','SafeDrive','safedrive_sdroute_agent'),
])
def test_shipped_configs_use_zoo_names_and_published_weights(tmp_path, monkeypatch, model, folder, config):
    from pathlib import Path
    from odyssey_runtime.launch import build_run
    root = Path(__file__).resolve().parents[1]
    if not scene001_available():
        pytest.skip('odyssey_scene001 not downloaded (ODYSSEY_SCENES_ROOT)')
    zoo=tmp_path/'zoo'
    (zoo/'models'/folder).mkdir(parents=True)
    (zoo/'sdroute').mkdir()
    (zoo/'sdroute/graph.py').write_text('')
    monkeypatch.setenv('ODYSSEY_ZOO_ROOT', str(zoo))
    monkeypatch.setenv('ODYSSEY_PLANNER_PY', '/envs/model/bin/python')
    # Weights beside the code when ODYSSEY_MODELS_ROOT is unset ...
    monkeypatch.delenv('ODYSSEY_MODELS_ROOT', raising=False)
    spec=build_run(root,root/f'OdysseyBenchmark/agents/{model}_sdroute.yaml','odyssey_scene001','nr',40,'0',tmp_path/'out')
    assert spec['env']['ODYSSEY_PLANNER_REPO'] == str(zoo/'models'/folder)
    assert spec['env']['ODYSSEY_PLANNER_CKPT'] == str(zoo/'models'/folder/'ckpts'/f'{model}_sdroute.ckpt')
    assert spec['env']['ODYSSEY_PLANNER_CFG'] == config
    assert spec['env']['ODYSSEY_ZOO_ROOT'] == str(zoo)
    # ... or from the published release kept outside the zoo (ADRLAB/odyssey-models layout).
    monkeypatch.setenv('ODYSSEY_MODELS_ROOT', str(tmp_path/'models'))
    spec=build_run(root,root/f'OdysseyBenchmark/agents/{model}_sdroute.yaml','odyssey_scene001','nr',40,'0',tmp_path/'out')
    assert spec['env']['ODYSSEY_PLANNER_REPO'] == str(zoo/'models'/folder)
    assert spec['env']['ODYSSEY_PLANNER_CKPT'] == str(tmp_path/'models/models'/folder/'ckpts'/f'{model}_sdroute.ckpt')
    spec=build_run(root,root/f'OdysseyBenchmark/agents/{model}_sdroute.yaml','odyssey_scene001','nr',40,'0',tmp_path/'out', checkpoint=str(tmp_path/'explicit.ckpt'))
    assert spec['env']['ODYSSEY_PLANNER_CKPT'] == str(tmp_path/'explicit.ckpt')


def test_relative_zoo_root_survives_worker_cwd_change(tmp_path, monkeypatch):
    from pathlib import Path
    from odyssey_runtime.launch import build_run
    root = Path(__file__).resolve().parents[1]
    if not scene001_available():
        pytest.skip('odyssey_scene001 not downloaded (ODYSSEY_SCENES_ROOT)')
    (tmp_path/'zoo/models/navsim').mkdir(parents=True)
    (tmp_path/'zoo/sdroute').mkdir()
    (tmp_path/'zoo/sdroute/graph.py').write_text('')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ODYSSEY_ZOO_ROOT', 'zoo')
    spec=build_run(root,root/'OdysseyBenchmark/agents/ltf_sdroute.yaml',
                   'odyssey_scene001','nr',40,'0',tmp_path/'out')
    assert spec['env']['ODYSSEY_ZOO_ROOT'] == str(tmp_path/'zoo')
