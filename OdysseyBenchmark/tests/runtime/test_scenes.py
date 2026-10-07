"""Scene inputs from the published release (ADRLAB/odyssey-scenes), by published name only."""
import os
import pickle

import pytest

from odyssey_runtime import scenes

ROW = dict(scene='odyssey_scene007', token='00aa11bb22cc33dd')


def release(root, token=ROW['token'], skip=(), key=ROW['scene']):
    (root / 'scenes.csv').write_text('scene,token,city\n' + f"{ROW['scene']},{ROW['token']},sg-one-north\n")
    folder = root / ROW['scene']
    folder.mkdir(parents=True)
    for name in scenes.RELEASE_FILES:
        if name not in skip:
            (folder / name).write_bytes(name.encode())
    metadata = dict(nuplan_lidar_pc_tokens=[token], omnire_timestamps_us=[0] * 10)
    (folder / 'scenario.pkl').write_bytes(pickle.dumps({key: {'metadata': metadata}}))
    return folder


@pytest.mark.parametrize('name', ['odyssey_scene007', 'scene007', '007', '7'])
def test_published_names_select_the_scene_and_every_input_is_a_file_in_its_folder(tmp_path, name):
    folder = release(tmp_path)
    got = scenes.from_release(tmp_path, name)
    assert (got.name, got.token, got.folder) == ('odyssey_scene007', ROW['token'], folder)
    assert got.scenario == folder / 'scenario.pkl' and got.checkpoint == folder / 'scene.ckpt'
    assert got.route == folder / 'route.npz' and got.road_surface == folder / 'road_surface.npz'
    assert got.tl_control == folder / 'tl_control.json' and got.inventory == folder / 'checkpoint_inventory.json'
    assert got.scenario_key == 'odyssey_scene007'


def test_the_download_is_only_read(tmp_path):
    release(tmp_path)
    before = sorted(os.listdir(tmp_path))
    scenes.from_release(tmp_path, '007')
    assert sorted(os.listdir(tmp_path)) == before


def test_unknown_names_are_rejected(tmp_path):
    release(tmp_path)
    for name in ('008', 'x007', 'scene', 'odyssey_scene1000'):
        with pytest.raises(ValueError, match='unknown scene'):
            scenes.from_release(tmp_path, name)


def test_scenario_must_belong_to_the_listed_scene(tmp_path):
    release(tmp_path, token='ffffffffffffffff')
    with pytest.raises(ValueError, match='scenes.csv lists'):
        scenes.from_release(tmp_path, '007')


def test_scenario_is_stored_under_the_published_name(tmp_path):
    release(tmp_path, key='scene_from_an_older_release')
    with pytest.raises(ValueError, match='holds scene'):
        scenes.from_release(tmp_path, '007')


def test_incomplete_download_names_the_missing_files(tmp_path):
    release(tmp_path, skip=('checkpoint_inventory.json', 'road_surface.json'))
    with pytest.raises(FileNotFoundError, match='checkpoint_inventory.json'):
        scenes.from_release(tmp_path, '007')


def test_resolve_requires_the_scenes_root(tmp_path):
    release(tmp_path)
    assert scenes.resolve('7', environ={'ODYSSEY_SCENES_ROOT': str(tmp_path)}).name == 'odyssey_scene007'
    with pytest.raises(ValueError, match='ODYSSEY_SCENES_ROOT'):
        scenes.resolve('7', environ={})


def test_launcher_runs_a_downloaded_scene_by_its_published_name(monkeypatch, tmp_path):
    from pathlib import Path
    from odyssey_runtime.launch import build_run
    release_root = os.environ.get('ODYSSEY_SCENES_ROOT')
    if not release_root or not (Path(release_root) / 'odyssey_scene041/scene.ckpt').is_file():
        pytest.skip('odyssey-scenes release not installed')
    root = Path(__file__).resolve().parents[3]
    folder = Path(release_root).resolve() / 'odyssey_scene041'
    monkeypatch.setenv('ODYSSEY_PLANNER_PY', '/envs/model/bin/python')
    spec = build_run(root, root / 'OdysseyBenchmark/agents/ltf_sdroute.yaml', '041', 'r', 40, '0', tmp_path / 'run')
    options = dict(x.split('=', 1) for x in spec['command'][2:])
    assert spec['scene'] == 'odyssey_scene041' and 'scene_name' not in spec
    assert options['data_file_path'] == str(folder / 'scenario.pkl')
    assert options['scene_checkpoint'] == spec['checkpoint'] == str(folder / 'scene.ckpt')
    assert spec['env']['ODYSSEY_ROUTE_FILE'] == str(folder / 'route.npz')
    assert options['spawn_ego_tight_ahead_gate'] == 'false'            # a SPAWN_GATE_OFF scene
    assert options['job_name'].startswith('odyssey_scene041_r_')
