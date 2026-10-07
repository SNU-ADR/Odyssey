"""tlc_timetable_set: one key turns the timetable mode on for the scenes of a set.

Covers: default null is identical to today (scene signal rows, planner/IDM/metric converter status, TLC snapshot,
render controller decisions and trace), the preset on an in-set scene (with the scene's own label as tl_control_path)
== the explicit keys, an out-of-set scene is a logged no-op, precedence of explicit keys, loud failures on a bad set,
and the benchmark's own timetable (OdysseyBenchmark/data/tlc_timetable).
"""
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import sys
# the simulator's TL-control helper (_manager) lives with the simulator tests
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'OdysseyTrafficAgent/tests'))
from test_signal_patch import _sm_scene, converter, doc as patch_doc  # noqa: E402
from test_tl_control_allow_excluded import _manager  # noqa: E402
from odyssey.manager import signal_patch as SP
from odyssey.manager import tlc_timetable_set as TS
from odyssey_benchmark import tl_sets
from odyssey_benchmark.tl_sets import DATA_DIR
from odyssey_benchmark.tlc_metric import capture_tlc_signal_snapshot
from odyssey_renderer.omnire.tl_control import TL_CONTROL_SCENE_EXCLUSIONS

SCENE = 'odyssey_scene060'


def label(scene, allow_unobserved=False):
    heads = {'h': {'node': True, 'representatives': {'red': 3, 'green': 7}, 'unusable': {}, 'hold_frame': 3},
             'g': {'node': True, 'representatives': {'red': 5, 'green': 5}, 'unusable': {}, 'hold_frame': 5}}
    if allow_unobserved:
        heads['h']['allow_unobserved'] = {'green': 7}
    return {'schema': 'tl_control/1', 'scene_id': scene, 'num_frames': 10, 'heads': heads,
            'connectors': {'A': ['h'], 'B': ['g']}, 'keep_natural': []}


def label_path(d, code):
    """The scene's own label, outside the set directory (as the published scene's tl_control.json is)."""
    return d.parent / f'{d.name}_scenes' / code / 'tl_control.json'


def make_set(root, scenes=None, name='tset'):
    """A small set: every scene gets connector A patched R on GT rows 1..3; its two-head label is written beside."""
    scenes = scenes or {'odyssey_scene060': {}}
    d = root / name
    d.mkdir(parents=True)
    for code, spec in scenes.items():
        (d / f'{code}.json').write_text(json.dumps(patch_doc(patches=[(1, 3, 'R')], scene_code=code)))
        label_path(d, code).parent.mkdir(parents=True)
        label_path(d, code).write_text(json.dumps(label(code, spec.get('allow_unobserved', False))))
    (d / TS.MANIFEST).write_text(json.dumps({'schema': TS.SCHEMA, 'name': name, 'scenes': scenes}))
    return d


def preset_keys(d, code, value=None):
    """The preset as the launcher passes it: the set plus the scene's own label."""
    return {TS.KEY: value or str(d), 'tl_control_path': str(label_path(d, code))}


def explicit_keys(d, code, allow_excluded=False):
    cfg = {'tl_signal_patch_path': str(d / f'{code}.json'),
           'tl_control_path': str(label_path(d, code)),
           'tl_control_pin_uncontrolled': True}
    if allow_excluded:
        cfg['tl_control_allow_excluded'] = [code]
    return cfg


def load_scene(monkeypatch, cfg, scene_id=SCENE):
    """The real ScenarioManager.__init__ (resample x2, then the patch hook) on the synthetic scene."""
    from odyssey.manager import base_manager, scenario_manager
    cfg = dict({'rollout_dt': 0.1}, **cfg)
    monkeypatch.setattr(base_manager, 'engine_initialized', lambda: True)
    monkeypatch.setattr(base_manager, 'get_engine', lambda: SimpleNamespace(global_random_seed=0))
    monkeypatch.setattr(base_manager, 'get_global_config', lambda: cfg)
    sc = _sm_scene()
    sc['id'] = scene_id
    sc['metadata']['scenario_id'] = scene_id
    sm = scenario_manager.ScenarioManager({scene_id: sc})
    return sm.scenes[scene_id]


def scene_fingerprint(sc):
    """Signal rows + what the planner / IDM / dense reward / metric (converter) and the TLC snapshot read."""
    c = converter(sc)
    n = len(sc['dynamic_map_states']['a']['state']['traffic_light_state'])
    out = dict(rows=[str(x) for x in sc['dynamic_map_states']['a']['state']['traffic_light_state']],
               status=[[str(x.status) for x in c.convert_to_traffic_lights(step)] for step in range(n)],
               snaps=[capture_tlc_signal_snapshot(sc, c, step) for step in range(n)],
               patch={k: v for k, v in (sc.get('tl_signal_patch') or {}).items() if k != 'path'},
               keys=sorted(sc))
    return hashlib.sha256(json.dumps(out, sort_keys=True, default=str).encode()).hexdigest()


def render(cfg, scene_code, sc=None):
    m = _manager(scene_code, cfg)
    if sc is not None:
        m.current_scene = sc
    m._configure_traffic_light_control()
    return m


def render_fingerprint(m):
    if m._tl_ctl is None:
        return json.dumps(m.traffic_light_control_trace, sort_keys=True)
    out = []
    for scope in (None, [], ['A'], ['B'], ['A', 'B']):
        for step in range(0, 25):
            r = m._tl_ctl.resolve(step, scope)
            out.append([step, scope, sorted(r.mapping.items()), sorted(r.report.items()), r.in_window])
    trace = {k: v for k, v in m.traffic_light_control_trace.items() if k != 'tlc_timetable_set'}
    return hashlib.sha256(json.dumps([out, trace], sort_keys=True).encode()).hexdigest()


# ---- default null: identical to today -------------------------------------------------------------------------------
def test_default_null_is_identical(tmp_path, monkeypatch, caplog):
    d = make_set(tmp_path)
    base = explicit_keys(d, 'odyssey_scene060')
    for extra in ({}, base):
        prints = set()
        for off in ({}, {TS.KEY: None}, {TS.KEY: ''}):
            with caplog.at_level(logging.INFO):
                sc = load_scene(monkeypatch, dict(extra, **off))
                m = render(dict(extra, **off), 'odyssey_scene060', sc)
            prints.add((scene_fingerprint(sc), render_fingerprint(m)))
            assert 'tlc_timetable_set' not in (m.traffic_light_control_trace or {})
            assert not any('[tlc-set]' in r.getMessage() for r in caplog.records)
            assert TS.resolve(dict(extra, **off), SCENE) is None
        assert len(prints) == 1, extra
    # and the default really is "no timetable": natural replay, DB rows
    sc = load_scene(monkeypatch, {TS.KEY: None})
    assert 'tl_signal_patch' not in sc and render({TS.KEY: None}, 'odyssey_scene060', sc)._tl_ctl is None


# ---- preset on (+ the scene's label) == the explicit keys ----------------------------------------------------------------------------
@pytest.mark.parametrize('code, allow_excluded', [('odyssey_scene060', False), ('odyssey_scene024', True)])
def test_preset_on_equals_explicit_keys(tmp_path, monkeypatch, caplog, code, allow_excluded):
    assert (code in TL_CONTROL_SCENE_EXCLUSIONS) is allow_excluded
    d = make_set(tmp_path, {code: {'allow_excluded': allow_excluded}})
    scene_id = code
    with caplog.at_level(logging.INFO, logger='odyssey.manager.tlc_timetable_set'):
        sc_p = load_scene(monkeypatch, preset_keys(d, code), scene_id)
    lines = [r.getMessage() for r in caplog.records]
    assert any(x.startswith(f'[tlc-set] scene={code} set=tset sha=') and 'pin_uncontrolled=on' in x
               and f'allow_excluded={allow_excluded}' in x for x in lines), lines
    sc_e = load_scene(monkeypatch, explicit_keys(d, code, allow_excluded), scene_id)
    assert sc_p['tl_signal_patch']['sha256'] == sc_e['tl_signal_patch']['sha256']
    assert scene_fingerprint(sc_p) == scene_fingerprint(sc_e)
    m_p = render(preset_keys(d, code), code, sc_p)
    m_e = render(explicit_keys(d, code, allow_excluded), code, sc_e)
    assert m_p._tl_ctl is not None and m_p._tl_ctl.pin_uncontrolled is True
    assert render_fingerprint(m_p) == render_fingerprint(m_e)
    trace = m_p.traffic_light_control_trace
    assert trace['tlc_timetable_set'] == dict(name='tset', dir=str(d.resolve()),
                                              sha256=hashlib.sha256((d / TS.MANIFEST).read_bytes()).hexdigest())
    assert 'tlc_timetable_set' not in m_e.traffic_light_control_trace
    assert trace.get('exclusion_overridden', False) is allow_excluded


def test_preset_by_name_resolves_under_the_sets_dir(tmp_path, monkeypatch):
    make_set(tmp_path, name='mini')
    monkeypatch.setattr(TS, 'SETS_DIR', tmp_path)
    rec = TS.resolve(preset_keys(tmp_path / 'mini', 'odyssey_scene060', 'mini'), SCENE)
    assert rec['in_set'] and rec['dir'] == str((tmp_path / 'mini').resolve())
    assert rec['tl_signal_patch_path'] == str((tmp_path / 'mini').resolve() / 'odyssey_scene060.json')
    assert rec['tl_control_path'] == str(label_path(tmp_path / 'mini', 'odyssey_scene060'))


def test_in_set_scene_without_its_label_fails_loudly(tmp_path, monkeypatch):
    d = make_set(tmp_path)
    with pytest.raises(ValueError, match='tl_control_path is not set'):
        TS.resolve({TS.KEY: str(d)}, SCENE)
    with pytest.raises(ValueError, match='tl_control_path is not set'):
        load_scene(monkeypatch, {TS.KEY: str(d)})


# ---- out-of-set scene: logged no-op ---------------------------------------------------------------------------------
def test_scene_not_in_the_set_is_a_logged_noop(tmp_path, monkeypatch, caplog):
    d = make_set(tmp_path)
    scene_id = 'odyssey_scene061'
    with caplog.at_level(logging.WARNING, logger='odyssey.manager.tlc_timetable_set'):
        sc = load_scene(monkeypatch, {TS.KEY: str(d)}, scene_id)
    assert any(r.getMessage().startswith('[tlc-set] scene=odyssey_scene061 is not in set tset') for r in caplog.records)
    assert scene_fingerprint(sc) == scene_fingerprint(load_scene(monkeypatch, {}, scene_id))
    assert 'tl_signal_patch' not in sc
    m = render({TS.KEY: str(d)}, 'odyssey_scene061', sc)
    assert m._tl_ctl is None and m.traffic_light_control_trace is None
    # an explicit key still works for that scene
    assert render(preset_keys(d, 'odyssey_scene060'), 'odyssey_scene060', None)._tl_ctl is not None


# ---- precedence -----------------------------------------------------------------------------------------------------
def test_explicit_keys_override_the_preset(tmp_path, monkeypatch, caplog):
    d = make_set(tmp_path)
    other = tmp_path / 'other_label.json'
    other.write_text(json.dumps(dict(label('odyssey_scene060'), heads={'h': {'node': True, 'representatives': {'red': 4},
                                                                 'unusable': {}, 'hold_frame': 4}})))
    cfg = {TS.KEY: str(d), 'tl_control_path': str(other), 'tl_control_allow_excluded': ['odyssey_scene024']}
    with caplog.at_level(logging.INFO, logger='odyssey.manager.tlc_timetable_set'):
        TS.log_scene(cfg, SCENE)
    assert any(f'tl_control={other} ' in r.getMessage() and 'explicit_overrides=none' in r.getMessage()
               for r in caplog.records)
    m = render(cfg, 'odyssey_scene060')
    assert m._tl_control_path == str(other.resolve()) and m._tl_ctl.node_heads == ['h']
    assert m._tl_ctl.pin_uncontrolled is True                     # still from the preset
    preset = TS.resolve(cfg, SCENE)
    assert TS.value_for(cfg, 'tl_signal_patch_path', preset) == str(d / 'odyssey_scene060.json')
    # an explicit timetable replaces the set's file, and the log says so
    other_patch = tmp_path / 'other_patch.json'
    other_patch.write_bytes((d / 'odyssey_scene060.json').read_bytes())
    cfg_p = dict(cfg, tl_signal_patch_path=str(other_patch))
    caplog.clear()
    with caplog.at_level(logging.INFO, logger='odyssey.manager.tlc_timetable_set'):
        TS.log_scene(cfg_p, SCENE)
    assert any("explicit_overrides=['tl_signal_patch_path']" in r.getMessage() for r in caplog.records)
    assert TS.value_for(cfg_p, 'tl_signal_patch_path', TS.resolve(cfg_p, SCENE)) == str(other_patch)
    # allow lists are combined; the preset adds its scene only when the set says allow_excluded
    d2 = make_set(tmp_path / 'x', {'odyssey_scene024': {'allow_excluded': True}})
    cfg2 = dict(preset_keys(d2, 'odyssey_scene024'), tl_control_allow_excluded=['odyssey_scene011'])
    assert TS.value_for(cfg2, 'tl_control_allow_excluded', TS.resolve(cfg2, 'odyssey_scene024_x')) == ['odyssey_scene011', 'odyssey_scene024']
    assert TS.value_for(cfg2, 'tl_control_allow_excluded', TS.resolve(cfg2, 'c072_x')) == ['odyssey_scene011']


# ---- loud failures --------------------------------------------------------------------------------------------------
def _broken(tmp_path, how):
    """-> the config of a broken set (the set plus the scene's label, unless the label is what is missing)."""
    d = make_set(tmp_path, {'odyssey_scene060': {'allow_unobserved': False}})
    m = json.loads((d / TS.MANIFEST).read_text())
    if how == 'no_manifest':
        (d / TS.MANIFEST).unlink()
    elif how == 'schema':
        (d / TS.MANIFEST).write_text(json.dumps(dict(m, schema='x')))
    elif how == 'not_json':
        (d / TS.MANIFEST).write_text('{')
    elif how == 'unknown_field':
        (d / TS.MANIFEST).write_text(json.dumps(dict(m, scenes={'odyssey_scene060': {'allow_excluded': 'yes'}})))
    elif how == 'missing_timetable':
        (d / 'odyssey_scene060.json').unlink()
    elif how == 'unobserved_mismatch':
        label_path(d, 'odyssey_scene060').write_text(json.dumps(label('odyssey_scene060', allow_unobserved=True)))
    elif how == 'no_label':
        return {TS.KEY: str(d)}
    return preset_keys(d, 'odyssey_scene060')


@pytest.mark.parametrize('value, msg', [
    ('no_such_set_name', 'neither a set name'),
    ('/no/such/dir', 'neither a set name'),
])
def test_unknown_name_or_path_fails_loudly(value, msg, monkeypatch):
    with pytest.raises(ValueError, match=msg):
        TS.resolve({TS.KEY: value}, SCENE)
    with pytest.raises(ValueError, match=msg):
        load_scene(monkeypatch, {TS.KEY: value})


@pytest.mark.parametrize('how, msg', [
    ('no_manifest', 'has no tlc_timetable_set.json'), ('schema', 'schema must be'), ('not_json', 'is not JSON'),
    ('unknown_field', 'only boolean'), ('no_label', 'tl_control_path is not set'),
    ('missing_timetable', 'has no odyssey_scene060.json'),
    ('unobserved_mismatch', 'has allow_unobserved heads but the set says allow_unobserved=False'),
])
def test_malformed_set_fails_loudly(tmp_path, monkeypatch, how, msg):
    cfg = _broken(tmp_path, how)
    with pytest.raises(ValueError, match=msg):
        load_scene(monkeypatch, cfg)
    with pytest.raises(ValueError, match=msg):
        render(cfg, 'odyssey_scene060')


# ---- the benchmark's timetable ------------------------------------------------------------------------------------------
TT_SCENES = ['odyssey_scene011', 'odyssey_scene021', 'odyssey_scene025', 'odyssey_scene032', 'odyssey_scene052', 'odyssey_scene056', 'odyssey_scene060', 'odyssey_scene061', 'odyssey_scene067', 'odyssey_scene072', 'odyssey_scene075', 'odyssey_scene081', 'odyssey_scene086']


def test_repo_timetable_is_complete(tmp_path):
    s = TS.load_set('tlc_timetable', DATA_DIR)
    assert s['name'] == tl_sets.NAME == 'tlc_timetable' and sorted(s['scenes']) == TT_SCENES
    assert 'set_version' not in s
    assert sorted(c for c, v in s['scenes'].items() if v['allow_excluded']) == ['odyssey_scene011', 'odyssey_scene021', 'odyssey_scene056', 'odyssey_scene081']
    # odyssey_scene075 is also allowed by the code (ALLOW_UNOBSERVED_SCENES) but its label uses no unobserved frame
    assert sorted(c for c, v in s['scenes'].items() if v['allow_unobserved']) == ['odyssey_scene060']
    for c in (c for c, v in s['scenes'].items() if v['allow_excluded']):
        assert c in TL_CONTROL_SCENE_EXCLUSIONS, c
    d = Path(s['dir'])
    on_disk = sorted(p.relative_to(d).as_posix() for p in d.rglob('*') if p.is_file())
    assert on_disk == sorted(['README.md', 'designated.json', TS.MANIFEST] + [f'{c}.json' for c in TT_SCENES])
    for c in TT_SCENES:
        own = tmp_path / c / 'tl_control.json'             # stands in for the published scene's label
        own.parent.mkdir()
        own.write_text(json.dumps(label(c, s['scenes'][c]['allow_unobserved'])))
        rec = TS.resolve({TS.KEY: 'tlc_timetable', TS.DIR_KEY: str(DATA_DIR), 'tl_control_path': str(own)},
                         c)
        assert rec['in_set'] and SP.load_for_scene(rec['tl_signal_patch_path'], c)[0]['scene'] == c
        assert rec['tl_control_path'] == str(own) and rec['tl_control_pin_uncontrolled'] is True


def test_repo_timetable_loads_in_the_scorer():
    from odyssey_benchmark import tlc_timetable
    tt = tlc_timetable.load_timetable(tl_sets.timetable_dir('tlc_timetable'))
    assert set(tt['scenes']) <= set(TT_SCENES)
    with pytest.raises(ValueError):
        tl_sets.timetable_dir('r7')
