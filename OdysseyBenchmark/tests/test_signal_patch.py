"""tl_signal_patch_path: camera signal patch applied once at the signal source.

Covers: strict file checks, scene check, application into traffic_light_state (row scale, unknown connector), every
consumer reading the same patched colour (converter -> planner/IDM/metric, TLC snapshot + pack, tl_control), the
render-only edge rules, and default-off identity (no key -> nothing changes, no new snapshot/pack/trace keys).
"""
import copy
import json
from types import SimpleNamespace

import pytest
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from shapely.geometry import box

from odyssey.components.agents.policy.pdm_planner.utils.odyssey_to_pdm_utils import OdysseyToNuPlanConverter
from odyssey.manager import signal_patch as SP
from odyssey.manager.agent_manager import BaseAgentManager
from odyssey.manager.render_manager import RenderManager
from odyssey_benchmark.tlc_metric import capture_tlc_signal_snapshot, pack_tlc
from odyssey_renderer.omnire.tl_control import TrafficLightController, normalize

U, R, G = 'TRAFFIC_LIGHT_UNKNOWN', 'TRAFFIC_LIGHT_RED', 'TRAFFIC_LIGHT_GREEN'


def scene(n=20):
    return {'id': 'odyssey_scene060', 'cadence': SimpleNamespace(sim_dt=0.1),
            'dynamic_map_states': {
                'a': dict(type='TRAFFIC_LIGHT', traffic_light_lane='A', state={'traffic_light_state': [U] * 10 + [G] * (n - 10)}),
                'b': dict(type='TRAFFIC_LIGHT', traffic_light_lane='B', state={'traffic_light_state': [R] * n})}}


def doc(patches=((0, 4, 'R'), (5, 9, 'G')), conns=('A',), scene_code='odyssey_scene060'):
    return {'schema': SP.SCHEMA, 'scene': scene_code,
            'signals': [{'roadblock': 'rb', 'connectors': list(conns),
                         'patches': [[a, b, c, 'camera', {}] for a, b, c in patches]}]}


def raw(d):
    return json.dumps(d)


# ---- file checks ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('bad, msg', [
    (dict(doc(), schema='x'), 'schema'),
    (dict(doc(), scene='085'), 'odyssey_sceneNNN'),
    (doc(patches=[(5, 4, 'R')]), 'first <= last'),
    (doc(patches=[(0, 1000, 'R')]), 'first <= last'),
    (doc(patches=[(0, 4, 'Y')]), 'first <= last'),
    (doc(patches=[(0, 4, 'R'), (4, 6, 'G')]), 'overlaps'),
    (dict(doc(), signals=[{'connectors': [], 'patches': []}]), 'connectors'),
])
def test_parse_rejects_malformed(bad, msg):
    with pytest.raises(ValueError, match=msg):
        SP.parse(raw(bad))


def test_load_for_scene_checks_the_scene(tmp_path):
    (tmp_path / 'odyssey_scene060.json').write_text(raw(doc()))
    assert SP.load_for_scene(tmp_path, 'odyssey_scene060')[0]['scene'] == 'odyssey_scene060'
    assert SP.load_for_scene(tmp_path, 'odyssey_scene061') is None          # directory without that scene
    with pytest.raises(ValueError, match='is for scene odyssey_scene060, not odyssey_scene061'):
        SP.load_for_scene(tmp_path / 'odyssey_scene060.json', 'odyssey_scene061')      # a file names its scene


# ---- application ------------------------------------------------------------------------------------------------
def test_apply_writes_rows_and_records_them():
    s = scene()
    rec = SP.apply_to_scene(s, doc(patches=[(0, 4, 'R'), (8, 12, 'G')]), 'sha', 'p')
    series = s['dynamic_map_states']['a']['state']['traffic_light_state']
    assert series[:5] == [R] * 5 and series[5:8] == [U] * 3 and series[8:13] == [G] * 5
    assert s['dynamic_map_states']['b']['state']['traffic_light_state'] == [R] * 20      # other connector untouched
    assert rec['rows'] == {'A': [[0, 4], [8, 12]]} and rec['spans'] == 2 and rec['sha256'] == 'sha'
    assert rec['cells_changed'] == 5 + 2 and rec['known_cells_overwritten'] == 0         # rows 10-12 were already G
    assert SP.is_patched(s, 'A', 4) and not SP.is_patched(s, 'A', 5) and not SP.is_patched(s, 'B', 0)


def test_apply_row_scale_and_log_end():
    s = scene(n=20)
    rec = SP.apply_to_scene(s, doc(patches=[(2, 3, 'R'), (9, 12, 'G')]), 'sha', 'p', row_scale=2)
    series = s['dynamic_map_states']['a']['state']['traffic_light_state']
    assert series[4:8] == [R] * 4 and series[3] == U                    # GT rows 2..3 -> scene rows 4..7
    assert rec['rows']['A'] == [[4, 7], [18, 19]]                        # 18..25 clipped at the log end (20 rows)


def test_apply_refuses_a_connector_without_db_signal():
    with pytest.raises(ValueError, match="no DB signal in scene odyssey_scene060: \\['Z'\\]"):
        SP.apply_to_scene(scene(), doc(conns=('A', 'Z')), 'sha', 'p')


def test_default_off_touches_nothing(tmp_path):
    s = scene()
    before = copy.deepcopy(s)
    assert SP.apply_configured({}, 'c085_x', s) is None
    assert SP.apply_configured({'tl_signal_patch_path': None}, 'c085_x', s) is None
    assert s == before and 'tl_signal_patch' not in s


def test_apply_configured_logs_and_applies(tmp_path, caplog):
    (tmp_path / 'odyssey_scene060.json').write_text(raw(doc()))
    s = scene()
    with caplog.at_level('INFO', logger='odyssey.manager.signal_patch'):
        rec = SP.apply_configured({'tl_signal_patch_path': str(tmp_path)}, 'odyssey_scene060', s)
    assert rec['spans'] == 2
    assert any(r.getMessage().startswith('[tl-patch] scene=odyssey_scene060 sha=') and 'spans=2' in r.getMessage()
               for r in caplog.records)
    with caplog.at_level('WARNING', logger='odyssey.manager.signal_patch'):
        assert SP.apply_configured({'tl_signal_patch_path': str(tmp_path)}, 'c086_x', scene()) is None
    assert any('no patch file' in r.getMessage() for r in caplog.records)


# ---- every consumer sees the same patched colour ----------------------------------------------------------------
def converter(s, rows=None):
    c = OdysseyToNuPlanConverter.__new__(OdysseyToNuPlanConverter)
    c.base_timestamp = 0
    c.scene = s
    c.engine = SimpleNamespace(agent_manager=SimpleNamespace(
        traffic_light_source_row=(lambda conn, step: step) if rows is None else rows), managers={})
    return c


def status(c, step, lane='A'):
    return next(x.status for x in c.convert_to_traffic_lights(step) if str(x.lane_connector_id) == lane)


def test_planner_idm_metric_snapshot_and_renderer_agree():
    s = scene()
    SP.apply_to_scene(s, doc(), 'sha', 'p')
    c = converter(s)
    # converter = planner (PDM), IDM, dense reward, online metric
    assert status(c, 2) == TrafficLightStatusType.RED and status(c, 7) == TrafficLightStatusType.GREEN
    assert status(c, 10) == TrafficLightStatusType.GREEN                  # unpatched DB row
    # renderer timetable
    ctl = TrafficLightController.from_scenario(
        {'schema': 'tl_control/1', 'num_frames': 20, 'heads': {}, 'connectors': {'A': []}, 'keep_natural': []}, s)
    assert normalize(ctl.timetable['A'][2]) == 'red' and normalize(ctl.timetable['A'][7]) == 'green'
    # TLC snapshot + pack
    snaps = [capture_tlc_signal_snapshot(s, c, step) for step in (2, 7, 12)]
    assert [x['states']['A'] for x in snaps] == ['RED', 'GREEN', 'GREEN']
    assert [x['patched']['A'] for x in snaps] == [True, True, False] and not snaps[0]['patched']['B']
    packed = pack_tlc(s, {'A': SimpleNamespace(polygon=box(0, 0, 2, 2))}, [2, 7, 12], .1, 1, None, None, {},
                      signal_snapshots=snaps)
    assert packed['signal_patched'] == [[True], [True], [False]]
    assert packed['signal_patched_all'] == [[True, False], [True, False], [False, False]]
    assert packed['signal_patch_sha256'] == 'sha'


def test_default_snapshot_and_pack_carry_no_patch_keys():
    s = scene()
    c = converter(s)
    snaps = [capture_tlc_signal_snapshot(s, c, step) for step in (2, 12)]
    assert all('patched' not in x for x in snaps)
    packed = pack_tlc(s, {'A': SimpleNamespace(polygon=box(0, 0, 2, 2))}, [2, 12], .1, 1, None, None, {},
                      signal_snapshots=snaps)
    assert not any(k.startswith('signal_patch') for k in packed)
    assert status(c, 2) == TrafficLightStatusType.UNKNOWN                 # DB unknown stays unknown


def test_sector_row_minus_one_stays_unknown_for_scoring():
    s = scene()
    SP.apply_to_scene(s, doc(), 'sha', 'p')
    c = converter(s, rows=lambda conn, step: -1)
    assert status(c, 3) == TrafficLightStatusType.UNKNOWN                 # planner / TLC: -1 is UNKNOWN
    snap = capture_tlc_signal_snapshot(s, c, 3)
    assert snap['inactive']['A'] and snap['patched']['A'] is False


# ---- render-only edge rules -------------------------------------------------------------------------------------
def test_spawn_row_from_the_sector_clock():
    m = object.__new__(BaseAgentManager)
    m._traffic_light_clock = None
    assert m.traffic_light_spawn_row('A') is None
    m._traffic_light_clock = SimpleNamespace(signal_sector={'A': 1}, spawn_rows=[3.0, 7.0])
    assert m.traffic_light_spawn_row('A') == 7 and m.traffic_light_spawn_row('Q') is None


def test_render_row_rules():
    m = object.__new__(RenderManager)
    m.current_scene = scene(n=20)
    agents = SimpleNamespace(traffic_light_spawn_row=lambda c: 7 if c == 'A' else None)
    rows = {0: -1, 1: 5, 2: 25}
    row = m._patched_render_row(agents, lambda c, s: rows[s])
    assert row('A', 0) == 7                  # sector not open: spawn-row colour (render only)
    assert row('B', 0) == -1                 # no spawn row known: stays -1
    assert row('A', 1) == 5                  # ordinary row unchanged
    assert row('A', 2) == 19                 # past the log: hold the last row


# ---- review follow-ups ---------------------------------------------------------------
def test_duplicate_connector_in_one_signal_is_refused():
    with pytest.raises(ValueError, match=r"more than once: \['A'\]"):
        SP.parse(raw(doc(conns=('A', 'A'))))


def test_spans_past_or_clipped_by_the_log_warn_and_are_counted(caplog):
    s = scene(n=20)
    with caplog.at_level('WARNING', logger='odyssey.manager.signal_patch'):
        rec = SP.apply_to_scene(s, doc(patches=[(2, 3, 'R'), (15, 30, 'G'), (40, 50, 'R')]), 'sha', 'p')
    assert rec['spans_past_log'] == [['A', 40, 50]] and rec['spans_clipped'] == [['A', 15, 30, 20]]
    msgs = [r.getMessage() for r in caplog.records]
    assert any('40-50 lies wholly past the log' in m for m in msgs)
    assert any('15-30 runs past the log (20 scene rows): clipped' in m for m in msgs)
    assert rec['rows']['A'] == [[2, 3], [15, 19]]


def test_idm_all_unknown_fallback_rows_are_counted_and_logged(caplog):
    s = scene(n=20)
    s['dynamic_map_states']['c'] = dict(type='TRAFFIC_LIGHT', traffic_light_lane='C',
                                        state={'traffic_light_state': [U] * 3 + [R] * 17})
    # signal {A, C}: rows 0..2 both UNKNOWN (fallback on) -> patched; rows 3..9 C known already (fallback was off)
    with caplog.at_level('WARNING', logger='odyssey.manager.signal_patch'):
        rec = SP.apply_to_scene(s, doc(patches=[(0, 9, 'R')], conns=('A', 'C')), 'sha', 'p')
    assert rec['idm_all_unknown_rows_filled'] == 3 and rec['idm_all_unknown_cells_filled'] == 6
    assert rec['idm_all_unknown_by_signal'] == {'rb': {'rows': 3, 'cells': 6}}
    assert any('IDM all-UNKNOWN fallback now off on 3 rows (6 connector cells)' in r.getMessage()
               for r in caplog.records)
    rec = SP.apply_to_scene(scene(n=20), doc(patches=[(10, 12, 'R')]), 'sha', 'p')    # known rows only
    assert rec['idm_all_unknown_rows_filled'] == 0 and rec['idm_all_unknown_by_signal'] == {}


def _sm_scene(n=11):
    import numpy as np
    from odyssey.scenario.scenarios.scenario_description import ScenarioDescription as SD
    pos = np.zeros((n, 3))
    pos[:, 0] = np.arange(n) * 0.5
    ego = {SD.TYPE: 'VEHICLE',
           SD.STATE: {SD.POSITION: pos, SD.HEADING: np.zeros(n), 'velocity': np.zeros((n, 2)),
                      SD.VALID: np.ones(n, dtype=bool), 'length': np.full(n, 4.5), 'width': np.full(n, 2.0),
                      'height': np.full(n, 1.5)},
           SD.METADATA: dict(object_id='ego', type='VEHICLE')}
    return {SD.ID: 'odyssey_scene060', SD.LENGTH: n, 'sample_rate': 4, SD.SDC_ID: 'ego',       # 0.2 s frames
            SD.METADATA: {SD.SDC_ID: 'ego', 'scenario_id': 'odyssey_scene060', 'dataset': 'test'},
            SD.OBJECT_TRACKS: {'ego': ego}, SD.MAP_FEATURES: {},
            SD.DYNAMIC_MAP_STATES: {'a': {SD.TYPE: 'TRAFFIC_LIGHT', 'traffic_light_lane': 'A',
                                          SD.STATE: {SD.TRAFFIC_LIGHT_STATES: [U] * 5 + [G] * (n - 5)}}}}


@pytest.mark.parametrize('with_patch', [False, True])
def test_scenario_manager_applies_the_patch_after_resampling(tmp_path, monkeypatch, with_patch):
    """Through the real ScenarioManager.__init__: resample x2 (0.2 s scene, 0.1 s rollout), then the patch."""
    from odyssey.manager import base_manager, scenario_manager
    (tmp_path / 'odyssey_scene060.json').write_text(raw(doc(patches=[(1, 3, 'R')])))
    cfg = {'rollout_dt': 0.1}
    if with_patch:
        cfg['tl_signal_patch_path'] = str(tmp_path)
    monkeypatch.setattr(base_manager, 'engine_initialized', lambda: True)
    monkeypatch.setattr(base_manager, 'get_engine', lambda: SimpleNamespace(global_random_seed=0))
    monkeypatch.setattr(base_manager, 'get_global_config', lambda: cfg)
    sm = scenario_manager.ScenarioManager({'odyssey_scene060': _sm_scene()})
    sc = sm.scenes['odyssey_scene060']
    assert sc['cadence'].upsample_n == 2
    series = [str(x) for x in sc['dynamic_map_states']['a']['state']['traffic_light_state']]
    assert len(series) == 21                                          # (11 - 1) * 2 + 1
    if not with_patch:
        assert series[:10] == [U] * 10 and series[10:] == [G] * 11 and 'tl_signal_patch' not in sc
        return
    # GT rows 1..3 -> scene rows 2..7; the rest is the resampled DB
    assert series[:2] == [U] * 2 and series[2:8] == [R] * 6 and series[8:10] == [U] * 2 and series[10:] == [G] * 11
    rec = sc['tl_signal_patch']
    assert rec['row_scale'] == 2 and rec['rows'] == {'A': [[2, 7]]} and rec['cells_changed'] == 6
    assert rec['idm_all_unknown_rows_filled'] == 6
    c = converter(sc)
    assert status(c, 2) == TrafficLightStatusType.RED and status(c, 8) == TrafficLightStatusType.UNKNOWN
