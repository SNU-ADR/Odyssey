"""tlc_timetable: the designated-set TLC score -- timetable file -> pinned snapshot -> P_TL, coverage and charges."""
import copy
import json

import numpy as np
import pytest
from shapely.geometry import box

from odyssey_benchmark import tlc_timetable as T
from odyssey_benchmark.tlc_metric import CODE_TABLE

VEHICLE = dict(width=2.297, front_length=4.049, rear_length=1.127, cog_position_from_rear_axle=1.67,
               wheel_base=3.089, vehicle_name='pacifica', vehicle_type='gen1', height=1.777)
A = box(-5, -5, 10, 5)            # holds the whole ego footprint (full overlap)
B = box(50, 0, 60, 10)            # a cross connector, never touched


def make_tlc(states, rows):
    n = len(states)
    steps = list(range(10, 10 + n))
    table = list(CODE_TABLE)
    conns = ('A', 'B')
    return dict(
        version='tl_events_v3', sample_dt=0.1, scene_frame_stride=1, sim_steps=steps,
        signal_code_table=table, signal_connector_ids=list(conns),
        signal_states_all=[[table.index(x) for x in r] for r in states], signal_source_rows_all=copy.deepcopy(rows),
        signal_inactive_all=[[r < 0 for r in row] for row in rows], signal_held_all=[[False, False]] * n,
        connector_ids=list(conns), connector_wkb={'A': A.wkb_hex, 'B': B.wkb_hex},
        connector_roadblock={'A': 'rb1', 'B': 'rb2'}, connector_paths={}, connector_maneuver={},
        map_location='us-nv-las-vegas-strip', apply_nearside_turn_filter=False,
        signal_states=[[table.index(x) for x in r] for r in states], signal_state_connector_ids=[list(conns)] * n,
        signal_source_rows=copy.deepcopy(rows), signal_inactive=[[r < 0 for r in row] for row in rows],
        signal_held=[[False, False]] * n, red_renderable={}, gt_rear_xy=None, gt_heading=None, vehicle=None,
        apply_gt_filter=False)


def write_timetable(d, show_g=True, spans=((0, 10, 'R'), (11, 11, 'G'), (12, 999, 'R')), conns=('A',)):
    d.mkdir(parents=True, exist_ok=True)
    (d / 'designated.json').write_text(json.dumps({'signals': [dict(
        scene='odyssey_scene060', roadblock='rb1', connectors=list(conns),
        showable={'R': dict(showable=True), 'G': dict(showable=show_g)})]}))
    (d / 'odyssey_scene060.json').write_text(json.dumps({'schema': 'tl_signal_patch/1', 'scene': 'odyssey_scene060', 'signals': [
        dict(roadblock='rb1', connectors=['A'], patches=[[a, b, c, 'camera', {}] for a, b, c in spans]),
        dict(roadblock='rb2', connectors=['B'], patches=[[0, 999, 'G', 'cross_of_ego', {}]], role='cross')]}))
    return d


def score_run(tlc, n, scene, tt):
    """Score a run the way driving_metrics.apply_rules does: the pinned TLC snapshot, the ego polygons of
    the dense states and the dense steps."""
    from odyssey_benchmark.tlc_metric import ego_polygons_from_states
    ego = ego_polygons_from_states(np.array([[1.0, 1.0, 0.0]] * n), VEHICLE)
    return T.score(tlc, ego, np.array(tlc['sim_steps']), scene, tt)


def test_timetable_to_npz_to_p_tl_coverage_and_charges(tmp_path):
    tt = T.load_timetable(write_timetable(tmp_path / 'tt'))
    assert tt['scenes'] == ['odyssey_scene060'] and 'set_version' not in tt
    # a DB-only run (offline): A unknown in the DB, B red; source rows 10, 11, 12, then -1 (sector not open) and 1000
    tlc = make_tlc([['UNKNOWN', 'RED']] * 5, [[10, 10], [11, 11], [12, 12], [-1, -1], [1000, 1000]])
    r, d = score_run(tlc, 5, 'odyssey_scene060', tt)
    assert r['version'] == T.VERSION and r['rule'] == 'timetable' and r['patch_applied'] == 'offline'
    # timetable colours on A: R (row 10), G (row 11), R (row 12) -> two red contacts, one passage -> one charge
    assert [d['colour'][(k, 0)] for k in range(3)] == ['R', 'G', 'R']
    assert r['charged_count'] == 1 and r['p_tl'] == pytest.approx(0.7)
    assert len(r['events']) == 1 and r['events'][0]['roadblock'] == 'rb1' and r['events'][0]['colour_source'] == 'camera'
    # row -1 and row > 999 are not scored; the cross connector B (green in the timetable) is never scored
    assert d['why_not'][(3, 0)] == 'row_before_open' and d['why_not'][(4, 0)] == 'row_past_log'
    assert all(d['why_not'][(k, 1)] == 'not_designated' for k in range(5))
    cov = r['coverage']
    assert cov['signals']['rb1'] == dict(touched=5, scored=3, coverage=0.6,
                                         not_scored={'row_before_open': 1, 'row_past_log': 1})
    assert cov['coverage'] == 0.6 and 'set_version' not in r['timetable'] and len(r['timetable']['sha256']) == 64


def test_unknown_and_not_showable_rows_are_not_scored(tmp_path):
    tt = T.load_timetable(write_timetable(tmp_path / 'tt', show_g=False, spans=((10, 10, 'R'), (11, 11, 'G'))))
    tlc = make_tlc([['UNKNOWN', 'RED']] * 3, [[10, 10], [11, 11], [12, 12]])
    r, d = score_run(tlc, 3, 'odyssey_scene060', tt)
    assert d['why_not'][(1, 0)] == 'not_showable:G' and d['why_not'][(2, 0)] == 'unknown'
    assert r['coverage']['signals']['rb1']['not_scored'] == {'not_showable:G': 1, 'unknown': 1}
    assert r['charged_count'] == 1 and r['coverage']['coverage'] == pytest.approx(1 / 3, abs=1e-4)


def test_online_run_is_checked_against_the_timetable_file(tmp_path):
    tt = T.load_timetable(write_timetable(tmp_path / 'tt'))
    sha = tt['files']['odyssey_scene060'][1]
    tlc = make_tlc([['RED', 'GREEN'], ['GREEN', 'GREEN'], ['RED', 'GREEN']], [[10, 10], [11, 11], [12, 12]])
    tlc['signal_patched'] = [[True, True]] * 3
    tlc['signal_patch_sha256'] = sha
    r, _ = score_run(tlc, 3, 'odyssey_scene060', tt)
    assert r['patch_applied'] == 'online' and r['online_check']['ok'] and r['charged_count'] == 1
    tlc['signal_patch_sha256'] = 'another file'
    r, _ = score_run(tlc, 3, 'odyssey_scene060', tt)
    assert r['online_check']['ok'] is False and r['online_check']['sha_matches'] is False


def test_load_timetable_fails_loudly(tmp_path):
    d = write_timetable(tmp_path / 'a', conns=('A', 'Z'))       # designated connectors differ from the file's entry
    with pytest.raises(ValueError, match='has no entry with the same connectors'):
        T.load_timetable(d)
    d = write_timetable(tmp_path / 'b')
    (d / 'odyssey_scene060.json').unlink()
    with pytest.raises(ValueError, match='has no timetable file'):
        T.load_timetable(d)
    with pytest.raises(FileNotFoundError):
        T.load_timetable(tmp_path / 'missing')
    tt = T.load_timetable(write_timetable(tmp_path / 'c'))
    with pytest.raises(ValueError, match='not in the designated set'):
        T.score(make_tlc([['RED', 'RED']], [[10, 10]]), [A], [10], 'odyssey_scene032', tt)
