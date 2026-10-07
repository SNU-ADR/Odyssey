"""tlc_timetable.evaluate: designated signals only, colour = recorded patched state (online) or DB + the timetable
through the recorded source rows (offline), v2 contact rule with its lateral margin, exclusions, no near-side waiver,
online check."""
import copy
import math
from types import SimpleNamespace

import pytest
from shapely import affinity
from shapely.geometry import Point, Polygon, box

from odyssey.components.agents.policy.pdm_planner.utils.odyssey_to_pdm_utils import OdysseyToNuPlanConverter
from odyssey.manager import signal_patch as SP
from odyssey_benchmark import tlc_timetable as P
from odyssey_benchmark.tlc_metric import CODE_TABLE, capture_tlc_signal_snapshot, pack_tlc, score_tlc

U, R, G = 'TRAFFIC_LIGHT_UNKNOWN', 'TRAFFIC_LIGHT_RED', 'TRAFFIC_LIGHT_GREEN'
ON_A = box(0.5, 0.5, 1.5, 1.5)
OFF = box(50, 50, 51, 51)


def designated(scene='odyssey_scene060', rb='rb1', conns=('A',), show_r=True, show_g=True):
    return {'signals': [dict(scene=scene, roadblock=rb, connectors=list(conns),
                             showable={'R': dict(showable=show_r), 'G': dict(showable=show_g)})]}


def patch(patches=((0, 9, 'R'),), conns=('A',), scene='odyssey_scene060'):
    return {'schema': SP.SCHEMA, 'scene': scene,
            'signals': [{'roadblock': 'rb1', 'connectors': list(conns),
                         'patches': [[a, b, c, 'camera', {}] for a, b, c in patches]}]}


def make_tlc(states, rows=None, held=None, conns=('A', 'B'), stride=1, rbs=('rb1', 'rb2')):
    """Route columns A (box 0..2), B (box 10..12); states per row as names."""
    n = len(states)
    steps = list(range(10, 10 + n))
    table = list(CODE_TABLE)
    rows = rows if rows is not None else [[s, s] for s in steps]
    return dict(
        version='tl_events_v3', sample_dt=0.1, scene_frame_stride=stride, sim_steps=steps,
        signal_code_table=table, signal_connector_ids=list(conns),
        signal_states_all=[[table.index(x) for x in r] for r in states], signal_source_rows_all=copy.deepcopy(rows),
        signal_inactive_all=[[r < 0 for r in row] for row in rows], signal_held_all=held or [[False, False]] * n,
        connector_ids=list(conns), connector_wkb={'A': box(0, 0, 2, 2).wkb_hex, 'B': box(10, 0, 12, 2).wkb_hex},
        connector_roadblock=dict(zip(conns, rbs)), connector_paths={}, connector_maneuver={},
        map_location='us-nv-las-vegas-strip', apply_nearside_turn_filter=False,
        signal_states=[[table.index(x) for x in r] for r in states], signal_state_connector_ids=[list(conns)] * n,
        signal_source_rows=rows, signal_inactive=[[r < 0 for r in row] for row in rows],
        signal_held=held or [[False, False]] * n, red_renderable={}, gt_rear_xy=None, gt_heading=None, vehicle=None,
        apply_gt_filter=False)


def score(tlc, ego, desig=None, doc=None, sha=None):
    return P.evaluate(tlc, ego, tlc['sim_steps'], 'odyssey_scene060', desig or designated(), doc, sha)


def test_only_designated_roadblocks_are_scored():
    tlc = make_tlc([['RED', 'RED']] * 3)
    res, d = score(tlc, [box(0.5, 0.5, 1.5, 1.5)] * 3)
    assert res['charged_count'] == 1 and res['events'][0]['roadblock'] == 'rb1'
    assert all(d['why_not'][(r, 1)] == 'not_designated' for r in range(3))       # B is rb2
    res, _ = score(tlc, [box(10.5, 0.5, 11.5, 1.5)] * 3)                         # ego on B only
    assert res['events'] == [] and res['counts']['not_scored_touched_cells'] == {'not_designated': 3}


def test_offline_patch_through_source_rows_and_stride():
    # DB unknown on A; the patch says R on GT rows 5..6. Source rows 10..13 with stride 2 -> GT rows 5, 5, 6, 6.
    tlc = make_tlc([['UNKNOWN', 'RED']] * 4, rows=[[r, r] for r in (10, 11, 12, 13)], stride=2)
    doc = patch(((5, 6, 'R'),))
    res, d = score(tlc, [ON_A] * 4, doc=doc, sha='s1')
    assert res['patch_applied'] == 'offline' and res['online_check'] is None
    assert d['patched'] == {(0, 0), (1, 0), (2, 0), (3, 0)} and d['gt_row'][(3, 0)] == 6
    assert res['charged_count'] == 1 and res['events'][0]['colour_source'] == 'camera'
    assert res['events'][0]['gt_rows'] == [5, 6]
    # the same run with the patch only on GT row 5: two contact cells from the patch
    res, d = score(tlc, [ON_A] * 4, doc=patch(((5, 5, 'R'),)))
    assert d['patched'] == {(0, 0), (1, 0)} and d['why_not'][(2, 0)] == 'unknown'
    assert res['events'][0]['colour_source'] == 'camera' and res['events'][0]['contacts'] == 2


def test_exclusions_row_minus_one_held_past_999_unknown_not_showable():
    rows = [[-1, -1], [5, 5], [6, 6], [1000, 1000], [7, 7]]
    held = [[False, False], [True, True], [False, False], [False, False], [False, False]]
    tlc = make_tlc([['RED', 'RED'], ['RED', 'RED'], ['UNKNOWN', 'RED'], ['RED', 'RED'], ['GREEN', 'RED']],
                   rows=rows, held=held)
    res, d = score(tlc, [ON_A] * 5, desig=designated(show_g=False))
    why = {r: d['why_not'].get((r, 0)) for r in range(5)}
    assert why == {0: 'row_before_open', 1: 'row_past_log', 2: 'unknown', 3: 'row_past_log', 4: 'not_showable:G'}
    assert res['events'] == [] and res['counts']['scored_cells'] == 0
    # a red the heads cannot show is not scored either
    tlc = make_tlc([['RED', 'RED']] * 3)
    res, d = score(tlc, [ON_A] * 3, desig=designated(show_r=False))
    assert res['events'] == [] and d['why_not'][(0, 0)] == 'not_showable:R'


def test_no_nearside_waiver_on_designated_signals():
    tlc = make_tlc([['RED', 'RED']] * 9)
    tlc.update(apply_nearside_turn_filter=True, connector_maneuver={'A': 'RIGHT'},
               connector_paths={'A': [[0.0, 1.0], [2.0, 1.0]]})
    ego = [box(x - 0.2, 0.8, x + 0.2, 1.2) for x in (0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.75)]
    v3 = score_tlc(tlc, ego, tlc['sim_steps'])
    assert v3['tl_violation_count'] == 0 and v3['tlc_nearside_turn_waived_event_count'] == 1   # v3 waives it
    res, _ = score(tlc, ego)
    assert res['charged_count'] == 1 and res['events'][0]['nearside_waived'] is False


def test_contact_rule_is_v2_full_overlap():
    tlc = make_tlc([['RED', 'RED']] * 2)
    res, d = score(tlc, [box(1.5, 0.5, 2.5, 1.5)] * 2)             # half outside A
    assert res['events'] == [] and res['counts']['touches_below_overlap'] == 2
    assert P.evaluate(tlc, [box(1.5, 0.5, 2.5, 1.5)] * 2, tlc['sim_steps'], 'odyssey_scene060', designated(), None,
                      overlap_min=0.5)[0]['charged_count'] == 1


# ---- lateral margin: connectors widened sideways, never behind a stop line nor past their exit ---------------
LANE_A = box(0, 0, 3.5, 20)                    # a 3.5 m lane entered at y = 0 (stop line), left at y = 20
PATH_A = [[1.75, 0.0], [1.75, 20.0]]


def lane_tlc(polygons, paths, rbs=('rb1', 'rb2')):
    tlc = make_tlc([['RED', 'RED']] * 2, rbs=rbs)
    tlc['connector_wkb'].update({c: p.wkb_hex for c, p in polygons.items()})
    tlc['connector_paths'] = dict(paths)
    return tlc


def car(x0, y0):
    return box(x0, y0, x0 + 2.3, y0 + 5.2)


def test_lateral_zones_widen_sideways_and_keep_the_stop_line_and_exit():
    zones, missing = P._lateral_zones(['A'], {'A': LANE_A}, {'A': PATH_A}, 4.0)
    assert zones['A'].bounds == pytest.approx((-4.0, 0.0, 7.5, 20.0), abs=1e-6) and missing == []
    assert P._lateral_zones(['A'], {'A': LANE_A}, {'A': PATH_A}, 0.0) == ({'A': LANE_A}, [])
    for path in (None, [[1.75, 0.0]], [[1.75, 0.0], [1.75, 0.0]], [[1.75, float('nan')], [1.75, 20.0]]):
        assert P._lateral_zones(['A'], {'A': LANE_A}, {'A': path}, 4.0) == ({'A': LANE_A}, ['A'])


def test_stop_line_is_where_the_baseline_enters_the_polygon():
    # A baseline can start several metres before its polygon, where the nearest edge is not reliably the stop line
    path = [[3.5, -8.0], [1.75, 0.0], [1.75, 20.0]]                          # starts off a corner, enters at y = 0
    zones, missing = P._lateral_zones(['A'], {'A': LANE_A}, {'A': path}, 4.0)
    assert zones['A'].bounds == pytest.approx((-4.0, 0.0, 7.5, 20.0), abs=1e-6) and missing == []


@pytest.mark.parametrize('path', [[[4.5, -3.0], [1.75, 10.0], [1.75, 20.0]],     # enters through the side x = 3.5
                                  [[3.5, 0.0], [1.75, 10.0], [1.75, 20.0]],      # enters at a corner
                                  [[1.75, 5.0], [1.75, 15.0], [1.75, 5.0]]])     # leaves where it came in
def test_baseline_that_does_not_cross_a_stop_line_is_not_widened(path):
    tlc = lane_tlc({'A': LANE_A}, {'A': path})
    res = score(tlc, [car(-1.5, -3.0)] * 2)[0]                              # 3 m of the car behind the stop line
    assert res['events'] == [] and res['unwidened_connectors'] == ['A']


def test_lateral_zone_at_utm_scale():
    def utm(g):
        return affinity.translate(affinity.rotate(g, 37.0, origin=(0, 0)), 331000.0, 4690000.0)
    path = [list(utm(Point(xy)).coords[0]) for xy in PATH_A]
    zone = P._lateral_zones(['A'], {'A': utm(LANE_A)}, {'A': path}, 4.0)[0]['A']
    assert zone.difference(utm(box(-4.0, 0.0, 7.5, 20.0))).area < 1e-6     # not behind the stop line, not past 4 m
    assert utm(box(-3.99, 0.01, 7.49, 19.99)).difference(zone).area < 1e-6  # and the full 4 m on both sides


def test_lateral_zone_on_a_turn_stays_between_its_own_end_edges():
    ts = [i * math.pi / 40 for i in range(21)]                              # a quarter turn: +y, then -x
    outer = [(11.75 * math.cos(t), 11.75 * math.sin(t)) for t in ts]
    inner = [(8.25 * math.cos(t), 8.25 * math.sin(t)) for t in reversed(ts)]
    turn = Polygon(outer + inner)                                            # stop line on y = 0, exit on x = 0
    path = [[10 * math.cos(t), 10 * math.sin(t)] for t in ts]
    zone = P._lateral_zones(['T'], {'T': turn}, {'T': path}, 4.0)[0]['T']
    assert zone.intersection(box(-50, -50, 50, 0)).area < 1e-9               # nothing behind the stop line
    assert zone.intersection(box(-50, -50, 0, 50)).area < 1e-9               # nothing past the exit edge
    assert zone.bounds[2] == pytest.approx(11.75 + 4.0, abs=1e-2)            # outer side widened by 4 m (arc chords)


def test_margin_never_reaches_behind_a_neighbouring_stop_line():
    a, b = LANE_A, box(3.5, 1.0, 7.0, 20)                                    # B's stop line is 1 m further on
    tlc = lane_tlc({'A': a, 'B': b}, {'A': PATH_A, 'B': [[5.25, 1.0], [5.25, 20.0]]}, rbs=('rb1', 'rb1'))
    desig = designated(conns=('A', 'B'))
    assert score(tlc, [car(-1.0, 0.5)] * 2, desig)[0]['events'] == []       # past A's stop line, not B's
    assert score(tlc, [car(-1.0, 1.0)] * 2, desig)[0]['charged_count'] == 1  # past both


def test_roadblock_with_an_unlocatable_stop_line_is_not_widened_at_all():
    b = box(3.5, 1.0, 7.0, 20)                                               # B's stop line is 1 m further on
    tlc = lane_tlc({'A': LANE_A, 'B': b}, {'A': PATH_A}, rbs=('rb1', 'rb1'))  # ... and B has no baseline
    res = score(tlc, [car(-1.0, 0.5)] * 2, designated(conns=('A', 'B')))[0]
    assert res['events'] == [] and res['unwidened_connectors'] == ['A', 'B']


def test_car_over_the_lane_edge_is_charged_only_with_the_margin():
    tlc = lane_tlc({'A': LANE_A}, {'A': PATH_A})
    ego = [car(2.5, 5.0)] * 2                                                # 1.3 m beyond the right edge of A
    assert score(tlc, ego)[0]['charged_count'] == 1
    res = P.evaluate(tlc, ego, tlc['sim_steps'], 'odyssey_scene060', designated(), None, lateral_margin=0)[0]
    assert res['events'] == [] and res['counts']['touches_below_overlap'] == 2


def test_car_in_the_margin_is_charged_without_touching_the_connector():
    tlc = lane_tlc({'A': LANE_A}, {'A': PATH_A})
    beside = [car(4.0, 5.0)] * 2                                             # next lane: 0.5 m clear of A
    res, d = score(tlc, beside)
    assert res['charged_count'] == 1 and res['events'][0]['contacts'] == 2 and not d['touch']
    assert P.evaluate(tlc, beside, tlc['sim_steps'], 'odyssey_scene060', designated(), None, lateral_margin=0)[0]['events'] == []
    assert score(tlc, [car(6.0, 5.0)] * 2)[0]['events'] == []              # 0.8 m beyond the 4 m margin


def test_margin_does_not_move_the_stop_line_or_the_exit():
    tlc = lane_tlc({'A': LANE_A}, {'A': PATH_A})
    assert score(tlc, [car(0.6, -1.0)] * 2)[0]['events'] == []             # 1 m of the car behind the stop line
    assert score(tlc, [car(0.6, 16.0)] * 2)[0]['events'] == []             # 1.2 m of the car past the exit
    assert score(tlc, [car(0.6, 0.0)] * 2)[0]['charged_count'] == 1        # wholly past the stop line


def test_margin_seals_the_gap_between_connectors_of_one_roadblock():
    a, b = box(0, 0, 3.4, 20), box(3.5, 0, 7.0, 20)                          # 10 cm apart
    tlc = lane_tlc({'A': a, 'B': b}, {'A': [[1.7, 0.0], [1.7, 20.0]], 'B': [[5.25, 0.0], [5.25, 20.0]]},
                   rbs=('rb1', 'rb1'))
    ego, desig = [car(2.3, 5.0)] * 2, designated(conns=('A', 'B'))          # straddles the gap
    assert P.evaluate(tlc, ego, tlc['sim_steps'], 'odyssey_scene060', desig, None, lateral_margin=0)[0]['events'] == []
    res = score(tlc, ego, desig)[0]
    assert res['charged_count'] == 1 and res['unwidened_connectors'] == []


def test_connector_without_a_pinned_path_is_not_widened_and_is_named():
    tlc = lane_tlc({'A': LANE_A}, {})
    res = score(tlc, [car(2.5, 5.0)] * 2)[0]
    assert res['events'] == [] and res['unwidened_connectors'] == ['A'] and res['lateral_margin_m'] == 4.0


@pytest.mark.parametrize('margin', [-1.0, float('nan'), float('inf')])
def test_invalid_lateral_margin_is_refused(margin):
    tlc = lane_tlc({'A': LANE_A}, {'A': PATH_A})
    with pytest.raises(ValueError, match='lateral_margin'):
        P.evaluate(tlc, [ON_A] * 2, tlc['sim_steps'], 'odyssey_scene060', designated(), None, lateral_margin=margin)


def test_patch_for_another_scene_is_refused():
    with pytest.raises(ValueError, match='not odyssey_scene060'):
        score(make_tlc([['RED', 'RED']]), [ON_A], doc=patch(scene='odyssey_scene075'))


# ---- offline == online on a synthetic run through the real snapshot and pack ----------------------------------
def scene():
    return {'id': 'odyssey_scene060', 'cadence': SimpleNamespace(sim_dt=0.1),
            'dynamic_map_states': {
                'a': dict(type='TRAFFIC_LIGHT', traffic_light_lane='A',
                          state={'traffic_light_state': [U] * 6 + [G] * 4 + [R] * 10}),
                'b': dict(type='TRAFFIC_LIGHT', traffic_light_lane='B', state={'traffic_light_state': [R] * 20})}}


def converter(s, rows):
    c = OdysseyToNuPlanConverter.__new__(OdysseyToNuPlanConverter)
    c.base_timestamp = 0
    c.scene = s
    c.engine = SimpleNamespace(agent_manager=SimpleNamespace(traffic_light_source_row=rows), managers={})
    return c


def run(s, steps, rows):
    c = converter(s, rows)
    snaps = [capture_tlc_signal_snapshot(s, c, st) for st in steps]
    lanes = {'A': SimpleNamespace(polygon=box(0, 0, 2, 2)), 'B': SimpleNamespace(polygon=box(10, 0, 12, 2))}
    blocks = {'rb1': SimpleNamespace(interior_edges=[SimpleNamespace(id='A')]),
              'rb2': SimpleNamespace(interior_edges=[SimpleNamespace(id='B')])}
    return pack_tlc(s, lanes, steps, .1, 1, None, None, {}, route_roadblock_dict=blocks, signal_snapshots=snaps)


@pytest.mark.parametrize('clock', ['identity', 'sector'])
def test_offline_equals_online(clock):
    doc = patch(((0, 3, 'R'), (4, 7, 'G'), (8, 9, 'R')))          # rows 0..5 DB unknown, 6..9 DB green
    steps = list(range(0, 18))
    rows = (lambda conn, step: step) if clock == 'identity' else (lambda conn, step: step - 3)   # -1.. before open
    ego = [ON_A if st % 5 else OFF for st in steps]
    online_scene = scene()
    SP.apply_to_scene(online_scene, doc, 'sha1', 'p')
    online = run(online_scene, steps, rows)
    offline = run(scene(), steps, rows)
    assert 'signal_patched' in online and 'signal_patched' not in offline
    ron, don = P.evaluate(online, ego, steps, 'odyssey_scene060', designated(), doc, 'sha1')
    roff, doff = P.evaluate(offline, ego, steps, 'odyssey_scene060', designated(), doc, 'sha1')
    assert ron['patch_applied'] == 'online' and roff['patch_applied'] == 'offline'
    assert ron['online_check']['ok'] and ron['online_check']['cells_patched'] > 0
    for k in ('code', 'why_not', 'patched', 'contact', 'touch'):
        assert don[k] == doff[k], k
    strip = ('patch_applied', 'online_check')
    assert {k: v for k, v in ron.items() if k not in strip} == {k: v for k, v in roff.items() if k not in strip}
    assert ron['events'] and ron['charged_count'] >= 1


def test_online_check_is_loud():
    doc = patch(((0, 3, 'R'),))
    s = scene()
    SP.apply_to_scene(s, doc, 'sha1', 'p')
    tlc = run(s, list(range(0, 6)), lambda conn, step: step)
    res, _ = P.evaluate(tlc, [ON_A] * 6, list(range(6)), 'odyssey_scene060', designated(), doc, 'other-sha')
    assert res['online_check']['ok'] is False and res['online_check']['sha_matches'] is False
    res, _ = P.evaluate(tlc, [ON_A] * 6, list(range(6)), 'odyssey_scene060', designated(),
                        patch(((0, 4, 'G'),)), 'sha1')
    chk = res['online_check']
    assert not chk['ok'] and chk['n_colour_mismatch'] == 4 and chk['n_unflagged'] == 1


def test_offline_equals_online_on_an_upsampled_scene():
    """Review Minor 1: S = 2 (0.2 s scene at 0.1 s steps). Online the patch writes GT row r to scene rows 2r, 2r+1;
    offline the scorer maps a recorded source row back with row // scene_frame_stride. They must agree."""
    def up():
        s = scene()
        for item in s['dynamic_map_states'].values():
            series = item['state']['traffic_light_state']
            item['state']['traffic_light_state'] = [series[min(i // 2, len(series) - 1)] for i in range(2 * len(series) - 1)]
        return s
    doc = patch(((1, 3, 'R'), (4, 6, 'G'), (7, 9, 'R')))
    steps = list(range(0, 30))
    rows = lambda conn, step: step                                 # scene rows (upsampled), identity clock
    ego = [ON_A if st % 4 else OFF for st in steps]

    def run2(s):
        c = converter(s, rows)
        snaps = [capture_tlc_signal_snapshot(s, c, st) for st in steps]
        lanes = {'A': SimpleNamespace(polygon=box(0, 0, 2, 2)), 'B': SimpleNamespace(polygon=box(10, 0, 12, 2))}
        blocks = {'rb1': SimpleNamespace(interior_edges=[SimpleNamespace(id='A')]),
                  'rb2': SimpleNamespace(interior_edges=[SimpleNamespace(id='B')])}
        return pack_tlc(s, lanes, steps, .1, 2, None, None, {}, route_roadblock_dict=blocks, signal_snapshots=snaps)
    online_scene = up()
    rec = SP.apply_to_scene(online_scene, doc, 'sha1', 'p', row_scale=2)
    assert rec['rows']['A'][0] == [2, 7]                           # GT rows 1..3 -> scene rows 2..7
    online, offline = run2(online_scene), run2(up())
    assert online['scene_frame_stride'] == rec['row_scale'] == 2
    ron, don = P.evaluate(online, ego, steps, 'odyssey_scene060', designated(), doc, 'sha1')
    roff, doff = P.evaluate(offline, ego, steps, 'odyssey_scene060', designated(), doc, 'sha1')
    assert ron['online_check']['ok'] and ron['online_check']['cells_patched'] == 18
    for k in ('code', 'why_not', 'patched', 'contact'):
        assert don[k] == doff[k], k
    assert doff['gt_row'][(3, 0)] == 1 and (1, 0) not in doff['patched'] and (2, 0) in doff['patched']
    strip = ('patch_applied', 'online_check')
    assert {k: v for k, v in ron.items() if k not in strip} == {k: v for k, v in roff.items() if k not in strip}
    assert ron['charged_count'] >= 1
