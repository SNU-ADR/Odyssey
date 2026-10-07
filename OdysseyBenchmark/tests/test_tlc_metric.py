import json
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import box

from odyssey_benchmark.tlc_metric import (CODE_TABLE, LEGACY_VERSION, VERSION,
                                            capture_tlc_signal_snapshot, pack_tlc, score_tlc)
from odyssey_benchmark.ds_formula import route_ds


def decode(packed, key):
    table = packed['signal_code_table']
    return [[table[c] for c in row] for row in packed[key]]


def pack_from_live(scene, converter, route_lane_dict, dense_steps, *args, **kwargs):
    snapshots = [capture_tlc_signal_snapshot(scene, converter, step)
                 for step in dense_steps]
    return pack_tlc(scene, route_lane_dict, dense_steps, *args,
                    signal_snapshots=snapshots, **kwargs)


def snapshot_v2(n, contact, flags=None, sample_dt=.1, connectors=('42',), start=15):
    """n sampled steps at ``sample_dt``; ``contact(i)`` says whether the ego overlaps at row i."""
    steps = [start + i for i in range(n)]
    code = CODE_TABLE.index('RED')
    data = dict(version=VERSION, sample_dt=sample_dt, scene_frame_stride=1, sim_steps=steps,
                signal_code_table=list(CODE_TABLE), signal_connector_ids=list(connectors),
                signal_states_all=[[code] * len(connectors)] * n,
                signal_held_all=[[False] * len(connectors)] * n,
                connector_ids=list(connectors),
                connector_wkb={c: box(0, 0, 2, 2).wkb_hex for c in connectors},
                signal_states=[[code] * len(connectors)] * n,
                signal_held=[[False] * len(connectors)] * n,
                red_renderable={c: (flags or {}).get(c, [None] * n) for c in connectors},
                gt_rear_xy=None, gt_heading=None, vehicle={}, apply_gt_filter=False)
    ego = [box(0, 0, 1, 1) if contact(i) else box(10, 10, 11, 11) for i in range(n)]
    return data, ego


def snapshot(flags=None, apply_gt_filter=False):
    flags = flags or {}
    return dict(version='tl_events_v1', score_stride=5,
        sim_steps=[15, 20, 25, 30, 35, 40], connector_ids=['42'],
        signal_connector_ids=['42'], signal_states_all=[['RED']] * 6,
        signal_held_all=[[False]] * 6,
        connector_wkb={'42': box(0, 0, 2, 2).wkb_hex},
        signal_states=[['RED']] * 6, signal_held=[[False]] * 6,
        red_renderable={'42': flags.get('42', [])}, gt_rear_xy=None,
        gt_heading=None, vehicle={}, apply_gt_filter=apply_gt_filter)


def test_continuous_overlap_counts_one_event_not_one_per_frame():
    data = snapshot()
    ego = [box(0, 0, 1, 1), box(0, 0, 1, 1), box(10, 10, 11, 11),
           box(0, 0, 1, 1), box(0, 0, 1, 1), box(10, 10, 11, 11)]
    result = score_tlc(data, ego, data['sim_steps'])
    assert result['tlc_red_contact_count'] == 4
    assert result['tlc_raw_event_count'] == result['tl_violation_count'] == 2
    assert result['P_TL'] == pytest.approx(.49)
    assert result['first_tlc_violation_step'] == 15
    assert [(e['first_step'], e['last_step']) for e in json.loads(result['tlc_events_json'])] == [
        (15, 20), (30, 35)]


def test_renderer_waives_only_the_event_it_cannot_show():
    data = snapshot({'42': [None, None, None, False, False, None, True, True, None]})
    ego = [box(0, 0, 1, 1), box(0, 0, 1, 1), box(10, 10, 11, 11),
           box(0, 0, 1, 1), box(0, 0, 1, 1), box(10, 10, 11, 11)]
    result = score_tlc(data, ego, data['sim_steps'])
    assert result['tlc_raw_event_count'] == 2
    assert result['tlc_render_waived_event_count'] == 1
    assert result['tl_violation_count'] == 1
    assert result['P_TL'] == pytest.approx(.7)


def two_connector_v2(roadblocks=None):
    """One red entry touching two route connectors on the same steps."""
    data, ego = snapshot_v2(6, lambda i: i < 2, connectors=('42', '43'))
    if roadblocks is not None:
        data['connector_roadblock'] = roadblocks
    return data, ego


def test_two_connectors_of_one_intersection_are_one_event():
    """nuPlan splits a wide turn across overlapping connectors; that is still one entry."""
    data, ego = two_connector_v2({'42': '900', '43': '900'})
    result = score_tlc(data, ego, data['sim_steps'])
    assert result['tl_violation_count'] == 1
    assert result['P_TL'] == pytest.approx(.7)
    event, = json.loads(result['tlc_events_json'])
    assert event['roadblock'] == '900'
    assert sorted(event['connectors']) == ['42', '43']


def test_connectors_of_different_intersections_are_separate_events():
    data, ego = two_connector_v2({'42': '900', '43': '901'})
    result = score_tlc(data, ego, data['sim_steps'])
    assert result['tl_violation_count'] == 2
    assert result['P_TL'] == pytest.approx(.49)


def test_unpinned_snapshot_keeps_the_per_connector_count():
    """No pinned groups must not merge connectors it knows nothing about."""
    data, ego = two_connector_v2()
    assert 'connector_roadblock' not in data
    result = score_tlc(data, ego, data['sim_steps'])
    assert result['tl_violation_count'] == 2


def test_merged_event_is_waived_only_when_no_member_could_show_red():
    """A missing/true flag on ANY connector of the intersection blocks the waiver."""
    data, ego = two_connector_v2({'42': '900', '43': '900'})
    data['red_renderable'] = {'42': [False] * 6, '43': [None] * 6}
    assert score_tlc(data, ego, data['sim_steps'])['tl_violation_count'] == 1
    data['red_renderable'] = {'42': [False] * 6, '43': [False] * 6}
    result = score_tlc(data, ego, data['sim_steps'])
    assert result['tl_violation_count'] == 0
    assert result['tlc_render_waived_event_count'] == 1


def test_pack_tlc_rejects_reconstruction_without_live_snapshots():
    scene = dict(id='scene', dynamic_map_states={})
    lane = SimpleNamespace(polygon=box(0, 0, 2, 2))
    with pytest.raises(ValueError, match='rollout-captured'):
        pack_tlc(scene, {'42': lane}, [15], .1, 1, None, None, {})


def test_snapshot_uses_simulation_steps_and_pins_route_polygons():
    calls = []
    class Converter:
        def convert_to_traffic_lights(self, step):
            calls.append(step)
            return [SimpleNamespace(lane_connector_id='42',
                                    status=SimpleNamespace(name='RED' if step >= 20 else 'GREEN'))]
    scene = dict(id='scene', dynamic_map_states={'x': dict(type='TRAFFIC_LIGHT',
        traffic_light_lane='42', state={'traffic_light_state': ['GREEN'] * 23})})
    lane = SimpleNamespace(polygon=box(0, 0, 2, 2))
    steps = list(range(15, 31))
    packed = pack_from_live(scene, Converter(), {'42': lane}, steps, 0.1, 5,
                      None, None, {}, apply_gt_filter=False)
    assert calls == steps                       # no thinning: the signal is queried every step
    assert decode(packed, 'signal_states') == [['GREEN']] * 5 + [['RED']] * 11
    assert packed['signal_held'][:8] == [[False]] * 8     # the record has 23 frames, so held from step 23
    assert packed['signal_held'][8] == [True] and packed['signal_held'][-1] == [True]
    assert packed['connector_wkb']['42'] == lane.polygon.wkb_hex
    assert packed['sample_dt'] == 0.1 and packed['sim_steps'] == steps


def test_snapshot_archives_connector_source_rows_and_distinguishes_inactive_from_exhausted():
    class Converter:
        def traffic_light_source_row(self, connector, step):
            return {0: -1, 1: 0, 2: 2}[step]

        def convert_to_traffic_lights(self, step):
            name = {0: 'UNKNOWN', 1: 'GREEN', 2: 'UNKNOWN'}[step]
            return [SimpleNamespace(lane_connector_id='42',
                                    status=SimpleNamespace(name=name))]

    scene = dict(id='scene', dynamic_map_states={'x': dict(
        type='TRAFFIC_LIGHT', traffic_light_lane='42',
        state={'traffic_light_state': ['GREEN', 'RED']})})
    lane = SimpleNamespace(polygon=box(0, 0, 2, 2))
    packed = pack_from_live(scene, Converter(), {'42': lane}, [0, 1, 2], .1, 1,
                      None, None, {})

    assert packed['signal_source_rows'] == [[-1], [0], [2]]
    assert packed['signal_inactive'] == [[True], [False], [False]]
    assert packed['signal_held'] == [[False], [False], [True]]


def test_actual_entry_lane_signal_replaces_route_connector_colour_but_not_contact_polygon():
    class Converter:
        def traffic_light_source_row(self, connector, step):
            return int(step) + (100 if connector == 'actual' else 0)

        def convert_to_traffic_lights(self, step):
            return [
                SimpleNamespace(lane_connector_id='planned',
                                status=SimpleNamespace(name='GREEN')),
                SimpleNamespace(lane_connector_id='actual',
                                status=SimpleNamespace(name='RED')),
            ]

    scene = dict(id='scene', dynamic_map_states={
        'p': dict(type='TRAFFIC_LIGHT', traffic_light_lane='planned',
                  state={'traffic_light_state': ['GREEN'] * 200}),
        'a': dict(type='TRAFFIC_LIGHT', traffic_light_lane='actual',
                  state={'traffic_light_state': ['RED'] * 200}),
    })
    planned_polygon = box(0, 0, 2, 2)
    lane = SimpleNamespace(polygon=planned_polygon)
    packed = pack_from_live(
        scene, Converter(), {'planned': lane}, [0, 1], .1, 1,
        None, None, {}, signal_source_overrides={1: {'planned': 'actual'}})

    # The route/contact geometry remains the planned connector, while only its signal source
    # changes on the step where lane matching selected the sibling entry lane.
    assert packed['connector_ids'] == ['planned']
    assert packed['connector_wkb']['planned'] == planned_polygon.wkb_hex
    assert decode(packed, 'signal_states') == [['GREEN'], ['RED']]
    assert packed['signal_state_connector_ids'] == [['planned'], ['actual']]
    assert packed['signal_source_rows'] == [[0], [101]]

    ego = [box(10, 10, 11, 11), box(0, 0, 1, 1)]
    result = score_tlc(packed, ego, [0, 1])
    assert result['tl_violation_count'] == 1


def test_route_ds_multiplies_a_measured_tl_factor_and_treats_an_unapplied_rule_as_one():
    row = dict(RC=.8, P_col=1., P_off=1., P_SD=1., P_TL=.49)
    assert route_ds(row) == pytest.approx(39.2)
    row.pop('P_TL')                         # the rule was not applied: no penalty, not unscored
    assert route_ds(row) == pytest.approx(80.)
    row['P_TL'] = 1.4                       # a factor outside [0, 1] is never multiplied in
    assert route_ds(row) is None
    row['P_TL'] = ''                        # an empty rule column means not applied
    assert route_ds(row) == pytest.approx(80.)


def test_same_violation_scores_the_same_at_0_5_and_0_1():
    """Cadence must not change what a violation is worth -- only how finely it is observed."""
    coarse = snapshot_v2(8, lambda i: 2 <= i <= 5, sample_dt=.5)        # 2.0 s of overlap
    fine = snapshot_v2(40, lambda i: 10 <= i <= 29, sample_dt=.1)       # the same 2.0 s
    a, b = score_tlc(*coarse, coarse[0]['sim_steps']), score_tlc(*fine, fine[0]['sim_steps'])
    assert a['tl_violation_count'] == b['tl_violation_count'] == 1
    assert a['P_TL'] == b['P_TL'] == pytest.approx(.7)
    assert a['tlc_red_contact_count'] == 4 and b['tlc_red_contact_count'] == 20
    assert a['tlc_red_contact_seconds'] == b['tlc_red_contact_seconds'] == pytest.approx(2.0)
    assert a['tlc_sample_dt'] == .5 and b['tlc_sample_dt'] == .1


def test_one_connector_is_one_violation_however_often_contact_breaks():
    data, ego = snapshot_v2(9, lambda i: i in (0, 1, 4, 5, 8))
    result = score_tlc(data, ego, data['sim_steps'])
    assert result['tlc_raw_event_count'] == result['tl_violation_count'] == 1
    assert result['tlc_red_contact_count'] == 5
    event = json.loads(result['tlc_events_json'])[0]
    assert (event['first_step'], event['last_step'], event['contacts']) == (15, 23, 5)


def test_separate_connectors_stay_separate_violations():
    data, ego = snapshot_v2(4, lambda i: True, connectors=('42', '43'))
    result = score_tlc(data, ego, data['sim_steps'])
    assert result['tlc_raw_event_count'] == result['tl_violation_count'] == 2
    assert result['P_TL'] == pytest.approx(.49)


def test_renderer_waiver_reads_the_flag_of_the_contact_step():
    """Flags arrive already on the step axis, so row i is the flag for sim_steps[i]."""
    unrenderable = snapshot_v2(4, lambda i: True, flags={'42': [False] * 4})
    partly = snapshot_v2(4, lambda i: True, flags={'42': [False, False, True, None]})
    waived = score_tlc(*unrenderable, unrenderable[0]['sim_steps'])
    kept = score_tlc(*partly, partly[0]['sim_steps'])
    assert waived['tl_violation_count'] == 0 and waived['tlc_filter_reason'] == 'rendering_sanity'
    assert kept['tl_violation_count'] == 1 and kept['tlc_filter_reason'] == 'unwaived_violation'
    assert kept['tlc_render_waived_contact_count'] == 2


def test_v1_archive_keeps_v1_rules_and_says_so():
    """An old snapshot is scored the way it was written: 0.5 s, and a gap splits the event."""
    data = snapshot()
    ego = [box(0, 0, 1, 1), box(0, 0, 1, 1), box(10, 10, 11, 11),
           box(0, 0, 1, 1), box(0, 0, 1, 1), box(10, 10, 11, 11)]
    result = score_tlc(data, ego, data['sim_steps'])
    assert result['tlc_version'] == LEGACY_VERSION
    assert result['tlc_sample_dt'] == .5
    assert result['tlc_raw_event_count'] == 2                 # v2 would call this one violation
    assert result['tlc_red_contact_seconds'] == pytest.approx(2.0)


def test_gt_poses_must_be_one_per_sampled_step():
    lane = SimpleNamespace(polygon=box(0, 0, 2, 2))
    scene = dict(id='scene', dynamic_map_states={})
    class Converter:
        def convert_to_traffic_lights(self, step):
            return []
    steps = list(range(15, 21))
    with pytest.raises(ValueError, match='one pose per sampled step'):
        pack_from_live(scene, Converter(), {'42': lane}, steps, .1, 1,
                 np.zeros((3, 2)), np.zeros(3), {})


def test_renderer_flags_are_converted_from_scene_frames_once():
    """A 0.5 s scene under a 0.1 s rollout: step 15 is scene frame 3, not score frame 3."""
    lane = SimpleNamespace(polygon=box(0, 0, 2, 2))
    scene = dict(id='scene', traffic_light_red_renderable={'42': [True, True, True, False, None]},
                 dynamic_map_states={'x': dict(type='TRAFFIC_LIGHT', traffic_light_lane='42',
                                               state={'traffic_light_state': ['RED'] * 25})})
    class Converter:
        def convert_to_traffic_lights(self, step):
            return [SimpleNamespace(lane_connector_id='42', status=SimpleNamespace(name='RED'))]
    steps = [15, 16, 20, 21]
    packed = pack_from_live(scene, Converter(), {'42': lane}, steps, .1, 5, None, None, {})
    assert packed['red_renderable']['42'] == [False, False, None, None]   # frames 3, 3, 4, 4


def test_capture_is_the_exact_dense_trace_used_for_scalar_score():
    """NPZ-facing trace and CSV-facing scalars must come from one score_tlc call."""
    data, ego = snapshot_v2(
        6, lambda i: i in (1, 2, 4),
        flags={'42': [None, False, False, None, True, None]},
    )
    capture = {}
    result = score_tlc(data, ego, data['sim_steps'], capture=capture)

    assert int(capture['tlc_trace_version']) == 4
    assert bool(capture['tlc_scorer_trace_valid'])
    np.testing.assert_array_equal(capture['tlc_trace_sim_steps'], data['sim_steps'])
    np.testing.assert_array_equal(
        capture['tlc_signal_source_rows'], np.asarray(data['sim_steps'])[:, None]
    )
    assert not capture['tlc_signal_inactive'].any()
    np.testing.assert_array_equal(capture['tlc_contact_sim_steps'], [16, 17, 19])
    np.testing.assert_array_equal(capture['tlc_contact_connector_ids'], ['42', '42', '42'])
    np.testing.assert_array_equal(capture['tlc_contact_render_flags'], [0, 0, 1])
    assert json.loads(str(capture['tlc_events_json'])) == json.loads(result['tlc_events_json'])
    assert len(capture['tlc_contact_sim_steps']) == result['tlc_red_contact_count']


def branch_case(maneuver, map_location='us-ma-boston', reaches_exit=True):
    """One raw contact followed by an observed choice at a fork."""
    if maneuver == 'RIGHT':
        points = [(0, 0), (1, 0), (2, 0), (3, -1), (3, -2), (3, -4)]
    else:
        points = [(0, 0), (1, 0), (2, 0), (3, 1), (3, 2), (3, 4)]
    data, _ = snapshot_v2(6, lambda _: False)
    data.update(
        connector_wkb={'42': box(-1, -1, 2, 1).wkb_hex},
        connector_paths={'42': points},
        connector_maneuver={'42': maneuver},
        map_location=map_location,
        apply_nearside_turn_filter=True,
    )
    driven = points if reaches_exit else points[:2] + [(1.2, 0), (1.4, 0), (1.6, 0), (1.8, 0)]
    ego = [box(x - .2, y - .2, x + .2, y + .2) for x, y in driven]
    return data, ego


def test_us_right_exit_keeps_raw_contact_but_waives_filtered_event():
    data, ego = branch_case('RIGHT')
    result = score_tlc(data, ego, data['sim_steps'])
    event, = json.loads(result['tlc_events_json'])
    assert result['tlc_raw_event_count'] == 1
    assert result['tlc_nearside_turn_waived_event_count'] == 1
    assert result['tl_violation_count'] == 0
    assert result['tlc_filter_reason'] == 'nearside_turn'
    assert event['selected_connector'] == '42'
    assert event['selected_maneuver'] == 'RIGHT'
    assert event['nearside_turn_waived'] is True


def test_us_left_exit_remains_a_violation():
    data, ego = branch_case('LEFT')
    result = score_tlc(data, ego, data['sim_steps'])
    assert result['tlc_raw_event_count'] == result['tl_violation_count'] == 1
    assert result['tlc_nearside_turn_waived_event_count'] == 0


def test_singapore_left_exit_is_the_nearside_turn():
    data, ego = branch_case('LEFT', map_location='sg-one-north')
    result = score_tlc(data, ego, data['sim_steps'])
    assert result['tlc_raw_event_count'] == 1
    assert result['tl_violation_count'] == 0
    assert result['tlc_nearside_turn_waived_event_count'] == 1


def test_touching_turn_polygon_without_taking_its_exit_grants_no_waiver():
    data, ego = branch_case('RIGHT', reaches_exit=False)
    result = score_tlc(data, ego, data['sim_steps'])
    event, = json.loads(result['tlc_events_json'])
    assert result['tlc_raw_event_count'] == result['tl_violation_count'] == 1
    assert result['tlc_nearside_turn_waived_event_count'] == 0
    assert 'selected_connector' not in event
