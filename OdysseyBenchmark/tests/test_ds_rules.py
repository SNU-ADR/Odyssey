"""Record-level Driving Score rules (traffic-light set, lane choice): pinned with the run, applied by replay.

Scene-end scoring and every rescore go through driving_metrics.replay, so these rules have one home.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# the rules reach into the benchmark package (odyssey_benchmark.lane_follow, odyssey_benchmark.ds_formula)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from odyssey_benchmark import driving_metrics  # noqa: E402
from odyssey_benchmark.driving_metrics import PinnedInputs, apply_rules, make_rules  # noqa: E402

SET = dict(name='tlc_timetable', sha256='sha-now', dir='/data/tlc_timetable', scenes={'odyssey_scene011': {}, 'odyssey_scene021': {}})
TIMETABLE = dict(files={'odyssey_scene011': None})


@pytest.fixture
def fake_set(monkeypatch):
    monkeypatch.setattr(driving_metrics, '_tl_set_and_timetable', lambda rules: (SET, TIMETABLE))


def scored(**over):
    row = dict(P_TL=0.49, tl_violation_count=2)
    row.update(over)
    return row


def pinned(scene='odyssey_scene001'):
    return PinnedInputs(scene=scene, ds_states=np.zeros((2, 11)), ds_sim_steps=np.array([0, 1]))


DATA = dict(tlc={'pinned': True}, map={'vehicle': {}})
TT = dict(tl_set='tlc_timetable', tl_set_sha256='sha-now')


def test_pinned_inputs_look_like_a_loaded_npz():
    z = PinnedInputs(scene='odyssey_scene001', ds_states=np.zeros(1))
    assert sorted(z.files) == ['ds_states', 'scene'] and 'scene' in z.files


def test_no_rules_marks_events_and_leaves_the_rule_columns_empty():
    row = apply_rules(scored(tl_rule='timetable', P_PLC=0.49), pinned(), DATA, {})
    assert row['tl_rule'] == 'events' and row['P_TL'] == 0.49
    assert row['tl_set'] is None and row['P_PLC'] is None and row['plc_rule'] is None


def test_scene_outside_the_set_gets_full_traffic_light_credit(fake_set):
    row = apply_rules(scored(), pinned('odyssey_scene001'), DATA, TT)
    assert (row['tl_rule'], row['P_TL'], row['tl_violation_count']) == ('not_signal_scene', 1.0, 0)
    assert (row['tl_events_penalty'], row['tl_events_violation_count']) == (0.49, 2)
    assert (row['tl_set'], row['tl_set_sha256']) == ('tlc_timetable', 'sha-now')


def test_set_scene_without_a_designated_signal_is_full_credit(fake_set):
    row = apply_rules(scored(), pinned('odyssey_scene021'), DATA, TT)
    assert row['tl_rule'] == 'no_designated_signal' and row['P_TL'] == 1.0


def test_designated_scene_takes_the_timetable_score(fake_set, monkeypatch):
    from odyssey_benchmark import tlc_metric, tlc_timetable
    monkeypatch.setattr(tlc_metric, 'ego_polygons_from_states', lambda states, vehicle: ['ego'])
    monkeypatch.setattr(tlc_timetable, 'score', lambda tlc, ego, steps, code, tt: (dict(
        patch_applied='online', online_check={'ok': True}, p_tl=0.7, charged_count=1,
        tlc_filter_reason='unwaived_violation', coverage={'coverage': 0.8}), {}))
    row = apply_rules(scored(), pinned('odyssey_scene011'), DATA, TT)
    assert (row['tl_rule'], row['P_TL'], row['tl_violation_count']) == ('timetable', 0.7, 1)
    assert row['tl_timetable_coverage'] == 0.8


@pytest.mark.parametrize('result', [dict(patch_applied='offline', online_check=None),
                                    dict(patch_applied='online', online_check={'ok': False})])
def test_run_not_driven_with_the_timetable_is_refused(fake_set, monkeypatch, result):
    """Scoring against colours the run did not drive under would judge a different drive."""
    from odyssey_benchmark import tlc_metric, tlc_timetable
    monkeypatch.setattr(tlc_metric, 'ego_polygons_from_states', lambda states, vehicle: ['ego'])
    monkeypatch.setattr(tlc_timetable, 'score', lambda *a: (dict(result, p_tl=1.0, charged_count=0), {}))
    with pytest.raises(ValueError, match='not driven with the tlc_timetable timetable'):
        apply_rules(scored(), pinned('odyssey_scene011'), DATA, TT)


def test_designated_signal_without_its_lateral_margin_is_refused(fake_set, monkeypatch):
    """Scoring a designated signal without the lateral margin would score it by another rule."""
    from odyssey_benchmark import tlc_metric, tlc_timetable
    monkeypatch.setattr(tlc_metric, 'ego_polygons_from_states', lambda states, vehicle: ['ego'])
    monkeypatch.setattr(tlc_timetable, 'score', lambda *a: (dict(
        patch_applied='online', online_check={'ok': True}, p_tl=1.0, charged_count=0,
        unwidened_connectors=['49662'], lateral_margin_m=4.0), {}))
    with pytest.raises(ValueError, match='lateral margin cannot be applied'):
        apply_rules(scored(), pinned('odyssey_scene011'), DATA, TT)


def test_lane_rule_sets_p_lane_and_hands_the_result_back(monkeypatch):
    from odyssey_benchmark import lane_follow
    result = SimpleNamespace(n_fail=2, n_late=1, n_reached=5)
    monkeypatch.setattr(lane_follow, 'score_pinned', lambda z: result)
    monkeypatch.setattr(lane_follow, 'lane_penalty', lambda r: 0.441)
    capture = {}
    row = apply_rules(scored(), pinned(), DATA, {'lane': 'plc'}, capture)
    assert (row['plc_rule'], row['P_PLC']) == ('plc', 0.441)
    assert (row['plc_fail'], row['plc_late'], row['plc_reached']) == (2, 1, 5)
    assert capture['lane_result'] is result


def test_a_changed_set_is_refused(monkeypatch):
    """The run pins the set's manifest sha256. Editing the set must not silently rescore old runs."""
    from odyssey.manager import tlc_timetable_set
    from odyssey_benchmark import tlc_timetable
    monkeypatch.setattr(tlc_timetable_set, 'load_set', lambda name, base=None: dict(SET, sha256='sha-edited'))
    monkeypatch.setattr(tlc_timetable, 'load_timetable', lambda d: TIMETABLE)
    with pytest.raises(ValueError, match='the set changed'):
        driving_metrics._tl_set_and_timetable(TT)


def test_make_rules_is_empty_by_default_and_loud_on_unknown_lane_rule():
    assert make_rules() == {} and make_rules(None, '') == {}
    with pytest.raises(ValueError, match='lane rule'):
        make_rules(lane='v9')


def test_rules_from_config_reads_the_two_keys(monkeypatch):
    seen = []
    monkeypatch.setattr(driving_metrics, 'make_rules', lambda tl_set=None, lane=None: seen.append((tl_set, lane)))
    driving_metrics.rules_from_config({'tl_set': 'tlc_timetable', 'plc_rule': 'plc'})
    driving_metrics.rules_from_config({})
    assert seen == [('tlc_timetable', 'plc'), (None, None)]


def test_the_timetable_is_the_thirteen_signal_scenes_with_eleven_designated():
    rules = make_rules('tlc_timetable', 'plc')
    s, timetable = driving_metrics._tl_set_and_timetable(rules)
    assert rules['tl_set'] == 'tlc_timetable' and rules['tl_set_sha256'] == s['sha256']
    assert sorted(s['scenes']) == 'odyssey_scene011 odyssey_scene021 odyssey_scene025 odyssey_scene032 odyssey_scene052 odyssey_scene056 odyssey_scene060 odyssey_scene061 odyssey_scene067 odyssey_scene072 odyssey_scene075 odyssey_scene081 odyssey_scene086'.split()
    assert set(s['scenes']) - set(timetable['files']) == {'odyssey_scene021', 'odyssey_scene081'}


# --- scene start: a designated scene must be driven with the set ------------------------------------


@pytest.mark.parametrize('scene, preset, ok', [
    ('odyssey_scene011', dict(name='tlc_timetable', in_set=True), True),
    ('odyssey_scene011', None, False),                                  # no timetable at all
    ('odyssey_scene011', dict(name='r8', in_set=True), False),          # another set
    ('odyssey_scene021', None, True),                                   # no designated signal
    ('odyssey_scene001', None, True)])                                  # not a signal scene
def test_designated_scene_must_be_driven_with_the_set(fake_set, monkeypatch, scene, preset, ok):
    from odyssey.manager import tlc_timetable_set
    monkeypatch.setattr(tlc_timetable_set, 'resolve', lambda cfg, scene_id: preset)
    if ok:
        driving_metrics.check_rules_drivable(TT, {}, scene)
    else:
        with pytest.raises(ValueError, match='must be driven with tlc_timetable_set=tlc_timetable'):
            driving_metrics.check_rules_drivable(TT, {}, scene)


def test_no_tl_rule_checks_nothing():
    driving_metrics.check_rules_drivable({'lane': 'plc'}, {}, 'odyssey_scene011')


# --- replay: pinned rules by default, an explicit dict overrides ------------------------------------


@pytest.fixture
def light_replay(monkeypatch):
    """replay with the heavy scorers stubbed; returns the rules apply_rules was handed."""
    from odyssey_benchmark.sdroute_score import SDRouteMetric
    monkeypatch.setattr(driving_metrics, 'score_dense', lambda *args: dict(
        P_col=1., P_off=1.))
    monkeypatch.setattr(driving_metrics, 'traffic_fields', lambda *args: {})
    metric = SimpleNamespace(score=lambda *args: dict(
        rc_method='hmm_prefix_1m', RC=1., rc_status='ok', rc_guard_reason='rollout_end',
        rc_reference_suspect=False, P_SD=1))
    monkeypatch.setattr(SDRouteMetric, 'from_snapshot', lambda *args: metric)
    seen = []

    def fake_rules(row, z, data, rules, capture=None):
        seen.append(rules)
        row['P_PLC'] = 0.5 if rules.get('lane') else None
        return row
    monkeypatch.setattr(driving_metrics, 'apply_rules', fake_rules)

    def run(pinned_rules, **kw):
        data = dict(map={}, actors=[], sim_dt=.1, rc={}, term_reason='time_limit',
                    departure_distance_m=30.)
        if pinned_rules is not None:
            data['rules'] = pinned_rules
        return driving_metrics.replay(PinnedInputs(
            driving_inputs_json=json.dumps(data), ds_states=np.zeros((2, 11)), ds_sim_steps=np.array([0, 1]),
            rc_sim_steps=np.array([0, 1]), rc_ego_xy=np.zeros((2, 2)), rc_ego_heading=np.zeros(2)), **kw)
    return run, seen


def test_replay_applies_the_pinned_rules_before_ds(light_replay):
    run, seen = light_replay
    row = run({'lane': 'plc'})
    assert seen == [{'lane': 'plc'}]
    assert row['RouteDS'] == pytest.approx(50.)          # P_lane is in DS, computed once after the rules
    assert row['scoring_error'] is None and row['static_mode'] == driving_metrics.STATIC_MODE


def test_replay_of_a_run_scored_before_pinning_uses_no_rules(light_replay):
    run, seen = light_replay
    assert run(None)['RouteDS'] == pytest.approx(100.) and seen == [{}]


def test_an_explicit_rules_dict_overrides_the_pinned_one(light_replay):
    run, seen = light_replay
    run({'lane': 'plc'}, rules={})
    assert seen == [{}]
