"""The arrival tolerance changes scored RC, never the measured route prefix."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from odyssey_benchmark import driving_metrics

from sdroute_fixtures import TestGraph


def row():
    return dict(rc_method='hmm_prefix_1m', RC=.9902,
                rc_status='full_sdf_one', rc_guard_reason='rollout_end',
                rc_reference_suspect=False, P_SD=1,
                goal_source='sd_route', sd_goal_progress_ratio=.9902,
                sd_goal_end_dist_m=4., P_col=.6,
                P_off=.8)


def test_clean_sd_goal_gets_full_rc_but_preserves_measurement():
    result = driving_metrics.apply_termination(row(), 'destination_arrival')
    assert result['RC'] == 1.
    assert result['RC_measured'] == pytest.approx(.9902)
    assert result['RC_goal_bonus'] is True
    assert result['RouteDS'] == pytest.approx(48.)
    assert driving_metrics.apply_termination(result, 'destination_arrival') == result


@pytest.mark.parametrize('reason,changes', [
    ('time_limit', {}),
    ('destination_arrival', {'goal_source': 'gt_fallback'}),
    ('destination_arrival', {'goal_source': None}),
    ('destination_arrival', {'P_SD': 0}),
    ('destination_arrival', {'P_SD': float('nan')}),
    ('destination_arrival', {'rc_status': 'first_prefix_zero'}),
    ('destination_arrival', {'rc_guard_reason': 'route_order_violation'}),
    ('destination_arrival', {'rc_reference_suspect': True}),
    ('destination_arrival', {'rc_reference_suspect': None}),
])
def test_bonus_requires_verified_complete_arrival(reason, changes):
    inputs = row()
    inputs.update(changes)
    measured = inputs['RC']
    result = driving_metrics.apply_termination(inputs, reason)
    assert result['RC'] == measured
    assert result['RC_goal_bonus'] is False


@pytest.mark.parametrize('measured', [.97725, .9, .5, .01])
def test_arrival_is_credited_however_short_the_prefix_scan_stopped(measured):
    """Once arrival is verified, RC is 1.0 whatever the measured value -- there is no floor.

    The run ends the moment the ego is within 10 m, so the last 10 m are never driven and
    cannot be credited. That shortfall is a fraction of the route length (10/route_len), so
    it differs per scene. SD_GOAL_PROGRESS_RATIO is a runtime progress threshold and measured is
    the prefix-scan RC -- different scales, no reason to share a number. Every route under
    1000 m has a ceiling below .99 (a 443.9 m route tops out at .97747).

    What gates the bonus here is the arrival verdict (goal_source) and match quality (sdf/rc_status/
    rc_guard_reason/rc_reference_suspect), not the credited distance.
    """
    inputs = row()
    inputs['RC'] = measured
    result = driving_metrics.apply_termination(inputs, 'destination_arrival')
    assert result['RC'] == 1.
    assert result['RC_goal_bonus'] is True
    assert result['RC_measured'] == pytest.approx(measured)


def test_already_complete_run_is_not_reported_as_bonused():
    """A run that reached 1.0 on its own is not recorded as bonused -- the field is an audit of whether
    the value actually moved, and it becomes unreadable if unmoved runs are True too."""
    inputs = row()
    inputs['RC'] = 1.
    result = driving_metrics.apply_termination(inputs, 'destination_arrival')
    assert result['RC'] == 1.
    assert result['RC_goal_bonus'] is False


@pytest.mark.parametrize('changes', [
    {'sd_goal_progress_ratio': .98},
    {'sd_goal_end_dist_m': 10.1},
    {'sd_goal_progress_ratio': .5, 'sd_goal_end_dist_m': 40.},
    {'sd_goal_progress_ratio': None, 'sd_goal_end_dist_m': None},
])
def test_arrival_tolerance_belongs_to_base_env_not_the_scorer(changes):
    """goal_source == 'sd_route' IS the arrival verdict; the scorer does not re-test it.

    base_env.done_function only emits that goal_source after its own progress/end-distance
    thresholds pass (defines.SD_GOAL_PROGRESS_RATIO / SD_GOAL_END_DIST_M). Re-checking them here
    against literal .99/10. was a second copy of a value owned elsewhere: it could disagree with
    the runtime but never with the ego. The fields may still ride along in the row -- they are diagnostics now, not gates.
    """
    inputs = row()
    inputs.update(changes)
    result = driving_metrics.apply_termination(inputs, 'destination_arrival')
    assert result['RC'] == 1.
    assert result['RC_goal_bonus'] is True
    assert result['RC_measured'] == pytest.approx(.9902)


def test_missing_goal_fields_do_not_block_a_verified_arrival():
    """A row that never carried the goal diagnostics still earns the tail."""
    inputs = row()
    del inputs['sd_goal_progress_ratio'], inputs['sd_goal_end_dist_m']
    result = driving_metrics.apply_termination(inputs, 'destination_arrival')
    assert result['RC'] == 1.
    assert result['RC_goal_bonus'] is True


def test_pinned_replay_uses_same_goal_metadata(monkeypatch):
    from odyssey_benchmark.sdroute_score import SDRouteMetric

    monkeypatch.setattr(driving_metrics, 'score_dense', lambda *args: dict(
        P_col=1., P_off=1.))
    monkeypatch.setattr(driving_metrics, 'traffic_fields', lambda *args: {})
    metric = SimpleNamespace(score=lambda *args: dict(
        rc_method='hmm_prefix_1m', RC=.9902,
        rc_status='full_sdf_one', rc_guard_reason='rollout_end',
        rc_reference_suspect=False, P_SD=1))
    monkeypatch.setattr(SDRouteMetric, 'from_snapshot', lambda *args: metric)
    data = dict(map={}, actors=[], sim_dt=.1, rc={}, term_reason='destination_arrival',
                departure_distance_m=30.,
                goal=dict(goal_source='sd_route', sd_goal_progress_ratio=.9902,
                          sd_goal_end_dist_m=4.))

    def replay(payload):
        return driving_metrics.replay(dict(
            driving_inputs_json=json.dumps(payload), ds_states=np.zeros((2, 11)),
            ds_sim_steps=np.array([0, 1]), rc_sim_steps=np.array([0, 1]),
            rc_ego_xy=np.zeros((2, 2)), rc_ego_heading=np.zeros(2)))

    result = replay(data)
    assert result['RC'] == 1.
    assert result['RC_measured'] == pytest.approx(.9902)
    assert result['RouteDS'] == 100.
    data.pop('goal')  # older pins must not guess which arrival rule was used
    old = replay(data)
    assert old['RC'] == pytest.approx(.9902)
    assert old['RC_goal_bonus'] is False


def test_live_finalizer_awards_only_the_last_percent(tmp_path, monkeypatch):
    from odyssey.manager import base_manager
    from odyssey_benchmark import sdroute_score as sdroute_metric
    from odyssey.manager.metric_manager import MetricManager
    from odyssey_benchmark import scorer

    monkeypatch.setenv('ODYSSEY_ROUTE_FILE', str(tmp_path / 'abc.npz'))
    monkeypatch.setattr(sdroute_metric, 'graph', lambda _: TestGraph(branching=True))
    route = np.c_[np.arange(0., 101., 2.), np.zeros(51)]
    np.savez(tmp_path / 'abc.npz', route_xy=route, sd_edges=[0, 1, 2], map_location='test')
    scene = dict(id='abc', metadata={'old_origin_in_current_coordinate': [0., 0.]},
                 cadence=SimpleNamespace(sim_dt=.1, score_stride_steps=5))
    gt = np.vstack([np.zeros((3, 2)), route])
    track = dict(position=np.repeat(gt, 5, axis=0) + [1.461, 0.],
                 heading=np.zeros(len(gt) * 5))
    metric = sdroute_metric.SDRouteMetric(scene, track, 15, 5, 1.461)
    xy = np.vstack([route[:-1], [99., 0.]])
    steps = np.arange(len(xy)) * 5 + 15
    steps[-1] -= 2  # exact non-grid terminal pose
    for step, point in zip(steps, xy):
        metric.observe(step, point, 0.)

    goal = dict(term_reason='destination_arrival', goal_source='sd_route',
                sd_goal_progress_ratio=.99, sd_goal_end_dist_m=1.)
    engine = SimpleNamespace(episode_step=int(steps[-1]), sim_dt=.1,
        global_config=dict(num_history=16, num_future=500, gt_budget_multiplier=1.),
        env=SimpleNamespace(SD_ROUTE_MAX_DIST_M=30., done_function=lambda: (True, goal)))
    monkeypatch.setattr(base_manager, 'get_engine', lambda: engine)
    manager = object.__new__(MetricManager)
    manager._scorer = scorer
    manager._rc_metric = metric
    manager._scored = False
    manager.current_step = int(steps[-1])
    manager.current_scene = scene
    manager.num_history, manager.num_future, manager._score_stride = 4, 100, 5
    manager._graded_init_done = False
    manager.first_NC_DAC_step = None
    manager.score_rows = []
    manager.ego_states_list = []
    manager.save_scores = Mock()
    manager._score_and_save({'token': scene['id'], 'step': int(steps[-1])})
    row = manager.score_rows[0]
    assert row['RC'] == 1.
    # SDF: an arrival run is matched only up to 90 m, where it entered the 10 m arrival radius -> the last 9 m of 99 m drop out.
    assert row['RC_measured'] == pytest.approx(.90, abs=1e-5)
    assert row['rc_arrival_trim_m'] == pytest.approx(9.)
    assert row['RC_goal_bonus'] is True
    assert row['P_SD'] == 1.
