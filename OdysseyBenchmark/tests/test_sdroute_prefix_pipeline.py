"""Runtime finalization -> CSV/NPZ/SDF -> launcher reader, using the real HMM."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from sdroute_fixtures import TestGraph
from odyssey.manager import base_manager
from odyssey_benchmark import sdroute_score as sdroute_metric
from odyssey.manager.metric_manager import MetricManager
from odyssey_benchmark import scorer
from odyssey_benchmark.sdroute_score import SDRouteMetric
from odyssey_benchmark.sdroute_prefix import arc
from odyssey_benchmark.sdroute_sdf import METHOD, matcher




def sample(vertices):
    p = np.asarray(vertices, float)
    s = arc(p)
    u = np.r_[np.arange(0, s[-1], 2.), s[-1]]
    return np.c_[np.interp(u, s, p[:, 0]), np.interp(u, s, p[:, 1])]


@pytest.mark.parametrize('reason,vertices,expected_sdf,expected_rc', [
    ('destination_arrival', [(0, 0), (70, 0)], 1, .7),
    ('time_limit', [(0, 0), (70, 0)], 1, .7),
    ('destination_arrival', [(0, 0), (40, 0), (40, 20), (70, 20), (70, 0), (100, 0)], 0, .4),
    ('route_deviation', [(0, 0), (20, 0), (20, -32)], 0, .2),
])
def test_real_finalizer_saves_new_rc_and_authoritative_sdf(
        tmp_path, monkeypatch, reason, vertices, expected_sdf, expected_rc):
    sc = tmp_path / 'sidecars'
    sc.mkdir()
    monkeypatch.setenv('ODYSSEY_ROUTE_FILE', str(sc / '0123456789abcdef.npz'))
    monkeypatch.setattr(sdroute_metric, 'graph', lambda _: TestGraph(branching=True))
    route = sample([(0, 0), (100, 0)])
    np.savez(sc / '0123456789abcdef.npz', route_xy=route,
             sd_edges=[0, 1, 2], map_location='test')
    scene = dict(id='scene_0123456789abcdef',
                 metadata={'old_origin_in_current_coordinate': [0., 0.]},
                 cadence=NS(sim_dt=.1, score_stride_steps=5))
    # 15 GT warmup transitions. Match start=0, end=100 after the handoff.
    gt = np.vstack([np.zeros((3, 2)), route])
    gt_sim = np.repeat(gt, 5, axis=0)
    track = dict(position=gt_sim + [1.461, 0], heading=np.zeros(len(gt_sim)))
    metric = SDRouteMetric(scene, track, 15, 5, 1.461)
    points = sample(vertices)
    # Score grid + a non-grid terminal pose; actual interval remains positive.
    steps = np.arange(len(points)) * 5 + 15
    steps[-1] -= 2
    for step, p in zip(steps, points):
        metric.observe(step, p, 0.)
    run = tmp_path / 'exp_0123456789abcdef_pipeline'
    out = run / 'scene/odyssey_output/openscene_format'
    engine = NS(episode_step=int(steps[-1]), sim_dt=.1, managers={},
                global_config=dict(num_history=16, num_future=500,
                                   gt_budget_multiplier=1., agent_policy='log_play_policy',
                                   data_output_dir=str(out)),
                env=NS(SD_ROUTE_MAX_DIST_M=30., done_function=lambda: (True, {'term_reason': reason})))
    monkeypatch.setattr(base_manager, 'get_engine', lambda: engine)
    m = object.__new__(MetricManager)
    m._scorer = scorer
    m._rc_metric = metric
    m._scored = False
    m.current_step = int(steps[-1])
    m.current_scene = scene
    m.num_history, m.num_future, m._score_stride = 4, 100, 5
    m._graded_init_done = False  # PDM length/setup independent of RC finalization.
    m.first_NC_DAC_step = None
    m.score_rows = []
    m.detection_tracks_list, m.agent_source_mode_list = [], []
    m.converter = NS(initial_ego_center=np.zeros(2))
    m._log_visibility_summary = Mock()
    def state(p):
        return NS(rear_axle=NS(x=p[0], y=p[1], heading=0.),
                  dynamic_car_state=NS(rear_axle_velocity_2d=NS(x=0., y=0.),
                                       rear_axle_acceleration_2d=NS(x=0., y=0.)))
    m.ego_states_list = [state(p) for p in np.vstack([np.zeros((3, 2)), points[:-1]])]
    m._score_and_save({'token': scene['id'], 'step': m.current_step})
    m._score_and_save({'token': scene['id'], 'step': m.current_step})
    saved = list(csv.DictReader((out / 'routeds_NR.csv').open()))
    assert len(saved) == 1 and len(m.score_rows) == 1   # one scene per run, never duplicated
    assert saved[0]['scene'] == scene['id'] and saved[0]['react'] == 'nr'
    assert float(saved[0]['P_SD']) == expected_sdf
    row = {k: ('' if v is None else str(v)) for k, v in m.score_rows[0].items()}   # the full scorer row
    assert row['rc_method'] == METHOD
    assert float(row['RC']) == pytest.approx(expected_rc, abs=1e-5)
    assert float(row['P_SD']) == expected_sdf
    assert row['term_reason'] == reason
    if expected_sdf == 1:
        assert float(row['rc_prefix_checks']) == 0
    elif reason != 'route_deviation':
        assert float(row['rc_first_zero_model_m']) == 49.
        assert float(row['rc_cutoff_model_m']) == 48.
    else:
        # Departure fails SDF and nothing else. The offroad columns stay absent rather than
        # claiming a measurement the run never made -- this row has no dense scoring at all.
        # DS is still 0, because SDF gates the product.
        assert not row.get('P_off')
        assert not row.get('RouteDS')
    with np.load(out / 'rollout_trajectory.npz', allow_pickle=False) as z:
        np.testing.assert_allclose(z['rc_ego_xy'], points)
        assert z['rc_sim_steps'][-1] == steps[-1]
        assert str(z['rc_method']) == METHOD
        assert 'agent_source_row' in z
        assert 'spawn_gate_event_source_rows' in z
        assert 'spawn_gate_event_reasons' in z


def test_a_failure_packing_the_scoring_inputs_is_reported_and_the_run_still_saved(tmp_path, monkeypatch):
    sc = tmp_path / 'sidecars'
    sc.mkdir()
    monkeypatch.setenv('ODYSSEY_ROUTE_FILE', str(sc / '0123456789abcdef.npz'))
    monkeypatch.setattr(sdroute_metric, 'graph', lambda _: TestGraph(branching=True))
    route = sample([(0, 0), (100, 0)])
    np.savez(sc / '0123456789abcdef.npz', route_xy=route, sd_edges=[0, 1, 2], map_location='test')
    scene = dict(id='scene_0123456789abcdef',
                 metadata={'old_origin_in_current_coordinate': [0., 0.]},
                 cadence=NS(sim_dt=.1, score_stride_steps=5))
    gt_sim = np.repeat(np.vstack([np.zeros((3, 2)), route]), 5, axis=0)
    metric = SDRouteMetric(scene, dict(position=gt_sim + [1.461, 0], heading=np.zeros(len(gt_sim))),
                           15, 5, 1.461)
    points = sample([(0, 0), (70, 0)])
    steps = np.arange(len(points)) * 5 + 15
    steps[-1] -= 2
    for step, p in zip(steps, points):
        metric.observe(step, p, 0.)
    out = tmp_path / 'run/odyssey_output'
    engine = NS(episode_step=int(steps[-1]), sim_dt=.1, managers={},
                global_config=dict(num_history=16, num_future=500, gt_budget_multiplier=1.,
                                   agent_policy='log_play_policy', data_output_dir=str(out)),
                env=NS(SD_ROUTE_MAX_DIST_M=30., done_function=lambda: (True, {'term_reason': 'time_limit'})))
    monkeypatch.setattr(base_manager, 'get_engine', lambda: engine)
    m = object.__new__(MetricManager)
    m._scorer = scorer
    m._rc_metric = metric
    m._scored = False
    m.current_step = int(steps[-1])
    m.current_scene = scene
    m.num_history, m.num_future, m._score_stride = 4, 100, 5
    m._graded_init_done = False
    m.first_NC_DAC_step = None
    m.score_rows = []
    m.detection_tracks_list, m.agent_source_mode_list = [], []
    m.converter = NS(initial_ego_center=np.zeros(2))
    m._log_visibility_summary = Mock()
    m.ego_states_list = [NS(rear_axle=NS(x=p[0], y=p[1], heading=0.),
                            dynamic_car_state=NS(rear_axle_velocity_2d=NS(x=0., y=0.),
                                                 rear_axle_acceleration_2d=NS(x=0., y=0.)))
                         for p in np.vstack([np.zeros((3, 2)), points[:-1]])]
    m._ds_steps = [int(steps[1])]          # dense frames were pinned, so the inputs are packed ...
    m.map_api, m._map_radius = None, 50.   # ... and packing fails: there is no map
    m._score_and_save({'token': scene['id'], 'step': m.current_step})
    saved = list(csv.DictReader((out / 'routeds_NR.csv').open()))
    assert len(saved) == 1 and saved[0]['scoring_error'] and not saved[0]['RouteDS']
    assert float(m.score_rows[0]['RC']) == pytest.approx(.7, abs=1e-5)
    with np.load(out / 'rollout_trajectory.npz', allow_pickle=False) as z:
        assert 'driving_inputs_json' not in z.files
        np.testing.assert_allclose(z['rc_ego_xy'], points)


def test_matcher_does_not_import_planner_navsim(monkeypatch):
    monkeypatch.setenv('ODYSSEY_PLANNER_REPO', '/nonexistent/planner')
    assert matcher().STEP == 5.


def test_missing_reference_never_falls_back_to_old_coverage(tmp_path, monkeypatch):
    monkeypatch.setenv('ODYSSEY_ROUTE_FILE', str(tmp_path / 'abc.npz'))
    route = sample([(0, 0), (100, 0)])
    np.savez(tmp_path / 'abc.npz', route_xy=route)
    scene = dict(id='abc', metadata={'old_origin_in_current_coordinate': [0., 0.]},
                 cadence=NS(sim_dt=.5))
    m = SDRouteMetric(scene, dict(position=route, heading=np.zeros(len(route))), 0, 1, 0.)
    for i, p in enumerate(route): m.observe(i, p, 0.)
    r = m.score(term_reason='route_deviation')
    assert np.isnan(r['RC'])
    assert r['rc_status'] == 'missing_sdf_reference'
    assert r['P_SD'] == 0
