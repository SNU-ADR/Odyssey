"""Runs that end by arrival are matched only up to where they entered the arrival radius."""
from types import SimpleNamespace

import numpy as np
import pytest

from odyssey_benchmark.sdroute_score import SDRouteMetric, arrival_zone_keep
from odyssey_benchmark import sdroute_score as sdroute_metric
from sdroute_fixtures import TestGraph


def line(xs):
    return np.c_[np.asarray(xs, float), np.zeros(len(xs))]


def test_keeps_up_to_the_first_pose_of_the_final_stretch_inside_the_radius():
    xy = line(range(0, 41, 5))                       # 0 5 ... 40, end at 40
    assert arrival_zone_keep(xy, (40, 0), radius=10) == 7          # keeps 0..30


def test_an_earlier_pass_near_the_end_is_not_the_arrival():
    xy = np.array([[0, 0], [38, 0], [60, 0], [60, 30], [30, 0], [35, 0], [40, 0]], float)
    assert arrival_zone_keep(xy, (40, 0), radius=10) == 5          # cut at (30, 0), not (38, 0)


def test_last_pose_outside_the_radius_keeps_everything():
    xy = line(range(0, 41, 5))
    assert arrival_zone_keep(xy, (60, 0), radius=10) == len(xy)


def test_too_little_left_to_match_keeps_everything():
    xy = line([28, 31, 34, 37, 40])
    assert arrival_zone_keep(xy, (40, 0), radius=10) == len(xy)    # 3 m kept < 5 m


@pytest.fixture
def metric(tmp_path, monkeypatch):
    monkeypatch.setenv('ODYSSEY_ROUTE_FILE', str(tmp_path / 'abc.npz'))
    route = np.c_[np.arange(0., 41.), np.zeros(41)]
    np.savez(tmp_path / 'abc.npz', route_xy=route, sd_edges=[0], map_location='test')
    monkeypatch.setattr(sdroute_metric, 'graph', lambda _: TestGraph())
    track = dict(position=route + [1.461, 0], heading=np.zeros(41))
    scene = dict(id='scene_abc', cadence=SimpleNamespace(sim_dt=.1),
                 metadata={'old_origin_in_current_coordinate': [0, 0]})
    m = SDRouteMetric(scene, track, 15, 5, 1.461)
    for step in range(15, 41):
        m.observe(step, [step, 0], 0)
    return m


def test_arrival_trims_the_arrival_radius(metric):
    row = metric.score('destination_arrival')
    assert row['rc_arrival_trim_m'] == pytest.approx(10)
    assert row['rc_final_step'] == 30
    assert row['P_SD'] == 1


@pytest.mark.parametrize('reason', [None, 'time_limit', 'route_exhausted'])
def test_runs_that_did_not_arrive_are_not_trimmed(metric, reason):
    row = metric.score(reason)
    assert row['rc_arrival_trim_m'] == 0
    assert row['rc_final_step'] == 40
