"""Exercise real MetricManager methods without a planner/renderer or score writes."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip('nuplan')
from odyssey.manager import base_manager
from odyssey.manager.metric_manager import MetricManager
from odyssey_benchmark import scorer
from odyssey_benchmark.sdroute_score import SDRouteMetric
from odyssey_benchmark import sdroute_score as sdroute_metric
from sdroute_fixtures import TestGraph


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setenv('ODYSSEY_ROUTE_FILE', str(tmp_path / 'abc.npz'))
    rear = np.c_[np.arange(0., 41.), np.zeros(41)]
    np.savez(tmp_path / 'abc.npz', route_xy=rear, sd_edges=[0], map_location='test')
    monkeypatch.setattr(sdroute_metric, 'graph', lambda _: TestGraph())
    track = dict(position=rear + [1.461, 0.], heading=np.zeros(41))
    scene = dict(id='abc', metadata={'old_origin_in_current_coordinate': [0, 0]},
                 cadence=SimpleNamespace(sim_dt=.1, score_stride_steps=5))
    engine = SimpleNamespace(episode_step=16, global_config={
        'num_history': 16, 'num_future': 25, 'gt_budget_multiplier': 2.},
        env=SimpleNamespace(done_function=Mock(return_value=(False, {}))))
    monkeypatch.setattr(base_manager, 'get_engine', lambda: engine)
    m = object.__new__(MetricManager)
    m._scorer = scorer
    m.agent = SimpleNamespace(rear_vehicle=SimpleNamespace(current_position=np.array([16., 0.])),
                              current_heading=0.)
    m.converter = SimpleNamespace(initial_ego_center=np.zeros(2))
    m._rc_metric = SDRouteMetric(scene, track, 15, 5, 1.461)
    m._rc_metric.observe(15, [15, 0], 0)
    m.current_scene = scene
    m.current_step = 15
    m._score_stride = 5
    m._scored = False
    m._graded_init_done = False
    m.num_history, m.num_future = 4, 5
    m.ego_states_list = []
    m.first_NC_DAC_step = None
    m.score_rows = []
    m.save_scores = Mock()
    return m, engine


@pytest.mark.parametrize('reason', ['destination_arrival', 'route_deviation'])
def test_real_non_grid_step_finalizes_last_pose_once(manager, reason):
    m, engine = manager
    info = {'term_reason': reason}
    if reason == 'destination_arrival':
        info.update(goal_source='sd_route', sd_goal_progress_ratio=1., sd_goal_end_dist_m=4.)
    engine.env.done_function.return_value = (True, info)
    m.step()
    m._score_and_save({'token': 'abc', 'step': 16})
    m.save_scores.assert_called_once()
    row = m.score_rows[0]
    assert row['term_reason'] == reason
    if reason == 'destination_arrival':
        assert row['goal_source'] == 'sd_route'
        assert row['sd_goal_end_dist_m'] == 4.
    assert row['rc_final_step'] == 16
    # A 1m episode is too short for the 5m SDF matcher; no old-RC fallback.
    # The original intent (no fallback to the old RC) stands; only the value changes from nan to 0.
    # A drive the matcher cannot resolve is not "unknown": it covered none of the route.
    # Left as nan, P_SD_status stays 'match_failed' and the whole run becomes a scoring failure.
    assert row['RC'] == 0. and row['rc_status'] == 'no_motion'
    assert row['rc_method'] == 'hmm_prefix_1m'
    assert 'scored_poses' not in row          # legacy PDMS is no longer measured
    if reason == 'route_deviation':
        assert 'P_off' not in row
        assert row['P_SD'] == 0
        assert row['RouteDS'] is None


def test_real_budget_finalization_does_not_force_rc_to_one(manager):
    m, engine = manager
    engine.episode_step = m.current_step = m._step_budget()
    m._rc_metric.observe(m.current_step, [25., 0.], 0.)
    assert m._rollout_ending()
    m._score_and_save({'token': 'abc', 'step': m.current_step})
    row = m.score_rows[0]
    assert row['term_reason'] == 'time_limit'
    assert row['RC'] == pytest.approx(.4)


def test_nominal_horizon_does_not_finalize_before_multiplied_budget(manager):
    m, engine = manager
    engine.episode_step = 40  # nominal horizon; actual budget is 80
    engine.managers = {}
    m._graded_init_done = True
    m.buffer_size = 5
    m.observations_list, m.detection_tracks_list = [], []
    m.agent_source_mode_list, m.traffic_light_data_list = [], []
    m.converter.convert_to_current_ego_state = lambda step: object()
    m.converter.convert_to_detections_tracks_from_agent_input = lambda step: object()
    m.step()
    assert not m._scored
    m.save_scores.assert_not_called()


def test_rc_no_longer_queries_hd_polygons(tmp_path, monkeypatch):
    from shapely.geometry import box, LineString
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as Layer
    monkeypatch.setenv('ODYSSEY_ROUTE_FILE', str(tmp_path / 'abc.npz'))
    route = np.array([[0., 0.], [10., 0.], [20., 0.]])
    np.savez(tmp_path / 'abc.npz', route_xy=route)
    road = SimpleNamespace(id='road', polygon=box(-1, -1, 21, 1), outgoing_edges=[],
        interior_edges=[SimpleNamespace(baseline_path=SimpleNamespace(linestring=LineString(route)))])
    def query(point, radius, layers):
        assert point.x == pytest.approx(10.)
        assert point.y == pytest.approx(0.)
        return {Layer.ROADBLOCK: [road], Layer.ROADBLOCK_CONNECTOR: []}
    scene = dict(id='abc', cadence=SimpleNamespace(sim_dt=.5),
                 metadata={'old_origin_in_current_coordinate': [0, 0]})
    track = dict(position=np.array([[1.461, 0], [11.461, 0], [21.461, 0]]), heading=np.zeros(3))
    m = SDRouteMetric(scene, track, 0, 1, 1.461,
                      map_api=SimpleNamespace(get_proximal_map_objects=query))
    assert m.topology is None
