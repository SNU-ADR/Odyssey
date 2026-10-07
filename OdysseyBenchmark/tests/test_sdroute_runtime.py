"""Runtime prefix RC contracts plus retained historical coverage unit tests."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from odyssey.manager.sdroute_metric import load_scene_route

from odyssey_benchmark.sdroute_score import SDRouteMetric
import odyssey
from odyssey.manager.sdroute_reference import build_coverage_reference
from odyssey_benchmark import scorer
from odyssey.manager import frenet
from odyssey_benchmark import sdroute_score as sdroute_metric
from sdroute_fixtures import TestGraph


@pytest.fixture
def metric(tmp_path, monkeypatch):
    route = np.c_[np.arange(0., 41.), np.zeros(41)]
    np.savez(tmp_path / 'abc.npz', route_xy=route, sd_edges=[0], map_location='test')
    monkeypatch.setenv('ODYSSEY_ROUTE_FILE', str(tmp_path / 'abc.npz'))
    monkeypatch.setattr(sdroute_metric, 'graph', lambda _: TestGraph())
    track = dict(position=route + [1.461, 0], heading=np.zeros(41))
    scene = dict(id='scene_abc', cadence=SimpleNamespace(sim_dt=.1),
                 metadata={'old_origin_in_current_coordinate': [0, 0]})
    return SDRouteMetric(scene, track, 15, 5, 1.461)


def test_rc_window_excludes_warmup_and_includes_terminal_partial_tick(metric):
    for step in range(23):
        metric.observe(step, [step, 0], 0)
    steps, xy, _, times = metric.arrays()
    assert steps.tolist() == [15, 20, 22]
    np.testing.assert_allclose(np.diff(times), [.5, .2])
    row = metric.score()
    assert row['rc_reference_m'] == pytest.approx(25)
    assert row['rc_covered_m'] == pytest.approx(7)
    assert row['RC'] == pytest.approx(7 / 25)
    assert row['rc_final_step'] == 22


def test_one_handoff_pose_is_real_zero_not_unknown(metric):
    metric.observe(15, [15., 0], 0)
    assert metric.score()['RC'] == 0.


def test_missing_handoff_pose_is_unknown(metric):
    metric.observe(16, [16., 0], 0)
    row = metric.score()
    assert row['rc_status'] == 'missing_handoff_pose'
    assert np.isnan(row['RC'])


def test_no_forced_completion_and_no_duplicate_endpoint_credit(metric):
    for step in range(15, 31):
        metric.observe(step, [step, 0], 0)
    before = metric.score()['RC']
    metric.observe(30, [30., 0], 0)
    assert metric.score()['RC'] == pytest.approx(before)
    assert before == pytest.approx(.6)


def method(name, path='envs/base_env.py', namespace=None):
    source = Path(odyssey.__file__).resolve().parent / path
    fn = next(n for n in ast.walk(ast.parse(source.read_text()))
              if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = dict(np=np, frenet=frenet, Dict=dict, Any=object, Tuple=tuple)
    ns.update(namespace or {})
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), ns)
    return ns[name]


def test_real_goal_source_uses_rear_query_and_is_idempotent():
    goal = method('_gt_goal_state')
    center = np.c_[np.arange(0., 101., 10.) + 1.461, np.zeros(11)]
    agent = SimpleNamespace(object_track={'position': center, 'heading': np.zeros(11)},
        current_position=np.array([99.461, 0.]),
        rear_vehicle=SimpleNamespace(current_position=np.array([98., 0.]), rear_axle_to_center_dist=1.461))
    scene = {'cadence': SimpleNamespace(score_stride_steps=1)}
    env = SimpleNamespace(engine=SimpleNamespace(episode_step=0, global_config={'num_history': 1},
        managers={'agent_manager': SimpleNamespace(ego_agent=agent),
                  'scenario_manager': SimpleNamespace(current_scene=scene)}))
    first = goal(env)
    assert goal(env) == first
    env.engine.episode_step = 1
    value = goal(env)
    assert value[0] == pytest.approx(.98)
    assert value[1] == pytest.approx(2.)


def test_sd_goal_uses_rc_crop_and_its_endpoint():
    goal = method('_sd_goal_state')
    done = method('done_function')
    route = np.c_[np.arange(151., dtype=float), np.zeros(151)]
    gt = route[:71]
    reference = build_coverage_reference(route, gt)
    scene = dict(metadata={'old_origin_in_current_coordinate': [0., 0.]})
    agent = SimpleNamespace(rear_vehicle=SimpleNamespace(current_position=np.array([0., 0.])))
    manager = SimpleNamespace(current_scene=scene, _rc_metric=SimpleNamespace(
        reference=reference, handoff=0, offset=np.zeros(2)))
    engine = SimpleNamespace(episode_step=0, global_config={}, managers={
        'scenario_manager': SimpleNamespace(current_scene=scene),
        'agent_manager': SimpleNamespace(ego_agent=agent), 'metric_manager': manager})
    env = SimpleNamespace(engine=engine, _sd_route_distance=lambda: None,
        _hopeless_stall_state=lambda: None,
        _gt_goal_state=lambda: pytest.fail('usable SD reference must determine arrival'),
        SD_ROUTE_MAX_DIST_M=30., SD_GOAL_PROGRESS_RATIO=.99, SD_GOAL_END_DIST_M=10.,
        logger=SimpleNamespace(info=lambda *args: None))
    env._sd_goal_state = lambda: goal(env)
    for step, x in enumerate((0., 30., 65.)):
        engine.episode_step = step
        agent.rear_vehicle.current_position = np.array([x, 0.])
        assert done(env)[0] is False
    engine.episode_step = 3
    agent.rear_vehicle.current_position = np.array([70., 11.])
    assert done(env)[0] is False  # progress alone is not arrival
    engine.episode_step = 4
    agent.rear_vehicle.current_position = np.array([70., 8.])
    state = goal(env)
    assert state[0] == pytest.approx(1.)
    assert state[1] == pytest.approx(8.)
    assert state[2] == pytest.approx(70.)  # not the full 150 m sidecar
    assert goal(env) == state  # two callers on one step do not advance the cursor
    ended, info = done(env)
    assert ended and info['goal_source'] == 'sd_route'
    assert info['sd_goal_end_dist_m'] == pytest.approx(8.)


def test_crop_geometry_and_credited_frontier_share_original_sd_arclength():
    route = np.array([[0., 0.], [10., 0.], [10., 10.], [20., 10.]])
    gt = np.array([[4., 0.], [10., 0.], [10., 5.]])
    reference = build_coverage_reference(route, gt)
    assert reference.s_start == pytest.approx(4.)
    assert reference.s_end == pytest.approx(15.)
    np.testing.assert_allclose(reference.crop_xy(), [[4., 0.], [10., 0.], [10., 5.]])
    credited = 7.5
    np.testing.assert_allclose(reference.crop_xy(reference.s_start + credited),
                               [[4., 0.], [10., 0.], [10., 1.5]])
    assert np.linalg.norm(np.diff(reference.crop_xy(reference.s_start + credited), axis=0),
                          axis=1).sum() == pytest.approx(credited)


@pytest.mark.parametrize('reference', [None, SimpleNamespace(quality={'suspect': True})])
def test_unusable_sd_goal_reference_uses_existing_gt_arrival(reference):
    goal = method('_sd_goal_state')
    done = method('done_function')
    scene = dict(metadata={'old_origin_in_current_coordinate': [0., 0.]})
    manager = SimpleNamespace(current_scene=scene, _rc_metric=SimpleNamespace(reference=reference))
    engine = SimpleNamespace(episode_step=3, global_config={}, managers={
        'scenario_manager': SimpleNamespace(current_scene=scene), 'metric_manager': manager})
    env = SimpleNamespace(engine=engine, _sd_route_distance=lambda: None,
        _gt_goal_state=lambda: (1., 2., 3), _gt_goal_ref=(None, np.array([0., 70.])),
        SD_ROUTE_MAX_DIST_M=30., GT_GOAL_PROGRESS_RATIO=.99, GT_GOAL_END_DIST_M=5.,
        logger=SimpleNamespace(info=lambda *args: None))
    env._sd_goal_state = lambda: goal(env)
    ended, info = done(env)
    assert ended and info['goal_source'] == 'gt_fallback'
    assert info['gt_end_dist_m'] == pytest.approx(2.)


def test_departure_cache_changes_scene_before_any_goal_call(tmp_path, monkeypatch):
    np.savez(tmp_path / 'abc.npz', route_xy=np.array([[0., 0.], [10., 0.]]))
    np.savez(tmp_path / 'def.npz', route_xy=np.array([[0., 100.], [10., 100.]]))
    monkeypatch.setenv('ODYSSEY_ROUTE_FILE', str(tmp_path / 'abc.npz'))
    distance = method('_sd_route_distance', namespace={'load_scene_route': load_scene_route})
    agent = SimpleNamespace(rear_vehicle=SimpleNamespace(current_position=np.array([5., 1.])))
    sm = SimpleNamespace(current_scene=dict(id='abc', metadata={'old_origin_in_current_coordinate': [0, 0]}))
    env = SimpleNamespace(engine=SimpleNamespace(
        managers={'agent_manager': SimpleNamespace(ego_agent=agent), 'scenario_manager': sm}))
    assert distance(env) == pytest.approx(1.)
    monkeypatch.setenv('ODYSSEY_ROUTE_FILE', str(tmp_path / 'def.npz'))
    sm.current_scene = dict(id='def', metadata={'old_origin_in_current_coordinate': [0, 0]})
    assert distance(env) == pytest.approx(99.)


def _departure_env(tmp_path, monkeypatch, route, reference):
    """Env that runs only the departure guard. metric_manager only needs to hold the RC reference."""
    np.savez(tmp_path / 'abc.npz', route_xy=route)
    monkeypatch.setenv('ODYSSEY_ROUTE_FILE', str(tmp_path / 'abc.npz'))
    agent = SimpleNamespace(rear_vehicle=SimpleNamespace(current_position=np.array([0., 0.])))
    scene = dict(id='abc', metadata={'old_origin_in_current_coordinate': [0, 0]})
    env = SimpleNamespace(engine=SimpleNamespace(managers={
        'agent_manager': SimpleNamespace(ego_agent=agent),
        'scenario_manager': SimpleNamespace(current_scene=scene),
        'metric_manager': SimpleNamespace(current_scene=scene,
                                          _rc_metric=SimpleNamespace(reference=reference))}))
    return env, agent


def test_departure_measures_the_scored_span_not_the_sidecar_tail(tmp_path, monkeypatch):
    """The sidecar continues past the GT end point. An ego that follows that tail stays on the full
    route (distance 0), so the guard would never fire and the run would continue to the step limit.
    Measured against the scored span (RC's denominator), driving onto the tail reads as departure."""
    distance = method('_sd_route_distance', namespace={'load_scene_route': load_scene_route})
    route = np.c_[np.arange(151., dtype=float), np.zeros(151)]
    env, agent = _departure_env(tmp_path, monkeypatch, route,
                               build_coverage_reference(route, route[:71]))
    agent.rear_vehicle.current_position = np.array([110., 0.])
    assert distance(env) == pytest.approx(40.)  # 40 m past the crop end at 70 m -- over the 30 m limit
    agent.rear_vehicle.current_position = np.array([95., 0.])
    assert distance(env) == pytest.approx(25.)  # same tail, but still within the limit
    agent.rear_vehicle.current_position = np.array([35., 4.])
    assert distance(env) == pytest.approx(4.)   # unchanged inside the scored span


@pytest.mark.parametrize('reference', [None, SimpleNamespace(quality={'suspect': True})])
def test_unusable_reference_keeps_the_full_sidecar_guard(tmp_path, monkeypatch, reference):
    """If the crop is untrusted, use the full sidecar as before -- ending a run on a wrong crop is worse."""
    distance = method('_sd_route_distance', namespace={'load_scene_route': load_scene_route})
    route = np.c_[np.arange(151., dtype=float), np.zeros(151)]
    env, agent = _departure_env(tmp_path, monkeypatch, route, reference)
    agent.rear_vehicle.current_position = np.array([110., 0.])
    assert distance(env) == pytest.approx(0.)


def test_real_done_source_gives_departure_priority_over_arrival():
    done = method('done_function')
    env = SimpleNamespace(engine=SimpleNamespace(global_config={}, episode_step=30),
        SD_ROUTE_MAX_DIST_M=30., GT_GOAL_PROGRESS_RATIO=.99, GT_GOAL_END_DIST_M=5.,
        _sd_route_distance=lambda: 30.001,
        _sd_goal_state=lambda: pytest.fail('arrival must not be queried after departure'),
        _gt_goal_state=lambda: pytest.fail('arrival must not be queried after departure'),
        logger=SimpleNamespace(info=lambda *args: None))
    ended, info = done(env)
    assert ended and info['term_reason'] == 'route_deviation'


def _client(plan):
    """Fake client with only what get_trajectory reads; the runtime answers every step with `plan`."""
    agent = SimpleNamespace(current_heading=0.5,
                            rear_vehicle=SimpleNamespace(current_position=np.array([7., -3.])))
    engine = SimpleNamespace(sim_dt=.1, env=SimpleNamespace())
    return SimpleNamespace(_runtime=SimpleNamespace(plan=lambda step: plan), agent=agent,
                           engine=engine, _cached_traj=None, config={'num_history': 1})


def test_route_exhausted_answer_holds_the_ego_instead_of_waiting_out_the_plan_timeout():
    """When the planner reports route exhaustion the sim does not wait for the timeout. It flags the
    env and holds the ego in place -- it does not invent forward motion in place of a missing plan."""
    from odyssey.components.agents.client.planner_client import PlannerClient
    client = _client(None)

    traj = PlannerClient.get_trajectory(client, 4)

    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    heading = client.agent.current_heading
    here = (client.agent.rear_vehicle.current_position            # waypoints are the vehicle centre
            + get_pacifica_parameters().rear_axle_to_center * np.array([np.cos(heading),
                                                                       np.sin(heading)]))
    assert client.engine.env._route_exhausted is True
    np.testing.assert_allclose(traj.waypoints, np.broadcast_to(here, traj.waypoints.shape))
    np.testing.assert_allclose(traj.velocities, 0.)
    np.testing.assert_allclose(traj.headings, heading)


def test_a_planner_answer_is_followed_without_flagging_exhaustion():
    """An ordinary answer is the plan the ego tracks; only a None answer ends the run."""
    from odyssey.components.agents.client.planner_client import PlannerClient
    plan = np.c_[np.arange(1., 41.) * .5, np.zeros(40), np.zeros(40)]    # straight ahead, 5 m/s
    client = _client(plan)

    traj = PlannerClient.get_trajectory(client, 4)

    assert not hasattr(client.engine.env, '_route_exhausted')
    assert len(traj.waypoints) == 41
    np.testing.assert_allclose(np.linalg.norm(traj.velocities[:-1], axis=1), 5.)


def test_route_exhausted_ends_the_run_instead_of_waiting_for_a_planner_answer():
    """With the baked route used up the planner cannot answer. Instead of waiting for the plan timeout
    and marking the whole run failed, the run ends at that step as a driving result, so the distance
    driven so far is scored. It does not wait for the scoring cadence either -- env has no cadence here."""
    done = method('done_function')
    env = SimpleNamespace(engine=SimpleNamespace(global_config={}, episode_step=31),
        SD_ROUTE_MAX_DIST_M=30., _route_exhausted=True,
        _sd_route_distance=lambda: 1.,
        _sd_goal_state=lambda: pytest.fail('arrival must not be queried after exhaustion'),
        _gt_goal_state=lambda: pytest.fail('arrival must not be queried after exhaustion'),
        _hopeless_stall_state=lambda: pytest.fail('stall must not be queried after exhaustion'),
        logger=SimpleNamespace(info=lambda *args: None))
    ended, info = done(env)
    assert ended and info['term_reason'] == 'route_exhausted'


def test_finalizer_scores_short_departure_and_is_write_once(metric):
    finalize = method('_score_and_save', 'manager/metric_manager.py', {'asdict': lambda x: x})
    metric.observe(15, [15, 0], 0)
    metric.observe(16, [15, 31], 0)
    writes = []
    manager = SimpleNamespace(_scored=False, _scorer=scorer, _rc_metric=metric, _graded_init_done=False,
        ego_states_list=[], num_history=4, current_step=16, first_NC_DAC_step=None,
        engine=SimpleNamespace(global_config={'num_history': 16, 'num_future': 100},
                               env=SimpleNamespace(SD_ROUTE_MAX_DIST_M=30.,
                                   done_function=lambda: (True, {'term_reason': 'route_deviation'}))),
        _term_reason=lambda: 'route_deviation', score_rows=[],
        save_scores=lambda: writes.append(True))
    finalize(manager, {'token': 'abc', 'step': 16})
    finalize(manager, {'token': 'abc', 'step': 16})
    assert len(writes) == 1
    row = manager.score_rows[0]
    assert 'scored_poses' not in row          # legacy PDMS is no longer measured
    assert row['RC'] == 0
    assert row['rc_final_step'] == 16
    assert 'P_off' not in row
    assert row['P_SD'] == 0
    assert row['RouteDS'] is None


def test_route_file_is_the_scene_route(tmp_path, monkeypatch):
    """A published scene's route.npz is passed by path (ODYSSEY_ROUTE_FILE); no name lookup.

    The file's tokens are not compared with the scene's token: three published scenes
    (odyssey_scene039, 080, 093) carry hand-edited routes built from a neighbouring window.
    """
    route = np.array([[0., 0.], [1., 0.], [2., 0.]])
    path = tmp_path / 'route.npz'
    np.savez(path, route_xy=route, tokens=np.array(['tok0', 'tok1']))
    monkeypatch.setenv('ODYSSEY_ROUTE_FILE', str(path))
    xy, match = load_scene_route()
    np.testing.assert_array_equal(xy, route)
    assert match == str(path)
    monkeypatch.delenv('ODYSSEY_ROUTE_FILE')
    assert load_scene_route() == (None, None)
