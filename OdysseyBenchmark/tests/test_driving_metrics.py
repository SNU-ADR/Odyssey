import json
from types import SimpleNamespace
import numpy as np
import pytest
from shapely.geometry import box
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import SemanticMapLayer as Layer
from odyssey.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from odyssey_benchmark.driving_metrics import pack_map, score_dense, apply_termination, replay, traffic_fields


def map_data(intersection=False):
    shapes = [box(-100, -20, 100, 20)]
    tokens, kinds = ['lane'], [Layer.LANE]
    if intersection:
        shapes.append(box(-10, -10, 10, 10))
        tokens.append('intersection'); kinds.append(Layer.INTERSECTION)
    return pack_map(PDMDrivableMap(tokens, kinds, shapes), ['lane'], get_pacifica_parameters())


def states(n=7):
    value = np.zeros((n, 11)); value[:, 0] = np.arange(n) * .5; value[:, 3] = 5.
    return value


def actor(x, y=0., speed=0.):
    return ['car', 'VEHICLE', x, y, 0., 4., 2., 1.5, speed, 0.]


def test_exact_terminal_contact_not_lost():
    poses = states(); actors = [[] for _ in poses]
    actors[-1] = [actor(poses[-1, 0] + 4.)]
    result = score_dense(poses, actors, np.arange(15, 22), .1, map_data())
    assert result['ds_final_step'] == 21
    assert result['no_at_fault_collisions'] == 0
    assert result['P_col'] == .6
    assert result['first_violation_step'] == 21
    assert score_dense(poses[:-1], actors[:-1], np.arange(15, 21), .1, map_data())['P_col'] == 1


def test_intersection_exemption_matches_mask():
    poses = states(4)
    md = map_data(True); md['lane_ids'] = []
    cap = {}
    result = score_dense(poses, [[]] * 4, np.arange(15, 19), .1, md, cap)
    assert result['P_off'] == 1.
    assert not cap['scorer']._offroute_mask.any()
    md = map_data(False); md['lane_ids'] = []
    assert score_dense(poses, [[]] * 4, np.arange(15, 19), .1, md)['P_off'] == 0.


def test_nondrivable_not_exempted_by_intersection():
    poses = states(4); poses[:, 1] = 22.
    assert score_dense(poses, [[]]*4, np.arange(15, 19), .1, map_data())['P_off'] == 0.


def test_dense_alignment_rejects_missing_step():
    with pytest.raises(ValueError, match='consecutive'):
        score_dense(states(3), [[]]*3, [15, 16, 18], .1, map_data())


def rc_until(step, stride=5):
    """An RC snapshot whose GT reference ends at `step` (rc['gt'][k] is step k*stride)."""
    return {'gt': [[0., 0.]] * (step // stride + 1), 'stride': stride}


def test_traffic_efficiency_scores_every_0p1_second_frame(tmp_path):
    from odyssey_benchmark.traffic_efficiency import score_npz
    poses = states(21)
    steps = np.arange(15, 36)
    # The old 0.5 s grid sees only 2 m/s; the 16 in-between frames are
    # 4 m/s and must now contribute to the denominator.
    actors = [[actor(50., speed=2. if step % 5 == 0 else 4.)] for step in steps]
    # The GT runs past the ego's last step (an early arrival): every frame is in the window.
    rc = rc_until(100)
    value = traffic_fields(poses, actors, steps, .1, rc)
    assert value['Eff'] == pytest.approx(105. / 74. * 100.)
    assert value['Eff_coverage'] == 1.
    assert value['Eff_dt_s'] == .1
    assert value['Eff_frames'] == 21

    short = traffic_fields(poses[:11], actors[:11], steps[:11], .1, rc)
    assert short['Eff'] is None
    assert short['Eff_status'] == 'too_few_scored_frames'

    archive = tmp_path / 'rollout_trajectory.npz'
    np.savez(archive, ds_states=poses, ds_sim_steps=steps,
             driving_inputs_json=json.dumps({'actors': actors, 'sim_dt': .1, 'rc': rc}),
             # Legacy score-grid arrays deliberately disagree; pinned replay
             # must use the same dense data as live scoring.
             ego_velocity=np.zeros((3, 2)), agent_xy=np.zeros((1, 3, 2)),
             agent_velocity=np.ones((1, 3, 2)), agent_types=np.array(['VEHICLE']),
             agent_frame_offset=0)
    replayed = score_npz(str(archive))
    assert replayed.efficiency == pytest.approx(value['Eff'])
    assert replayed.n_frames == value['Eff_frames']

    legacy = tmp_path / 'legacy_rollout_trajectory.npz'
    np.savez(legacy, ego_velocity=np.array([[5., 0.]] * 3),
             ego_xy=np.zeros((3, 2)), ego_heading=np.zeros(3),
             agent_xy=np.zeros((1, 3, 2)), agent_heading=np.zeros((1, 3)),
             agent_velocity=np.array([[[2., 0.]] * 3]),
             agent_types=np.array(['VEHICLE']), agent_tokens=np.array(['car']),
             agent_frame_offset=0, score_dt=.5)
    # No pinned RC snapshot means no GT window, so the old grid arrays are not scored.
    assert score_npz(str(legacy)) is None


def test_traffic_efficiency_is_measured_inside_the_gt_window_only():
    # The run outlives the GT by 20 steps: the log's car is gone and the ego crawls at 1 m/s.
    steps = np.arange(15, 56)
    poses = states(len(steps)); poses[steps > 35, 3] = 1.
    actors = [[actor(50., speed=4.)] if step <= 35 else [] for step in steps]
    value = traffic_fields(poses, actors, steps, .1, rc_until(35))
    assert value['Eff'] == pytest.approx(5. / 4. * 100.)
    # 21 of the 21 window frames -- not 21 of the 41 run frames, which read as sparse traffic.
    assert value['Eff_coverage'] == 1.
    assert value['Eff_low_coverage'] == 0
    assert value['Eff_frames'] == 21

    # A car still driven after the GT ends (IDM keeps log-expired cars) is outside the window.
    kept = [[actor(50., speed=4. if step <= 35 else 10.)] for step in steps]
    assert traffic_fields(poses, kept, steps, .1, rc_until(35))['Eff'] \
        == pytest.approx(125.)
    # So is a car that only starts moving after the GT ends: inside the window it is parked.
    late = [[actor(50., speed=0. if step <= 35 else 4.)] for step in steps]
    assert traffic_fields(poses, late, steps, .1, rc_until(35))['Eff_status'] \
        == 'all_parked'


def test_traffic_efficiency_without_a_gt_reference_is_unmeasured():
    steps = np.arange(15, 36)
    actors = [[actor(50., speed=4.)] for _ in steps]
    value = traffic_fields(states(len(steps)), actors, steps, .1, {'gt': None, 'stride': 5})
    assert value['Eff'] is None
    assert value['Eff_status'] == 'no_gt_window'
    # A snapshot without the key at all is a broken pin, not an unmeasured run.
    with pytest.raises(KeyError):
        traffic_fields(states(len(steps)), actors, steps, .1, {'stride': 5})


def test_0p1_comfort_is_separate_replayable_and_catches_brief_impulse(tmp_path):
    from odyssey_benchmark.comfort import score_npz as score_comfort_npz
    poses = states(101)
    poses[:, 3:5] = 0.
    steps = np.arange(15, 116)
    poses[50, 5] = 20.  # one 0.1 s acceleration impulse
    result = score_dense(poses, [[] for _ in steps], steps, .1, map_data())
    assert result['Comf'] == 0.
    assert result['Comf_first_violation_step'] is not None
    assert 'lon_accel' in result['Comf_violation_types']
    assert result['Comf_dt_s'] == .1
    assert result['Comf_pose_count'] == 101
    assert 'comfort' not in result  # PDM's score-grid field is untouched.

    archive = tmp_path / 'rollout_trajectory.npz'
    np.savez(archive, ds_states=poses, ds_sim_steps=steps,
             driving_inputs_json=json.dumps({'sim_dt': .1}))
    offline = score_comfort_npz(str(archive))
    for key in ('Comf', 'Comf_first_violation_step',
                'Comf_violation_types', 'Comf_max_jerk'):
        assert offline[key] == result[key]


def test_0p1_comfort_limits_and_sampling_guards():
    from odyssey_benchmark.comfort import score_states
    steps = np.arange(15, 36)
    poses = states(len(steps))
    assert score_states(poses, steps, .1)['Comf'] == 1.

    poses[:, 5] = 3.
    acceleration = score_states(poses, steps, .1)
    assert acceleration['Comf'] == 0.
    assert acceleration['Comf_first_violation_step'] == 15
    assert acceleration['Comf_violation_types'] == 'lon_accel'

    poses[:, 5] = 0.
    poses[:, 2] = 1.2 * np.arange(len(steps)) * .1
    turning = score_states(poses, steps, .1)
    assert turning['Comf'] == 0.
    assert 'yaw_rate' in turning['Comf_violation_types']

    assert score_states(poses, steps, .5)['Comf_status'] == 'requires_0p1_rollout'
    assert score_states(poses[:6], steps[:6], .1)['Comf_status'] == 'insufficient_poses'
    with pytest.raises(ValueError, match='consecutive'):
        score_states(poses, np.r_[steps[:-1], 37], .1)


@pytest.mark.parametrize('reason,sdf', [('route_deviation',0), ('destination_arrival',1), ('time_limit',1)])
def test_common_termination(reason, sdf):
    row = dict(rc_method='hmm_prefix_1m', RC=.6, P_SD=1,
               P_col=.6, P_off=.8, offroad_ratio=.2)
    apply_termination(row, reason)
    assert row['P_SD'] == sdf
    assert row['P_off'] == .8
    assert row['offroad_ratio'] == .2
    assert row['RouteDS'] == pytest.approx(.6 * .6 * .8 * sdf * 100)
    assert row['RC'] == .6


def test_departure_preserves_measured_partial_offroad_penalty():
    poses = states()
    poses[-2:, 1] = 25.
    measured = score_dense(poses, [[] for _ in poses], np.arange(15, 22), .1, map_data())
    assert 0. < measured['offroad_ratio'] < 1.
    measured.update(rc_method='hmm_prefix_1m', RC=.6, P_SD=1)
    penalty, ratio = measured['P_off'], measured['offroad_ratio']
    apply_termination(measured, 'route_deviation')
    assert measured['P_off'] == pytest.approx(penalty)
    assert measured['offroad_ratio'] == pytest.approx(ratio)
    assert measured['P_SD'] == 0
    assert measured['RouteDS'] == 0.


def test_pack_map_json_roundtrip():
    md = json.loads(json.dumps(map_data()))
    assert score_dense(states(), [[]]*7, np.arange(15, 22), .1, md)['P_off'] == 1.


def test_current_object_dynamics():
    from odyssey.components.agents.policy.pdm_planner.observation.pdm_observation import PDMObservation
    obj = PDMObservation.__new__(PDMObservation)
    obj._global_to_local_idcs = [0, 1]
    a, b = SimpleNamespace(speed=0), SimpleNamespace(speed=5)
    obj._frame_objects = [{'car': a}, {'car': b}]
    assert obj.object_at(1, 'car') is b


def test_nc_uses_current_actor_and_contact_is_judged_at_its_origin():
    """An overlap keeps the verdict of the frame it began on, however it is reclassified later.

    The actor here starts beside the ego and moving (non-fault) and then stops while the
    polygons are still overlapping, which reclassifies it STOPPED_TRACK. That is not a new
    collision: the contact never broke, and after it began the geometry is an artefact of a
    simulation with no physics -- actors are log replay and pass straight through the ego, so
    there is no frame after first contact whose pose pair could exist in the world modelled.
    """
    poses = states(4); poses[:, 0] = 0.
    moving = ['car', 'VEHICLE', -.5, 1., 0., 1., 1., 1., 5., 0.]
    stopped = moving.copy(); stopped[8] = 0.
    actors = [[moving], [moving], [moving], [stopped]]
    result = score_dense(poses, actors, np.arange(15, 19), .1, map_data())
    assert result['P_col'] == 1.
    assert result['first_violation_step'] is None
    # A track that is ALREADY stopped when contact begins is at fault from its first frame.
    actors = [[stopped.copy()], [moving], [moving], [moving]]
    actors[0][0][2] = -30.  # stopped at first observation, no contact
    result = score_dense(poses, actors, np.arange(15, 19), .1, map_data())
    assert result['P_col'] == 1.


def test_contact_origin_verdict_holds_until_polygon_overlap_breaks():
    poses = states(7); poses[:, 0] = 0.
    # A moving car starts behind the ego, then passes through to its front.
    # All five first frames still geometrically overlap the ego.
    x = [-3., -2., 0., 2., 4., 10., 4.]
    actors = [[actor(value, speed=5.)] for value in x]
    steps = np.arange(15, 22)
    continuous = score_dense(poses[:5], actors[:5], steps[:5], .1, map_data())
    assert continuous['collision_count'] == 0
    assert continuous['P_col'] == 1.
    assert continuous['no_at_fault_collisions'] == 1.
    assert continuous['first_violation_step'] is None

    # The gap at step 20 ends the episode. A new front contact
    # with the same token at step 21 must be charged normally.
    separated = score_dense(poses, actors, steps, .1, map_data())
    assert separated['collision_count'] == 1
    assert separated['P_col'] == pytest.approx(.6)
    assert separated['no_at_fault_collisions'] == 0.
    assert separated['first_violation_step'] == 21


def test_lateral_origin_is_immunized_and_latch_is_per_actor():
    """A side overlap that sweeps to the ego's front is one contact, not a new collision.

    This is the measured case: an actor overtaking a near-stopped ego in the next lane begins
    ACTIVE_LATERAL (non-fault) and turns ACTIVE_FRONT the moment its polygon crosses the ego's
    front-bumper segment, while the overlap is already shrinking.
    """
    poses = states(3); poses[:, 0] = 0.
    lateral = [[actor(0., speed=5.)], [actor(2., speed=5.)],
               [actor(4., speed=5.)]]
    result = score_dense(poses, lateral, np.arange(15, 18), .1, map_data())
    assert result['collision_count'] == 0
    assert result['first_violation_step'] is None

    # The latch is PER ACTOR: a second actor already overlapping the ego's front on its own
    # first frame is charged, and the immunised one beside it does not shield it.
    mixed = [[actor(-3., speed=5.), ['other', 'VEHICLE', 4., 0., 0., 4., 2., 1.5, 5., 0.]],
             [actor(0., speed=5.)], [actor(4., speed=5.)]]
    result = score_dense(poses, mixed, np.arange(15, 18), .1, map_data())
    assert result['collision_count'] == 1  # front actor only
    assert result['P_col'] == pytest.approx(.6)
    assert result['first_violation_step'] == 15


def test_v2_rearm_requires_actual_contact_break():
    """A break is necessary, and on its own it is not sufficient: the ego must have LEFT.

    A STATIONARY ego (poses[:, 0] = 0.) whose contact merely lasts longer than MAX_ID_TIME is
    charged once: re-arming on time alone would let a long jam whose actor polygon flickers
    charge one car many times. The rule is the same on both branches -- a break plus the ego
    actually having moved away.
    """
    poses = states(63); poses[:, 0] = 0.
    moving = ['car', 'VEHICLE', -.5, 1., 0., 1., 1., 1., 5., 0.]
    stopped = moving.copy(); stopped[8] = 0.
    frames = [[stopped]] + [[moving]] * 61 + [[stopped]]
    value = score_dense(poses, frames, np.arange(15, 78), .1, map_data())
    assert value['collision_count'] == 1  # 6.2s, but continuously touching
    frames[-2] = []
    value = score_dense(poses, frames, np.arange(15, 78), .1, map_data())
    assert value['collision_count'] == 1  # the ego never went anywhere
    assert value['P_col'] == pytest.approx(.6)


def test_v2_rearm_when_the_ego_leaves_and_comes_back():
    """The counterpart: a real departure and return IS two events."""
    poses = states(63); poses[:, 0] = np.arange(63) * .5     # 31.5 m of travel
    near = lambda x: ['car', 'VEHICLE', x, 1., 0., 1., 1., 1., 0., 0.]
    # Touch at the start, a long clear gap while the ego drives on, then touch again.
    frames = [[near(poses[i, 0] - .5)] if (i < 3 or i > 59) else [near(9e3)]
              for i in range(63)]
    value = score_dense(poses, frames, np.arange(15, 78), .1, map_data())
    assert value['collision_count'] == 2
    assert value['P_col'] == pytest.approx(.36)


def test_v2_subcentimetre_flicker_is_not_a_contact_break():
    """`intersects` is a knife edge; a 1 mm gap for one tick is not a separation.

    Millimetre gaps in the middle of one long scrape must not re-arm the charge, even when the
    ego has rolled more than 5 m forward ALONG the same car and so passes the distance test.
    One sideswipe is charged once.
    """
    poses = states(40); poses[:, 0] = np.arange(40) * .37    # 3.7 m/s, like the real run
    # The actor sits just beside the ego's path: overlapping, except for two ticks where it
    # is nudged a fraction of a millimetre clear.
    frames = [[['car', 'VEHICLE', poses[i, 0], 1.49 if i in (20, 21) else 1.4,
                0., 1., 1., 1., .03, 0.]] for i in range(40)]
    value = score_dense(poses, frames, np.arange(15, 55), .1, map_data())
    assert value['collision_count'] == 1
    assert value['P_col'] == pytest.approx(.6)


def test_no_motion_known_rc_does_not_fabricate_sdf():
    row = dict(rc_method='hmm_prefix_1m', RC=0, P_SD=float('nan'),
               P_col=1, P_off=1)
    assert apply_termination(row, 'destination_arrival')['RouteDS'] is None


def _stalled_metric(tmp_path, monkeypatch, driven_m):
    """Build one run that never (or barely) started; only the distance driven varies."""
    from odyssey_benchmark import sdroute_score as module
    from sdroute_fixtures import TestGraph
    monkeypatch.setenv('ODYSSEY_ROUTE_FILE', str(tmp_path / 'abc.npz'))
    monkeypatch.setattr(module, 'graph', lambda *_, **__: TestGraph())
    route = np.c_[np.arange(101.), np.zeros(101)]
    np.savez(tmp_path / 'abc.npz', route_xy=route, sd_edges=[0], map_location='test')
    metric = module.SDRouteMetric(dict(id='abc', cadence=SimpleNamespace(sim_dt=.1),
        metadata={'old_origin_in_current_coordinate': [0, 0]}),
        dict(position=route, heading=np.zeros(101)), 15, 5, 0.)
    steps = 60
    for k in range(steps + 1):                      # constant speed, covering only driven_m
        metric.observe(15 + k, [driven_m * k / steps, 0], 0.)
    return metric.score('destination_arrival')


def test_a_car_that_never_started_scores_zero_rather_than_failing(tmp_path, monkeypatch):
    """A run that moved too little for the matcher to resolve scores RC 0 and SDF 0, not a scoring failure.

    The matcher resamples at STEP (5 m), so a shorter drive collapses to one point and match()
    returns None, as its contract says. That None must not turn the run into SCORING_FAILED: a
    car that did not move has a known answer, it covered none of the route, so 0.
    """
    row = _stalled_metric(tmp_path, monkeypatch, 3.0)
    assert row['rc_status'] == 'no_motion'
    assert row['RC'] == 0. and row['rc_covered_m'] == 0.
    # SDF is 0, not nan: a car that did not move has a known answer.
    assert row['P_SD'] == 0. and row['P_SD_status'] == 'no_motion'


def test_a_car_that_did_move_is_still_matched_normally(tmp_path, monkeypatch):
    """The 5 m threshold is the smallest motion the matcher can resolve -- runs that went farther must not be pushed to 0."""
    row = _stalled_metric(tmp_path, monkeypatch, 40.0)
    assert row['rc_status'] != 'no_motion'
    assert row['P_SD_status'] != 'no_motion'
    assert row['RC'] > 0


def test_pinned_reference_and_graph_replay_without_sidecar(tmp_path, monkeypatch):
    from odyssey_benchmark import sdroute_score as module
    from sdroute_fixtures import TestGraph
    monkeypatch.setenv('ODYSSEY_ROUTE_FILE', str(tmp_path / 'abc.npz'))
    monkeypatch.setattr(module, 'graph', lambda _: TestGraph())
    route = np.c_[np.arange(101.), np.zeros(101)]
    np.savez(tmp_path / 'abc.npz', route_xy=route, sd_edges=[0], map_location='test')
    metric = module.SDRouteMetric(dict(id='abc', cadence=SimpleNamespace(sim_dt=.1),
        metadata={'old_origin_in_current_coordinate':[0,0]}),
        dict(position=route, heading=np.zeros(101)), 15, 5, 0.)
    for step in range(15, 77):
        metric.observe(step, [step, 0], 0.)
    before = metric.score('destination_arrival')
    snapshot = json.loads(json.dumps(metric.snapshot()))
    # No path lookup on replay, and no accidental load of today's graph.
    snapshot['sidecar'] = '/does/not/exist.npz'
    monkeypatch.setattr(module, 'graph', lambda _: pytest.fail('live graph accessed'))
    steps, xy, heading, _ = metric.arrays()
    restored = module.SDRouteMetric.from_snapshot(snapshot, steps, xy, heading)
    after = restored.score('destination_arrival')
    assert after['RC'] == pytest.approx(before['RC'])
    assert after['P_SD'] == before['P_SD'] == 1
    # Exercise the official NPZ dispatcher, with the legacy scene loader disabled.
    from odyssey_benchmark.driving_metrics import replay
    dense_steps = np.arange(15, 77)
    dense_states = states(len(dense_steps)); dense_states[:, 0] = dense_steps
    # A car beside the route, whose source ends with the unset-pose row: no velocity, back at a
    # far-off start point.
    cars = [[['car', 'VEHICLE', float(step), 30., 0., 4., 2., 1.5, 4., 0.]] for step in dense_steps]
    cars[-1] = [['car', 'VEHICLE', -500., 0., 0., 4., 2., 1.5, 0., 0.]]
    inputs = dict(rc=snapshot, map=map_data(), actors=cars, sim_dt=.1,
                  term_reason='destination_arrival', departure_distance_m=30.)
    path = tmp_path / 'saved.npz'
    np.savez(path, driving_inputs_json=json.dumps(inputs), ds_states=dense_states,
             ds_sim_steps=dense_steps, rc_sim_steps=steps, rc_ego_xy=xy, rc_ego_heading=heading,
             scene='abc', map_location='test')
    with np.load(path, allow_pickle=False) as saved:
        result = replay(saved)
    assert result['RC'] == pytest.approx(before['RC'])
    assert result['ds_final_step'] == 76 and result['P_SD'] == 1
    # The TE CLI reads the same pins through the same row filter and the same GT window, so it
    # reproduces the replayed value. The unset row would otherwise add a 0 m/s sample.
    from odyssey_benchmark.traffic_efficiency import score_npz
    cli = score_npz(str(path))
    assert result['Eff'] == pytest.approx(5. / 4. * 100.)
    assert cli.efficiency == pytest.approx(result['Eff'])
    assert cli.coverage == result['Eff_coverage']
    assert cli.n_frames == result['Eff_frames']


def test_pinned_manual_graph_preserves_stable_ids_and_replaced_edge_mask():
    from odyssey_benchmark.sdroute_sdf import graph
    from odyssey_benchmark.driving_metrics import pack_graph, unpack_graph
    from odyssey_bridge.sd_route import replaced_sd_edges

    source = graph('sg-one-north', include_manual=True)
    restored = unpack_graph(json.loads(json.dumps(pack_graph(source))))
    manual = source.edge_index[-1]
    assert restored.edge_ids[manual] == -1
    assert manual in restored.candidates(source.geom[manual][0], 1.)
    replaced = replaced_sd_edges('sg-one-north')
    assert replaced
    assert not replaced.intersection(restored.active)
    assert not replaced.intersection(e for edges in restored.out.values() for e in edges)


def test_legacy_archive_vocabulary_is_normalized_before_scoring():
    from odyssey_benchmark.driving_metrics import normalize_legacy_inputs
    data = {'term_reason': 'gt_reached', 'rules': {'lane': 'v4_penalty', 'tl': 'r7'},
            'rc': {'method': 'sd_sdf_prefix_1m_v1'}, 'tlc': {'version': 'tlc_pinned_events_v3'}}
    out = normalize_legacy_inputs(data)
    assert out is data
    assert data['term_reason'] == 'destination_arrival'
    assert data['rules'] == {'lane': 'plc', 'tl': 'r7'}
    assert data['rc']['method'] == 'hmm_prefix_1m'
    assert data['tlc']['version'] == 'tl_events_v3'
    # Current vocabulary passes through untouched; missing blocks are tolerated.
    assert normalize_legacy_inputs({'term_reason': 'time_limit'}) == {'term_reason': 'time_limit'}
