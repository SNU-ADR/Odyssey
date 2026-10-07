"""Both ends of lane-follow scoring: the lane_graph the rollout bakes, and the scorer that runs on it alone.

Each piece being right on its own is not enough, so one file chains them -- fake nuPlan lane
objects are baked by `pack_lane_graph`, and `odyssey_benchmark/lane_follow.py` reads the result and
produces the verdict. Runs without the simulator or a map.

The geometry is one straight two-lane roadblock. Only the right lane continues into the next
connector, so an ego that drives in the left lane must fail and one in the right lane must pass.
The stop line is the roadblock end (x=0) extended sideways by HALF_WIDTH on each side, and the
verdict is taken when the ego crosses it in the lane direction -- an approach from outside the
route lanes (oncoming lane, off road) is still judged.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import box

from odyssey_benchmark.driving_metrics import pack_lane_graph

from odyssey_benchmark import lane_follow as LF

LANE_W = 3.4
RB_LEN = 40.0


def _lane(lane_id, y0, x0, x1, exits, curve=None):
    """Fake nuPlan lane with only a polygon, a centreline and outgoing edges."""
    path = curve if curve is not None else [(x, y0) for x in np.linspace(x0, x1, 21)]
    half = LANE_W / 2
    ring = ([(x, y + half) for x, y in path] + [(x, y - half) for x, y in reversed(path)])
    ring.append(ring[0])
    return SimpleNamespace(
        id=lane_id,
        polygon=SimpleNamespace(exterior=SimpleNamespace(coords=ring)),
        baseline_path=SimpleNamespace(
            discrete_path=[SimpleNamespace(x=x, y=y) for x, y in path]),
        outgoing_edges=[SimpleNamespace(id=e) for e in exits],
    )


@pytest.fixture
def world():
    """Roadblock 'A' (2 lanes) -> connector 'C' (1 lane). Only the right lane continues."""
    left = _lane('L0', LANE_W, -RB_LEN, 0.0, exits=['X9'])        # exits elsewhere
    right = _lane('L1', 0.0, -RB_LEN, 0.0, exits=['CL'])          # continues into the connector
    conn = _lane('CL', 0.0, 0.0, 30.0, exits=['Z0'])
    blocks = {'A': SimpleNamespace(interior_edges=[left, right]),
              'C': SimpleNamespace(interior_edges=[conn])}
    map_api = SimpleNamespace(
        get_map_object=lambda rb, layer: SimpleNamespace() if rb == 'A' else None)
    return blocks, map_api


def _graph(world):
    blocks, map_api = world
    return LF.LaneGraph({k: {'rb': v['rb'], 'type': v['type'],
                             'polygon': np.asarray(v['polygon'], float),
                             'polyline': np.asarray(v['polyline'], float),
                             'exit': v['exit'], 'left_count': None}
                         for k, v in pack_lane_graph(blocks, map_api).items()})


def _drive(y, x_end=12.0):
    """ds_states driving left to right, with the rear axle shifted back so the centre runs along y."""
    xs = np.arange(-RB_LEN + 2.0, x_end, 0.5) - LF.REAR_TO_CENTRE
    states = np.zeros((len(xs), 11))
    states[:, 0], states[:, 1], states[:, 2] = xs, y, 0.0
    return states


def test_pack_lane_graph_uses_the_scenario_builders_vocabulary(world):
    blocks, map_api = world
    packed = pack_lane_graph(blocks, map_api)
    assert set(packed) == {'L0', 'L1', 'CL'}
    # One scorer reads both pkl and npz, so the type names must not diverge.
    assert packed['L1']['type'] == 'LANE_SURFACE_STREET'
    assert packed['CL']['type'] == 'LANE_SURFACE_UNSTRUCTURE'
    assert packed['L1']['exit'] == ['CL'] and packed['L0']['exit'] == ['X9']
    assert packed['L1']['rb'] == 'A' and packed['CL']['rb'] == 'C'
    # What the rollout bakes is shipped as json, so it must be serializable.
    json.dumps(packed)


def test_a_pose_outside_every_lane_is_not_credited_to_the_nearest(world):
    """Failure mode of the old 4 m threshold: a pose outside the polygon is outside the lane."""
    graph = _graph(world)
    inside, _d, _rem = graph.assign(np.array([-20.0, 0.0]), 0.0)
    assert inside == 'L1'
    # 2.6 m from the right lane centre: outside half the 3.4 m lane width, but the old 4 m rule caught it.
    outside, _d, _rem = graph.assign(np.array([-20.0, -2.6]), 0.0)
    assert outside is None


BLOCKS = [('A', box(-RB_LEN, -LANE_W / 2, 0.0, LANE_W * 1.5)),            # route roadblock
          ('B', box(-RB_LEN, LANE_W * 1.5, 0.0, LANE_W * 3.5))]           # adjacent oncoming lane


def _log(states):
    xy, h = LF.ego_centre(states)
    return xy, h, np.arange(len(states)) + 15


def _score(world, states, log=None):
    """Without `log`, the log passes the stop line in the right (feasible) lane, so the stop line is scored."""
    graph = _graph(world)
    gs, nf = LF.gates(graph, ['A', 'C'])
    d = dict(route=['A', 'C'], graph=graph, blocks=BLOCKS, gates=gs, no_feasible=nf,
             log=_log(_drive(0.0) if log is None else log))
    xy, h = LF.ego_centre(states)
    return LF.score(d, xy, h, np.arange(len(states)) + 15, token='t')


def test_judges_the_lane_at_the_stop_line(world):
    right = _score(world, _drive(0.0))
    assert right.n_reached == 1 and right.lane_score == pytest.approx(1.0)
    assert right.stops[0].how == 'passed' and right.stops[0].where == 'feasible'
    left = _score(world, _drive(LANE_W))
    assert left.n_reached == 1 and left.lane_score == pytest.approx(0.0)
    assert left.stops[0].where == 'wrong_lane'


def test_an_oncoming_approach_is_judged_and_fails(world):
    """An approach from outside the route lanes is still judged when it crosses the stop line. The old rule
    (assign only to route lanes, judge at the last sample in that roadblock) dropped this approach entirely."""
    result = _score(world, _drive(LANE_W * 2.5))       # middle of the oncoming lane
    assert result.n_reached == 1 and result.n_fail == 1 and result.stops[0].where == 'off_route'


def test_the_stop_line_reaches_half_width_to_each_side(world):
    """Driving off road is still judged if it crosses within HALF_WIDTH sideways of the stop line; beyond that it is not."""
    near = _score(world, _drive(-(LF.HALF_WIDTH - 5.0)))
    assert near.n_reached == 1 and near.stops[0].where == 'off_road'
    far = _score(world, _drive(-(LF.HALF_WIDTH + 5.0)))
    assert far.n_reached == 0 and far.lane_score is None


def test_crossing_the_line_backwards_is_not_passing_it(world):
    states = _drive(0.0)[::-1].copy()
    states[:, 2] = np.pi                               # drives the route backwards
    assert _score(world, states).n_reached == 0


def test_stop_lines_without_a_lane_choice_are_not_gates(world):
    """If both lanes continue there is no choice to make, so it is not a gate (this is not a wrong-way metric)."""
    blocks, _ = world
    blocks['A'].interior_edges[0].outgoing_edges = [SimpleNamespace(id='CL')]
    result = _score(world, _drive(LANE_W * 2.5))       # even when passing in the oncoming lane
    assert result.n_stops == 0 and result.lane_score is None


def test_a_rollout_that_starts_past_the_stop_line_is_judged_at_its_first_sample(world):
    """Judged even if warm-up already crossed the stop line, as long as the ego is still in its own lane
    (e.g. the infeasible lane ends first so the stop line is there, and the ego starts past it)."""
    blocks, _ = world
    blocks['A'].interior_edges[0] = _lane('L0', LANE_W, -RB_LEN, -3.0, exits=['X9'])  # ends 3 m earlier
    states = _drive(0.0)
    x = states[:, 0] + LF.REAR_TO_CENTRE
    result = _score(world, states[x > -2.0])           # starts past the stop line (x=-3), still inside L1
    assert result.n_reached == 1 and result.stops[0].how == 'started_past'
    assert result.stops[0].passed is True
    in_connector = _score(world, states[x > 1.0])      # already in the connector -- warm-up made the choice
    assert in_connector.n_reached == 0


def test_ending_just_short_of_the_line_is_judged_far_short_is_not(world):
    near = _score(world, _drive(LANE_W, x_end=-LF.NEAR_TOL / 2))
    assert near.n_reached == 1 and near.stops[0].how == 'ended_near' and near.n_fail == 1
    far = _score(world, _drive(LANE_W, x_end=-12.0))
    assert far.n_reached == 0 and far.lane_score is None
    assert far.reason == 'no_stop_reached'


def _change_lane_at(x_change, y_before=LANE_W, y_after=0.0):
    """Drive at y_before, then move to y_after once the centre passes x_change."""
    states = _drive(y_after)
    x = states[:, 0] + LF.REAR_TO_CENTRE
    states[x < x_change, 1] = y_before
    return states


def test_a_lane_change_inside_the_last_10_m_gets_half_credit(world):
    """In the feasible lane at the stop line (x=0), but it entered that lane only 5 m before the line."""
    result = _score(world, _change_lane_at(-5.0))
    v = result.stops[0]
    assert v.passed is True and v.credit == LF.LATE_CREDIT
    assert 4.0 <= v.entered_m < LF.EARLY_M
    assert (result.n_pass, result.n_late) == (1, 1) and result.lane_score == pytest.approx(0.5)


def test_a_lane_change_before_the_last_10_m_gets_full_credit(world):
    result = _score(world, _change_lane_at(-15.0))
    assert result.stops[0].credit == 1.0 and result.stops[0].entered_m >= LF.EARLY_M
    assert result.n_late == 0 and result.lane_score == pytest.approx(1.0)


def test_off_road_inside_the_last_10_m_is_not_in_the_correct_lane(world):
    """Driving in the oncoming lane and cutting into the correct lane 5 m before the stop line is a late pass."""
    result = _score(world, _change_lane_at(-5.0, y_before=LANE_W * 2.5))
    assert result.stops[0].passed is True and result.stops[0].credit == LF.LATE_CREDIT


def test_a_wrong_lane_at_the_line_stays_zero(world):
    result = _score(world, _change_lane_at(-5.0, y_before=0.0, y_after=LANE_W))
    assert result.stops[0].credit == 0.0 and result.n_late == 0


def test_a_rollout_that_starts_inside_the_last_10_m_is_not_late(world):
    """The model did not drive the part before the first sample -- the correct lane from the first sample is not late."""
    states = _drive(0.0)
    x = states[:, 0] + LF.REAR_TO_CENTRE
    v = _score(world, states[x > -6.0]).stops[0]
    assert v.passed is True and v.entered_m is None and v.credit == 1.0


def test_on_a_short_roadblock_the_correct_lane_starts_in_the_one_before(world):
    """When the stop-line roadblock is only 6 m long, most of the 10 m before it is in the previous
    roadblock 'P'. P1, which leads into the feasible lane (L1) without a lane change, is a correct
    lane; P0, which leads into the infeasible lane (L0), is not."""
    blocks, _ = world
    blocks['A'].interior_edges[:] = [_lane('L0', LANE_W, -6.0, 0.0, exits=['X9']),
                                     _lane('L1', 0.0, -6.0, 0.0, exits=['CL'])]
    blocks['P'] = SimpleNamespace(interior_edges=[_lane('P0', LANE_W, -RB_LEN, -6.0, exits=['L0']),
                                                  _lane('P1', 0.0, -RB_LEN, -6.0, exits=['L1'])])
    map_api = SimpleNamespace(
        get_map_object=lambda rb, layer: SimpleNamespace() if rb in ('A', 'P') else None)
    graph = LF.LaneGraph({k: {'rb': v['rb'], 'type': v['type'],
                              'polygon': np.asarray(v['polygon'], float),
                              'polyline': np.asarray(v['polyline'], float),
                              'exit': v['exit'], 'left_count': None}
                          for k, v in pack_lane_graph(blocks, map_api).items()})
    route = ['P', 'A', 'C']
    gs, nf = LF.gates(graph, route)
    assert [g.rb for g in gs] == ['A'] and set(gs[0].approach) == {'L1', 'P1'}
    d = dict(route=route, graph=graph, blocks=BLOCKS, gates=gs, no_feasible=nf, log=_log(_drive(0.0)))

    def credit(states):
        xy, h = LF.ego_centre(states)
        return LF.score(d, xy, h, np.arange(len(states)) + 15).stops[0].credit

    assert credit(_drive(0.0)) == 1.0                  # P1 -> L1: correct lane from the start
    assert credit(_change_lane_at(-3.0)) == LF.LATE_CREDIT   # P0 -> L0 -> L1 3 m before the stop line


# --------------------------------------------------------------------------- log reference
STRIDE, HANDOFF = 5, 15


def _rc(log_states):
    """rc snapshot: gt[k] is the log rear axle at step k*stride. The log is assumed to start at step HANDOFF."""
    pts = log_states[::STRIDE, :2]
    lead = np.repeat(pts[:1], HANDOFF // STRIDE, axis=0)      # before handoff it stood at the first point
    return {'gt': np.vstack([lead, pts]).tolist(), 'stride': STRIDE, 'handoff': HANDOFF}


def _expert(log_states):
    """The expert metric_manager bakes: 0.5 s samples from handoff, relative to the first point, plus the origin."""
    pts, h = log_states[::STRIDE, :2], log_states[::STRIDE, 2]
    centre = np.array([100.0, 50.0, 0.0])                      # global = expert_xy + origin + centre[:2]
    return dict(expert_xy=pts - pts[0], expert_heading=h, expert_origin=pts[0] - centre[:2],
                initial_ego_center=centre, expert_source_pose_count=np.array(len(pts)))


def test_a_stop_line_the_log_passed_late_is_not_scored(world):
    """A stop line the human driver entered late (5 m before it, credit 0.5) is not scored -- no verdict, not counted in N."""
    result = _score(world, _drive(LANE_W), log=_change_lane_at(-5.0))
    assert result.n_stops == 0 and result.n_reached == 0 and result.lane_score is None
    assert [v.rb for v in result.excluded] == ['A'] and result.excluded[0].credit == LF.LATE_CREDIT


def test_a_stop_line_the_log_got_wrong_or_never_reached_is_not_scored(world):
    wrong = _score(world, _drive(0.0), log=_drive(LANE_W))
    assert wrong.n_stops == 0 and wrong.excluded[0].where == 'wrong_lane'
    states = _drive(0.0)
    x = states[:, 0] + LF.REAR_TO_CENTRE
    started_past = _score(world, states[x > 1.0], log=states[x > 1.0])   # the log starts past the line
    assert started_past.n_stops == 0 and started_past.excluded[0].reached is False


def test_a_stop_line_the_log_passed_is_scored_as_before(world):
    result = _score(world, _drive(LANE_W), log=_drive(0.0))
    assert result.n_stops == 1 and result.n_fail == 1 and result.excluded == []


def test_no_log_means_no_value(world):
    graph = _graph(world)
    gs, nf = LF.gates(graph, ['A', 'C'])
    d = dict(route=['A', 'C'], graph=graph, blocks=BLOCKS, gates=gs, no_feasible=nf, log=None)
    xy, h = LF.ego_centre(_drive(0.0))
    r = LF.score(d, xy, h, np.arange(len(xy)) + 15)
    assert r.lane_score is None and r.reason == 'no_log' and r.n_stops == 0


def test_log_track_takes_its_steps_from_rc_gt(tmp_path):
    log = _drive(0.0)
    np.savez(tmp_path / 'z.npz', **_expert(log))
    with np.load(tmp_path / 'z.npz') as z:
        xy, h, steps = LF.log_track(z, _rc(log))
        assert steps[0] == HANDOFF and np.all(np.diff(steps) == 1)
        cxy, _ = LF.ego_centre(log)
        assert np.allclose(xy[0], cxy[0]) and np.allclose(xy[STRIDE], cxy[STRIDE])
        bad = _rc(log)
        bad['handoff'] = HANDOFF + STRIDE                   # shifting the handoff by one stride misaligns the points
        with pytest.raises(ValueError):
            LF.log_track(z, bad)
