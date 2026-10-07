"""What the sector replay clock has to guarantee.

The hybrid clock ties the traffic's playback rate to the ego's speed. These pin the
property that buys back -- inside a sector the log runs at log speed and the offset is
re-anchored only at the boundaries -- along with the two the clock keeps from the
viewer it was ported from: an ego that drives the recording sees the recording, and
progress never rewinds.
"""

import numpy as np
import pytest

from odyssey.scenario.hybrid_replay import (SECTOR_LEAD_M, SECTOR_LEN_S, HybridReplay,
                                                SectorReplay)


def straight_route(n=100, dx=1.0):
    xy = np.stack([np.arange(n) * dx, np.zeros(n)], axis=1)
    return xy, np.zeros(n)


def track(k0, k1, x0, dx, y):
    k = np.arange(k0, k1)
    xy = np.stack([x0 + (k - k0) * dx, np.full(len(k), float(y))], axis=1)
    return k, xy


def test_sectors_span_the_default_twenty_seconds_of_log_time():
    route_xy, _ = straight_route()
    r = SectorReplay(route_xy, {'a': track(0, 100, 0.0, 1.0, 10.0)}, src_dt=0.5)

    assert SECTOR_LEN_S == 20.0
    # 20 s of log time at 0.5 s a row is 40 rows, whatever distance that covers
    assert list(r.bound_rows) == [0, 40, 80]


def test_the_log_keeps_running_while_the_ego_stands_still():
    """The reason this clock exists: a stopped ego must not stop the city."""
    route_xy, route_h = straight_route()
    tracks = {'a': track(0, 100, 0.0, 1.0, 10.0)}
    sector = SectorReplay(route_xy, tracks, src_dt=0.5)
    hybrid = HybridReplay(route_xy, route_h, tracks, src_dt=0.5)

    held = (10.0, 0.0)
    assert [sector.step(t, held)['a'] for t in range(4)] == [0.0, 1.0, 2.0, 3.0]
    # the parent freezes at the row the ego's progress calls for, which is the point
    assert [hybrid.step(t, held)['a'] for t in range(4)] == [10.0] * 4


def test_a_sector_plays_at_log_speed_and_re_anchors_at_the_boundary():
    route_xy, _ = straight_route()
    tracks = {'a': track(0, 100, 45.0, 0.0, 10.0)}      # first seen at x=45 -> sector 1
    r = SectorReplay(route_xy, tracks, src_dt=0.5)
    assert r.sector['a'] == 1

    half_pace = lambda t: (0.5 * t, 0.0)
    # sector 1 starts at row 40, which is 40 m along; at half pace the ego needs 80 steps
    assert r.step(79, half_pace(79))['a'] == -1.0
    assert r.step(80, half_pace(80))['a'] == pytest.approx(40.0)
    # from there the log runs at its own speed: 10 steps later it is 10 rows on, even
    # though the ego has covered only 5 m
    assert r.step(90, half_pace(90))['a'] == pytest.approx(50.0)


def test_actors_in_one_sector_keep_the_gap_the_log_gave_them():
    """One offset a sector, so within it the geometry is the log's."""
    route_xy, _ = straight_route()
    tracks = {'lead': track(0, 100, 26.0, 1.0, 0.5),
              'follow': track(0, 100, 20.0, 1.0, 0.5)}
    r = SectorReplay(route_xy, tracks, src_dt=0.5)
    assert r.sector['lead'] == r.sector['follow']

    for t in (0, 5, 20):
        out = r.step(t, (0.6 * t, 0.0))
        assert out['lead'] == out['follow']        # same row -> the recorded 6 m gap


def test_an_unreached_sector_is_not_on_the_road_and_backing_up_does_not_open_it():
    route_xy, _ = straight_route()
    tracks = {'near': track(0, 100, 5.0, 0.0, 10.0),
              'far': track(0, 100, 85.0, 0.0, 10.0)}
    r = SectorReplay(route_xy, tracks, src_dt=0.5)
    assert (r.sector['near'], r.sector['far']) == (0, 2)

    out = r.step(0, (60.0, 0.0))
    assert out['near'] == pytest.approx(0.0)
    assert out['far'] == -1.0
    # backing up neither rewinds the sectors already entered nor opens a new one
    assert r.step(1, (20.0, 0.0))['far'] == -1.0
    assert r.step(2, (85.0, 0.0))['far'] == pytest.approx(80.0)


def test_reverse_moving_actor_belongs_to_the_earliest_sector_its_track_reaches():
    """A later first sighting must not make an oncoming actor pop in behind the ego.

    The actor is first observed in sector 2 but subsequently travels back into sector 0.
    Assigning only its first pose would keep it absent until the sector-2 gate opens, at
    which point its current source row can already be beside the ego in sector 0.
    """
    route_xy, _ = straight_route()
    reverse = (np.arange(60), np.stack([np.linspace(85.0, 25.0, 60),
                                        np.ones(60)], axis=1))
    r = SectorReplay(route_xy, {'reverse': reverse}, src_dt=0.5)

    assert r.sector['reverse'] == 0
    assert r.step(0, (0.0, 0.0))['reverse'] == pytest.approx(0.0)


def test_replaying_the_recording_reproduces_it():
    """The identity check: every sector's offset is zero when the ego drives the log."""
    route_xy, _ = straight_route()
    tracks = {'a': track(0, 100, 0.0, 1.0, 10.0),
              'lead': track(0, 100, 20.0, 1.0, 1.0)}
    r = SectorReplay(route_xy, tracks, src_dt=0.5)
    for step in range(100):
        out = r.step(step, (float(step), 0.0))
        assert out['a'] == pytest.approx(float(step))
        assert out['lead'] == pytest.approx(float(step))


def test_signals_share_their_sector_actor_clock_and_unopened_sectors_are_unknown():
    route_xy, _ = straight_route()
    tracks = {
        'near': track(0, 100, 5.0, 0.0, 10.0),
        'far': track(0, 100, 85.0, 0.0, 10.0),
    }
    replay = SectorReplay(
        route_xy,
        tracks,
        signals={'near-light': [5.0, 0.0], 'far-light': [85.0, 0.0]},
        src_dt=0.5,
    )

    actor_rows = replay.step(0, (0.0, 0.0))
    signal_rows = replay.signal_rows(0)
    assert signal_rows['near-light'] == actor_rows['near'] == pytest.approx(0.0)
    assert signal_rows['far-light'] == actor_rows['far'] == -1.0

    actor_rows = replay.step(120, (85.0, 0.0))
    signal_rows = replay.signal_rows(120)
    assert signal_rows['far-light'] == actor_rows['far'] == pytest.approx(80.0)


def test_signal_rows_retain_historical_sector_activation():
    route_xy, _ = straight_route()
    replay = SectorReplay(
        route_xy,
        {'far': track(0, 100, 85.0, 0.0, 10.0)},
        signals={'far-light': [85.0, 0.0]},
        src_dt=0.5,
    )

    replay.step(0, (0.0, 0.0))
    replay.step(120, (85.0, 0.0))

    # TLC is packed after the rollout. Opening sector 2 at step 120 must not rewrite
    # history and make its signal appear at step 119.
    assert replay.signal_rows(119)['far-light'] == -1.0
    assert replay.signal_rows(120)['far-light'] == pytest.approx(80.0)


def test_a_lead_puts_the_actors_on_the_road_before_the_boundary():
    """Placing actors right at the boundary makes them pop up beside the ego or where it has already passed."""
    route_xy, _ = straight_route()
    tracks = {'a': track(0, 100, 45.0, 0.0, 10.0)}      # sector 1 (boundary at 40 m)
    plain = SectorReplay(route_xy, tracks, src_dt=0.5)
    lead = SectorReplay(route_xy, tracks, lead_m=15.0, src_dt=0.5)
    assert SECTOR_LEAD_M == 15.0
    assert list(lead.spawn_s) == [0.0, 25.0, 65.0]      # boundary - 15 m, clipped at 0

    # drive at 0.5 m/step from the start -- sector entry must register when the ego actually passes
    rows = {r: [r.step(t, (0.5 * t, 0.0))['a'] for t in range(100)] for r in (plain, lead)}
    # no lead: off the road until step 80, when the ego reaches the boundary (40 m)
    assert rows[plain][79] == -1.0 and rows[plain][80] != -1.0
    # 15 m lead: on the road from 25 m, i.e. step 50
    assert rows[lead][49] == -1.0 and rows[lead][50] != -1.0


def test_nothing_happens_when_the_ego_crosses_the_sector_line_itself():
    """With lead spawning an actor enters once at the spawn line, and the sector line is a non-event.

    Re-anchoring the offset at the sector line makes the row jump there, and an actor whose valid
    interval falls on that jump pops up at the boundary -- the artefact lead spawning was meant to
    remove just moves to the boundary.
    """
    route_xy, _ = straight_route()
    tracks = {'a': track(0, 100, 45.0, 0.0, 10.0)}     # sector 1 (line at 40 m, spawn line at 25 m)
    r = SectorReplay(route_xy, tracks, lead_m=15.0, src_dt=0.5)
    rows = [r.step(t, (0.5 * t, 0.0))['a'] for t in range(100)]

    # enters at the spawn line (25 m = step 50) on that line's log row
    assert rows[49] == -1.0
    assert rows[50] == pytest.approx(25.0)             # the route is 1 m per row, so 25 m = row 25
    assert r.is_eligible('a')
    # crossing the sector line (40 m = step 80) changes nothing: still one row per step
    live = rows[50:]
    assert all(b - a == pytest.approx(1.0) for a, b in zip(live, live[1:]))


def test_a_lead_still_reproduces_the_recording_when_the_ego_drives_it():
    """Both offset terms must come from the same point -- otherwise driving the recording exactly still misses.

    Subtracting the sector line's log row from the spawn line's step leaves the time spent crossing
    the lead span in the offset, so offsets drift by many rows per sector and some steps end up with
    empty surroundings.
    """
    route_xy, _ = straight_route()
    tracks = {'a': track(0, 100, 45.0, 0.0, 10.0),
              'b': track(0, 100, 85.0, 0.0, 10.0)}
    for lead in (0.0, 15.0, 30.0):
        r = SectorReplay(route_xy, tracks, lead_m=lead, src_dt=0.5)
        for t in range(100):
            out = r.step(t, (float(t), 0.0))           # exactly the recording: 1 m/step
            for oid, row in out.items():
                assert row == -1.0 or row == pytest.approx(float(t)), (lead, t, oid, row)


def test_no_lead_is_the_default_and_changes_nothing():
    route_xy, _ = straight_route()
    tracks = {'a': track(0, 100, 45.0, 0.0, 10.0)}
    r = SectorReplay(route_xy, tracks, src_dt=0.5)
    assert r.lead_m == 0.0
    assert list(r.spawn_s) == list(r.bound_s)


def test_the_manager_builds_the_clock_the_mode_asks_for():
    """Which clock each agent_replay value actually builds -- if the branch dies, hybrid runs silently."""
    pytest.importorskip("omegaconf")
    from types import SimpleNamespace

    from odyssey.manager.agent_manager import BaseAgentManager
    from odyssey.scenario.scenarios.scenario_description import ScenarioDescription as SD

    n = 100
    xy = np.stack([np.arange(n) * 1.0, np.zeros(n)], axis=1)
    ego_state = {SD.VALID: np.ones(n, dtype=bool),
                 SD.POSITION: np.concatenate([xy, np.zeros((n, 1))], axis=1),
                 SD.HEADING: np.zeros(n)}
    actor_state = {SD.VALID: np.ones(n, dtype=bool),
                   SD.POSITION: np.concatenate([xy + [0.0, 10.0], np.zeros((n, 1))], axis=1),
                   SD.HEADING: np.zeros(n)}

    class _Manager(BaseAgentManager):
        """Only the two things _build_replay_clock reads: the config and the tracks."""

        def __init__(self, cfg):
            self._cfg = cfg

        @property
        def engine(self):
            return SimpleNamespace(
                global_config=self._cfg, sim_dt=0.5,
                current_scene={SD.OBJECT_TRACKS: {'ego': {'type': 'VEHICLE'},
                                                  'a': {'type': 'VEHICLE'}}})

        def _process_agent_track(self, obj_id, _):
            return (ego_state if obj_id == 'ego' else actor_state), None

    def build(cfg):
        return _Manager(cfg)._build_replay_clock(cfg['agent_replay'])

    hybrid = build({'agent_replay': 'hybrid'})
    assert isinstance(hybrid, HybridReplay) and not isinstance(hybrid, SectorReplay)

    default = build({'agent_replay': 'sector'})
    assert isinstance(default, SectorReplay)
    assert list(default.bound_rows) == [0, 40, 80]        # 20 s of log time at 0.5 s a row

    tuned = build({'agent_replay': 'sector', 'sector_len_s': 10})
    assert list(tuned.bound_rows) == [0, 20, 40, 60, 80]


def test_parked_cars_and_road_furniture_stay_on_the_log_clock():
    """Parked cars and cones stay off the sector clock -- the original viewer's (nr.js) static_always.

    The viewer draws static objects for the whole run, regardless of clock or trigger. Putting them
    in sectors makes parked cars appear all at once when the ego crosses the spawn line.
    The static test is the same one the simulator uses to make an agent static.
    """
    pytest.importorskip("omegaconf")
    from types import SimpleNamespace

    from odyssey.manager.agent_manager import BaseAgentManager
    from odyssey.scenario.scenarios.scenario_description import ScenarioDescription as SD

    n = 100
    xy = np.stack([np.arange(n) * 1.0, np.zeros(n)], axis=1)

    def state(pos):
        return {SD.VALID: np.ones(n, dtype=bool),
                SD.POSITION: np.concatenate([pos, np.zeros((n, 1))], axis=1),
                SD.HEADING: np.zeros(n)}

    states = {'ego': state(xy),
              'moving': state(xy + [0.0, 10.0]),
              'parked': state(np.tile([60.0, 5.0], (n, 1))),
              'cone': state(np.tile([70.0, -3.0], (n, 1)))}
    tracks = {'ego': {'type': 'VEHICLE'}, 'moving': {'type': 'VEHICLE'},
              'parked': {'type': 'VEHICLE'}, 'cone': {'type': 'TRAFFIC_CONE'}}

    class _Manager(BaseAgentManager):
        def __init__(self, cfg):
            self._cfg = cfg

        @property
        def engine(self):
            return SimpleNamespace(global_config=self._cfg, sim_dt=0.5,
                                   current_scene={SD.OBJECT_TRACKS: tracks})

        def _process_agent_track(self, obj_id, _):
            return states[obj_id], None

    clock = _Manager({'agent_replay': 'sector'})._build_replay_clock('sector')
    assert set(clock.sector) == {'moving'}
    rows = clock.step(0, xy[0])
    assert 'parked' not in rows and 'cone' not in rows      # run on the log clock (simulation_step)


def test_an_absent_actor_is_not_handed_the_logs_first_row():
    """-1 marks 'not on the road'. Clamping it to 0 puts a ghost at the local origin.

    The ghost vanishes once the sector opens and the row turns non-negative, so on screen a car
    disappears at every boundary.
    """
    pytest.importorskip("omegaconf")
    from odyssey.manager.agent_manager import BaseAgentManager

    written = {}

    class _Agent:
        def __init__(self, oid): self.oid = oid
        def set_traj_step(self, v): written[self.oid] = v

    class _Manager(BaseAgentManager):
        def __init__(self): pass

        @property
        def all_agents(self):
            return {'ego': _Agent('ego'), 'a': _Agent('a'), 'b': _Agent('b')}

    _Manager()._write_replay_rows({'ego': 7.0, 'a': -1.0, 'b': 12.0})
    assert 'a' not in written        # no row is written for an actor that is not on the road
    assert 'ego' not in written      # this clock never touches the ego
    assert written == {'b': 12}


def test_the_boundary_goes_between_two_junctions_not_through_one():
    """Changing the time offset in the middle of an intersection splits the traffic crossing it."""
    from odyssey.scenario.hybrid_replay import intersection_bounds

    route_xy, _ = straight_route(400)                  # straight line 0..399 m, 1 m spacing
    cum = np.arange(400, dtype=float)
    box = lambda x0, x1: np.array([[x0, -5.], [x1, -5.], [x1, 5.], [x0, 5.]])
    # three intersections: 100-110, 200-210, and 120-130 right next to the first
    polys = [box(100, 110), box(120, 130), box(200, 210)]
    b = intersection_bounds(route_xy, cum, polys, approach_m=0.5, min_m=50.0)

    # the midpoint of 110-120 (115) is at least 50 m from the previous boundary (0), so it is kept,
    # and the midpoint of 130-200 (165) is at least 50 m from 115, so it is kept too
    assert [round(float(x)) for x in b] == [0, 115, 165]
    # no boundary falls inside an intersection
    for s in b[1:]:
        for poly in polys:
            assert not (poly[:, 0].min() <= s <= poly[:, 0].max())


def test_a_sector_never_comes_out_shorter_than_the_minimum():
    """The minimum length is a floor on sector length -- which is why nearby intersections share a sector."""
    from odyssey.scenario.hybrid_replay import intersection_bounds

    route_xy, _ = straight_route(400)
    cum = np.arange(400, dtype=float)
    box = lambda x0, x1: np.array([[x0, -5.], [x1, -5.], [x1, 5.], [x0, 5.]])
    # three intersections 10 m apart. The first candidate (115) is far from the start and kept,
    # but the next (135) is only 20 m from 115 and dropped -> intersections 2 and 3 share a sector.
    b = intersection_bounds(route_xy, cum, [box(100, 110), box(120, 130), box(140, 150)],
                            approach_m=0.5, min_m=50.0)
    assert [round(float(x)) for x in b] == [0, 115]
    assert float(np.diff(b).min()) >= 50.0

    # a tail shorter than the minimum length drops its boundary (it joins the previous sector).
    # accepting candidate 355 would leave the last sector only 44 m long.
    tail = intersection_bounds(route_xy, cum, [box(340, 350), box(360, 370)],
                               approach_m=0.5, min_m=50.0)
    assert [round(float(x)) for x in tail] == [0]


def test_the_approach_margin_catches_a_junction_the_route_skirts():
    """Margin for junctions, such as T-junctions, whose polygon does not cover the ego's lane."""
    from odyssey.scenario.hybrid_replay import intersection_bounds

    route_xy, _ = straight_route(400)                  # drives along y=0
    cum = np.arange(400, dtype=float)
    # two intersections 3 m off the route (y 3-8)
    off = lambda x0, x1: np.array([[x0, 3.], [x1, 3.], [x1, 8.], [x0, 8.]])
    polys = [off(100, 110), off(200, 210)]
    assert list(intersection_bounds(route_xy, cum, polys, approach_m=1.0, min_m=50.0)) == [0.0]
    b = intersection_bounds(route_xy, cum, polys, approach_m=5.0, min_m=50.0)
    assert [round(float(x)) for x in b] == [0, 155]
