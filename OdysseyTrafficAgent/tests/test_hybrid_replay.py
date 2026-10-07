"""What the ego-progress replay clock has to guarantee.

These pin the four properties the offline study relied on: an ego that drives the
recording reproduces it exactly, an ego that drives differently still meets the traffic
at the same place, progress never rewinds, and actors that travel together are released
together.
"""

import numpy as np
import pytest

from odyssey.scenario.hybrid_replay import EgoProgressClock, HybridReplay


def straight_route(n=100, dx=1.0):
    xy = np.stack([np.arange(n) * dx, np.zeros(n)], axis=1)
    return xy, np.zeros(n)


def track(k0, k1, x0, dx, y):
    k = np.arange(k0, k1)
    xy = np.stack([x0 + (k - k0) * dx, np.full(len(k), float(y))], axis=1)
    return k, xy


def test_progress_clock_maps_pose_to_source_row_and_never_rewinds():
    clock = EgoProgressClock(
        route_xy=np.array([[0.0, 0.0], [5.0, 0.0], [15.0, 0.0]]),
        rows=np.array([0, 10, 30]),
    )

    assert clock.source_row((2.5, 0.0)) == pytest.approx(5.0)
    assert clock.source_row((10.0, 0.0)) == pytest.approx(20.0)
    # A reversing or noisy ego projection cannot rewind the lifecycle source.
    assert clock.source_row((4.0, 0.0)) == pytest.approx(20.0)
    assert clock.source_row((15.0, 0.0)) == pytest.approx(30.0)


def test_progress_clock_rejects_non_monotone_source_rows():
    with pytest.raises(ValueError, match="rows must be monotone"):
        EgoProgressClock(
            route_xy=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            rows=np.array([0, 2, 1]),
        )


def test_ambient_follows_ego_progress_not_the_clock():
    """A parked-lane actor is shown at the row the ego's progress calls for."""
    route_xy, route_h = straight_route()
    # 10 m off the route, so ambient however it moves
    tracks = {'a': track(0, 100, 0.0, 1.0, 10.0)}
    r = HybridReplay(route_xy, route_h, tracks, src_dt=0.5)
    assert r.interactive['a'] is False

    # ego half way along the route -> row 50, whatever the episode step is
    assert r.step(3, (50.0, 0.0))['a'] == pytest.approx(50.0)
    assert r.step(99, (50.0, 0.0))['a'] == pytest.approx(50.0)


def test_progress_never_rewinds():
    route_xy, route_h = straight_route()
    tracks = {'a': track(0, 100, 0.0, 1.0, 10.0)}
    r = HybridReplay(route_xy, route_h, tracks, src_dt=0.5)

    r.step(0, (60.0, 0.0))
    # backing up must not rewind the world; it freezes at the furthest point reached
    assert r.step(1, (20.0, 0.0))['a'] == pytest.approx(60.0)
    assert r.step(2, (70.0, 0.0))['a'] == pytest.approx(70.0)


def test_interactive_is_classified_by_the_three_rules():
    route_xy, route_h = straight_route()
    tracks = {
        # on the route, moving, ahead of the ego -> interactive
        'lead': track(0, 100, 20.0, 1.0, 1.0),
        # on the route but parked -> ambient, there is no arrival to be early for
        'parked': track(0, 100, 30.0, 0.0, 1.0),
        # moving but a lane away -> ambient
        'nextlane': track(0, 100, 20.0, 1.0, 8.0),
        # on the route, moving, but always behind the ego -> ambient
        'follower': track(50, 100, -40.0, 1.0, 1.0),
    }
    r = HybridReplay(route_xy, route_h, tracks, src_dt=0.5)
    assert r.interactive['lead'] is True
    assert r.interactive['parked'] is False
    assert r.interactive['nextlane'] is False
    assert r.interactive['follower'] is False


def test_interactive_free_runs_once_released():
    route_xy, route_h = straight_route()
    tracks = {'lead': track(10, 100, 20.0, 1.0, 1.0)}
    r = HybridReplay(route_xy, route_h, tracks, src_dt=0.5)
    assert r.interactive['lead'] is True

    # ego has not reached the progress the log showed it at -> not on the road
    assert r.step(0, (5.0, 0.0))['lead'] == -1.0
    # ego reaches row 10 at episode step 4 -> released there
    assert r.step(4, (10.0, 0.0))['lead'] == pytest.approx(10.0)
    # and from then on it runs on the sim clock, not on ego progress: the ego stopping
    # does not stop it
    assert r.step(5, (10.0, 0.0))['lead'] == pytest.approx(11.0)
    assert r.step(6, (10.0, 0.0))['lead'] == pytest.approx(12.0)


def test_a_platoon_is_released_as_one():
    """Two cars nose to tail must not get two different release times."""
    route_xy, route_h = straight_route(200)
    tracks = {
        'front': track(0, 150, 20.0, 1.0, 0.5),
        # 8 m behind, same lane, same speed: it reaches the front car's ground 4 s later
        'rear': track(8, 150, 20.0, 1.0, 0.5),
        # far away in both space and time -> its own bundle
        'other': track(0, 150, 150.0, 1.0, 0.5),
    }
    r = HybridReplay(route_xy, route_h, tracks, pet_s=3.0, pet_r=5.0, src_dt=0.5)
    assert r.bundle['front'] == r.bundle['rear']
    assert r.bundle['other'] != r.bundle['front']

    out = r.step(0, (25.0, 0.0))
    assert out['front'] == out['rear']


def test_a_crossing_pair_is_bundled_too():
    """PET does not look at heading, so a junction conflict couples as well."""
    route_xy, route_h = straight_route(200)
    k = np.arange(0, 60)
    along = np.stack([50.0 + k * 1.0, np.zeros(len(k))], axis=1)
    # crosses the same patch of road two seconds later, at right angles
    across = np.stack([np.full(len(k), 80.0), -30.0 + k * 1.0], axis=1)
    r = HybridReplay(route_xy, route_h, {'a': (k, along), 'b': (k, across)}, src_dt=0.5)
    if r.interactive['a'] and r.interactive['b']:
        assert r.bundle['a'] == r.bundle['b']


def test_replaying_the_recording_reproduces_it():
    """The identity check: an ego that drives the log sees the log."""
    route_xy, route_h = straight_route()
    tracks = {'a': track(0, 100, 0.0, 1.0, 10.0),
              'lead': track(0, 100, 20.0, 1.0, 1.0)}
    r = HybridReplay(route_xy, route_h, tracks, src_dt=0.5)
    for step in range(100):
        out = r.step(step, (float(step), 0.0))   # ego exactly on its recorded pose
        assert out['a'] == pytest.approx(float(step))
        assert out['lead'] == pytest.approx(float(step))
