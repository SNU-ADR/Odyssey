"""The baked lidar ground has to reproduce a plane it was given, and admit where it has none."""
import numpy as np
import pytest

from odyssey_renderer.lidar_ground import GroundField
from ground_fixture import bake, save_field

# A realistic Boston UTM patch: northing near 4.69e6 is where float32 steps by 0.5 m.
EAST, NORTH, UP = 331_329.0, 4_691_072.0, 7.5


def tilted(n=200_000, gx=0.03, gy=-0.02, span=40.0, noise=0.0, seed=0):
    """Points on z = UP + gx dx + gy dy over a `span` metre square, in UTM."""
    rng = np.random.default_rng(seed)
    u, v = rng.uniform(0, span, n), rng.uniform(0, span, n)
    z = UP + gx * u + gy * v + (noise * rng.standard_normal(n) if noise else 0.0)
    return np.stack([EAST + u, NORTH + v, z], axis=-1)


def test_recovers_a_plane_to_a_millimetre():
    gx, gy = 0.03, -0.02
    field = bake(tilted(gx=gx, gy=gy), cell=0.25, sigmas=(1.0,))
    # Query away from the edges, where the kernel runs off the patch.
    q = np.stack(np.meshgrid(EAST + np.linspace(8, 32, 25),
                             NORTH + np.linspace(8, 32, 25)), -1).reshape(-1, 2)
    z, n = field.ground(q)
    want_z = UP + gx * (q[:, 0] - EAST) + gy * (q[:, 1] - NORTH)
    assert np.isfinite(z).all()
    assert np.abs(z - want_z).max() < 1e-3

    want_n = np.array([-gx, -gy, 1.0])
    want_n /= np.linalg.norm(want_n)
    cos = np.clip(n @ want_n, -1.0, 1.0)
    assert np.degrees(np.arccos(cos)).max() < 0.1


def test_normal_is_unit_length():
    field = bake(tilted(gx=0.08, gy=0.05), cell=0.25, sigmas=(1.0,))
    _, n = field.ground([[EAST + 20.0, NORTH + 20.0], [EAST + 21.0, NORTH + 19.0]])
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0)


def test_outside_the_support_is_nan_not_a_guess():
    field = bake(tilted(span=40.0), cell=0.25, sigmas=(1.0,))
    z, n = field.ground([[EAST + 20.0, NORTH + 20.0],     # inside
                         [EAST + 200.0, NORTH + 20.0],    # far past the grid in x
                         [EAST - 50.0, NORTH - 50.0]])    # before the origin
    assert np.isfinite(z[0])
    assert np.isnan(z[1:]).all()
    assert np.isnan(n[1:]).all()


def test_a_hole_wider_than_the_kernels_stays_unknown():
    p = tilted(span=40.0)
    keep = np.linalg.norm(p[:, :2] - [EAST + 20.0, NORTH + 20.0], axis=1) > 6.0
    field = bake(p[keep], cell=0.25, sigmas=(0.5, 1.0, 2.0))
    z, _ = field.ground([[EAST + 20.0, NORTH + 20.0],     # hole centre
                         [EAST + 30.0, NORTH + 20.0]])    # ordinary road
    assert np.isnan(z[0])
    assert np.isfinite(z[1])


def test_a_small_hole_is_filled_by_a_wider_kernel():
    p = tilted(span=40.0)
    keep = np.linalg.norm(p[:, :2] - [EAST + 20.0, NORTH + 20.0], axis=1) > 0.8
    field = bake(p[keep], cell=0.25, sigmas=(0.5, 1.0, 2.0))
    z, _ = field.ground([[EAST + 20.0, NORTH + 20.0]])
    assert np.isfinite(z[0])
    assert abs(z[0] - (UP + 0.03 * 20.0 - 0.02 * 20.0)) < 5e-3
    centre = field.sigma[field.sigma.shape[0] // 2, field.sigma.shape[1] // 2]
    assert centre > 0.5     # the narrow kernel could not cover it


def test_curvature_is_tracked_not_flattened():
    """A road crown is the case the baseline gets wrong, so the fit must follow it."""
    rng = np.random.default_rng(1)
    u, v = rng.uniform(0, 40, 400_000), rng.uniform(0, 40, 400_000)
    crown = 0.06 * (1.0 - ((v - 20.0) / 5.0) ** 2)          # 6 cm over a 5 m half-width
    p = np.stack([EAST + u, NORTH + v, UP + crown], axis=-1)
    field = bake(p, cell=0.25, sigmas=(0.5,))
    q = np.stack([np.full(21, EAST + 20.0), NORTH + np.linspace(15, 25, 21)], -1)
    z, _ = field.ground(q)
    want = UP + 0.06 * (1.0 - ((q[:, 1] - NORTH - 20.0) / 5.0) ** 2)
    assert np.abs(z - want).max() < 4e-3


def test_save_round_trips_through_float32_within_a_millimetre(tmp_path):
    """The file stores float32. UTM in float32 steps by 0.5 m, so it must store offsets."""
    field = bake(tilted(gx=0.03, gy=-0.02), cell=0.25, sigmas=(1.0,))
    path = tmp_path / "field.npz"
    save_field(field, path)
    back = GroundField.load(path)

    q = np.stack(np.meshgrid(EAST + np.linspace(8, 32, 25),
                             NORTH + np.linspace(8, 32, 25)), -1).reshape(-1, 2)
    before, n_before = field.ground(q)
    after, n_after = back.ground(q)
    assert np.abs(after - before).max() < 1e-3
    assert np.abs(n_after - n_before).max() < 1e-5
    assert np.isfinite(after).all()


def test_slope_survives_realistic_registration_noise():
    gx, gy = 0.03, -0.02
    field = bake(tilted(gx=gx, gy=gy, noise=0.022), cell=0.25, sigmas=(1.0,))
    q = np.stack(np.meshgrid(EAST + np.linspace(8, 32, 25),
                             NORTH + np.linspace(8, 32, 25)), -1).reshape(-1, 2)
    z, n = field.ground(q)
    want_z = UP + gx * (q[:, 0] - EAST) + gy * (q[:, 1] - NORTH)
    assert np.abs(z - want_z).max() < 5e-3          # 2.2 cm of noise, averaged down
    want_n = np.array([-gx, -gy, 1.0]) / np.linalg.norm([-gx, -gy, 1.0])
    assert np.degrees(np.arccos(np.clip(n @ want_n, -1, 1))).max() < 1.0


def test_bake_rejects_the_wrong_shape():
    with pytest.raises(ValueError):
        bake(np.zeros((10, 2)))


# --- standing in for LogPathLift ------------------------------------------------------------

ANCHOR = np.array([331_300.0, 4_691_000.0, 5.0])   # a scene's recon2global_translation
BODY = 0.31                                        # ego body origin above the road


def road(u, v):
    return UP + 0.03 * u - 0.02 * v


def path_lift_along(span=40.0, n=60, body=BODY, drift=0.0):
    """LogPathLift over a straight drive across the patch, in RECON-LOCAL coordinates."""
    from odyssey_renderer.ego_lift import LogPathLift
    u = np.linspace(4.0, span - 4.0, n)
    v = np.full(n, span / 2)
    z = road(u, v) + body + drift * np.linspace(0.0, 1.0, n)
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = EAST + u - ANCHOR[0]
    poses[:, 1, 3] = NORTH + v - ANCHOR[1]
    poses[:, 2, 3] = z - ANCHOR[2]
    return LogPathLift(poses)


def utm_field(span=40.0, hole=None):
    rng = np.random.default_rng(3)
    u, v = rng.uniform(0, span, 400_000), rng.uniform(0, span, 400_000)
    p = np.stack([EAST + u, NORTH + v, road(u, v)], axis=-1)
    if hole is not None:
        p = p[np.linalg.norm(p[:, :2] - hole, axis=1) > 7.0]
    return bake(p, cell=0.25, sigmas=(0.5, 1.0))


def test_road_surface_carries_the_path_lift_datum():
    """The field measures the road; LogPathLift reports the body origin above it."""
    from odyssey_renderer.lidar_ground import RoadSurface
    lift = path_lift_along()
    surface = RoadSurface(utm_field(), lift, ANCHOR)
    assert abs(surface.offset - BODY) < 5e-3
    assert surface.spread < 0.01
    assert surface.cover == 1.0

    # off the path, where LogPathLift has to guess, the surface still follows the real road
    q = np.array([[EAST + 20.0 - ANCHOR[0], NORTH + 8.0 - ANCHOR[1]]])
    z, n = surface.ground(q, [0.0])
    assert abs(z[0] - (road(20.0, 8.0) + BODY - ANCHOR[2])) < 5e-3
    want = np.array([-0.03, 0.02, 1.0]) / np.linalg.norm([-0.03, 0.02, 1.0])
    assert np.degrees(np.arccos(np.clip(n[0] @ want, -1, 1))) < 0.2


def test_road_surface_rejects_the_wrong_anchor():
    from odyssey_renderer.lidar_ground import RoadSurface
    with pytest.raises(ValueError, match="covers only"):
        RoadSurface(utm_field(), path_lift_along(), ANCHOR + [500.0, 0.0, 0.0])


def test_road_surface_reports_the_datum_drift_but_does_not_reject_it():
    """A drifting gap is the logged pose's own height error, which is what this removes."""
    from odyssey_renderer.lidar_ground import RoadSurface
    surface = RoadSurface(utm_field(), path_lift_along(drift=1.2), ANCHOR)
    assert surface.spread > 0.4                       # the drift is visible ...
    assert abs(surface.offset - (BODY + 0.6)) < 0.05   # ... and the datum is its median


def test_road_surface_carries_the_nearest_plane_into_a_hole_and_counts_it():
    """A hole is answered from the road around it, not from the logged path."""
    from odyssey_renderer.lidar_ground import RoadSurface
    surface = RoadSurface(utm_field(hole=[EAST + 20.0, NORTH + 8.0]), path_lift_along(), ANCHOR)
    q = np.array([[EAST + 20.0 - ANCHOR[0], NORTH + 8.0 - ANCHOR[1]],   # in the hole
                  [EAST + 30.0 - ANCHOR[0], NORTH + 20.0 - ANCHOR[1]]])  # on the road
    z, n = surface.ground(q, [0.0, 0.0])
    assert surface.fallbacks == 1 and surface.queries == 2
    # 7 m from the nearest node and carried only `reach` = 5 m of it: a few cm of a 3% grade
    # go unrealised. The old fallback answered from the logged path, metres off the road.
    assert abs(z[0] - (road(20.0, 8.0) + BODY - ANCHOR[2])) < 0.10
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0)


def test_walking_into_a_hole_is_not_a_step():
    """An actor walking off the edge of ground coverage must not drop metres in one frame."""
    from odyssey_renderer.lidar_ground import RoadSurface
    surface = RoadSurface(utm_field(hole=[EAST + 20.0, NORTH + 8.0]), path_lift_along(), ANCHOR)
    u = np.linspace(4.0, 20.0, 161)                    # 0.1 m steps from road into the hole
    q = np.stack([EAST + u - ANCHOR[0], np.full_like(u, NORTH + 8.0 - ANCHOR[1])], axis=-1)
    z, _ = surface.ground(q, np.zeros(len(q)))
    assert surface.fallbacks > 0                       # it did cross into the hole
    fz, _ = surface.field.ground(q)
    edge = np.flatnonzero(np.isfinite(fz))[-1]         # last step the field itself answered
    step = np.abs(np.diff(z))
    # The step at the edge of coverage must be no bigger than the road's own grade over one 0.1 m step.
    assert step[max(0, edge - 3):edge + 3].max() < 0.01
    # Deep inside a hole wider than `reach` the nearest node can change sides, a few cm. Never
    # a plunge.
    assert step.max() < 0.05


def test_a_long_carry_holds_instead_of_running_away():
    from odyssey_renderer.lidar_ground import RoadSurface
    surface = RoadSurface(utm_field(), path_lift_along(), ANCHOR, reach=5.0)
    far = np.array([[EAST + 40.0 + 50.0 - ANCHOR[0], NORTH + 20.0 - ANCHOR[1]],
                    [EAST + 40.0 + 200.0 - ANCHOR[0], NORTH + 20.0 - ANCHOR[1]]])
    z, _ = surface.ground(far, [0.0, 0.0])
    assert abs(z[1] - z[0]) < 1e-3                     # 50 m and 200 m out: the same height


def test_road_surface_lift_stands_a_body_on_it():
    from odyssey_renderer.lidar_ground import RoadSurface
    surface = RoadSurface(utm_field(), path_lift_along(), ANCHOR)
    x, y = EAST + 20.0 - ANCHOR[0], NORTH + 8.0 - ANCHOR[1]
    pose = surface.lift(x, y, 0.4)
    assert pose.shape == (4, 4)
    assert np.allclose(pose[:3, 3][:2], [x, y])
    assert abs(pose[2, 3] - surface.ground([[x, y]], [0.4])[0][0]) < 1e-9
    assert np.allclose(np.linalg.norm(pose[:3, :3], axis=0), 1.0)   # orthonormal columns
