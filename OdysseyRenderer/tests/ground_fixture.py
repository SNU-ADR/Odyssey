"""Synthetic ground assets for query/loader regression tests only."""
import numpy as np
from scipy.ndimage import gaussian_filter
from odyssey_renderer.lidar_ground import GroundField


def _fit_one_scale(moments, sigma_cells, cell):
    """WLS plane at every node for one kernel width. -> (z, dz/dx, dz/dy, cover, spread)."""
    b = [gaussian_filter(m, sigma_cells, mode="constant", cval=0.0) for m in moments]
    occ, w, sx, sy, sxx, sxy, syy, tz, tx, ty = b

    with np.errstate(invalid="ignore", divide="ignore"):
        inv = 1.0 / w
        ux, uy, uz = sx * inv, sy * inv, tz * inv          # kernel-weighted centroid
        # Central moments. Fitting about the centroid rather than the node centre keeps the
        # 2x2 system well scaled: the shift to the node is then at most about one sigma.
        cxx = sxx * inv - ux * ux
        cxy = sxy * inv - ux * uy
        cyy = syy * inv - uy * uy
        cxz = tx * inv - ux * uz
        cyz = ty * inv - uy * uz
        det = cxx * cyy - cxy * cxy
        gx = (cyy * cxz - cxy * cyz) / det
        gy = (cxx * cyz - cxy * cxz) / det

        # Smaller eigenvalue of the weighted xy covariance: how thin the support is. A node
        # whose points all lie on one line (a kerb, the rim of the mask) has no cross-slope
        # information, and the solve above would amplify noise into a slope.
        half = 0.5 * (cxx + cyy)
        gap = np.sqrt(np.maximum(half * half - det, 0.0))
        spread = np.sqrt(np.maximum(half - gap, 0.0))

        nodes_y, nodes_x = w.shape
        cx = (np.arange(nodes_x) + 0.5) * cell
        cy = (np.arange(nodes_y) + 0.5) * cell
        z = uz + gx * (cx[None, :] - ux) + gy * (cy[:, None] - uy)
    return z, gx, gy, occ, spread


def bake(xyz, cell=0.25, sigmas=(0.5, 1.0, 2.0, 4.0), min_cover=0.5, min_spread=0.25,
         origin=None):
    """Fit the road surface everywhere the points support one. -> GroundField

    `xyz` is world (UTM) float64. `sigmas` are kernel widths in metres, tried narrowest
    first; a node takes the first width whose window is at least `min_cover` covered by
    occupied cells and whose support is at least `min_spread * sigma` wide in its thinnest
    direction. Nodes no width satisfies are left NaN.

    There is no robust reweighting here, and that is a measured choice. The fit is plain
    least squares, so a kerb top inside the window does pull the plane: 5.1% of nodes come
    out steeper than 10 deg, in a thin line along the edge of the drivable mask. Re-testing
    every point against the fitted plane and refitting (two rounds, keeping -0.10/+0.08 m)
    thins that line, but on held-out sweeps it moved |z| p90 by 3 mm, made p99 worse
    (0.218 -> 0.268 m) and cost 0.6 pt of coverage and 3x the bake time. The steep line is
    the kerb, which really is steep; it is not an artifact to regularise away.
    """
    p = np.asarray(xyz, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("bake needs (N, 3) world points")
    # UTM northing is ~4.7e6, where float32 steps by 0.5 m. Everything below is float64 and
    # origin-relative; the float32 cast happens only on save, on numbers under 1 km.
    origin = p.min(axis=0) if origin is None else np.asarray(origin, dtype=np.float64)
    u = p[:, 0] - origin[0]
    v = p[:, 1] - origin[1]
    z = p[:, 2] - origin[2]

    nx = int(np.floor(u.max() / cell)) + 1
    ny = int(np.floor(v.max() / cell)) + 1
    flat = np.floor(v / cell).astype(np.int64) * nx + np.floor(u / cell).astype(np.int64)

    size = nx * ny
    # float64 throughout: gaussian_filter keeps its input's dtype, and an integer count grid
    # would come back rounded, which wrecks the centred moments below (they are a difference
    # of two numbers near u^2, so a 1-in-8 error in the weight is a 400x error in cxx).
    count = np.bincount(flat, minlength=size).reshape(ny, nx).astype(np.float64)
    moments = [(count > 0).astype(np.float64), count]      # occupancy, then weight
    moments += [np.bincount(flat, weights=wt, minlength=size).reshape(ny, nx)
                for wt in (u, v, u * u, u * v, v * v, z, u * z, v * z)]

    out_z = np.full((ny, nx), np.nan)
    out_gx = np.full((ny, nx), np.nan)
    out_gy = np.full((ny, nx), np.nan)
    out_sigma = np.full((ny, nx), np.nan, dtype=np.float32)
    for sigma in sigmas:
        todo = ~np.isfinite(out_z)
        if not todo.any():
            break
        z_s, gx_s, gy_s, cover, spread = _fit_one_scale(moments, sigma / cell, cell)
        ok = todo & (cover >= min_cover) & (spread >= min_spread * sigma) & np.isfinite(z_s)
        out_z[ok], out_gx[ok], out_gy[ok], out_sigma[ok] = z_s[ok], gx_s[ok], gy_s[ok], sigma
    return GroundField(origin, cell, out_z, out_gx, out_gy, out_sigma)


def save_field(field, path):
    np.savez_compressed(path, origin=field.origin, cell=field.cell,
                        z=field.z.astype(np.float32), gx=field.gx.astype(np.float32),
                        gy=field.gy.astype(np.float32), sigma=field.sigma)
