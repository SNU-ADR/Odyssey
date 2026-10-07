"""Load exported road-height fields and query height/normal for simulation.

Asset fitting belongs to the separate production pipeline; this module consumes
the precomputed grid and maps it into the reconstruction-local frame.
"""
from __future__ import annotations

import numpy as np

from odyssey_renderer.ego_lift import attitude


class GroundField:
    """A baked road surface: node grid of (height, slope), queried by (x, y).

    Node `(j, i)` sits at world `origin + ((i + 0.5) cell, (j + 0.5) cell)` -- the centre of
    the accumulator cell the kernel was centred on, not its corner.
    """

    def __init__(self, origin, cell, z, gx, gy, sigma):
        self.origin = np.asarray(origin, dtype=np.float64)
        self.cell = float(cell)
        self.z = np.asarray(z, dtype=np.float64)          # height above origin[2]
        self.gx = np.asarray(gx, dtype=np.float64)        # dz/dx
        self.gy = np.asarray(gy, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float32)  # kernel width that supplied the node
        self.valid = np.isfinite(self.z)

    def ground(self, xy, headings=None):
        """-> (road height z, world normal n) under each (x, y). Batched; NaN where unknown.

        `headings` is accepted and ignored so this can stand in for `LogPathLift.ground`,
        which needs it to tell a revisited stretch of path from a nearby one. A field indexed
        by position has no such ambiguity to resolve.

        The four surrounding nodes each carry a plane, and the answer is those four planes
        evaluated at the query point and blended bilinearly -- not a bilinear interpolation
        of the node heights, which would ignore the slopes the fit already measured and cut
        the corner of every crown. Where some of the four are unknown the weights are
        renormalised over the rest, so coverage ends at the data instead of a cell early;
        with none of the four known the answer is NaN.
        """
        q = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        g = (q - self.origin[:2]) / self.cell - 0.5        # continuous node index (x, y)
        i0 = np.floor(g).astype(np.int64)
        f = g - i0
        ny, nx = self.z.shape

        num_z = np.zeros(len(q))
        num_gx = np.zeros(len(q))
        num_gy = np.zeros(len(q))
        den = np.zeros(len(q))
        for dj in (0, 1):
            for di in (0, 1):
                i, j = i0[:, 0] + di, i0[:, 1] + dj
                inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
                w = (f[:, 0] if di else 1.0 - f[:, 0]) * (f[:, 1] if dj else 1.0 - f[:, 1])
                ii, jj = np.where(inside, i, 0), np.where(inside, j, 0)
                w = np.where(inside & self.valid[jj, ii], w, 0.0)
                # the node's own plane, evaluated at the query point
                cx = self.origin[0] + (ii + 0.5) * self.cell
                cy = self.origin[1] + (jj + 0.5) * self.cell
                plane = (self.z[jj, ii] + self.gx[jj, ii] * (q[:, 0] - cx)
                         + self.gy[jj, ii] * (q[:, 1] - cy))
                num_z += w * plane
                num_gx += w * self.gx[jj, ii]
                num_gy += w * self.gy[jj, ii]
                den += w

        known = den > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(known, self.origin[2] + num_z / den, np.nan)
            gx = np.where(known, num_gx / den, np.nan)
            gy = np.where(known, num_gy / den, np.nan)
        n = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        return z, n

    def lift(self, x, y, heading):
        """-> 4x4 pose standing on the road under (x, y). Rotation columns: forward, left, up."""
        z, n = self.ground([[x, y]])
        pose = np.eye(4)
        pose[:3, :3] = attitude(heading, n[0])
        pose[:3, 3] = [x, y, z[0]]
        return pose

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["origin"], float(d["cell"]), d["z"], d["gx"], d["gy"], d["sigma"])


class RoadSurface:
    """The lidar field answering in the logged path's frame and datum, everywhere a body can be.

    Two things have to be reconciled before a baked field can stand in for `LogPathLift`.

    FRAME. The renderer works in reconstruction-local coordinates -- world minus the scene's
    anchor -- while the field was baked in world UTM. That is one translation, applied here to
    the field's origin so nothing downstream has to know about it.

    DATUM. `LogPathLift.ground` does not return the road. It returns the height of the EGO BODY
    ORIGIN, and every caller adds its own measured constant on top (`_measure_actor_height` for
    the actors). The field returns the road itself, about 0.3 m lower. Swapping one for the
    other without closing that gap drops every body in the scene by it, which is exactly the
    failure this class exists to remove. So the gap is measured once along the logged path and
    carried as one constant.

    Measuring it is also the frame check: if the anchor were wrong the logged path would fall
    outside the field's coverage, and `cover` collapsing raises rather than rendering something
    plausible-looking -- a silent 0.3 m is invisible in a video and fatal in a score.

    How much the gap VARIES along the path is reported but deliberately not a gate. That
    variation is not disagreement about where the ground is -- it is the
    logged pose's own height error against the measured road, which is the thing this class
    exists to stop propagating. Failing on it would reject exactly the scenes that need it. A
    wrong anchor *height*, the other thing it might have caught, is absorbed by the constant and
    is harmless, because the datum is defined relative to the logged path either way.

    Where the field has no answer -- a street end the sweep never reached, the shadow behind a
    parked car -- the NEAREST ANSWERED NODE's plane is carried out to the query point. Not the
    logged path: switching between two sources that disagree is a step in the rendered height,
    and off the path is exactly where the logged path disagrees most.

    Carrying ONE nearest plane is not enough either: where the nearest node flips between an
    ordinary road node and a kerb node (the steep line along the edge of the drivable mask), a
    kerb plane carried a metre or two is metres of height. So the carry is two things:

      * a blend of the `k` nearest nodes with Franke-Little weights, ((R - d) / (R d))^2 where R
        is the distance to the (k+1)-th node. A node's weight reaches zero exactly as it leaves
        the candidate set, so the answer is continuous as the query moves -- nearest-single-node
        is a Voronoi map, and every Voronoi edge is a step;
      * each node's slope clipped to `max_grade` before it is carried. Only the carry clips: a
        kerb IS steep and the field keeps it where it has data, but it is not the slope of the
        road a car drives on past the end of the coverage. Real road grades are a few degrees,
        so 15% leaves them untouched.

    A plane carried a long way is still a guess -- a 3 deg grade carried 100 m is 5 m of error --
    so each node's carry stops at `reach` metres and holds from there. Every such answer is
    still a substitution, and still counted in `fallbacks`.
    """

    def __init__(self, field, path_lift, anchor, min_cover=0.8, reach=5.0, k=12, max_grade=0.15):
        from scipy.spatial import cKDTree
        origin = np.asarray(field.origin, dtype=np.float64) - np.asarray(anchor, dtype=np.float64)
        self.field = GroundField(origin, field.cell, field.z, field.gx, field.gy, field.sigma)
        self.reach, self.k, self.max_grade = float(reach), int(k), float(max_grade)
        j, i = np.nonzero(self.field.valid)
        self._node_xy = origin[:2] + (np.stack([i, j], axis=-1) + 0.5) * field.cell
        self._node = (j, i)
        self._tree = cKDTree(self._node_xy)

        xy = path_lift.a
        heading = np.arctan2(path_lift.t[:, 1], path_lift.t[:, 0])
        z_path, _ = path_lift.ground(xy, heading)
        z_field, _ = self.field.ground(xy)
        seen = np.isfinite(z_field)
        self.cover = float(seen.mean())
        if self.cover < min_cover:
            raise ValueError(
                "road surface covers only %.1f%% of the logged path; the field and the scene "
                "are probably not in the same frame (anchor %s)" % (100 * self.cover, anchor))
        gap = z_path[seen] - z_field[seen]
        self.offset = float(np.median(gap))
        self.spread = float(np.percentile(np.abs(gap - self.offset), 90))
        self.queries = 0
        self.fallbacks = 0

    def ground(self, xy, headings):
        """-> (z, world normal n) under each (x, y), in the logged path's datum. Batched.

        `headings` is accepted for parity with `LogPathLift.ground` and not used: a field
        indexed by position has no revisited stretch of path to tell apart.
        """
        q = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        z, n = self.field.ground(q)
        z += self.offset
        miss = ~np.isfinite(z)
        self.queries += len(z)
        if miss.any():
            self.fallbacks += int(miss.sum())
            z[miss], n[miss] = self._carry(q[miss])
        return z, n

    def _carry(self, q):
        """-> (z, n) blended from the `k` nearest answered nodes' planes (see the class doc)."""
        d, idx = self._tree.query(q, k=self.k + 1)
        radius = d[:, -1:]                                   # the (k+1)-th node: weight 0 there
        d, idx = np.maximum(d[:, :-1], 1e-6), idx[:, :-1]
        w = ((radius - d) / (radius * d)) ** 2
        w /= w.sum(axis=1, keepdims=True)

        j, i = self._node[0][idx], self._node[1][idx]
        gx, gy = self.field.gx[j, i], self.field.gy[j, i]
        grade = np.hypot(gx, gy)
        clip = np.minimum(1.0, self.max_grade / np.maximum(grade, 1e-9))
        gx, gy = gx * clip, gy * clip

        step = q[:, None, :] - self._node_xy[idx]
        far = np.linalg.norm(step, axis=-1, keepdims=True)
        step = step * np.minimum(1.0, self.reach / np.maximum(far, 1e-9))
        z = (w * (self.field.z[j, i] + gx * step[..., 0] + gy * step[..., 1])).sum(axis=1)
        gxb, gyb = (w * gx).sum(axis=1), (w * gy).sum(axis=1)
        n = np.stack([-gxb, -gyb, np.ones_like(gxb)], axis=-1)
        return (self.field.origin[2] + z + self.offset,
                n / np.linalg.norm(n, axis=-1, keepdims=True))

    def lift(self, x, y, heading):
        """-> 4x4 pose standing on the road under (x, y). Rotation columns: forward, left, up."""
        z, n = self.ground([[x, y]], [heading])
        pose = np.eye(4)
        pose[:3, :3] = attitude(heading, n[0])
        pose[:3, 3] = [x, y, z[0]]
        return pose
