"""Read the road's height and normal off the logged path, and stand a vehicle on it.

The simulator never decides z, roll or pitch; the renderer has to borrow them from the
reconstruction. The stock rules borrow ONE logged pose -- the OmniRe engine the pose of the same
tick, mtgs the nearest-xy pose -- and carry its z and its body-frame roll/pitch. Both go
wrong as soon as the ego is not where the log was at that moment: on a hilly road, a run
that pulls ahead of the log renders from metres above the road.

This lift projects (x, y) onto the logged path, reads z and the WORLD normal at that arc
length s (interpolated between the two neighbouring logged poses), and builds the attitude
from (heading, normal) so roll/pitch follow the ground rather than the logged car. z and n
depend on s only: a lateral term (d times the crown slope) moved the z residual against
learned actor poses by only a few centimetres, so there is none. Continuous along the path
and independent of timing.

The same road serves the NON-EGO actors, whose learned z can wobble by up to a metre
around their own median height, which reads as a car floating over the road. The
caller gives each actor one constant (its body origin above the road) and this gives the
rest.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def colmap_ego(camera_to_global, camera_names, cameras):
    """-> (ego poses [rows, 4, 4], camera residual [rows, cameras, 4, 4]) in camera_to_global's frame.

    The reconstructed ego is the checkpoint's own CAM_F0 (camera_to_global) with the scenario's
    CAM_F0 sensor2ego taken off; against it every camera's residual is the rig itself. The
    renderer lifts the ego along this path and the anchor-surface bake stands the ego anchors on
    it, so both read the same ego (see OmniReRenderEngine._colmap_ego for why not the log).
    """
    if "CAM_F0" not in cameras:
        raise ValueError("the reconstructed ego needs the scenario's CAM_F0 sensor2ego")
    w, x, y, z = np.asarray(cameras["CAM_F0"]["sensor2ego_rotation"], dtype=np.float64)
    sensor2ego = np.eye(4)
    sensor2ego[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    sensor2ego[:3, 3] = np.asarray(cameras["CAM_F0"]["sensor2ego_translation"], dtype=np.float64)
    camera_to_global = np.asarray(camera_to_global, dtype=np.float64)
    front = list(camera_names).index("CAM_F0")
    ego = camera_to_global[:, front] @ np.linalg.inv(sensor2ego)
    residual = np.linalg.inv(ego)[:, None] @ camera_to_global
    return ego, residual


def attitude(headings, normals):
    """Rotations whose columns are forward, left, up for a body standing on `normals`.

    Yaw is the caller's: the forward column is `heading` projected onto the tangent plane,
    which leaves its xy direction within 0.2 deg of `heading` for a road tilted up to 5 deg
    (measured over 2,000 random pairs). Roll and pitch come entirely from the normal.
    Batched; a single (heading, normal) returns one 3x3.
    """
    n = np.asarray(normals, dtype=np.float64)
    single = n.ndim == 1
    n = np.atleast_2d(n)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)   # a direction, however the caller scaled it
    h = np.atleast_1d(np.asarray(headings, dtype=np.float64))
    f = np.stack([np.cos(h), np.sin(h), np.zeros_like(h)], axis=-1)
    f = f - (f * n).sum(-1, keepdims=True) * n
    f = f / np.linalg.norm(f, axis=-1, keepdims=True)
    R = np.stack([f, np.cross(n, f), n], axis=-1)
    return R[0] if single else R


class LogPathLift:
    """One logged trajectory read as a road surface (4x4 poses, x forward / y left / z up)."""

    def __init__(self, poses):
        P = np.asarray(poses, dtype=np.float64)
        if P.ndim != 3 or P.shape[1:] != (4, 4) or len(P) < 2:
            raise ValueError("LogPathLift needs at least two 4x4 logged poses")
        xy = P[:, :2, 3]
        seg = np.diff(xy, axis=0)
        length = np.linalg.norm(seg, axis=1)
        keep = length > 1e-3                       # a stopped log leaves zero-length segments
        if not keep.any():
            raise ValueError("logged path has no length")
        self.a = xy[:-1][keep]
        self.t = seg[keep] / length[keep][:, None]  # unit tangent in xy
        self.len = length[keep]
        self.z0, self.z1 = P[:-1, 2, 3][keep], P[1:, 2, 3][keep]
        self.n0, self.n1 = P[:-1, :3, 2][keep], P[1:, :3, 2][keep]
        self.s0 = np.concatenate([[0.0], np.cumsum(self.len)[:-1]])

    def project(self, xy, headings):
        """-> (segment index, position u along it, arc length s, lateral offset d), batched."""
        q = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        h = np.atleast_1d(np.asarray(headings, dtype=np.float64))
        u = np.clip(((q[:, None, :] - self.a) * self.t).sum(-1) / self.len, 0.0, 1.0)  # (N, S)
        foot = self.a + u[..., None] * self.t * self.len[:, None]                      # (N, S, 2)
        dist = np.linalg.norm(q[:, None, :] - foot, axis=-1)
        s = self.s0 + u * self.len
        rows = np.arange(len(q))
        # Nearest segment wins. Only a REVISIT -- another stretch of the path, far away in s
        # but about as close in xy -- is decided by heading. Deciding every near-tie by
        # heading let the pick hop among neighbouring segments on a straight road (their
        # distances differ by millimetres at d = 2 m) and s jumped by metres per tick.
        i = np.argmin(dist, axis=1)
        revisit = ((dist <= dist[rows, i][:, None] + 0.5)
                   & (np.abs(s - s[rows, i][:, None]) > 20.0))
        if revisit.any():
            revisit[rows, i] = True
            forward = np.stack([np.cos(h), np.sin(h)], axis=-1) @ self.t.T              # (N, S)
            i = np.argmax(np.where(revisit, forward, -np.inf), axis=1)
        left = np.stack([-self.t[i, 1], self.t[i, 0]], axis=-1)
        d = ((q - foot[rows, i]) * left).sum(-1)
        return i, u[rows, i], s[rows, i], d

    def ground(self, xy, headings):
        """-> (road height z, world normal n) under each (x, y), batched."""
        i, u, _, _ = self.project(xy, headings)
        n = (1.0 - u)[:, None] * self.n0[i] + u[:, None] * self.n1[i]
        n = n / np.linalg.norm(n, axis=-1, keepdims=True)
        return (1.0 - u) * self.z0[i] + u * self.z1[i], n

    def lift(self, x, y, heading):
        """-> 4x4 pose standing on the road under (x, y). Rotation columns: forward, left, up."""
        z, n = self.ground([[x, y]], [heading])
        pose = np.eye(4)
        pose[:3, :3] = attitude(heading, n[0])
        pose[:3, 3] = [x, y, z[0]]
        return pose
