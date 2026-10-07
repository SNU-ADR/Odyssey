"""OmniReSMPLSubModel -- a pedestrian whose canonical shape is posed by SMPL LBS.

MTGS has NO SMPL node type at all (MODEL_MAPPING has five keys and none of them
is one), so without this class an SMPL actor has nowhere to go and every SMPL
pedestrian exports as nothing -- in a busy scene that can be most of the
pedestrians and a large share of the checkpoint's gaussians.

WHY NOT A RIGID NODE CARRYING PER-FRAME POSED GAUSSIANS. Two reasons, and the
first is fatal on its own:

  * It is not expressible. A RigidSubModel holds ONE canonical geometry and a
    per-frame RIGID pose. SMPL deformation is per-gaussian per-frame, so there
    is no field of a rigid node that can carry it. One node per frame is barred
    by asset_token uniqueness.
  * It would not be cheap even if it were. Baking posed means and quats for
    every frame takes several times the memory of the whole SMPLNodes class
    as trained.

WHAT THIS NODE CARRIES INSTEAD, and why it is small. For a sample scene:

    W        (P, 24)        11.35 MiB total   replaces 216.0 MiB of voxel grids
    A_rel    (F, 24, 4, 4)   5.64 MiB total   replaces  18.8 MiB of SMPL layer
    gaussians                10.9  MiB
                            ------
                             ~28 MiB against 259 MiB as trained, 709 MiB baked.

  * W is the LBS skinning weight per gaussian per joint. Upstream evaluates it
    every frame through a pair of voxel grids, but VoxelDeformer.forward takes
    (xc, instance_indices) and nothing else -- no theta, no frame -- and the
    canonical positions never move at inference. So it is a constant, evaluated
    once at export. NOTE: it is NOT the stored `template.W` buffer, which is the
    non-voxel fallback path and differs from the voxel answer by up to 0.946.
  * A_rel is the SMPL kinematic chain evaluated with an IDENTITY ROOT rotation,
    so this node's own rigid pose supplies the root -- which is what makes this
    a rigid node with a per-frame canonical shape, exactly like the deformable
    node, rather than a special case in the renderer.

THE ROOT FACTORISATION, WHICH IS NOT THE OBVIOUS ONE. Upstream folds the root
rotation into theta, so A carries it. Pulling it back out is NOT `A = R0 @ A_rel`
-- that is wrong by up to 0.259 m, because the root joint sits 0.21-0.24 m from
the canonical origin. The identity that holds, verified at 2.98e-07:

    A_true = [ R0 | J0 - R0 @ J0 ] @ A_rel

So the node applies its rigid rotation to the LBS output and then corrects the
translation by `J0 - R @ J0`. Because that term is recomputed from whatever
rotation is actually used, it stays exact for a SIMULATED pedestrian whose
heading the planner changed, not only for a replaying one.

THE STATIC CACHE. Same trap as the deformable node: `_is_static_model` is an
exact `type(model) is VanillaModel` check, so this subclass is correctly
re-evaluated every frame. If that ever becomes isinstance, this class must be
excluded explicitly or every pedestrian freezes in one pose, silently.
"""
import logging

import torch

from odyssey_renderer.mtgs.gaussian_model.rigid_object import RigidPortableSubModel
from odyssey_renderer.mtgs.utils.gaussian_utils import quat_mult, quat_to_rotmat

logger = logging.getLogger(__name__)


def _matrix_to_quaternion(m):
    """Rotation matrices (..., 3, 3) -> wxyz quaternions, Shepperd's method.

    Local rather than imported: gaussian_utils exports quat_to_rotmat but not
    its inverse, and adding one there would modify an s12 file.
    """
    m = m.reshape(-1, 3, 3)
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = torch.zeros(m.shape[0], 4, device=m.device, dtype=m.dtype)

    big = t > 0
    if big.any():
        s = torch.sqrt(t[big] + 1.0) * 2.0
        q[big, 0] = 0.25 * s
        q[big, 1] = (m[big, 2, 1] - m[big, 1, 2]) / s
        q[big, 2] = (m[big, 0, 2] - m[big, 2, 0]) / s
        q[big, 3] = (m[big, 1, 0] - m[big, 0, 1]) / s

    rest = ~big
    if rest.any():
        mm = m[rest]
        d = torch.stack([mm[:, 0, 0], mm[:, 1, 1], mm[:, 2, 2]], dim=-1)
        k = torch.argmax(d, dim=-1)
        out = torch.zeros(mm.shape[0], 4, device=m.device, dtype=m.dtype)
        for axis in (0, 1, 2):
            sel = k == axis
            if not sel.any():
                continue
            a, b, c = axis, (axis + 1) % 3, (axis + 2) % 3
            n = mm[sel]
            s = torch.sqrt(1.0 + n[:, a, a] - n[:, b, b] - n[:, c, c]) * 2.0
            out[sel, 0] = (n[:, c, b] - n[:, b, c]) / s
            out[sel, 1 + a] = 0.25 * s
            out[sel, 1 + b] = (n[:, b, a] + n[:, a, b]) / s
            out[sel, 1 + c] = (n[:, c, a] + n[:, a, c]) / s
        q[rest] = out
    return q / q.norm(dim=-1, keepdim=True)


class OmniReSMPLSubModel(RigidPortableSubModel):

    MODEL_TYPE = "smpl"

    def _update_default_config(self, config):
        config = super()._update_default_config(config)
        config["num_timesteps"] = config.get("num_timesteps", None)
        return config

    def load_state_dict(self, sd: dict):
        sd = dict(sd)                  # the base pops keys; do not mutate the asset
        for key in ("lbs_weights", "joint_transforms", "root_joint"):
            if key not in sd:
                raise ValueError(
                    "no %r in the asset; OmniReSMPLSubModel requires it "
                    "(convert.smpl emits it)" % key)
        self._lbs_weights = sd.pop("lbs_weights")            # (P, 24)
        # Point corrections are added directly by upstream and are not
        # normalized.  Root translation in T_root @ A_rel is consequently
        # scaled by sum_j(w_pj) for each Gaussian, which can reach about 1.7.
        self._lbs_weight_sum = self._lbs_weights.sum(dim=-1)  # (P,)
        self._joint_transforms = sd.pop("joint_transforms")  # (F, 24, 4, 4)
        self._root_joint = sd.pop("root_joint").reshape(3)   # (3,)
        msg = super().load_state_dict(sd)
        self._pose_cache = (None, None, None)
        self._warned_no_timestamp = False
        return msg

    # -- frame resolution ----------------------------------------------------

    def _frame_index(self, timestamp):
        """Nearest stored frame for this timestamp, in this node's numbering.

        Deliberately NEAREST rather than interpolated. A_rel is a rigid
        transform per joint; linearly blending two of them is not a rigid
        transform and would shrink limbs. Upstream interpolates the joint
        QUATERNIONS instead, and only on its held-out split -- these runs have
        none. Snapping is the honest approximation at 10 Hz.
        """
        if timestamp is None:
            return None
        ts = self.config.log_timestamps
        ts = ts.squeeze() if torch.is_tensor(ts) else torch.as_tensor(ts)
        ts = ts.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        if ts.numel() == 0:
            return None
        return int(torch.argmin((ts - float(timestamp)).abs()))

    def _posed(self, frame):
        """(R, t) per gaussian at this frame, root rotation NOT applied."""
        if frame is None:
            return None, None
        if self._pose_cache[0] == frame:
            return self._pose_cache[1], self._pose_cache[2]
        dev = self.gauss_params["means"].device
        W = self._lbs_weights.to(dev)                        # (P, 24)
        A = self._joint_transforms.to(dev)[frame]            # (24, 4, 4)
        T = torch.einsum("pj,jrc->prc", W, A[:, :3, :])      # (P, 3, 4)
        R, t = T[..., :3], T[..., 3]
        self._pose_cache = (frame, R, t)
        return R, t

    def get_global_gaussians(self, quat=None, trans=None, timestamp=None, **kwargs):
        if timestamp is None and not self._warned_no_timestamp:
            logger.warning(
                "%s: get_global_gaussians called with timestamp=None; rendering "
                "the CANONICAL (unposed) body. The closed-loop renderer always "
                "passes a timestamp, so this is a viewer-path defect.",
                self.model_name)
            self._warned_no_timestamp = True
        self._frame = self._frame_index(timestamp)
        return super().get_global_gaussians(quat=quat, trans=trans,
                                            timestamp=timestamp, **kwargs)

    def get_means(self, global_quat, global_trans):
        R, t = self._posed(getattr(self, "_frame", None))
        local = self.gauss_params["means"]
        if R is not None:
            local = torch.einsum("pij,pj->pi", R, local) + t
        rot = quat_to_rotmat(global_quat[None, ...])[0, ...]
        # A_rel was built with an identity root, so the rotation applied here is
        # the root -- and the root joint offset has to travel with it.
        # A_true = [R0 | J0 - R0 J0] @ A_rel, verified to 2.98e-07.
        j0 = self._root_joint.to(local.device, local.dtype)
        root_shift = j0 - rot @ j0
        self.global_means = (local @ rot.T + global_trans
                             + self._lbs_weight_sum.to(local.device, local.dtype)[:, None]
                             * root_shift[None, :])
        return self.global_means

    def get_quats(self, global_quat, global_trans=None):
        R, _ = self._posed(getattr(self, "_frame", None))
        local = self.quats / self.quats.norm(dim=-1, keepdim=True)
        if R is not None:
            rq = _matrix_to_quaternion(R)
            local = quat_mult(rq, local)
            local = local / local.norm(dim=-1, keepdim=True)
        return quat_mult(global_quat[None, ...], local)
