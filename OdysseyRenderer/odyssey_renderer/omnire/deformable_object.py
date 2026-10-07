"""OmniReDeformableSubModel -- a rigid actor whose canonical shape deforms.

MTGS maps the string "DeformableSubModel" to None, so an asset must never emit
it; ours is namespaced "OmniReDeformableSubModel" and registered additively.

WHY THIS IS NOT A RIGID NODE WITH EXTRA STEPS. The learned deformation is
typically several times a gaussian's own sigma, so most gaussians move further
than their own size, and discarding it costs a large PSNR drop over the actor.

THE STATIC CACHE. mtgs.py `_is_static_model` is `type(model) is VanillaModel`,
an exact type check, and its docstring says the check is by type "so a future
dynamic model is excluded by default". This class is a RigidPortableSubModel
subclass, so it is correctly excluded and re-evaluated every frame. If that
check ever becomes isinstance, this class MUST be excluded explicitly or it
will render one frozen pose forever -- silently.

NORMALISED TIME IS INDEX-BASED. scene_graph.py registers
`torch.linspace(0, 1, num_timestamps)` and deformable.py indexes it with
`cur_frame`, so the MLP's t is `frame_index / (num_timesteps - 1)`, NOT
`(t - t0) / (t1 - t0)`. Those differ wherever the 10 Hz log jitters. This class
recovers a FRACTIONAL index by interpolating between bracketing log_timestamps,
which equals upstream exactly at a stored timestamp and stays continuous in
between -- the MLP is continuous in t by construction, so a rollout stepping
between frames gets a smooth deformation rather than a staircase.
"""
import logging

import torch

from odyssey_renderer.mtgs.gaussian_model.rigid_object import RigidPortableSubModel
from odyssey_renderer.mtgs.utils.gaussian_utils import quat_mult, quat_to_rotmat
from odyssey_renderer.omnire.deform_network import ConditionalDeformNetwork

logger = logging.getLogger(__name__)


class OmniReDeformableSubModel(RigidPortableSubModel):

    MODEL_TYPE = "deformable"

    def _update_default_config(self, config):
        config = super()._update_default_config(config)
        config["num_timesteps"] = config.get("num_timesteps", None)
        config["deform_network"] = config.get("deform_network", None)
        # Index of this node OWN first frame within log_timestamps. Zero for a
        # single-chunk asset, where the two numberings coincide. In a merged
        # multi-chunk asset log_timestamps spans the whole 796-frame log while
        # the deform MLP was only ever trained on its owner 214-frame window,
        # so the offset is what keeps normalised time meaning the same thing.
        config["frame_offset"] = config.get("frame_offset", 0)
        return config

    @staticmethod
    def _flatten(obj, prefix=""):
        """Undo AttrDict's dotted-key nesting.

        portable_utils.AttrDict splits every key on its first dot and then
        recursively re-wraps the value, so a flat `deform_network.linear.0.weight`
        arrives as four levels of nesting. That is also why the base loader has a
        `if "gauss_params" in dict` branch alongside the flat one. Flatten back
        rather than guessing which form we were handed.
        """
        out = {}
        for k, v in obj.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(OmniReDeformableSubModel._flatten(v, key + "."))
            else:
                out[key] = v
        return out

    def load_state_dict(self, sd: dict):
        sd = dict(sd)                  # the base pops keys; do not mutate the asset
        prefix = "deform_network."
        net_nested = {k: sd.pop(k) for k in list(sd) if k.startswith("deform_network")}
        net_sd = {k[len(prefix):]: v
                  for k, v in self._flatten(net_nested).items()
                  if k.startswith(prefix)}
        if not net_sd:
            raise ValueError(
                "no deform_network.* weights in the asset; this node type "
                "requires them (convert.deformable emits them)")
        self._instance_embedding = sd.pop("instance_embedding").reshape(-1)
        self._instance_size = sd.pop("instance_size").reshape(-1)
        msg = super().load_state_dict(sd)

        cfg = self.config.deform_network or {}
        w0 = net_sd["linear.0.weight"]
        n_linear = len({int(k.split(".")[1]) for k in net_sd
                        if k.startswith("linear.") and k.endswith(".weight")})
        self.deform_network = ConditionalDeformNetwork(
            D=n_linear,
            W=int(w0.shape[0]),
            input_ch=3,
            embed_dim=int(self._instance_embedding.numel()),
            x_multires=int(cfg.get("x_multires", 10)),
            t_multires=int(cfg.get("t_multires", 10)),
            deform_quat="gaussian_rotation.weight" in net_sd,
            deform_scale="gaussian_scaling.weight" in net_sd,
        )
        got, want = int(self.deform_network.input_ch), int(w0.shape[1])
        if got != want:
            raise ValueError(
                "deform_network input width %d rebuilt from config does not match "
                "the checkpoint's %d; x_multires/t_multires/embed_dim disagree "
                "with the trained weights" % (got, want))
        self.deform_network.load_state_dict(net_sd)
        self.deform_network.eval()
        for p in self.deform_network.parameters():
            p.requires_grad_(False)
        self._deform_cache = (None, None, None)
        self._norm_t = None
        self._warned_no_timestamp = False
        # NOTE: do NOT set config.fourier_features_dim as a "colours are
        # time-varying" hint. RigidPortableSubModel.get_true_features_dc would
        # then route features_dc through get_fourier_features, an IDFT that
        # expects features_dc to be (N, fourier_dim, 3). Ours is (N, 3) -- a
        # plain SH DC band -- so it would corrupt every colour, not annotate it.
        return msg

    # -- normalised time -----------------------------------------------------

    def _normalized_time(self, timestamp):
        """Fractional frame index / (num_timesteps - 1), matching upstream."""
        n = self.config.num_timesteps
        if n is None or int(n) < 2 or timestamp is None:
            return None
        ts = self.config.log_timestamps
        ts = ts.squeeze() if torch.is_tensor(ts) else torch.as_tensor(ts)
        ts = ts.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        t = float(timestamp)
        if t <= float(ts[0]):
            idx = 0.0
        elif t >= float(ts[-1]):
            idx = float(len(ts) - 1)
        else:
            hi = int(torch.searchsorted(ts, torch.tensor(t, dtype=torch.float64)))
            hi = min(max(hi, 1), len(ts) - 1)
            lo = hi - 1
            span = float(ts[hi] - ts[lo])
            idx = lo + ((t - float(ts[lo])) / span if span > 0 else 0.0)
        idx = idx - float(int(self.config.frame_offset or 0))
        t_norm = idx / float(int(n) - 1)
        # THE DOMAIN ASSERTION. scene_graph.py resolves cur_frame by argmin over
        # linspace(0, 1, num_timesteps), which CLAMPS any out-of-range time to
        # frame 0 or frame N-1 and renders the nearest end of the window --
        # no error, no log, a plausible and wrong picture. The MLP time is
        # normalised WITHIN its own chunk, so a merged asset can ask chunk 0
        # network for t = 3.29. Refuse instead: returning None renders the
        # canonical shape, which is visibly undeformed rather than silently
        # misdeformed, and says so once.
        if t_norm < -1e-6 or t_norm > 1.0 + 1e-6:
            # Record, do not warn here. get_global_gaussians resolves the time
            # BEFORE the base class decides whether this node is drawn at all,
            # and a merged asset parks an out-of-window actor at the absence
            # sentinel -- so the common case is a node that is about to be
            # dropped, where the out-of-domain time is never used. Warning here
            # would fire once per such node per frame and bury the case that
            # matters. _deformation warns instead, which is only reached when
            # the node IS being drawn.
            self._out_of_domain = t_norm
            return None
        self._out_of_domain = None
        return t_norm

    # -- the deformation -----------------------------------------------------

    def _deformation(self, norm_t):
        """(delta_xyz, delta_quat) for every gaussian at this normalised time."""
        key = None if norm_t is None else round(float(norm_t), 12)
        if self._deform_cache[0] == key and key is not None:
            return self._deform_cache[1], self._deform_cache[2]
        if norm_t is None:
            ood = getattr(self, "_out_of_domain", None)
            if ood is not None and not getattr(self, "_warned_out_of_domain", False):
                # Reached only while this node is actually being rendered, so
                # this is the real fault: a deformable is on screen at a time
                # its own MLP was never trained for. scene_graph.py would have
                # CLAMPED to frame 0 or N-1 and drawn a plausible wrong pose;
                # this draws the canonical shape and says so.
                n = self.config.num_timesteps
                base = int(self.config.frame_offset or 0)
                logger.warning(
                    "%s: DRAWN at normalised time %.4f, outside [0, 1]. This "
                    "node deform MLP was trained on log frames %d..%d. "
                    "Refusing to extrapolate; rendering the canonical "
                    "(undeformed) shape. A merged asset should have parked "
                    "this actor at the absence sentinel outside its owner "
                    "window -- check the time-cut in convert/merge.py.",
                    self.model_name, ood, base,
                    base + (int(n) - 1 if n else 0))
                self._warned_out_of_domain = True
            return None, None

        means = self.gauss_params["means"]
        dev = means.device
        if next(self.deform_network.parameters()).device != dev:
            self.deform_network.to(dev)
        emb = self._instance_embedding.to(dev)
        height = self._instance_size.to(dev)[2]
        # deformable.py:43 -- the MLP input is the canonical position normalised
        # by the instance's HEIGHT, times two. instances_size is load-bearing.
        x = means.detach() / height * 2.0
        t = torch.full((means.shape[0], 1), float(norm_t),
                       device=dev, dtype=means.dtype)
        cond = emb.unsqueeze(0).expand(means.shape[0], -1)
        with torch.no_grad():
            d_xyz, d_quat, _ = self.deform_network(x, t, cond)
        self._deform_cache = (key, d_xyz, d_quat)
        return d_xyz, d_quat

    def get_global_gaussians(self, quat=None, trans=None, timestamp=None, **kwargs):
        # Resolve the time BEFORE the base class calls get_means / get_quats,
        # neither of which receives the timestamp.
        if timestamp is None and not self._warned_no_timestamp:
            # interactive/renderer.py:221 (FreeDriveRenderer, the VIEWER path)
            # calls get_global_gaussians(..., timestamp=None) unconditionally,
            # and its gate at :128 is `isinstance(model, RigidPortableSubModel)`
            # -- which this class satisfies, so subclassing cannot dodge it.
            # With no time there is no deformation and the actor renders in its
            # canonical pose: a silent, plausible, WRONG picture. The closed-loop
            # renderer (MTGSRenderEngine.update_world) always passes a timestamp,
            # so this is a viewer-only defect. Say so once, loudly, rather than
            # letting it pass as "the cyclist looks a bit stiff".
            logger.warning(
                "%s: get_global_gaussians called with timestamp=None; rendering "
                "the CANONICAL (undeformed) shape. If this is FreeDriveRenderer, "
                "its line 128 isinstance gate needs to exclude "
                "OmniReDeformableSubModel for the viewer to show deformation.",
                self.model_name)
            self._warned_no_timestamp = True
        self._norm_t = self._normalized_time(timestamp)
        if self._norm_t is None and getattr(self, "_out_of_domain", None) is not None:
            # OUT OF DOMAIN MEANS ABSENT, and it has to be enforced HERE.
            #
            # A merged asset parks this actor at the absence sentinel outside its
            # owner window, and for an actor the simulator does not drive that is
            # enough: _decide_global_pose falls through to
            # _get_log_pose_from_timestamp, gets None, and the node is skipped.
            #
            # But rigid_object.py:236 short-circuits the moment a quat AND a trans
            # are supplied, and in a log-replay rollout the agent manager supplies
            # both for EVERY tracked actor. So the sentinel is bypassed exactly
            # when it matters, and the node renders its canonical, undeformed
            # shape at a time its own MLP never saw.
            #
            # Found by driving the whole 796-frame merged scene rather than
            # sampling frames at the cuts: one actor, 57 frames, 5.7 s.
            # A chunk owns an actor for its whole life or not at all; where the
            # owner has no model for an instant, the actor is absent at that
            # instant, and that is what this returns.
            if not getattr(self, "_warned_absent_out_of_domain", False):
                logger.warning(
                    "%s: absent outside its owner window (log frames %d..%d); "
                    "not drawn. The merged asset parks it at the sentinel, but a "
                    "driven actor bypasses that, so it is enforced here.",
                    self.model_name, int(self.config.frame_offset or 0),
                    int(self.config.frame_offset or 0)
                    + int(self.config.num_timesteps or 1) - 1)
                self._warned_absent_out_of_domain = True
            return None
        return super().get_global_gaussians(quat=quat, trans=trans,
                                            timestamp=timestamp, **kwargs)

    def get_means(self, global_quat, global_trans):
        d_xyz, _ = self._deformation(getattr(self, "_norm_t", None))
        local_means = self.gauss_params["means"]
        if d_xyz is not None:
            local_means = local_means + d_xyz
        rot = quat_to_rotmat(global_quat[None, ...])[0, ...]
        self.global_means = local_means @ rot.T + global_trans
        return self.global_means

    def get_quats(self, global_quat, global_trans=None):
        _, d_quat = self._deformation(getattr(self, "_norm_t", None))
        local = self.quats / self.quats.norm(dim=-1, keepdim=True)
        if d_quat is not None:
            # deformable.py:67 adds the delta to the NORMALISED quat, and
            # transform_quats then normalises again.
            local = local + d_quat
            local = local / local.norm(dim=-1, keepdim=True)
        return quat_mult(global_quat[None, ...], local)
