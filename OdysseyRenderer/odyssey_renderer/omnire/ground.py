"""Portable GroundNodes: a road surface as its own Gaussian class.

Equations: drivestudio `models/nodes/ground.py` (GroundNodes.get_scaling,
quat_act/get_quats, and the surface_follow composition in get_gaussians).

The class is static in the world frame -- it has no per-frame pose and no
time-conditioned appearance -- so everything that separates it from an ordinary
background is in two activations:

  scaling   the two in-plane axes are exp() then clamped to
            [min_planar_extent, max_planar_extent]; the third axis is NOT a
            parameter at all, it is the constant ctrl.thickness. Letting the
            renderer take exp() of the stored third column would give a disc a
            thickness of e^s metres, which is how a road turns into gravel.
  rotation  yaw about world Z only (keep w and z, drop x and y, normalise), and
            with surface_follow the LiDAR surface normal rotation multiplied on
            the LEFT of that yaw.

Why the normal arrives as a tensor rather than being recomputed here: it is not
a model parameter. In training it comes from a height grid rebuilt from the
stored means (`build_surface`), which needs scipy and a pass over every disc --
10.9 M of them in a large asset. The exporter does that once, with the same code
path the training loader uses, and ships the composed rotation as
`surface_quats`. Recomputing it per asset load would put a grid build on the
rollout's critical path and give a second implementation to keep in step.

A missing `surface_quats` under surface_follow is an error, never a fallback to
bare yaw: bare yaw renders a *flat* road that looks entirely plausible from the
ego view and would be found, if ever, as "the road is slightly wrong on hills".
"""
from __future__ import annotations

import torch

from odyssey_renderer.mtgs.gaussian_model.vanilla_gaussian_splatting import (
    VanillaPortableGaussianModel,
)
from odyssey_renderer.mtgs.utils.gaussian_utils import quat_mult

import logging

logger = logging.getLogger(__name__)

# No single reconstruction is larger than this. A road-surface centre farther away means the
# coordinate frame is wrong, not the coordinates (an asset exported with the anchor subtracted).
_RECON_SPAN_M = 10_000.0

# ctrl values that have no safe default. drivestudio defaults exist, but a
# checkpoint trained with other values would then render with the defaults and
# look merely a bit off -- so require them and say which one is missing.
REQUIRED_CTRL = ("thickness", "min_planar_extent", "max_planar_extent")


class OmniReGroundSubModel(VanillaPortableGaussianModel):
    """Road-surface Gaussians with drivestudio's constrained activations."""

    MODEL_TYPE = "ground"

    # The renderer memoises models whose output cannot change (mtgs
    # _is_static_model). Subclassing alone drops out of that set, which would
    # re-run exp/clamp/norm/sigmoid over every disc on every frame. This class
    # genuinely cannot change between frames, so it says so.
    STATIC_IN_WORLD = True

    def _update_default_config(self, config):
        config = super()._update_default_config(config)
        missing = [k for k in REQUIRED_CTRL if config.get(k) is None]
        if missing:
            raise ValueError(
                "OmniReGroundSubModel config is missing %s; these come from the "
                "training ctrl block and have no safe default" % ", ".join(missing)
            )
        lo = float(config["min_planar_extent"])
        hi = float(config["max_planar_extent"])
        if not 0.0 < lo < hi:
            raise ValueError(
                "OmniReGroundSubModel needs 0 < min_planar_extent (%r) < "
                "max_planar_extent (%r)" % (lo, hi)
            )
        config["thickness"] = float(config["thickness"])
        config["min_planar_extent"] = lo
        config["max_planar_extent"] = hi
        config["surface_follow"] = bool(config.get("surface_follow", False))
        return config

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self._restore_recon_frame()
        quats = state.get("surface_quats")
        if self.config["surface_follow"]:
            if quats is None:
                raise ValueError(
                    "OmniReGroundSubModel was trained with surface_follow but the "
                    "asset carries no surface_quats; rendering bare yaw would give "
                    "a flat road that still looks plausible"
                )
            quats = torch.as_tensor(quats)
            if quats.shape != self.gauss_params["means"].shape[:1] + (4,):
                raise ValueError(
                    "surface_quats %s does not match %d discs"
                    % (tuple(quats.shape), self.gauss_params["means"].shape[0])
                )
            # A buffer, not a Parameter: it is measured geometry, never optimised,
            # and it must ride .to(device) with the rest of the model.
            self.register_buffer("surface_quats", quats.float(), persistent=False)
        elif quats is not None:
            raise ValueError(
                "asset carries surface_quats but config.surface_follow is false; "
                "one of the two is from a different run"
            )

    def _restore_recon_frame(self):
        """Move the road-surface means back into the reconstruction frame used by the background.

        The exporter subtracts recon2world_translation from the road surface only. The background
        is stored without it, so the two load offset by the anchor and the renderer corrects
        neither (recon2global_translation is applied only to ego/camera poses).

        Correctly exported assets are left untouched: if the centre lies within the
        reconstruction span, nothing is changed.
        """
        means = self.gauss_params["means"]
        centre = means.detach().double().mean(0)
        far = float(torch.linalg.norm(centre))
        if far <= _RECON_SPAN_M:
            logger.info("[ground] frame OK (centre %.1f m) -- no correction", far)
            return
        anchor = self.config.get("recon2world_translation")
        if anchor is None:
            raise ValueError(
                "road-surface centre is %.0f m away, outside the reconstruction span, and the "
                "asset has no recon2world_translation to restore it" % far)
        shift = torch.as_tensor(anchor, dtype=means.dtype, device=means.device).reshape(3)
        with torch.no_grad():
            means.add_(shift)
        after = float(torch.linalg.norm(means.detach().double().mean(0)))
        if after > _RECON_SPAN_M:
            raise ValueError(
                "road-surface centre is still %.0f m away after adding the anchor (%.0f m "
                "before); the anchor is not the cause of this offset" % (after, far))
        logger.warning(
            "[ground] road surface was outside the reconstruction frame -- centre %.0f m -> "
            "%.1f m. Restored by adding anchor %s (asset exporter bug)",
            far, after, [round(float(v), 1) for v in shift.tolist()])

    def get_scales(self):
        planar = torch.exp(self.scales[:, :2]).clamp(
            min=self.config["min_planar_extent"], max=self.config["max_planar_extent"])
        thin = planar.new_full((planar.shape[0], 1), self.config["thickness"])
        return torch.cat([planar, thin], dim=-1)

    def get_quats(self):
        raw = self.quats
        zero = torch.zeros_like(raw[:, 0])
        yaw = torch.stack([raw[:, 0], zero, zero, raw[:, 3]], dim=-1)
        yaw = yaw / yaw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        if not self.config["surface_follow"]:
            return yaw
        return quat_mult(self.surface_quats, yaw)
