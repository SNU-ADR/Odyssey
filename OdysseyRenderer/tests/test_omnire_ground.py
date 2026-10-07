"""The GroundNodes port produces the same geometry as in training.

The reference is three rules from drivestudio `models/nodes/ground.py`:
  get_scaling  exp(s[:, :2]).clamp(min, max) with a constant thickness on the third axis
  get_quats    yaw-only quaternion (w, 0, 0, z), normalized
  with surface_follow, the LiDAR surface-normal rotation is multiplied on the left

The only check is that our node reproduces these formulas. Reference values are built by
restating the formulas in the test; comparing against the implementation would pass if both were wrong.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from odyssey_renderer.omnire.ground import OmniReGroundSubModel  # noqa: E402


def _asset(n=64, *, thickness=0.01, lo=0.005, hi=0.06, surface=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    state = {
        "gauss_params.means": torch.randn(n, 3, generator=g),
        # deliberately include values outside both clamp bounds
        "gauss_params.scales": torch.randn(n, 3, generator=g) * 2 + math.log(0.02),
        "gauss_params.quats": torch.randn(n, 4, generator=g),
        "gauss_params.features_dc": torch.randn(n, 3, generator=g),
        "gauss_params.features_rest": torch.randn(n, 15, 3, generator=g),
        "gauss_params.opacities": torch.randn(n, 1, generator=g),
    }
    config = {"type": "OmniReGroundSubModel", "sh_degree": 3, "scale_dim": 3,
              "thickness": thickness, "min_planar_extent": lo, "max_planar_extent": hi,
              "surface_follow": surface}
    if surface:
        q = torch.randn(n, 4, generator=g)
        state["surface_quats"] = q / q.norm(dim=-1, keepdim=True)
    from odyssey_renderer.mtgs.utils.portable_utils import convert_to_attribute_dict
    # AttrDict nests "gauss_params.means", so the raw state is returned separately for comparison.
    return convert_to_attribute_dict({"config": config, "state_dict": state}), state


def _quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def test_scales_clamp_planar_and_freeze_thickness():
    asset, raw_state = _asset(thickness=0.01, lo=0.005, hi=0.06)
    model = OmniReGroundSubModel(asset=asset, model_name="ground")
    got = model.get_scales()
    raw = raw_state["gauss_params.scales"]
    want_planar = torch.exp(raw[:, :2]).clamp(min=0.005, max=0.06)
    assert torch.allclose(got[:, :2], want_planar, atol=0, rtol=0)
    assert torch.equal(got[:, 2], torch.full((got.shape[0],), 0.01))
    # both clamp bounds must actually trigger, otherwise this test checks nothing
    assert (torch.exp(raw[:, :2]) > 0.06).any() and (torch.exp(raw[:, :2]) < 0.005).any()


def test_quats_are_yaw_only_when_surface_follow_off():
    asset, raw_state = _asset(surface=False)
    model = OmniReGroundSubModel(asset=asset, model_name="ground")
    got = model.get_quats()
    raw = raw_state["gauss_params.quats"]
    zero = torch.zeros_like(raw[:, 0])
    want = torch.stack([raw[:, 0], zero, zero, raw[:, 3]], dim=-1)
    want = want / want.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    assert torch.allclose(got, want, atol=1e-6)
    assert torch.allclose(got[:, 1:3], torch.zeros_like(got[:, 1:3]), atol=1e-7)


def test_quats_compose_surface_normal_on_the_left():
    asset, raw_state = _asset(surface=True)
    model = OmniReGroundSubModel(asset=asset, model_name="ground")
    raw = raw_state["gauss_params.quats"]
    zero = torch.zeros_like(raw[:, 0])
    yaw = torch.stack([raw[:, 0], zero, zero, raw[:, 3]], dim=-1)
    yaw = yaw / yaw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    want = _quat_mul(raw_state["surface_quats"], yaw)
    assert torch.allclose(model.get_quats(), want, atol=1e-6)


def test_surface_follow_without_surface_quats_fails_loudly():
    """Silently using yaw only would render a plausible but flattened road that nothing else catches."""
    asset, raw_state = _asset(surface=True)
    # The node reads from asset. raw_state is a separate copy made by AttrDict,
    # so deleting the key there would leave it visible to the node.
    del asset.state_dict["surface_quats"]
    with pytest.raises(ValueError, match="surface_quats"):
        OmniReGroundSubModel(asset=asset, model_name="ground")


def test_missing_ctrl_values_fail_loudly():
    asset, raw_state = _asset()
    del asset.config["thickness"]
    with pytest.raises(ValueError, match="thickness"):
        OmniReGroundSubModel(asset=asset, model_name="ground")


def test_means_and_opacity_are_untouched():
    asset, raw_state = _asset()
    model = OmniReGroundSubModel(asset=asset, model_name="ground")
    assert torch.equal(model.get_means(), raw_state["gauss_params.means"])
    want = torch.sigmoid(raw_state["gauss_params.opacities"]).squeeze(-1)
    assert torch.allclose(model.get_opacity(), want)


def test_declares_itself_static_in_world():
    """Without the static declaration, activation reruns over 10.9M rows every frame."""
    assert OmniReGroundSubModel.STATIC_IN_WORLD is True


def test_registered_under_its_own_namespaced_type():
    from odyssey_renderer.omnire.registry import register_all
    assert register_all()["OmniReGroundSubModel"] is OmniReGroundSubModel


def test_engine_treats_ground_as_scenery_not_an_actor():
    """The road must remain after the ego passes the end of the log (actors disappear)."""
    from odyssey_renderer.omnire import engine as omnire_engine
    assert "ground" in omnire_engine.SCENERY_TOKENS
