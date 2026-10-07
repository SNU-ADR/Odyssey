"""CPU contracts for timed traffic-light appearance and explicit head masking."""
import copy
import importlib.util

import pytest
import torch
from torch import nn
from torch.nn import functional as F


# Import actual native leaf modules without importing renderer/env/CUDA engines.
import sys
import types
from pathlib import Path
_root = Path(__file__).resolve().parents[1] / "odyssey_renderer"
_previous = {k:v for k,v in sys.modules.items() if k == "odyssey_renderer" or k.startswith("odyssey_renderer.")}
for _name, _directory in (("odyssey_renderer",_root),
    ("odyssey_renderer.omnire",_root/"omnire"),
    ("odyssey_renderer.mtgs",_root/"mtgs"),
    ("odyssey_renderer.mtgs.utils",_root/"mtgs/utils")):
    _package=types.ModuleType(_name); _package.__path__=[str(_directory)]
    sys.modules[_name]=_package
try:
    from odyssey_renderer.omnire.deform_network import ConditionalDeformNetwork
    from odyssey_renderer.mtgs.utils.portable_utils import convert_to_attribute_dict
    from odyssey_renderer.omnire.traffic_light import OmniReTrafficLightSubModel as TrafficLightNodes
finally:
    for _name in list(sys.modules):
        if _name == "odyssey_renderer" or _name.startswith("odyssey_renderer."):
            del sys.modules[_name]
    sys.modules.update(_previous)


def build_traffic_light_asset(state, config, timestamps, step=10):
    """Small synthetic portable fixture; no exporter/runtime dependency."""
    metadata=copy.deepcopy(state["_traffic_light_metadata"])
    ctrl=config["ctrl"]; degree=ctrl["sh_degree"]
    portable={"gauss_params."+name:state["_"+name].clone() for name in
              ("means","scales","quats","features_dc","features_rest","opacities")}
    portable.update({k:v.clone() for k,v in state.items() if k.startswith(("deform_network.","appearance_network."))})
    for key in ("points_ids","instances_size","instances_embedding"):
        portable[key]=state[key].clone()
    portable.update(installation_trans=state["instances_trans"][0].clone(),
                    installation_quats=state["instances_quats"][0].clone(),_traffic_light_metadata=metadata)
    return dict(config=dict(type="OmniReTrafficLightSubModel",schema_version=1,
        sh_degree=degree,effective_sh_degree=min(step//ctrl["sh_degree_interval"],degree),
        temporal_sh_mode=ctrl["temporal_sh_mode"],max_displacement=ctrl["max_displacement"],
        scale_dim=3,networks=config["networks"],training_timestamps_us=timestamps,
        source_scene_id=metadata["source_context"]["scene_id"],
        source_context=metadata["source_context"],recon2world_translation=[0.,0.,0.],checkpoint_step=step),
        state_dict=portable)



def raw_fixture(mode="full", degree=1):
    # Interleaved head rows catch masks that reorder geometry without RGB.
    generator = torch.Generator().manual_seed(37)
    network = ConditionalDeformNetwork(D=3, W=8, embed_dim=2, x_multires=1,
                                      t_multires=1, deform_quat=True, deform_scale=True)
    appearance = nn.Sequential(nn.Linear(14, 8), nn.ReLU(), nn.Linear(8, 8),
                               nn.ReLU(), nn.Linear(3 * ((degree + 1)**2 if mode == "full" else 1) + 1, 1))
    # Replace output with the actual source architecture.
    appearance[4] = nn.Linear(8, 3 * ((degree + 1)**2 if mode == "full" else 1) + 1)
    for module in (network, appearance):
        for p in module.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=generator) * .18 + .08)
    state = {
        "_means": torch.tensor([[.2, .3, .4], [.5, -.2, .1], [-.1, .4, .2]]),
        "_scales": torch.tensor([[-2., -3., -4.]]).repeat(3, 1),
        "_quats": torch.tensor([[2., .1, .2, -.1]]).repeat(3, 1),
        "_features_dc": torch.tensor([[.1, .2, .3], [.2, -.1, .1], [.3, .1, -.2]]),
        "_features_rest": torch.randn((3, (degree+1)**2-1, 3), generator=generator) * .08,
        "_opacities": torch.tensor([[-1.], [.5], [1.]]),
        "points_ids": torch.tensor([[1], [0], [1]]),
        "instances_size": torch.tensor([[1., 2., 3.], [2., 4., 1.]]),
        "instances_embedding": torch.tensor([[.1, .2], [.3, .5]]),
        "instances_trans": torch.tensor([[[10., 0., 1.], [0., 20., 2.]]]).repeat(3, 1, 1),
        # Non-unit installation quaternion catches missing normalization for rotation.
        "instances_quats": torch.tensor([[[2., 0., 0., 0.], [2., 0., 0., 2.]]]).repeat(3, 1, 1),
        "instances_fv": torch.ones(3, 2, dtype=torch.bool),
        "_traffic_light_metadata": {
            "version": 1, "appearance_mode": mode, "head_ids": ["a", "b"],
            "heads": {"a": {"observed_frames": [0, 2], "representatives": {"red": 0}},
                      "b": {"observed_frames": [1, 2], "representatives": {}}},
            "source_context": {"version": 1, "coordinate_frame": "reconstruction",
                               "scene_id": "fixture", "start_timestep": 0, "num_frames": 3},
            # Deliberately neither timestamp-ratio nor linspace.
            "normalized_timestamps": torch.tensor([0., .2, 1.]),
        },
    }
    state.update({"deform_network." + k: v.detach().clone() for k, v in network.state_dict().items()})
    state.update({"appearance_network." + k: v.detach().clone() for k, v in appearance.state_dict().items()})
    config = {"networks": {"D": 3, "W": 8, "embed_dim": 2, "x_multires": 1,
                            "t_multires": 1, "deform_quat": True, "deform_scale": True},
              "ctrl": {"sh_degree": degree, "sh_degree_interval": 1,
                       "temporal_sh_mode": mode, "max_displacement": .05}}
    return state, config, [100, 120, 200]


def make_model(mode="full", degree=1, nested=False):
    state, config, timestamps = raw_fixture(mode, degree)
    asset = build_traffic_light_asset(state, config, timestamps, step=10)
    return TrafficLightNodes(convert_to_attribute_dict(asset) if nested else asset), state


def test_kernel_is_available():
    assert TrafficLightNodes.MODEL_TYPE == "traffic_light"


def embed(x):
    return torch.cat((x, x.sin(), x.cos()), -1)


def source_reference(state, frames, degree=1, mode="full"):
    """Independent evaluation of the source equations, not portable model helpers."""
    ids = state["points_ids"][:, 0]
    x = state["_means"] / state["instances_size"][ids] * 2
    t = state["_traffic_light_metadata"]["normalized_timestamps"][torch.tensor(frames)[ids]][:, None]
    inputs = torch.cat((embed(x), embed(t), state["instances_embedding"][ids]), -1)
    h = inputs
    for index in range(3):
        h = F.relu(F.linear(h, state[f"deform_network.linear.{index}.weight"],
                           state[f"deform_network.linear.{index}.bias"]))
        if index == 1:
            h = torch.cat((inputs, h), -1)
    delta = {}
    for name in ("warp", "rotation", "scaling"):
        delta[name] = F.linear(h, state[f"deform_network.gaussian_{name}.weight"],
                              state[f"deform_network.gaussian_{name}.bias"])
    local = state["_means"] + .05 * delta["warp"].tanh()
    # The fixture installations are identity and a 90-degree Z rotation.
    means = local.clone()
    means[ids == 1] = torch.stack((-local[ids == 1, 1], local[ids == 1, 0],
                                   local[ids == 1, 2]), -1)
    means += state["instances_trans"][0, ids]
    local_q = F.normalize(F.normalize(state["_quats"], dim=-1) + .05 * delta["rotation"].tanh(), dim=-1)
    q = state["instances_quats"][0, ids]
    # Hamilton product independent of portable quat_mult.
    w = q[:, :1] * local_q[:, :1] - (q[:, 1:] * local_q[:, 1:]).sum(-1, keepdim=True)
    v = q[:, :1] * local_q[:, 1:] + local_q[:, :1] * q[:, 1:] + torch.cross(q[:, 1:], local_q[:, 1:], dim=-1)
    quats = F.normalize(torch.cat((w, v), -1), dim=-1)
    scales = state["_scales"].exp() * (.2 * delta["scaling"].tanh()).exp()
    a = inputs
    for index in (0, 2, 4):
        a = F.linear(a, state[f"appearance_network.{index}.weight"],
                     state[f"appearance_network.{index}.bias"])
        if index != 4:
            a = a.relu()
    coeffs = torch.cat((state["_features_dc"][:, None], state["_features_rest"]), 1).clone()
    if mode == "full":
        coeffs += a[:, :-1].reshape_as(coeffs)
    else:
        coeffs[:, 0] += a[:, :3]
    return dict(means=means, quats=quats, scales=scales,
                opacities=(state["_opacities"] + a[:, -1:]).sigmoid().squeeze(-1)), coeffs


@pytest.mark.parametrize("mode,degree", [("full", 1), ("dc", 1), ("dc", 0)])
def test_geometry_and_rgb_match_source_equations(mode, degree):
    model, state = make_model(mode, degree, nested=True)
    model.set_source_frames({"a": 2, "b": 1})
    expected, coeffs = source_reference(state, [2, 1], degree, mode)
    got = model.get_global_gaussians(timestamp=100)
    for key in expected:
        torch.testing.assert_close(got[key], expected[key], atol=2e-6, rtol=2e-6)
    camera = torch.eye(4)[None]
    rgb = model.get_gaussian_rgbs(camera, timestamp=100)
    if degree == 0:
        want = coeffs[:, 0].sigmoid()
    else:
        dirs = F.normalize(expected["means"], dim=-1)
        # Explicit degree-one real SH basis and signs.
        bases = torch.stack((torch.full_like(dirs[:, 0], .28209479177387814),
                             -.4886025119029199*dirs[:, 1], .4886025119029199*dirs[:, 2],
                             -.4886025119029199*dirs[:, 0]), -1)
        want = ((bases[:, :, None] * coeffs).sum(1) + .5).clamp(0, 1)
    torch.testing.assert_close(rgb, want, atol=2e-6, rtol=2e-6)


def test_strict_default_outside_window_requires_every_head():
    model, _ = make_model()
    model.set_source_frames({"a": 2})
    with pytest.raises(ValueError):
        model.get_global_gaussians(timestamp=201)
    model.set_source_frames({"a": 2, "b": 1})
    assert model.get_global_gaussians(timestamp=201)["means"].shape == (3, 3)


def test_active_mask_preserves_interleaved_geometry_and_rgb_and_empty_is_legal():
    model, _ = make_model()
    model.set_source_frames({"a": 2, "b": 1})
    full = model.get_global_gaussians(timestamp=201)
    full_rgb = model.get_gaussian_rgbs(torch.eye(4)[None], timestamp=201)
    model.set_source_frames({"b": 1}, active_head_ids=["b"])
    selected = model.get_global_gaussians(timestamp=201)
    for key, value in full.items():
        torch.testing.assert_close(selected[key], value[[0, 2]])
    torch.testing.assert_close(model.get_gaussian_rgbs(torch.eye(4)[None], timestamp=201), full_rgb[[0, 2]])
    assert model.active_head_ids == ("b",)
    model.set_source_frames({}, active_head_ids=[])
    for key, value in model.get_global_gaussians(timestamp=201).items():
        assert value.shape == ((0,) if key == "opacities" else (0, 4 if key == "quats" else 3))
    assert model.get_gaussian_rgbs(torch.eye(4)[None], timestamp=201).shape == (0, 3)


def test_mask_mode_does_not_change_in_training_natural_replay():
    model, _ = make_model()
    natural = model.get_global_gaussians(timestamp=120)
    natural_rgb = model.get_gaussian_rgbs(torch.eye(4)[None], timestamp=120)
    model.set_source_frames({}, active_head_ids=[])
    for k, v in model.get_global_gaussians(timestamp=120).items():
        torch.testing.assert_close(v, natural[k])
    torch.testing.assert_close(model.get_gaussian_rgbs(torch.eye(4)[None], timestamp=120), natural_rgb)


@pytest.mark.parametrize("frames,mask,error", [
    ({"unknown": 0}, None, KeyError), ({"a": True}, None, ValueError),
    ({"a": 1.0}, None, ValueError), ({"a": 1}, None, ValueError),
    ({"a": 3}, None, ValueError), ({"a": 0}, ["b"], ValueError),
    ({"a": 0}, ["a", "a"], ValueError), ({}, ["unknown"], KeyError),
])
def test_invalid_update_is_atomic(frames, mask, error):
    model, _ = make_model()
    model.set_source_frames({"b": 2}, active_head_ids=["b"])
    before = model.get_global_gaussians(timestamp=300)
    with pytest.raises(error):
        model.set_source_frames(frames, active_head_ids=mask)
    for k, v in model.get_global_gaussians(timestamp=300).items():
        torch.testing.assert_close(v, before[k])
    assert model.active_head_ids == ("b",)


def test_clear_and_training_remove_mask_and_overrides():
    model, _ = make_model()
    model.set_source_frames({}, active_head_ids=[])
    model.clear_source_frames()
    with pytest.raises(ValueError):
        model.get_global_gaussians(timestamp=300)
    model.set_source_frames({"a": 2, "b": 1})
    model.train(True)
    with pytest.raises(ValueError):
        model.set_source_frames({"a": 0})
    model.eval()
    model.set_frame(0)
    assert model.source_frames().tolist() == [0, 0]


def test_source_frame_and_camera_changes_do_not_reuse_stale_cache():
    model, _ = make_model()
    model.set_frame(0)
    first = model.get_global_gaussians()["opacities"].clone()
    model.set_frame(2)
    assert not torch.equal(model.get_global_gaussians()["opacities"], first)
    cam = torch.eye(4)[None]
    rgb = model.get_gaussian_rgbs(cam)
    cam[:, :3, 3] = torch.tensor([30., 10., 20.])
    assert not torch.equal(model.get_gaussian_rgbs(cam), rgb)
    model.double()
    assert model.get_global_gaussians()["means"].dtype == torch.float64
    assert all(t.dtype == torch.float64 for t in model.buffers() if t.is_floating_point())


def test_missing_representative_and_inbetween_timestamp_rejected():
    model, _ = make_model()
    with pytest.raises(ValueError):
        model.set_representative_states({"b": "green"})
    with pytest.raises(ValueError):
        model.get_global_gaussians(timestamp=110)
    with pytest.raises(ValueError):
        model.get_global_gaussians(quat=torch.ones(4), timestamp=100)


def test_canonical_quaternion_magnitude_does_not_change_deformed_orientation():
    state, cfg, times = raw_fixture()
    original = TrafficLightNodes(build_traffic_light_asset(state, cfg, times, step=10))
    state["_quats"] *= 1e-14
    scaled = TrafficLightNodes(build_traffic_light_asset(state, cfg, times, step=10))
    torch.testing.assert_close(scaled.get_global_gaussians(timestamp=100)["quats"],
                               original.get_global_gaussians(timestamp=100)["quats"],
                               rtol=2e-6, atol=2e-6)


def test_effective_degree_zero_keeps_configured_degree_sh_activation():
    state, cfg, times = raw_fixture()
    model = TrafficLightNodes(build_traffic_light_asset(state, cfg, times, step=0))
    _, coeffs = source_reference(state, [0, 0])
    want = (.28209479177387814 * coeffs[:, 0] + .5).clamp(0, 1)
    torch.testing.assert_close(model.get_gaussian_rgbs(torch.eye(4)[None], timestamp=100), want)


def test_standard_state_reload_invalidates_temporal_cache():
    model, _ = make_model()
    before = model.get_global_gaussians(timestamp=100)["opacities"].clone()
    changed = copy.deepcopy(model.state_dict())
    changed["_opacities"] += 2
    model.load_state_dict(changed)
    after = model.get_global_gaussians(timestamp=100)["opacities"]
    torch.testing.assert_close(after, torch.sigmoid(torch.logit(before) + 2))


def test_inference_mode_setup_can_render_and_does_not_cache_untracked_mutation():
    with torch.inference_mode():
        model, _ = make_model()
        before = model.get_global_gaussians(timestamp=100)["opacities"].clone()
        model._opacities.add_(2)
        after = model.get_global_gaussians(timestamp=100)["opacities"]
        torch.testing.assert_close(after, torch.sigmoid(torch.logit(before) + 2))


def test_native_type_device_constructor_and_no_refactor_dependency():
    state,config,times=raw_fixture()
    asset=build_traffic_light_asset(state,config,times)
    model=TrafficLightNodes(convert_to_attribute_dict(asset),model_name="traffic_lights",device="cpu")
    assert model.device == torch.device("cpu")
    assert model.model_name == "traffic_lights"
    assert model.model_type == "traffic_light"
    assert not torch.cuda.is_initialized()


def test_legacy_portable_type_accepted_without_mutation():
    state,config,times=raw_fixture()
    asset=build_traffic_light_asset(state,config,times)
    asset["config"]["type"]="TrafficLightNodes"
    model=TrafficLightNodes(asset)
    assert asset["config"]["type"]=="TrafficLightNodes"
    assert model.get_global_gaussians(timestamp=times[0])["means"].shape==(3,3)
