"""Camera-independent scenario vehicle geometry must not change camera colour or clocks."""
import numpy as np
import pytest
import torch
from test_omnire_engine import _reactive_on_road, Model, module
from types import SimpleNamespace


def scene(monkeypatch, static=False, held=False):
    class Rigid(Model):
        def get_means(self, global_quat, global_trans):
            self.geometry_calls += 1
            self.global_means = global_trans[None].clone()
            return self.global_means
        def get_scales(self): return torch.ones(1, 3)
        def get_quats(self, global_quat, global_trans): return global_quat[None].clone()
        def get_opacity(self): return torch.ones(1)
        def get_gaussian_rgbs(self, camera_to_worlds, **kwargs):
            self.color_calls += 1
            return self.global_means + camera_to_worlds[:, :3, 3]
    e = _reactive_on_road(monkeypatch, static)
    e._held_at_log_pose = frozenset({'car'} if held else ())
    e.submodel_names = {'car': 'r'}
    model = Rigid()
    e.gaussian_models = {'r': model}
    e._actor_visibility = {'car': np.ones(3, dtype=bool)}
    e._actor_road = e._smooth_actor_road()
    e._render_agent_states = {'car': np.array([20., 0., 0., 0., 0., .2])}
    e._sh_colors_cached = lambda *args: None
    cameras = [torch.eye(4) for _ in range(3)]
    for i, camera in enumerate(cameras): camera[0, 3] = float(i)
    return e, model, cameras


@pytest.mark.parametrize('static,held', [(False, False), (True, True), (True, False)])
def test_scenario_road_pose_reused_but_scripted_geometry_still_runs_per_camera(monkeypatch, static, held):
    e, model, cameras = scene(monkeypatch, static, held)
    separate = [e._collect_camera(1, i, cam, {})[0] for i, cam in enumerate(cameras)]
    model.geometry_calls = model.color_calls = 0
    e._vehicle_pose_refs = {}
    road_calls = []
    ground = e._path_lift.ground
    def counted_ground(*args, **kwargs):
        road_calls.append(1)
        return ground(*args, **kwargs)
    e._path_lift.ground = counted_ground
    common = {}
    shared = []
    for i, cam in enumerate(cameras):
        if i:
            model.global_means = torch.full((1, 3), -999.)
        shared.append(e._collect_camera(1, i, cam, common)[0])
    for a, b in zip(separate, shared):
        for key in a: assert torch.equal(a[key], b[key]), key
    assert model.geometry_calls == 3
    assert len(road_calls) == (0 if static and held else 1)
    assert model.color_calls == 3
    assert not torch.equal(shared[0]['rgbs'], shared[1]['rgbs'])
    e._render_agent_states['car'][0] = 30.
    e._vehicle_pose_refs = {}
    next_frame, _ = e._collect_camera(2, 0, cameras[0], {})
    assert model.geometry_calls == 4
    assert len(road_calls) == (0 if static and held else 2)
    assert float(next_frame['means'][0, 0]) == 30.
    e._suppressed_actors = frozenset({'car'})
    with pytest.raises(ValueError, match='no Gaussian geometry'):
        e._collect_camera(2, 1, cameras[1], common)


def test_logged_rigid_camera_exposure_geometry_is_not_shared(monkeypatch):
    e, model, cameras = scene(monkeypatch)
    e.actor_pose_source = 'checkpoint'
    e._actor_road = None
    e._packed_rigid[0][:, 0, 0] = torch.tensor([0., 5., 10.])
    times = e._calibration['training_timestamps_us']
    images = np.repeat(times[:, None], 8, axis=1)
    images[1, 0] += 10000
    images[1, 1] -= 20000
    e._calibration['image_timestamps_us'] = images
    common = {}
    a, _ = e._collect_camera(1, 0, cameras[0], common)
    b, _ = e._collect_camera(1, 1, cameras[1], common)
    assert model.geometry_calls == 2
    assert not torch.equal(a['means'], b['means'])


@pytest.mark.parametrize('cls', [module.mtgs.RigidModel, module.mtgs.MirroredModel])
def test_concrete_rigid_attributes_are_exact_cached_and_reset(monkeypatch, cls):
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(scale_dim=3)
    model.gauss_params = torch.nn.ParameterDict({
        'scales': torch.nn.Parameter(torch.tensor([[.2, -.4, .8], [1., -1., 0.]])),
        'opacities': torch.nn.Parameter(torch.tensor([[.5], [-.2]])),
    })
    expected = (model.get_scales().detach(), model.get_opacity().detach())
    calls = []
    original = model.get_scales
    monkeypatch.setattr(model, 'get_scales', lambda: (calls.append(1), original())[1])
    e = module.OmniReRenderEngine.__new__(module.OmniReRenderEngine)
    e._clear_asset_caches()
    first = e._rigid_attributes_cached('car', model)
    second = e._rigid_attributes_cached('car', model)
    for want, a, b in zip(expected, first, second):
        assert torch.equal(want, a)
        assert a is b
        assert not a.requires_grad
    assert len(calls) == 1
    with torch.no_grad(): model.gauss_params['scales'].add_(.1)
    e._clear_asset_caches()
    third = e._rigid_attributes_cached('car', model)
    assert torch.equal(third[0], original())
    assert len(calls) == 2


def test_unknown_rigid_subclasses_do_not_get_persistent_attribute_cache():
    class Animated:
        count = 0
        def get_scales(self):
            self.count += 1
            return torch.full((1, 3), self.count)
        def get_opacity(self): return torch.ones(1)
    model = Animated()
    e = module.OmniReRenderEngine.__new__(module.OmniReRenderEngine)
    a = e._rigid_attributes_cached('car', model)
    b = e._rigid_attributes_cached('car', model)
    assert not torch.equal(a[0], b[0])
