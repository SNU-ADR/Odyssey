"""Catch stale slices, reordered actors, and unsafe reuse of inference tensors."""
import pytest
import torch
from odyssey_renderer.omnire import engine as omnire_engine


def assembler():
    cls = getattr(omnire_engine, "GaussianAssembly", None)
    assert cls is not None, "renderer has no reusable Gaussian assembly"
    return cls()


def test_static_slices_are_not_written_again_and_dynamic_slices_change():
    a = assembler()
    fixed = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    moving = torch.tensor([[7., 8., 9.]])
    first = a.assemble("means", [fixed, moving], [True, True])
    version = first._version
    same = a.assemble("means", [fixed, moving], [True, True])
    assert same is first and same._version == version
    moving.add_(10)
    updated = a.assemble("means", [fixed, moving], [True, True])
    assert updated is first
    assert torch.equal(updated, torch.tensor([[1.,2.,3.],[4.,5.,6.],[17.,18.,19.]]))


def test_alias_mutation_and_replacement_at_same_shape_are_copied():
    a = assembler()
    fixed = torch.tensor([1., 2.])
    mutable = torch.tensor([3., 4.])
    a.assemble("opacity", [fixed, mutable], [True, True])
    mutable.view(-1)[0] = 9
    got = a.assemble("opacity", [fixed, mutable], [True, True])
    assert torch.equal(got, torch.tensor([1., 2., 9., 4.]))
    got = a.assemble("opacity", [fixed, torch.tensor([5., 6.])], [True, True])
    assert torch.equal(got, torch.tensor([1., 2., 5., 6.]))


def test_actor_order_population_shape_dtype_and_empty_segments():
    a = assembler()
    x, y = torch.tensor([1., 2.]), torch.tensor([3., 4.])
    a.assemble("x", [x,y])
    assert torch.equal(a.assemble("x", [y,x]), torch.tensor([3.,4.,1.,2.]))
    assert torch.equal(a.assemble("x", [x], [True]), torch.tensor([1.,2.]))
    assert torch.equal(a.assemble("x", [torch.empty(0),y,x]), torch.tensor([3.,4.,1.,2.]))
    z = a.assemble("x", [x.double(),y.double()])
    assert z.dtype == torch.float64 and torch.equal(z, torch.tensor([1.,2.,3.,4.], dtype=torch.float64))
    assert a.assemble("x", [torch.empty(0)]).numel() == 0


def test_output_mutation_does_not_poison_cached_input():
    a = assembler()
    x = torch.tensor([1., 2.])
    a.assemble("x", [x], [True]).fill_(99)
    assert torch.equal(a.assemble("x", [x], [True]), x)


def test_inference_tensors_without_version_are_always_refreshed():
    a = assembler()
    with torch.inference_mode():
        x = torch.tensor([1.,2.])
        a.assemble("x", [x], [True])
        x.add_(3)
        got = a.assemble("x", [x], [True])
        assert torch.equal(got, torch.tensor([4.,5.]))


def test_noncontiguous_sources_and_separate_attributes():
    a = assembler()
    x = torch.arange(12.).reshape(3,4).T
    y = torch.full((1,3), 99.)
    got = a.assemble("means", [x,y])
    assert torch.equal(got, torch.cat([x,y]))
    a.assemble("scales", [torch.zeros_like(x), y])
    assert torch.equal(got, torch.cat([x,y]))


def test_untrusted_same_tensor_is_refreshed_and_mixed_dtype_matches_cat():
    a = assembler()
    x = torch.tensor([1.,2.])
    a.assemble("x", [x], [False])
    x.data.add_(5)  # Native extensions may write without incrementing a counter.
    assert torch.equal(a.assemble("x", [x], [False]), torch.tensor([6.,7.]))
    y = torch.tensor([8.], dtype=torch.float64)
    got = a.assemble("x", [x,y])
    assert got.dtype == torch.float64 and torch.equal(got, torch.tensor([6.,7.,8.], dtype=torch.float64))


def test_collect_assembled_cameras_match_independent_arrays_and_reset(monkeypatch):
    from tests.test_camera_reuse import scene
    e, model, cameras = scene(monkeypatch)
    common = {}
    for i, camera in enumerate(cameras):
        expected, actors = e._collect_camera(1, i, camera, {})
        actual, assembled_actors = e._collect_camera(1, i, camera, common, assembler=e._gaussian_assembly)
        assert actors == assembled_actors
        for key in expected:
            assert torch.equal(actual[key], expected[key]), key
    old = e._gaussian_assembly
    e._clear_asset_caches()
    assert e._gaussian_assembly is not old
