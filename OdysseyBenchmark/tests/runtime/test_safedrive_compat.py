import importlib
import importlib.util
import pytest
import torch

def load_compat():
    name = "odyssey_runtime.safedrive_compat"
    assert importlib.util.find_spec(name) is not None, "native SafeDrive compatibility helper missing"
    return importlib.import_module(name)

def test_padding_cannot_overwrite_real_query_zero():
    compat = load_compat()
    dest = torch.zeros(1, 3, 2)
    index = torch.tensor([[[0, 1], [0, 0]]])
    valid = torch.tensor([[[True, True], [False, False]]])
    src = torch.tensor([[[7., 11.], [0., 0.]]])
    result = compat.scatter_valid(dest, 1, index, src, valid)
    assert torch.equal(result, torch.tensor([[[7., 0.], [0., 11.], [0., 0.]]]))
    assert result.is_contiguous()
    assert result.shape == dest.shape

def test_boolean_mask_keeps_valid_zero_slot():
    compat = load_compat()
    index = torch.tensor([[[0], [0]]])
    valid = torch.tensor([[[True], [False]]])
    result = compat.scatter_valid(torch.zeros(1, 2, 1, dtype=torch.bool), 1, index, valid, valid)
    assert result[:, 0].all()
    assert not result[:, 1].any()

def test_pairwise_scatter_broadcast_mask_and_invalid_nan():
    compat = load_compat()
    dest = torch.zeros(2, 1, 4, 3, 2)
    index = torch.tensor([[[0, 1, 2], [0, 0, 0]]])[None, ..., None].expand(2, 1, 2, 3, 2)
    valid = torch.tensor([[[True, True, True], [False, False, False]]])[None, ..., None]
    src = torch.ones(2, 1, 2, 3, 2)
    src[:, :, 1] = float("nan")
    result = compat.scatter_valid(dest, 2, index, src, valid)
    assert torch.isfinite(result).all()
    for proposal in range(3):
        assert torch.equal(result[:, :, proposal, proposal], torch.ones(2, 1, 2))
    assert result.sum().item() == 12

def test_empty_selection():
    compat = load_compat()
    dest = torch.zeros(1, 3, 2)
    index = torch.empty(1, 0, 2, dtype=torch.long)
    assert torch.equal(compat.scatter_valid(dest, 1, index, torch.empty(1, 0, 2),
                       torch.empty(1, 0, 2, dtype=torch.bool)), dest)

@pytest.mark.parametrize("matmul,cudnn", [(False, True), (True, False), (True, True), (False, False)])
@pytest.mark.parametrize("raises", [False, True])
def test_precision_settings_restore_independently_even_on_exception(matmul, cudnn, raises):
    compat = load_compat()
    original = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
        try:
            with compat.tf32_disabled():
                assert not torch.backends.cuda.matmul.allow_tf32
                assert not torch.backends.cudnn.allow_tf32
                if raises:
                    raise RuntimeError("projection failed")
        except RuntimeError:
            assert raises
        assert torch.backends.cuda.matmul.allow_tf32 == matmul
        assert torch.backends.cudnn.allow_tf32 == cudnn
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original
