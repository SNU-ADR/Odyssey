"""Pure-PyTorch stand-in for NVIDIA transformer_engine.

Why this exists: cosmos_predict2 imports transformer_engine at module level, but
the real package needs `transformer_engine_torch`, which ships source-only on
PyPI and does not build here (system nvcc 13.1 vs torch cu12.8), while the
conda-forge prebuilds are pinned to torch versions that either predate B200
(sm_100) support or postdate what cosmos_predict2 accepts.

Only three symbols are actually used by cosmos_predict2:
    te.pytorch.RMSNorm
    transformer_engine.pytorch.attention.DotProductAttention
    transformer_engine.pytorch.attention.apply_rotary_pos_emb

apply_rotary_pos_emb here is a line-by-line port of TransformerEngine v2.5
(transformer_engine/pytorch/attention/rope.py, Apache-2.0) so the numerics match
the fused CUDA kernel; `fused=True` simply falls through to the same math.
"""
from . import pytorch  # noqa: F401

__version__ = "2.5.0+shim"
