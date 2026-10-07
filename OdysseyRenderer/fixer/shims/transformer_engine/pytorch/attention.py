"""Ported from TransformerEngine v2.5 (Apache-2.0), pure-PyTorch."""
import torch
from torch import nn


def _rotate_half(x: torch.Tensor, interleaved: bool) -> torch.Tensor:
    if not interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x_new = torch.stack((-x2, x1), dim=-1)
    return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def _get_freqs_on_this_cp_rank(freqs, seqlen, cp_size, cp_rank):
    if cp_size > 1:
        cp_seg = seqlen // 2
        full_seqlen = cp_size * seqlen
        return torch.cat([
            freqs[cp_rank * cp_seg:(cp_rank + 1) * cp_seg],
            freqs[full_seqlen - (cp_rank + 1) * cp_seg:full_seqlen - cp_rank * cp_seg],
        ])
    return freqs


def _apply_rotary_pos_emb_base(t, freqs, start_positions=None,
                               tensor_format="sbhd", interleaved=False):
    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[1] if tensor_format == "bshd" else t.shape[0]

    if start_positions is not None:
        max_offset = torch.max(start_positions)
        assert max_offset + cur_seq_len <= max_seq_len
        freqs = torch.concatenate([freqs[i:i + cur_seq_len] for i in start_positions], dim=1)

    assert cur_seq_len <= max_seq_len, (
        f"Rotary Embeddings only supported up to {max_seq_len} sequence length!")
    freqs = freqs[:cur_seq_len]

    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * cos_) + (_rotate_half(t, interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)


def apply_rotary_pos_emb(t, freqs, tensor_format="sbhd", start_positions=None,
                         interleaved=False, fused=False, cu_seqlens=None,
                         cp_size=1, cp_rank=0):
    # `fused` only selects the CUDA kernel upstream; the math is identical.
    assert not (cp_size > 1 and start_positions is not None)
    assert tensor_format != "thd" or cu_seqlens is not None

    if tensor_format == "thd":
        cu_seqlens = cu_seqlens // cp_size
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        return torch.cat([
            _apply_rotary_pos_emb_base(
                x.unsqueeze(1),
                _get_freqs_on_this_cp_rank(freqs, x.size(0), cp_size, cp_rank),
                start_positions=(start_positions[idx:idx + 1]
                                 if start_positions is not None else None),
                interleaved=interleaved,
            )
            for idx, x in enumerate(torch.split(t, seqlens))
        ]).squeeze(1)

    if tensor_format == "sbhd":
        seqlen = t.size(0)
    elif tensor_format == "bshd":
        seqlen = t.size(1)
    else:
        raise ValueError(f"Unsupported tensor_format: {tensor_format}.")
    return _apply_rotary_pos_emb_base(
        t, _get_freqs_on_this_cp_rank(freqs, seqlen, cp_size, cp_rank),
        start_positions, tensor_format, interleaved=interleaved)


class DotProductAttention(nn.Module):
    """SDPA stand-in for TE's DotProductAttention, no_mask / non-causal only."""

    def __init__(self, num_attention_heads, kv_channels, num_gqa_groups=None,
                 attention_dropout=0.0, qkv_format="sbhd", attn_mask_type="no_mask",
                 softmax_scale=None, **kwargs):
        super().__init__()
        if attn_mask_type not in ("no_mask", "padding", None):
            raise NotImplementedError(f"attn_mask_type={attn_mask_type} not supported by shim")
        self.qkv_format = qkv_format
        self.num_attention_heads = num_attention_heads
        self.kv_channels = kv_channels
        self.attention_dropout = attention_dropout
        self.softmax_scale = softmax_scale

    def forward(self, q, k, v, **kwargs):
        fmt = self.qkv_format
        if fmt == "bshd":       # (b, s, h, d)
            qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))
        elif fmt == "sbhd":     # (s, b, h, d)
            qt, kt, vt = (x.permute(1, 2, 0, 3) for x in (q, k, v))
        else:
            raise NotImplementedError(f"qkv_format={fmt} not supported by shim")

        o = torch.nn.functional.scaled_dot_product_attention(
            qt, kt, vt,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.softmax_scale,
        )                                   # (b, h, s, d)
        b, h, s, d = o.shape
        if fmt == "bshd":
            return o.transpose(1, 2).reshape(b, s, h * d)
        return o.permute(2, 0, 1, 3).reshape(s, b, h * d)
