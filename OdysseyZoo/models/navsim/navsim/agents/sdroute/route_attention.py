"""SD route -> 24 route tokens -> gated cross-attention onto planner queries.

The route tensor is the (P,5) ego-local polyline built by the shared SD-route builder
(``[x, y, dx, dy, heading]`` + a (P,) bool mask; see OdysseyZoo/sdroute/route_centerline.py):
P=120 points at ~1 m spacing, 120 m ahead, in raw metres without normalization.

The polyline is kept as a sequence -- 120 points grouped 5 at a time into 24 segment
tokens (~5 m each) -- so the decoder can attend to "the part of the route 40 m ahead"
instead of one pooled summary of the whole route.

Two design choices matter:

* The route attention is a separate layer, not extra tokens spliced into the host
  model's keyval. Spliced tokens would compete with the BEV tokens inside one softmax
  and change the host's own attention; a separate layer leaves it untouched.
* The residual gates are zero-initialised, so at step 0 this module contributes
  nothing and the host model is numerically identical to the same model without the
  route branch.

Only torch is imported, so the same file is copied verbatim into each repo that uses it.
"""
from __future__ import annotations

import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

#: Route polyline length, fixed by the (P,5) target the SD-route builder emits.
ROUTE_NUM_POINTS: int = 120
#: Points folded into one route token (~5 m of route at the 1 m point spacing).
ROUTE_SEG_LEN: int = 5
#: 120 / 5 -- divides exactly, so no ragged tail segment.
ROUTE_NUM_SEGMENTS: int = ROUTE_NUM_POINTS // ROUTE_SEG_LEN


# FourierEmbedding is adapted from QCNet (https://github.com/ZikangZhou/QCNet,
# layers/fourier_embedding.py), which PLUTO also uses:
#   Copyright (c) 2023, Zikang Zhou. All rights reserved.
#   Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
# Changes: the forward is condensed, and input_dim=0 is not supported.
class FourierEmbedding(nn.Module):
    """Per-dim learnable frequency bands -> MLP -> sum (QCNet's continuous embedding, as in PLUTO).

    Used for the segment start because the route is in raw metres (up to ~120 m):
    the Fourier bands cover that range without a normalization constant to tune per
    host model.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_freq_bands: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.freqs = nn.Embedding(input_dim, num_freq_bands)
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(input_dim)
            ]
        )
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., input_dim) -> (..., hidden_dim)."""
        z = x.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
        z = torch.cat([z.cos(), z.sin(), x.unsqueeze(-1)], dim=-1)
        out = torch.stack([self.mlps[i](z[..., i, :]) for i in range(self.input_dim)])
        return self.to_out(out.sum(dim=0))


class SDRouteSegmentEncoder(nn.Module):
    """(B,120,5) route + (B,120) mask -> (B,24,D) tokens + (B,24) valid mask.

    Per segment: 6-channel point features -> Linear -> masked max-pool. A masked
    max-pool rather than PLUTO's two-stage PointsEncoder, because a segment holds
    only 5 points.

    The 6 channels are [x-x0, y-y0, dx, dy, cos(theta), sin(theta)], with (x0, y0)
    the line's first point, so all 24 tokens share one coordinate frame. The
    per-segment Fourier term below also encodes where each segment starts; it partly
    overlaps the coordinates and is kept because the released checkpoints use it.
    """

    def __init__(self, d_model: int, num_freq_bands: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.point_mlp = nn.Linear(6, d_model)
        # Fourier over the segment start [x, y, heading] -- the absolute anchor that
        # the translation-invariant point features deliberately dropped.
        self.seg_pos_emb = FourierEmbedding(3, d_model, num_freq_bands)
        # Learned ordering ("how far along the route this segment sits"). The host's
        # attention has no other way to tell segment 0 from segment 23.
        self.seg_index_emb = nn.Embedding(ROUTE_NUM_SEGMENTS, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, route: torch.Tensor, route_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """route: (B,120,5) float, route_mask: (B,120) bool -> (B,24,D), (B,24) bool."""
        B, P, C = route.shape
        assert P == ROUTE_NUM_POINTS and C == 5, f"expected (B,{ROUTE_NUM_POINTS},5), got {tuple(route.shape)}"
        S, L = ROUTE_NUM_SEGMENTS, ROUTE_SEG_LEN

        seg = route.view(B, S, L, 5)
        seg_mask = route_mask.view(B, S, L)
        # A segment is usable iff it holds >=1 valid point. The last point of the
        # whole polyline is always invalid (the builder gives it no forward vector), so
        # the final segment carries at most 4 valid points, not 5.
        seg_valid = seg_mask.any(dim=-1)

        pos, vec, ori = seg[..., 0:2], seg[..., 2:4], seg[..., 4]
        # One origin for the whole polyline (its first point), not one per segment:
        # per-segment frames would disagree with the absolute cos/sin heading channels.
        # Coordinates then run up to ~120 m.
        origin = pos[:, 0:1, 0:1, :]  # (B,1,1,2) line origin, broadcast over all segments
        feat = torch.cat([pos - origin, vec, ori.cos()[..., None], ori.sin()[..., None]], dim=-1)  # (B,S,L,6)

        # Zero the invalid points before the MLP: the builder zero-pads them, but
        # subtracting the origin made them non-zero again. The -inf fill below is what
        # keeps them out of the max; this keeps their features clean regardless.
        feat = feat * seg_mask[..., None].to(feat.dtype)
        pt = self.point_mlp(feat)  # (B,S,L,D)

        # Masked max-pool. -inf on invalid points so they can never win the max; an
        # all-invalid segment then yields all -inf, which is scrubbed to 0 right after.
        pt = pt.masked_fill(~seg_mask[..., None], float("-inf"))
        tok = pt.max(dim=2).values  # (B,S,D)
        tok = torch.where(seg_valid[..., None], tok, torch.zeros_like(tok))

        seg_start = torch.cat([pos[:, :, 0, :], ori[:, :, 0, None]], dim=-1)  # (B,S,3)
        tok = tok + self.seg_pos_emb(seg_start) * seg_valid[..., None].to(tok.dtype)
        tok = tok + self.seg_index_emb.weight[None]
        return self.norm(tok), seg_valid


class SDRouteCrossAttention(nn.Module):
    """Gated cross-attention: queries attend to the 24 route tokens. Zero-init output.

    Pre-norm residual with a zero-initialised gate, so the module is an exact no-op
    until training moves the gate off zero. Pre-norm is required for that: with
    post-norm the final LayerNorm would rescale the residual even when the gate
    contributes zero.

    The always-visible learned "no route" token keeps attention defined:
    nn.MultiheadAttention returns NaN when a row's key_padding_mask is all-True, which
    happens whenever the SD-route builder emits an all-False mask (no match onto the SD
    graph, or the ego within 1 m of the route end).
    """

    def __init__(self, d_model: int, num_heads: int = 8, d_ffn: Optional[int] = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.no_route_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.no_route_token, std=0.02)

        d_ffn = d_ffn or 4 * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(d_ffn, d_model),
        )
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # The gate: both residual branches start at exactly zero.
        self.gate_attn = nn.Linear(d_model, d_model)
        self.gate_ffn = nn.Linear(d_model, d_model)
        for g in (self.gate_attn, self.gate_ffn):
            nn.init.zeros_(g.weight)
            nn.init.zeros_(g.bias)

    def forward(self, query: torch.Tensor, route_tokens: torch.Tensor,
                seg_valid: torch.Tensor) -> torch.Tensor:
        """query: (B,Q,D); route_tokens: (B,S,D); seg_valid: (B,S) bool -> (B,Q,D)."""
        B = query.shape[0]
        memory = torch.cat([self.no_route_token.expand(B, -1, -1), route_tokens], dim=1)  # (B,1+S,D)
        # MultiheadAttention wants True = ignore, hence the inversion; column 0 (the
        # always-visible token) is False for every row.
        pad = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=query.device), ~seg_valid], dim=1
        )
        attn_out, _ = self.attn(self.norm_attn(query), memory, memory, key_padding_mask=pad, need_weights=False)
        query = query + self.dropout(self.gate_attn(attn_out))
        query = query + self.dropout(self.gate_ffn(self.ffn(self.norm_ffn(query))))
        return query


class SDRouteDecoderLayer(nn.Module):
    """A stock nn.TransformerDecoderLayer with one extra route cross-attention inside it.

    Order per layer: self-attn -> memory cross-attn -> route cross-attn -> FFN.

    The route is read inside every layer, the way the scene memory is, rather than once by
    a layer appended after the decoder. nn.TransformerDecoderLayer cannot be extended, so
    its body is reproduced here; `assert_matches_stock` checks that this layer without a
    route gives bitwise-equal output to nn.TransformerDecoderLayer with the same weights.
    Run it whenever this class changes.

    `plan_slice` restricts the route attention to the planner's own queries (TransFuser,
    for example, packs 1 trajectory query with 30 detection queries in one tensor); the
    other queries skip the route branch. They are untouched within a layer, but from the
    second layer on they can read the route-conditioned plan query through self-attention.

    Only norm_first=False (post-norm) is implemented: the default, and what every host
    model here uses.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation=None, layer_norm_eps: float = 1e-5,
                 batch_first: bool = True, route_num_heads: Optional[int] = None,
                 plan_slice: Optional[slice] = None, route_init_seed: int = 0) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation or nn.functional.relu

        self.plan_slice = plan_slice
        # The route branch is built under a forked RNG so that turning the route on does
        # not consume draws from the global stream. Everything above this line must draw
        # in exactly the same order as a stock nn.TransformerDecoderLayer, or the host's
        # own weights change when the route is enabled.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(route_init_seed)
            self.route_attn = SDRouteCrossAttention(
                d_model, num_heads=route_num_heads or nhead, d_ffn=dim_feedforward, dropout=dropout
            )

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                route_tokens: Optional[torch.Tensor] = None,
                seg_valid: Optional[torch.Tensor] = None,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None) -> torch.Tensor:
        """Stock post-norm decoder layer, with the route branch spliced before the FFN."""
        x = tgt
        x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
        x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))

        if route_tokens is not None and seg_valid is not None:
            # The route branch carries its own gate/norms and is a no-op until trained,
            # so the two lines above stay bitwise-identical to the stock layer at init.
            if self.plan_slice is None:
                x = self.route_attn(x, route_tokens, seg_valid)
            else:
                conditioned = self.route_attn(x[:, self.plan_slice], route_tokens, seg_valid)
                x = x.clone()
                x[:, self.plan_slice] = conditioned

        x = self.norm3(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn(x, mem, mem, attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout2(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class SDRouteDecoder(nn.Module):
    """Drop-in replacement for nn.TransformerDecoder that threads the route through layers.

    Owns the route encoder, so the host passes the raw (B,120,5) route + mask and never
    sees the tokenisation. With route=None every layer's route branch is skipped and this
    is a stock decoder.
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, plan_slice: Optional[slice] = None,
                 route_num_heads: Optional[int] = None, num_freq_bands: int = 64,
                 route_init_seed: int = 0) -> None:
        super().__init__()
        # Encoder is route-only, so it forks too (see SDRouteDecoderLayer).
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(route_init_seed)
            self.encoder = SDRouteSegmentEncoder(d_model, num_freq_bands=num_freq_bands)
        # Build one prototype and deep-copy it, as nn.TransformerDecoder does: building N
        # layers would draw from the RNG N times instead of once and shift the init of
        # every host weight created after the decoder.
        proto = SDRouteDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                    plan_slice=plan_slice, route_num_heads=route_num_heads,
                                    route_init_seed=route_init_seed)
        self.layers = nn.ModuleList([copy.deepcopy(proto) for _ in range(num_layers)])

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                route: Optional[torch.Tensor] = None,
                route_mask: Optional[torch.Tensor] = None, **kw) -> torch.Tensor:
        route_tokens = seg_valid = None
        if route is not None and route_mask is not None:
            if route_mask.dtype != torch.bool:
                route_mask = route_mask.bool()
            route_tokens, seg_valid = self.encoder(route.to(tgt.dtype), route_mask)
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, route_tokens=route_tokens, seg_valid=seg_valid, **kw)
        return x


def assert_matches_stock(d_model: int = 64, nhead: int = 4, dim_feedforward: int = 128,
                         batch: int = 3, n_query: int = 7, n_mem: int = 11) -> None:
    """Check that SDRouteDecoderLayer == nn.TransformerDecoderLayer when the route is absent.

    Guards the hand-copied layer body against silent divergence from torch's own.
    Raises AssertionError on any mismatch; returns None on success.
    """
    torch.manual_seed(0)
    stock = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout=0.0,
                                       batch_first=True).eval()
    ours = SDRouteDecoderLayer(d_model, nhead, dim_feedforward, dropout=0.0).eval()
    missing, unexpected = ours.load_state_dict(stock.state_dict(), strict=False)
    assert not missing or all(k.startswith("route_attn.") for k in missing), missing
    assert not unexpected, unexpected

    tgt, mem = torch.randn(batch, n_query, d_model), torch.randn(batch, n_mem, d_model)
    with torch.no_grad():
        a, b = stock(tgt, mem), ours(tgt, mem)
    assert torch.equal(a, b), f"diverged from stock layer: max |diff| = {(a - b).abs().max()}"
