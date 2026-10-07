# PointsEncoder below is ported from PLUTO (https://github.com/jchengai/pluto, commit b9964b6,
# src/models/pluto/layers/embedding.py). Changes: LayerNorm instead of BatchNorm, and zero buffers
# that follow the input dtype.
"""SD route -> one conditioning token for the ReCogDrive diffusion planner.

PLUTO's reference-line embedding, built from PLUTO's PointsEncoder and QCNet's FourierEmbedding
(see the notices in this file). The route is cast to the encoder's parameter dtype, so the module
also runs under DeepSpeed bf16, where the weights themselves are bf16.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn


class PointsEncoder(nn.Module):
    """PLUTO PointsEncoder: per-point MLP -> masked max-pool -> concat -> MLP -> max-pool.

    Max-pool runs over the point axis (dim=1). Invalid points are written as zeros before each pool,
    so they take part in the max as 0 (as in PLUTO).
    """

    def __init__(self, feat_channel: int, encoder_channel: int) -> None:
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_mlp = nn.Sequential(
            nn.Linear(feat_channel, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
        )
        self.second_mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.encoder_channel),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, M, feat); mask: (B, M) bool -> (B, encoder_channel)."""
        bs, n, _ = x.shape
        device = x.device

        # Only valid points go through the MLP; invalid ones stay exactly zero.
        x_valid = self.first_mlp(x[mask])
        x_features = torch.zeros(bs, n, 256, device=device, dtype=x_valid.dtype)
        x_features[mask] = x_valid

        pooled_feature = x_features.max(dim=1)[0]
        x_features = torch.cat(
            [x_features, pooled_feature.unsqueeze(1).repeat(1, n, 1)], dim=-1
        )

        x_features_valid = self.second_mlp(x_features[mask])
        res = torch.zeros(bs, n, self.encoder_channel, device=device, dtype=x_features_valid.dtype)
        res[mask] = x_features_valid

        res = res.max(dim=1)[0]
        return res


# FourierEmbedding is adapted from QCNet (https://github.com/ZikangZhou/QCNet,
# layers/fourier_embedding.py), which PLUTO also uses:
#   Copyright (c) 2023, Zikang Zhou. All rights reserved.
#   Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
class FourierEmbedding(nn.Module):
    """PLUTO FourierEmbedding over continuous inputs (per-dim frequency bands + MLP).

    Present because the route carries raw metres: a line starting 90 m ahead would saturate a plain
    Linear, and the bands cover the range with no normalization constant to tune.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_freq_bands: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.freqs = nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None
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

    def forward(self, continuous_inputs: torch.Tensor) -> torch.Tensor:
        x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
        x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
        continuous_embs: List[Optional[torch.Tensor]] = [None] * self.input_dim
        for i in range(self.input_dim):
            continuous_embs[i] = self.mlps[i](x[..., i, :])
        x = torch.stack(continuous_embs).sum(dim=0)
        return self.to_out(x)


class RouteCenterlineEncoder(nn.Module):
    """Routed centerline -> one (B, dim) token, like PLUTO's reference-line embedding.

    Point positions are taken relative to the line's first point; the absolute start [x, y, heading]
    is restored by the Fourier embedding of the line start.
    """

    def __init__(self, dim: int, num_freq_bands: int = 64) -> None:
        super().__init__()
        self.r_encoder = PointsEncoder(6, dim)
        self.r_pos_emb = FourierEmbedding(3, dim, num_freq_bands)

    def forward(self, route_centerline: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """route_centerline: (B, P, 5) = [x, y, dx, dy, heading]; mask: (B, P) bool -> (B, dim)."""
        B, P, _ = route_centerline.shape
        dim = self.r_encoder.encoder_channel
        # Whole-batch early out: with nothing valid anywhere, x[mask] is empty.
        if mask is None or not mask.any():
            return route_centerline.new_zeros(B, dim)

        pos = route_centerline[..., 0:2]
        vec = route_centerline[..., 2:4]
        ori = route_centerline[..., 4]
        r_feature = torch.cat(
            [pos - pos[:, 0:1, :2], vec, torch.stack([ori.cos(), ori.sin()], dim=-1)], dim=-1
        )  # (B, P, 6)

        r_emb = self.r_encoder(r_feature, mask)                    # (B, dim)
        r_pos = torch.cat([pos[:, 0], ori[:, 0, None]], dim=-1)    # (B, 3) line start [x, y, heading]
        r_emb = r_emb + self.r_pos_emb(r_pos)
        # Zero the token for samples with no valid route (per sample).
        return r_emb * mask.any(dim=-1, keepdim=True).to(r_emb.dtype)


class SDRouteStatusToken(nn.Module):
    """Route -> one token, to be added to the planner's status conditioning.

    `route` / `route_mask` may be None; the call then returns an exact zero of `batch_size` rows.
    """

    def __init__(self, d_model: int, num_freq_bands: int = 64, init_seed: int = 0) -> None:
        super().__init__()
        # Forked RNG: turning the route on must not consume draws from the global stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            self.encoder = RouteCenterlineEncoder(d_model, num_freq_bands=num_freq_bands)
        self.d_model = d_model

    def forward(self, route: Optional[torch.Tensor],
                route_mask: Optional[torch.Tensor],
                batch_size: Optional[int] = None,
                device=None, dtype=None) -> torch.Tensor:
        """route: (B,P,5) or None; route_mask: (B,P) or None -> (B, d_model), zero when absent."""
        if route is None or route_mask is None:
            if batch_size is None:
                raise ValueError(
                    "SDRouteStatusToken called with route=None and no batch_size; the caller must "
                    "say how wide the zero should be."
                )
            return torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)
        if route_mask.dtype != torch.bool:
            route_mask = route_mask.bool()
        param_dtype = next(self.encoder.parameters()).dtype
        return self.encoder(route.to(param_dtype), route_mask).to(
            dtype if dtype is not None else route.dtype
        )
