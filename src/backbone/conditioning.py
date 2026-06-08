from __future__ import annotations

import math

import torch
from torch import nn

from backbone.blocks import RMSNorm


class TimeEmbedding(nn.Module):
    """Sinusoidal scalar-time embedding followed by a small MLP."""

    def __init__(self, dim: int, features: int | None = None, time_scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.time_scale = float(time_scale)
        self.features = features or min(256, max(16, dim))
        if self.features % 2:
            self.features += 1
        hidden = max(dim, self.features)
        self.mlp = nn.Sequential(nn.Linear(self.features, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2 and t.shape[1] == self.dim:
            return t
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        elif t.ndim != 1:
            raise ValueError(f"Expected scalar time [B] or embedded time [B, {self.dim}], got {tuple(t.shape)}")
        if not torch.is_floating_point(t):
            t = t.to(dtype=torch.get_default_dtype())
        t = t * self.time_scale
        half = self.features // 2
        freqs = torch.exp(torch.linspace(0, math.log(10000.0), half, device=t.device, dtype=t.dtype))
        args = t[:, None] / freqs[None]
        return self.mlp(torch.cat([torch.sin(args), torch.cos(args)], dim=1))


class ConditioningEmbedding(nn.Module):
    """External conditioning embedding projected to `cond_dim` by a small MLP."""

    def __init__(self, embed_dim: int, cond_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(embed_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.mlp(cond)


class ConditioningCombiner(nn.Module):
    """RMS-normalise the embedded timestep and the (projected) global conditioning
    independently, then sum into one [B, cond_dim] tensor. Per-path normalisation
    bounds the time-embedding magnitude (it otherwise runs away to ~200 and drowns
    the conditioning) and balances the two before the add. At least one of the two
    must be present."""

    def __init__(self, cond_dim: int):
        super().__init__()
        self.cond_dim = cond_dim
        self.time_norm = RMSNorm(cond_dim)
        self.cond_norm = RMSNorm(cond_dim)

    def forward(self, t_embed: torch.Tensor | None, cond: torch.Tensor | None) -> torch.Tensor:
        if cond is not None:
            if cond.ndim == 1:
                cond = cond[None]  # bare [cond_dim] -> [1, cond_dim]
            if cond.ndim != 2 or cond.shape[1] != self.cond_dim:
                raise ValueError(f"Conditioning must be shaped [B, {self.cond_dim}], got {tuple(cond.shape)}")
            cond = self.cond_norm(cond)
        if t_embed is None:
            if cond is None:
                raise ValueError("Backbone forward needs a timestep `t` and/or conditioning `cond`")
            return cond
        t_embed = self.time_norm(t_embed)
        return t_embed if cond is None else t_embed + cond
