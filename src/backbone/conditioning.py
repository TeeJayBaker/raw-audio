from __future__ import annotations

import math

import torch
from torch import nn


def conditioning_mode(conditioning: dict | None) -> str:
    return (conditioning or {}).get("mode", "none")


def combine_time_conditioning(t: torch.Tensor | None, cond: torch.Tensor | None, mode: str) -> torch.Tensor | None:
    if t is None:
        return cond
    if mode == "none":
        raise ValueError("Received t but this backbone has no configured conditioning path")
    if cond is None:
        return t
    return t + cond


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, features: int | None = None, time_scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.time_scale = float(time_scale)
        self.features = features or min(256, max(16, dim))
        if self.features % 2:
            self.features += 1
        hidden_dim = hidden_dim or max(dim, self.features)
        self.mlp = nn.Sequential(nn.Linear(self.features, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim))

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


class Conditioning(nn.Module):
    def __init__(
        self,
        channels: int,
        mode: str = "none",
        cond_dim: int | None = None,
        pool: str | None = None,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.mode = mode
        self.channels = channels
        self.cond_dim = cond_dim or channels
        self.pool = pool
        hidden_dim = hidden_dim or max(channels, self.cond_dim)
        if mode == "none":
            self.proj = None
        elif mode == "add":
            self.proj = self._mlp(channels, hidden_dim)
        elif mode == "film":
            self.proj = self._mlp(channels * 2, hidden_dim)
        elif mode in {"adaln", "adaln_zero"}:
            self.proj = self._mlp(channels * 2, hidden_dim, zero_final=mode == "adaln_zero")
        elif mode == "context_tokens":
            self.proj = None
        else:
            raise ValueError(f"Unknown conditioning mode: {mode}")

    def _mlp(self, out_dim: int, hidden_dim: int, zero_final: bool = False) -> nn.Sequential:
        final = nn.Linear(hidden_dim, out_dim)
        if zero_final:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        return nn.Sequential(nn.Linear(self.cond_dim, hidden_dim), nn.SiLU(), final)

    def _prepare(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim == 1:
            cond = cond[:, None]
        elif cond.ndim == 3 and cond.shape[-1] == 1:
            cond = cond.squeeze(-1)
        elif cond.ndim > 2:
            if self.pool == "mean":
                cond = cond.mean(dim=tuple(range(2, cond.ndim)))
            else:
                raise ValueError(
                    "Conditioning must be global [B], [B, C], or [B, C, 1]. "
                    "Set conditioning.pool: mean to accept frame-rate conditioning."
                )
        if cond.ndim != 2:
            raise ValueError(f"Unsupported conditioning shape: {tuple(cond.shape)}")
        if cond.shape[1] == 1 and self.cond_dim != 1:
            cond = cond.expand(-1, self.cond_dim)
        if cond.shape[1] != self.cond_dim:
            raise ValueError(f"Expected conditioning dim {self.cond_dim}, got {cond.shape[1]}")
        return cond

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode in {"none", "context_tokens"} or cond is None:
            return x
        cond = self._prepare(cond)
        params = self.proj(cond)
        while params.ndim < x.ndim:
            params = params.unsqueeze(-1)
        if self.mode == "add":
            return x + params
        scale, shift = params.chunk(2, dim=1)
        return x * (1 + scale) + shift


def make_conditioning(config: dict | None, channels: int) -> Conditioning:
    config = config or {"mode": "none"}
    return Conditioning(
        channels,
        mode=config.get("mode", "none"),
        cond_dim=config.get("cond_dim"),
        pool=config.get("pool"),
        hidden_dim=config.get("hidden_dim"),
    )
