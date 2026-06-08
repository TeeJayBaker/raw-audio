from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * self.weight


class AdaLN(nn.Module):
    """AdaLN-Zero modulation: maps a global conditioning vector to `groups`
    zero-initialised modulation tensors of width `channels` (so the block is an
    identity at initialisation)."""

    def __init__(self, cond_dim: int, channels: int, groups: int):
        super().__init__()
        self.groups = groups
        self.proj = nn.Linear(cond_dim, groups * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.proj(F.silu(cond)).chunk(self.groups, dim=-1)


def _rope(x: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    if dim % 2:
        return x
    pos = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
    freq = torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim
    inv = 1.0 / (10000**freq)
    angles = pos[:, None] * inv[None, :]
    cos = angles.cos()[None, None].to(dtype=x.dtype)
    sin = angles.sin()[None, None].to(dtype=x.dtype)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0, rope: bool = True, qk_norm: bool = True):
        super().__init__()
        if dim % heads:
            raise ValueError(f"Transformer dim {dim} must divide heads {heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.rope = rope
        self.qk_norm = qk_norm
        self.dropout = dropout
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.rope:
            q = _rope(q)
            k = _rope(k)
        if self.qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.w12 = nn.Linear(dim, hidden * 2, bias=False)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.w12(x).chunk(2, dim=-1)
        return self.out(self.drop(F.silu(gate) * x))


class TransformerBlock(nn.Module):
    """Pre-norm RoPE attention + SwiGLU, modulated by AdaLN-Zero (scale/shift/gate per sub-block)."""

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        heads: int = 4,
        mlp_ratio: float = 8 / 3,
        dropout: float = 0.0,
        rope: bool = True,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads=heads, dropout=dropout, rope=rope, qk_norm=qk_norm)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio), dropout=dropout)
        self.ada = AdaLN(cond_dim, dim, groups=6)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale1, shift1, gate1, scale2, shift2, gate2 = self.ada(cond)
        h = self.norm1(x) * (1 + scale1[:, None]) + shift1[:, None]
        x = x + gate1[:, None] * self.attn(h)
        h = self.norm2(x) * (1 + scale2[:, None]) + shift2[:, None]
        return x + gate2[:, None] * self.mlp(h)


class ConvNeXtBlock1d(nn.Module):
    """ConvNeXt block on [B, C, T]: depthwise conv, channel LayerNorm, AdaLN-Zero
    modulation (scale/shift + zero-init gate on the residual branch),
    inverted-bottleneck pointwise MLP, LayerScale residual."""

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        kernel_size: int = 7,
        expansion: int = 4,
        layer_scale: float | None = 1e-6,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.ada = AdaLN(cond_dim, channels, groups=3)
        hidden = channels * expansion
        self.pointwise = nn.Sequential(nn.Conv1d(channels, hidden, 1), nn.GELU(), nn.Conv1d(hidden, channels, 1))
        self.layer_scale = nn.Parameter(torch.ones(1, channels, 1) * layer_scale) if layer_scale is not None else None

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.depthwise(x)
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        scale, shift, gate = self.ada(cond)
        h = h * (1 + scale[..., None]) + shift[..., None]
        h = self.pointwise(h)
        if self.layer_scale is not None:
            h = h * self.layer_scale
        return x + gate[..., None] * h
