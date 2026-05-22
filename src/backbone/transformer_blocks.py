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

    def forward(self, x: torch.Tensor, rope_start: int = 0) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.rope:
            if rope_start:
                q = torch.cat([q[:, :, :rope_start], _rope(q[:, :, rope_start:])], dim=2)
                k = torch.cat([k[:, :, :rope_start], _rope(k[:, :, rope_start:])], dim=2)
            else:
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
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        mlp_ratio: float = 8 / 3,
        dropout: float = 0.0,
        rope: bool = True,
        qk_norm: bool = True,
        norm: str = "rms",
        conditioning: str = "none",
        cond_dim: int | None = None,
    ):
        super().__init__()
        norm_cls = RMSNorm if norm == "rms" else nn.LayerNorm
        self.norm1 = norm_cls(dim)
        self.attn = Attention(dim, heads=heads, dropout=dropout, rope=rope, qk_norm=qk_norm)
        self.norm2 = norm_cls(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio), dropout=dropout)
        self.conditioning = conditioning
        self.ada = None
        if conditioning == "adaln_zero":
            self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim or dim, dim * 6))
            nn.init.zeros_(self.ada[-1].weight)
            nn.init.zeros_(self.ada[-1].bias)
        elif conditioning != "none":
            raise ValueError(f"Unknown TransformerBlock conditioning: {conditioning}")

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None, rope_start: int = 0) -> torch.Tensor:
        if self.ada is None:
            x = x + self.attn(self.norm1(x), rope_start=rope_start)
            return x + self.mlp(self.norm2(x))
        if cond is None:
            raise ValueError("adaln_zero transformer blocks require conditioning")
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.ada(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + gamma1[:, None]) + beta1[:, None]
        x = x + alpha1[:, None] * self.attn(h, rope_start=rope_start)
        h = self.norm2(x) * (1 + gamma2[:, None]) + beta2[:, None]
        return x + alpha2[:, None] * self.mlp(h)
