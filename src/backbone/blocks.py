from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    from backbone.jvp_flash_attention import flash_attn_jvp
except ImportError:  # triton unavailable (e.g. CPU-only env); JVP falls back to the MATH backend
    flash_attn_jvp = None


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


def _rope_angles(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Rotation angles [n, dim//2] for token `positions` over a `dim`-wide pair grid."""
    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim))
    return positions.float()[:, None] * inv[None, :]


def rope_tables(npf: int, npt: int, head_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) tables [npf*npt, head_dim//2] for the freq-major token raster.

    `npf == 1` rotates the full head dim by time index (bit-for-bit the original 1-D RoPE,
    so `column` and its checkpoints are unchanged). `npf > 1` is axial 2-D RoPE: the first
    head_dim//2 dims rotate by freq-patch index, the second half by time-patch index."""
    idx = torch.arange(npf * npt, device=device)
    freq_idx, time_idx = idx // npt, idx % npt
    if npf == 1:
        angles = _rope_angles(time_idx, head_dim)
    else:
        if head_dim % 4:
            raise ValueError(f"axial RoPE needs head_dim divisible by 4, got {head_dim}")
        half = head_dim // 2
        angles = torch.cat([_rope_angles(freq_idx, half), _rope_angles(time_idx, half)], dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate [b, heads, n, head_dim] by precomputed (cos, sin) tables [n, head_dim//2]."""
    cos = cos[None, None].to(dtype=x.dtype)
    sin = sin[None, None].to(dtype=x.dtype)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    # Forward-AD attention path, enabled only around a torch.func.jvp call (MeanFlow's jvp
    # tangent). Class-level so MeanFlow.u_and_dudt can flip every layer at once.
    _use_triton_jvp: bool = False

    @classmethod
    def set_triton_jvp(cls, enabled: bool) -> None:
        """Toggle the forward-AD attention path. On CUDA it routes to the Triton
        flash-attention-JVP kernel; otherwise to the MATH SDPA backend (the fused kernels
        lack forward-AD). Normal training (flag off) keeps the optimised fused SDPA."""
        cls._use_triton_jvp = enabled

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

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.rope:
            if rope is None:  # 1-D fallback for callers that don't supply a token grid
                rope = rope_tables(1, n, self.head_dim, x.device)
            q = apply_rope(q, *rope)
            k = apply_rope(k, *rope)
        if self.qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        if self._use_triton_jvp:
            if self.training and self.dropout:
                raise ValueError("JVP attention path does not support dropout; set block.dropout=0")
            if flash_attn_jvp is not None and q.is_cuda:
                y = flash_attn_jvp(q.float(), k.float(), v.float()).to(v.dtype)
            else:
                with sdpa_kernel(SDPBackend.MATH):
                    y = F.scaled_dot_product_attention(q, k, v)
        else:
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

    def forward(self, x: torch.Tensor, cond: torch.Tensor, rope=None) -> torch.Tensor:
        scale1, shift1, gate1, scale2, shift2, gate2 = self.ada(cond)
        h = self.norm1(x) * (1 + scale1[:, None]) + shift1[:, None]
        x = x + gate1[:, None] * self.attn(h, rope)
        h = self.norm2(x) * (1 + scale2[:, None]) + shift2[:, None]
        return x + gate2[:, None] * self.mlp(h)
