from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, activation: str = "gelu"):
        super().__init__()
        if activation == "geglu":
            self.fc1 = nn.Linear(dim, hidden_dim * 2)
            self.act = None
        else:
            self.fc1 = nn.Linear(dim, hidden_dim)
            self.act = nn.GELU()
        self.activation = activation
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.activation == "geglu":
            x, gate = x.chunk(2, dim=-1)
            x = x * F.gelu(gate)
        else:
            x = self.act(x)
        return self.fc2(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 65536, theta: float = 10000.0):
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE dim must be even")
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, freqs).float()
        freqs = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("freqs_cos", freqs.cos(), persistent=False)
        self.register_buffer("freqs_sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None):
        if position_ids is None:
            return self.freqs_cos[: x.shape[1]], self.freqs_sin[: x.shape[1]]
        identity = position_ids == -1
        safe_ids = position_ids.clamp(min=0)
        cos, sin = self.freqs_cos[safe_ids], self.freqs_sin[safe_ids]
        cos = cos.masked_fill(identity.unsqueeze(-1), 1.0)
        sin = sin.masked_fill(identity.unsqueeze(-1), 0.0)
        return cos, sin


class RotaryEmbedding2D(nn.Module):
    def __init__(self, dim: int, n_freq_patches: int = 8, max_time_patches: int = 256, theta: float = 10000.0):
        super().__init__()
        if dim % 4:
            raise ValueError("2D RoPE dim must be divisible by 4")
        half_dim = dim // 2
        freqs = 1.0 / (theta ** (torch.arange(0, half_dim, 2).float() / half_dim))
        freq_freqs = torch.outer(torch.arange(n_freq_patches).float(), freqs)
        time_freqs = torch.outer(torch.arange(max_time_patches).float(), freqs)
        self.register_buffer("freq_cos", torch.cat([freq_freqs, freq_freqs], dim=-1).cos(), persistent=False)
        self.register_buffer("freq_sin", torch.cat([freq_freqs, freq_freqs], dim=-1).sin(), persistent=False)
        self.register_buffer("time_cos", torch.cat([time_freqs, time_freqs], dim=-1).cos(), persistent=False)
        self.register_buffer("time_sin", torch.cat([time_freqs, time_freqs], dim=-1).sin(), persistent=False)
        self.n_freq_patches = n_freq_patches

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None):
        if position_ids is None:
            seq_len = x.shape[1]
            n_time = max(1, (seq_len + self.n_freq_patches - 1) // self.n_freq_patches)
            ids = torch.arange(seq_len, device=x.device)
        else:
            identity = position_ids == -1
            ids = position_ids.clamp(min=0)
            n_time = ((ids.max() + 1 + self.n_freq_patches - 1) // self.n_freq_patches).clamp(min=1)
        freq_idx = ids // n_time
        time_idx = ids % n_time
        cos = torch.cat([self.freq_cos[freq_idx], self.time_cos[time_idx]], dim=-1)
        sin = torch.cat([self.freq_sin[freq_idx], self.time_sin[time_idx]], dim=-1)
        if position_ids is not None:
            cos = cos.masked_fill(identity.unsqueeze(-1), 1.0)
            sin = sin.masked_fill(identity.unsqueeze(-1), 0.0)
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    else:
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        use_rope: bool = True,
        rope_2d: bool = False,
        n_freq_patches: int = 8,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if not use_rope:
            self.rope = None
        elif rope_2d:
            self.rope = RotaryEmbedding2D(head_dim, n_freq_patches=n_freq_patches)
        else:
            self.rope = RotaryEmbedding(head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, channels // self.num_heads)
        q, k, v = qkv.permute(2, 0, 1, 3, 4)
        if self.rope is not None:
            cos, sin = self.rope(q, position_ids=position_ids)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn_mask = None
        if mask is not None:
            attn_mask = torch.zeros(bsz, 1, 1, seq_len, device=x.device, dtype=q.dtype)
            attn_mask.masked_fill_(mask.view(bsz, 1, 1, seq_len) == 0, float("-inf"))
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(x.transpose(1, 2).reshape(bsz, seq_len, channels))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_rope: bool = True,
        activation: str = "gelu",
        qkv_bias: bool = True,
        rope_2d: bool = False,
        n_freq_patches: int = 8,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias, use_rope, rope_2d, n_freq_patches)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio), activation=activation)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask, position_ids=position_ids)
        return x + self.mlp(self.norm2(x))
