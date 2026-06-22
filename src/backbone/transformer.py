from __future__ import annotations

import torch
from torch import nn

from backbone.blocks import TransformerBlock, rope_tables
from backbone.conditioning import ConditioningCombiner, ConditioningEmbedding, TimeEmbedding
from backbone.io import STFTConfig
from backbone.patching import build_patcher


def _zero_init_time_embed(cond_dim: int, time_scale: float) -> TimeEmbedding:
    emb = TimeEmbedding(cond_dim, time_scale=time_scale)
    nn.init.zeros_(emb.mlp[-1].weight)
    nn.init.zeros_(emb.mlp[-1].bias)
    return emb


class Transformer(nn.Module):
    """Spectrogram-in / spectrogram-out transformer (no STFT inside -> scripts to C++).

    Input and output are channelised STFTs ``[B, 2*C*F, frames]`` (the
    ``complex_to_channels`` layout). A pure-reshape patcher turns the spectrogram into
    tokens, a learned ``in_proj`` (optionally low-rank via ``bottleneck``) lifts them to
    ``dim``, AdaLN-Zero blocks with axial RoPE process them, and ``out_proj`` + ``unpatch``
    return the spectrogram. The waveform<->spectrogram crossing lives in
    ``RectifiedFlow._predict``, not here. Conditioned throughout via per-block AdaLN-Zero.
    """

    def __init__(
        self,
        channels: int = 1,
        out_channels: int | None = None,
        patching: dict | None = None,
        bottleneck: int | None = None,
        block: dict | None = None,
        conditioning: dict | None = None,
        stft: dict | None = None,
        sample_rate: int = 48000,
        name: str | None = None,
    ):
        super().__init__()
        block = block or {}
        conditioning = conditioning or {}
        self.sample_rate = sample_rate
        self.name = name
        self.channels = int(channels)
        self.out_channels = int(out_channels or channels)
        self.stft = STFTConfig.from_dict(stft)
        self.patcher = build_patcher(patching, self.stft.freq_bins)

        dim = int(block["dim"])
        self.dim = dim
        self.heads = int(block.get("heads", 4))
        self.head_dim = dim // self.heads
        self.cond_dim = int(conditioning["cond_dim"])
        self.time_embed = TimeEmbedding(self.cond_dim, time_scale=conditioning.get("time_scale", 1.0))
        time_scale = float(conditioning.get("time_scale", 1.0))
        self.gap_embed = (
            _zero_init_time_embed(self.cond_dim, time_scale)
            if conditioning.get("gap_embed", False)
            else None
        )
        self.omega_embed = (
            _zero_init_time_embed(self.cond_dim, float(conditioning.get("omega_scale", 1.0)))
            if conditioning.get("guidance_embed", False)
            else None
        )
        self.lo_embed = (
            _zero_init_time_embed(self.cond_dim, time_scale)
            if conditioning.get("interval_embed", False)
            else None
        )
        self.hi_embed = (
            _zero_init_time_embed(self.cond_dim, time_scale)
            if conditioning.get("interval_embed", False)
            else None
        )
        self.cond_embed = ConditioningEmbedding(int(conditioning.get("embed_dim", self.cond_dim)), self.cond_dim)
        self.cond_combine = ConditioningCombiner(self.cond_dim)

        feat_in = self.patcher.feat(self.channels)
        feat_out = self.patcher.feat(self.out_channels)
        if bottleneck:  # JiT/pMF input-only low-rank waist; output stays full-rank
            self.in_proj = nn.Sequential(
                nn.Conv1d(feat_in, int(bottleneck), 1, bias=False),
                nn.Conv1d(int(bottleneck), dim, 1),
            )
        else:
            self.in_proj = nn.Conv1d(feat_in, dim, 1)

        def make_block() -> TransformerBlock:
            return TransformerBlock(
                dim,
                self.cond_dim,
                heads=self.heads,
                mlp_ratio=block.get("mlp_ratio", 8 / 3),
                dropout=block.get("dropout", 0.0),
                rope=block.get("rope", True),
                qk_norm=block.get("qk_norm", True),
            )

        depth = int(block.get("depth", 2))
        self.aux_depth = int(block.get("aux_depth", 0))
        if not 0 <= self.aux_depth <= depth:
            raise ValueError(f"block.aux_depth must be in [0, depth={depth}], got {self.aux_depth}")
        self.trunk_depth = depth - self.aux_depth
        # `blocks` spans the full depth: a shared trunk `[:trunk_depth]` then the u-head tail. The
        # MeanFlow v-head (instantaneous velocity, training-only) branches off the shared-trunk
        # activation through a parallel `v_blocks`/`v_out_proj`. aux_depth=0 leaves the module
        # bit-identical to the single-head RF transformer (no v-head, RF checkpoints load as-is).
        self.out_proj = nn.Conv1d(dim, feat_out, 1)
        self.blocks = nn.ModuleList([make_block() for _ in range(depth)])
        self.v_blocks = nn.ModuleList([make_block() for _ in range(self.aux_depth)]) if self.aux_depth else None
        self.v_out_proj = nn.Conv1d(dim, feat_out, 1) if self.aux_depth else None

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        omega: torch.Tensor | None = None,
        t_lo: torch.Tensor | None = None,
        t_hi: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        t_embed = self.time_embed(t) if t is not None else None
        if h is not None and self.gap_embed is None:
            raise ValueError("Backbone got `h` but conditioning.gap_embed is false")
        if omega is not None and self.omega_embed is None:
            raise ValueError("Backbone got `omega` but conditioning.guidance_embed is false")
        if (t_lo is not None or t_hi is not None) and self.lo_embed is None:
            raise ValueError("Backbone got interval bounds but conditioning.interval_embed is false")
        if t_embed is not None:
            if self.gap_embed is not None:
                t_embed = t_embed + self.gap_embed(torch.zeros_like(t) if h is None else h)
            if self.omega_embed is not None:
                t_embed = t_embed + self.omega_embed(torch.ones_like(t) if omega is None else omega)
            if self.lo_embed is not None:
                t_embed = t_embed + self.lo_embed(torch.zeros_like(t) if t_lo is None else t_lo)
                t_embed = t_embed + self.hi_embed(torch.ones_like(t) if t_hi is None else t_hi)
        cond = self.cond_embed(cond)
        cond = self.cond_combine(t_embed, cond)

        frames = x.shape[-1]
        z = self.in_proj(self.patcher.patch(x)).transpose(1, 2)  # [B, N, dim]
        npt = z.shape[1] // self.patcher.npf
        rope = rope_tables(self.patcher.npf, npt, self.head_dim, z.device)
        want_aux = return_aux and self.aux_depth > 0
        z_shared = None
        for i, block in enumerate(self.blocks):
            if want_aux and i == self.trunk_depth:
                z_shared = z  # shared-trunk activation feeding both heads
            z = block(z, cond, rope)
        u_spec = self.patcher.unpatch(self.out_proj(z.transpose(1, 2)), frames)
        if not want_aux:
            return u_spec
        for block in self.v_blocks:
            z_shared = block(z_shared, cond, rope)
        v_spec = self.patcher.unpatch(self.v_out_proj(z_shared.transpose(1, 2)), frames)
        return u_spec, v_spec
