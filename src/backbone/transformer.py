from __future__ import annotations

import torch
from torch import nn

from backbone.blocks import TransformerBlock
from backbone.conditioning import ConditioningCombiner, ConditioningEmbedding, TimeEmbedding
from backbone.io import (
    STFTConfig,
    as_waveform,
    center_crop_or_pad,
    channels_to_complex,
    complex_to_channels,
    stft_to_waveform,
    waveform_to_stft,
)


def _zero_init_time_embed(cond_dim: int, time_scale: float) -> TimeEmbedding:
    emb = TimeEmbedding(cond_dim, time_scale=time_scale)
    nn.init.zeros_(emb.mlp[-1].weight)
    nn.init.zeros_(emb.mlp[-1].bias)
    return emb


class ConvHead(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 13):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Transformer(nn.Module):
    """Waveform-in / waveform-out transformer. With an `stft` config it operates on
    the (real/imag channelised) spectrogram internally; otherwise it patches the raw
    waveform. Conditioned throughout via per-block AdaLN-Zero."""

    def __init__(
        self,
        channels: int = 1,
        out_channels: int | None = None,
        patching: dict | None = None,
        block: dict | None = None,
        conditioning: dict | None = None,
        head: dict | None = None,
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
        self.stft = STFTConfig.from_dict(stft) if stft is not None else None
        self.patch_size = int((patching or {}).get("patch_size", 1))

        dim = int(block["dim"])
        self.dim = dim
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

        if self.stft is not None:
            in_channels = 2 * self.channels * self.stft.freq_bins
            out_channels = 2 * self.out_channels * self.stft.freq_bins
        else:
            in_channels = self.channels
            out_channels = self.out_channels
        self.in_proj = nn.Conv1d(in_channels, dim, self.patch_size, stride=self.patch_size)
        self.out_proj = nn.ConvTranspose1d(dim, out_channels, self.patch_size, stride=self.patch_size)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    self.cond_dim,
                    heads=block.get("heads", 4),
                    mlp_ratio=block.get("mlp_ratio", 8 / 3),
                    dropout=block.get("dropout", 0.0),
                    rope=block.get("rope", True),
                    qk_norm=block.get("qk_norm", True),
                )
                for _ in range(int(block.get("depth", 2)))
            ]
        )

        head = head or {}
        head_type = head.get("type", "identity")
        if self.stft is not None or head_type == "identity":
            self.head = nn.Identity()
        elif head_type == "conv":
            self.head = ConvHead(self.out_channels, kernel_size=int(head.get("kernel_size", 13)))
        else:
            raise ValueError(f"Unknown Transformer head type: {head_type}")

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        omega: torch.Tensor | None = None,
        t_lo: torch.Tensor | None = None,
        t_hi: torch.Tensor | None = None,
        length: int | None = None,
        return_spec: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        x = as_waveform(x)
        target = int(length or x.shape[-1])
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

        if self.stft is not None:
            spec = waveform_to_stft(x, self.stft)
            h_in = complex_to_channels(spec)
        else:
            h_in = x

        h = self.in_proj(h_in).transpose(1, 2)
        for block in self.blocks:
            h = block(h, cond)
        y = self.out_proj(h.transpose(1, 2))

        if self.stft is not None:
            y = center_crop_or_pad(y, h_in.shape[-1])
            spec = channels_to_complex(y, self.out_channels, self.stft.freq_bins)
            wav = stft_to_waveform(spec, self.stft, length=target)  # fp32
            return (wav, spec) if return_spec else wav
        # fp32 audio out regardless of AMP state (ConvHead/Identity stays bf16 under AMP).
        out = center_crop_or_pad(self.head(y), target).float()
        return (out, None) if return_spec else out
