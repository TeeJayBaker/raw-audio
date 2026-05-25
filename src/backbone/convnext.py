from __future__ import annotations

import torch
from torch import nn

from backbone.blocks import ConvNeXtBlock1d
from backbone.conditioning import TimeEmbedding, prepare_conditioning
from backbone.io import (
    STFTConfig,
    as_waveform,
    center_crop_or_pad,
    channels_to_complex,
    complex_to_channels,
    stft_to_waveform,
    waveform_to_stft,
)


class IStftHead(nn.Module):
    def __init__(self, dim: int, out_channels: int, stft: STFTConfig, parameterisation: str = "magphase"):
        super().__init__()
        if parameterisation not in {"realimag", "magphase"}:
            raise ValueError(f"Unknown iSTFT parameterisation: {parameterisation}")
        self.out_channels = out_channels
        self.stft = stft
        self.parameterisation = parameterisation
        self.proj = nn.Conv1d(dim, 2 * out_channels * stft.freq_bins, 1)

    def spec(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x)
        if self.parameterisation == "magphase":
            b, _cf, t = y.shape
            mag, phase = y.split(self.out_channels * self.stft.freq_bins, dim=1)
            mag = mag.reshape(b, self.out_channels, self.stft.freq_bins, t)
            phase = phase.reshape(b, self.out_channels, self.stft.freq_bins, t)
            return torch.polar(torch.exp(torch.clamp(mag, max=1e2)), phase)
        return channels_to_complex(y, self.out_channels, self.stft.freq_bins)

    def forward(self, x: torch.Tensor, length: int) -> torch.Tensor:
        return stft_to_waveform(self.spec(x), self.stft, length=length)


class WaveNeXtHead(nn.Module):
    def __init__(self, dim: int, out_channels: int, stft: STFTConfig):
        super().__init__()
        if out_channels != 1:
            raise ValueError("WaveNeXt head emits mono waveform output")
        self.stft = stft
        self.proj_fft = nn.Linear(dim, stft.n_fft)
        self.proj_hop = nn.Linear(stft.n_fft, stft.hop_length, bias=False)

    def forward(self, x: torch.Tensor, length: int) -> torch.Tensor:
        h = self.proj_hop(self.proj_fft(x.transpose(1, 2)))
        return center_crop_or_pad(h.reshape(h.shape[0], 1, -1), length)


def _layer_scale(value, depth: int) -> float | None:
    if value is None:
        return None
    if value == "1/depth":
        return 1.0 / max(1, depth)
    return float(value)


def _make_head(head: dict | None, dim: int, out_channels: int, stft: STFTConfig) -> nn.Module:
    head = head or {}
    head_type = head.get("type", "istft")
    if head_type == "istft":
        return IStftHead(dim, out_channels, stft, head.get("parameterisation", "magphase"))
    if head_type == "wavenext":
        return WaveNeXtHead(dim, out_channels, stft)
    raise ValueError(f"Unknown ConvNeXt head type: {head_type}")


def _resolve_branches(branches: dict | None, stft: dict | None) -> list[STFTConfig]:
    branches = branches or {"mode": "single"}
    mode = branches.get("mode", "single")
    if mode == "single":
        return [STFTConfig.from_dict(stft)]
    if mode == "multi_resolution":
        resolutions = branches.get("resolutions")
        if not resolutions:
            raise ValueError("branches.resolutions is required when branches.mode is multi_resolution")
        return [STFTConfig.from_dict(res) for res in resolutions]
    raise ValueError(f"Unknown ConvNeXt branch mode: {mode}")


class STFTBranch(nn.Module):
    """One STFT-domain ConvNeXt path: STFT, in-projection, ConvNeXt trunk, waveform head."""

    def __init__(self, stft: STFTConfig, channels: int, out_channels: int, block: dict, head: dict | None, cond_dim: int):
        super().__init__()
        self.stft = stft
        width = int(block["channels"])
        depth = int(block.get("depth", 4))
        kernel = int(block.get("kernel_size", 7))
        scale = _layer_scale(block.get("layer_scale", 1e-6), depth)
        self.in_proj = nn.Conv1d(2 * channels * stft.freq_bins, width, kernel, padding=kernel // 2)
        self.trunk = nn.ModuleList(
            [
                ConvNeXtBlock1d(width, cond_dim, kernel_size=kernel, expansion=int(block.get("expansion", 4)), layer_scale=scale)
                for _ in range(depth)
            ]
        )
        self.head = _make_head(head, width, out_channels, stft)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, length: int) -> torch.Tensor:
        h = self.in_proj(complex_to_channels(waveform_to_stft(x, self.stft)))
        for block in self.trunk:
            h = block(h, cond)
        return self.head(h, length)


class ConvNeXt(nn.Module):
    """Waveform-in / waveform-out ConvNeXt vocoder. One STFT branch (`single`) or
    several at different scales whose waveforms are summed (`multi_resolution`)."""

    def __init__(
        self,
        channels: int = 1,
        out_channels: int | None = None,
        branches: dict | None = None,
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
        self.cond_dim = int(conditioning["cond_dim"])
        self.time_embed = TimeEmbedding(self.cond_dim, time_scale=conditioning.get("time_scale", 1.0))
        self.branches = nn.ModuleList(
            [
                STFTBranch(res, self.channels, self.out_channels, block, head, self.cond_dim)
                for res in _resolve_branches(branches, stft)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        length: int | None = None,
    ) -> torch.Tensor:
        x = as_waveform(x)
        target = int(length or x.shape[-1])
        t_embed = self.time_embed(t) if t is not None else None
        cond = prepare_conditioning(t_embed, cond, self.cond_dim)
        return sum(branch(x, cond, target) for branch in self.branches)
