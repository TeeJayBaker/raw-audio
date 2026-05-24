from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from backbone.audio_ops import (
    STFTConfig,
    as_waveform,
    channels_to_complex,
    complex_to_channels,
    stft_to_channels,
    stft_to_waveform,
    waveform_to_stft,
)
from backbone.blocks import center_crop_or_pad
from backbone.conditioning import TimeEmbedding, make_conditioning, prepare_conditioning
from backbone.convnext_blocks import ConvNeXtBlock1d
from backbone.transformer_blocks import TransformerBlock


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


class ConvNeXtHead(nn.Module):
    def __init__(self, channels: int, hidden: int = 16, kernel_size: int = 7):
        super().__init__()
        self.expand = nn.Conv1d(channels, hidden, 1)
        self.block = ConvNeXtBlock1d(hidden, kernel_size=kernel_size)
        self.project = nn.Conv1d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.block(self.expand(x)))


class Transformer(nn.Module):
    def __init__(
        self,
        io: dict | None = None,
        patching: dict | str | None = None,
        block: dict | None = None,
        conditioning: dict | None = None,
        head: dict | None = None,
        sample_rate: int = 48000,
        name: str | None = None,
    ):
        super().__init__()
        self.io = io or {"type": "waveform", "channels": 1}
        patching = {"type": patching} if isinstance(patching, str) else (patching or {"type": "1d", "patch_size": 16})
        block = block or {}
        conditioning = conditioning or {"mode": "none"}
        self.sample_rate = sample_rate
        self.name = name
        self.patch_type = patching.get("type", "1d")
        self.patch_size = int(patching.get("patch_size", 16))
        patch_shape = patching.get("patch_shape")
        if patch_shape is None:
            self.patch_shape = (self.patch_size, self.patch_size)
        else:
            # STFT 2-D patches are ordered [freq, time].
            if len(patch_shape) != 2:
                raise ValueError("patching.patch_shape must be [freq, time]")
            self.patch_shape = (int(patch_shape[0]), int(patch_shape[1]))
        self.in_channels = int(self.io.get("channels", 1))
        self.out_channels = int(self.io.get("out_channels", self.in_channels))
        self.is_stft = self.io.get("type") == "stft" or self.patch_type == "2d"
        self.stft = STFTConfig.from_dict(self.io) if self.is_stft else None
        in_channels = self.in_channels
        dim = int(block.get("dim", 64))
        self.dim = dim
        if self.is_stft and self.patch_type == "2d":
            self.freq_bins = int(self.io.get("freq_bins", self.stft.freq_bins))
            self.in_proj = nn.Conv2d(2 * in_channels, dim, self.patch_shape, stride=self.patch_shape)
        else:
            if self.is_stft:
                in_channels *= 2 * int(self.io.get("freq_bins", self.stft.freq_bins))
            self.in_proj = nn.Conv1d(in_channels, dim, self.patch_size, stride=self.patch_size)
        self.cond_mode = conditioning.get("mode", "none")
        pre_cond = {"mode": "none"} if self.cond_mode in {"context_tokens", "adaln_zero"} else conditioning
        self.cond = make_conditioning(pre_cond, dim)
        self.cond_dim = int(conditioning.get("cond_dim", dim))
        self.time_embed = (
            TimeEmbedding(self.cond_dim, conditioning.get("time_hidden_dim"), time_scale=conditioning.get("time_scale", 1.0))
            if self.cond_mode != "none"
            else None
        )
        self.context_proj = None
        if self.cond_mode == "context_tokens":
            self.context_proj = nn.Sequential(
                nn.Linear(int(conditioning.get("cond_dim", dim)), dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    heads=block.get("heads", 4),
                    mlp_ratio=block.get("mlp_ratio", 8 / 3),
                    dropout=block.get("dropout", 0.0),
                    rope=block.get("rope", True),
                    qk_norm=block.get("qk_norm", True),
                    norm=block.get("norm", "rms"),
                    conditioning="adaln_zero" if self.cond_mode == "adaln_zero" else "none",
                    cond_dim=self.cond_dim,
                )
                for _ in range(int(block.get("depth", 2)))
            ]
        )
        if self.is_stft and self.patch_type == "2d":
            self.out_proj = nn.ConvTranspose2d(dim, 2 * self.out_channels, self.patch_shape, stride=self.patch_shape)
        elif self.is_stft:
            freq_bins = int(self.io.get("freq_bins", self.stft.freq_bins))
            self.out_proj = nn.ConvTranspose1d(dim, 2 * self.out_channels * freq_bins, self.patch_size, stride=self.patch_size)
            self.freq_bins = freq_bins
        else:
            self.out_proj = nn.ConvTranspose1d(dim, self.out_channels, self.patch_size, stride=self.patch_size)
        head = head or {"type": "identity"}
        head_type = head.get("type", "identity")
        if head_type == "identity":
            self.head = nn.Identity()
        elif head_type == "conv":
            if self.is_stft or self.patch_type != "1d":
                raise ValueError("head.type: conv is only supported for 1d transformers")
            self.head = ConvHead(self.out_channels, kernel_size=int(head.get("kernel_size", 13)))
        elif head_type == "convnext":
            if self.is_stft or self.patch_type != "1d":
                raise ValueError("head.type: convnext is only supported for 1d transformers")
            self.head = ConvNeXtHead(
                self.out_channels,
                hidden=int(head.get("hidden", 16)),
                kernel_size=int(head.get("kernel_size", 7)),
            )
        else:
            raise ValueError(f"Unknown Transformer head type: {head_type}")

    def _context_token(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim == 1:
            cond = cond[:, None]
        elif cond.ndim == 3 and cond.shape[-1] == 1:
            cond = cond.squeeze(-1)
        elif cond.ndim > 2:
            raise ValueError("context_tokens conditioning must be global")
        return self.context_proj(cond)[:, None, :]

    def _prepare_block_cond(self, cond: torch.Tensor | None) -> torch.Tensor | None:
        if cond is None:
            return None
        if cond.ndim == 1:
            cond = cond[:, None]
        elif cond.ndim == 3 and cond.shape[-1] == 1:
            cond = cond.squeeze(-1)
        elif cond.ndim > 2:
            raise ValueError("Transformer block conditioning must be global")
        if cond.shape[1] == 1 and self.cond_dim != 1:
            cond = cond.expand(-1, self.cond_dim)
        if cond.shape[1] != self.cond_dim:
            raise ValueError(f"Expected conditioning dim {self.cond_dim}, got {cond.shape[1]}")
        return cond

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None, cond: torch.Tensor | None = None, length: int | None = None) -> torch.Tensor:
        if t is not None and self.time_embed is not None:
            t = self.time_embed(t)
        cond = prepare_conditioning(t, cond, self.cond_mode, self.cond_dim)
        if self.is_stft:
            if x.ndim == 4 and not torch.is_complex(x):
                raise ValueError("Real STFT tensors [B, C, F, T] are ambiguous; pass complex STFT or channelized real/imag input")
            if torch.is_complex(x):
                h_in = stft_to_channels(x)
                target = int(length or h_in.shape[-1] * self.stft.hop_length)
            elif x.ndim == 3 and x.shape[1] == 2 * self.in_channels * self.freq_bins:
                h_in = x
                target = int(length or h_in.shape[-1] * self.stft.hop_length)
            else:
                wav = as_waveform(x)
                target = int(length or wav.shape[-1])
                spec = waveform_to_stft(wav, self.stft)
                h_in = complex_to_channels(spec)
        else:
            h_in = as_waveform(x)
            target = int(length or h_in.shape[-1])

        stft_grid = None
        stft_input_shape = None
        if self.is_stft and self.patch_type == "2d":
            b = h_in.shape[0]
            real, imag = h_in.split(self.in_channels * self.freq_bins, dim=1)
            real = real.reshape(b, self.in_channels, self.freq_bins, h_in.shape[-1])
            imag = imag.reshape(b, self.in_channels, self.freq_bins, h_in.shape[-1])
            h_grid = torch.cat([real, imag], dim=1)
            stft_input_shape = h_grid.shape[-2:]
            pad_f = (-h_grid.shape[-2]) % self.patch_shape[0]
            pad_t = (-h_grid.shape[-1]) % self.patch_shape[1]
            if pad_f or pad_t:
                h_grid = F.pad(h_grid, (0, pad_t, 0, pad_f))
            h = self.in_proj(h_grid)
            stft_grid = h.shape[-2:]
            h = h.flatten(2)
        else:
            h = self.in_proj(h_in)
        h = self.cond(h, cond).transpose(1, 2)
        context_added = False
        if self.context_proj is not None and cond is not None:
            h = torch.cat([self._context_token(cond), h], dim=1)
            context_added = True
        block_cond = self._prepare_block_cond(cond) if self.cond_mode == "adaln_zero" else None
        rope_start = 1 if context_added else 0
        for block in self.blocks:
            h = block(h, cond=block_cond, rope_start=rope_start)
        if context_added:
            h = h[:, 1:]
        if self.is_stft and self.patch_type == "2d":
            if stft_grid is None:
                raise RuntimeError("Missing STFT patch grid")
            y = self.out_proj(h.transpose(1, 2).reshape(h.shape[0], self.dim, *stft_grid))
            if stft_input_shape is None:
                raise RuntimeError("Missing STFT input shape")
            y = y[..., : stft_input_shape[0], : stft_input_shape[1]]
            real, imag = y.split(self.out_channels, dim=1)
            y = torch.cat(
                [
                    real.reshape(y.shape[0], self.out_channels * self.freq_bins, y.shape[-1]),
                    imag.reshape(y.shape[0], self.out_channels * self.freq_bins, y.shape[-1]),
                ],
                dim=1,
            )
        else:
            y = self.out_proj(h.transpose(1, 2))
        if self.is_stft:
            y = center_crop_or_pad(y, h_in.shape[-1])
            spec = channels_to_complex(y, self.out_channels, self.freq_bins)
            return spec if self.io.get("type") == "stft" else stft_to_waveform(spec, self.stft, length=target)
        y = self.head(y)
        return center_crop_or_pad(y, target)
