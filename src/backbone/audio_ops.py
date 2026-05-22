from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def as_waveform(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(1)
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected waveform tensor [B, T] or [B, C, T], got {tuple(x.shape)}")


@dataclass(frozen=True)
class STFTConfig:
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int | None = None

    @classmethod
    def from_dict(cls, cfg: dict | None) -> STFTConfig:
        cfg = cfg or {}
        n_fft = int(cfg.get("n_fft", 1024))
        freq_bins = cfg.get("freq_bins")
        if freq_bins is not None and int(freq_bins) != n_fft // 2 + 1:
            raise ValueError(f"freq_bins={freq_bins} does not match n_fft // 2 + 1 ({n_fft // 2 + 1})")
        hop_length = int(cfg.get("hop_length", max(1, n_fft // 4)))
        win_length = cfg.get("win_length", n_fft)
        return cls(n_fft=n_fft, hop_length=hop_length, win_length=int(win_length))

    @property
    def freq_bins(self) -> int:
        return self.n_fft // 2 + 1


def _window(cfg: STFTConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.hann_window(cfg.win_length or cfg.n_fft, device=device, dtype=dtype)


def waveform_to_stft(x: torch.Tensor, cfg: STFTConfig) -> torch.Tensor:
    """Return complex STFT as [B, C, F, frames]."""
    x = as_waveform(x)
    b, c, length = x.shape
    flat = x.reshape(b * c, length)
    spec = torch.stft(
        flat,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=_window(cfg, x.device, x.dtype),
        center=True,
        return_complex=True,
    )
    return spec.reshape(b, c, cfg.freq_bins, spec.shape[-1])


def stft_to_waveform(spec: torch.Tensor, cfg: STFTConfig, length: int | None = None) -> torch.Tensor:
    """Invert complex STFT shaped [B, C, F, frames] to [B, C, T]."""
    if not torch.is_complex(spec) or spec.ndim != 4:
        raise ValueError(f"Expected complex STFT [B, C, F, T], got {tuple(spec.shape)}")
    b, c, _f, frames = spec.shape
    flat = spec.reshape(b * c, spec.shape[-2], frames)
    wav = torch.istft(
        flat,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=_window(cfg, spec.device, spec.real.dtype),
        center=True,
        length=length,
    )
    return wav.reshape(b, c, wav.shape[-1])


def complex_to_channels(spec: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(spec) or spec.ndim != 4:
        raise ValueError(f"Expected complex STFT [B, C, F, T], got {tuple(spec.shape)}")
    b, c, f, t = spec.shape
    real = spec.real.reshape(b, c * f, t)
    imag = spec.imag.reshape(b, c * f, t)
    return torch.cat([real, imag], dim=1)


def channels_to_complex(x: torch.Tensor, channels: int, freq_bins: int) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected channel STFT [B, 2*C*F, T], got {tuple(x.shape)}")
    expected = 2 * channels * freq_bins
    if x.shape[1] != expected:
        raise ValueError(f"Expected {expected} STFT channels, got {x.shape[1]}")
    b, _cf, t = x.shape
    real, imag = x.split(channels * freq_bins, dim=1)
    real = real.reshape(b, channels, freq_bins, t)
    imag = imag.reshape(b, channels, freq_bins, t)
    return torch.complex(real, imag)


def pad_or_trim_frames(x: torch.Tensor, frames: int) -> torch.Tensor:
    diff = x.shape[-1] - frames
    if diff == 0:
        return x
    if diff > 0:
        return x[..., :frames]
    return F.pad(x, (0, -diff))


def stft_to_channels(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        return complex_to_channels(x)
    if x.ndim == 4:
        b, c, f, t = x.shape
        return x.reshape(b, c * f, t)
    if x.ndim == 3:
        return x
    raise ValueError(f"Unsupported STFT-like tensor shape: {tuple(x.shape)}")


def channels_to_stft_like(x: torch.Tensor, freq_bins: int | None = None) -> torch.Tensor:
    if freq_bins is None:
        return x
    b, c, t = x.shape
    if c % freq_bins != 0:
        return x
    return x.reshape(b, c // freq_bins, freq_bins, t)
