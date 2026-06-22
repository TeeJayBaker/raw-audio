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
        hop_length = int(cfg.get("hop_length", max(1, n_fft // 4)))
        win_length = int(cfg.get("win_length", n_fft))
        return cls(n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    @property
    def freq_bins(self) -> int:
        return self.n_fft // 2 + 1


def waveform_to_stft(x: torch.Tensor, cfg: STFTConfig) -> torch.Tensor:
    """Complex STFT [B, C, F, frames] from waveform [B, C, T]. Always runs in fp32
    (torch.stft doesn't support bf16/half), so this is safe inside an autocast block."""
    x = as_waveform(x)
    b, c, length = x.shape
    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        window = torch.hann_window(cfg.win_length or cfg.n_fft, device=x.device, dtype=torch.float32)
        spec = torch.stft(
            x.reshape(b * c, length),
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window=window,
            center=True,
            return_complex=True,
        )
    return spec.reshape(b, c, cfg.freq_bins, spec.shape[-1])


def stft_to_waveform(spec: torch.Tensor, cfg: STFTConfig, length: int | None = None) -> torch.Tensor:
    """Waveform [B, C, T] from complex STFT [B, C, F, frames]. Always runs / returns
    in fp32 (torch.istft doesn't support bf16/half) — safe inside an autocast block."""
    if not torch.is_complex(spec) or spec.ndim != 4:
        raise ValueError(f"Expected complex STFT [B, C, F, T], got {tuple(spec.shape)}")
    b, c, _f, frames = spec.shape
    with torch.amp.autocast(device_type=spec.device.type, enabled=False):
        if spec.dtype != torch.complex64:
            spec = spec.to(torch.complex64)
        window = torch.hann_window(cfg.win_length or cfg.n_fft, device=spec.device, dtype=torch.float32)
        wav = torch.istft(
            spec.reshape(b * c, spec.shape[-2], frames),
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window=window,
            center=True,
            length=length,
        )
    return wav.reshape(b, c, wav.shape[-1])


def complex_to_channels(spec: torch.Tensor) -> torch.Tensor:
    """[B, C, F, T] complex -> [B, 2*C*F, T] real (real bands then imag bands)."""
    if not torch.is_complex(spec) or spec.ndim != 4:
        raise ValueError(f"Expected complex STFT [B, C, F, T], got {tuple(spec.shape)}")
    b, c, f, t = spec.shape
    real = spec.real.reshape(b, c * f, t)
    imag = spec.imag.reshape(b, c * f, t)
    return torch.cat([real, imag], dim=1)


def channels_to_complex(x: torch.Tensor, channels: int, freq_bins: int) -> torch.Tensor:
    """Inverse of complex_to_channels. Casts to fp32 because torch.complex doesn't
    accept bf16/half inputs — keeps this safe inside an autocast block."""
    if x.ndim != 3:
        raise ValueError(f"Expected channel STFT [B, 2*C*F, T], got {tuple(x.shape)}")
    expected = 2 * channels * freq_bins
    if x.shape[1] != expected:
        raise ValueError(f"Expected {expected} STFT channels, got {x.shape[1]}")
    b, _cf, t = x.shape
    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        real, imag = x.split(channels * freq_bins, dim=1)
        real = real.reshape(b, channels, freq_bins, t)
        imag = imag.reshape(b, channels, freq_bins, t)
        return torch.complex(real, imag)


def stft_channels(wav: torch.Tensor, cfg: STFTConfig) -> torch.Tensor:
    """Waveform [B, C, T] -> channelised STFT [B, 2*C*F, frames] (fp32). The single
    waveform->spectrogram crossing for the spec-only backbone."""
    return complex_to_channels(waveform_to_stft(wav, cfg))


def istft_channels(
    channels: torch.Tensor, cfg: STFTConfig, out_channels: int, length: int | None = None
) -> torch.Tensor:
    """Channelised STFT [B, 2*C*F, frames] -> waveform [B, C, T] (fp32). Inverse of
    :func:`stft_channels`."""
    spec = channels_to_complex(channels, out_channels, cfg.freq_bins)
    return stft_to_waveform(spec, cfg, length=length)


def center_crop_or_pad(x: torch.Tensor, length: int) -> torch.Tensor:
    diff = x.shape[-1] - length
    if diff == 0:
        return x
    if diff > 0:
        start = diff // 2
        return x[..., start : start + length]
    left = (-diff) // 2
    return F.pad(x, (left, -diff - left))
