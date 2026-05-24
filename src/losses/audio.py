from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_MR_STFT_RESOLUTIONS = [
    {"n_fft": 1024, "hop_length": 256, "win_length": 1024},
    {"n_fft": 512, "hop_length": 128, "win_length": 512},
    {"n_fft": 256, "hop_length": 64, "win_length": 256},
]

_WINDOW_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
_FILTERBANK_CACHE: dict[tuple[int, int, str, torch.dtype], torch.Tensor] = {}


def _cached_hann_window(win_length: int, like: torch.Tensor) -> torch.Tensor:
    key = (int(win_length), str(like.device), like.dtype)
    window = _WINDOW_CACHE.get(key)
    if window is None:
        window = torch.hann_window(int(win_length), device=like.device, dtype=like.dtype)
        _WINDOW_CACHE[key] = window
    return window


def _linear_filterbank(freq_bins: int, out_bins: int, like: torch.Tensor) -> torch.Tensor:
    if out_bins <= 0:
        raise ValueError("filterbank_bins must be positive")
    key = (int(freq_bins), int(out_bins), str(like.device), like.dtype)
    fb = _FILTERBANK_CACHE.get(key)
    if fb is not None:
        return fb

    if out_bins == freq_bins:
        fb = torch.eye(freq_bins, device=like.device, dtype=like.dtype)
    else:
        positions = torch.linspace(0, freq_bins - 1, out_bins + 2, device=like.device, dtype=like.dtype)
        freqs = torch.arange(freq_bins, device=like.device, dtype=like.dtype)
        rows = []
        for i in range(out_bins):
            left, center, right = positions[i], positions[i + 1], positions[i + 2]
            up = (freqs - left) / (center - left).clamp_min(torch.finfo(like.dtype).eps)
            down = (right - freqs) / (right - center).clamp_min(torch.finfo(like.dtype).eps)
            row = torch.minimum(up, down).clamp_min(0.0)
            rows.append(row / row.sum().clamp_min(torch.finfo(like.dtype).eps))
        fb = torch.stack(rows, dim=0)
    _FILTERBANK_CACHE[key] = fb
    return fb


def _flatten_audio_channels(audio: torch.Tensor, stereo_policy: str) -> torch.Tensor:
    if audio.ndim == 2:
        return audio
    if audio.ndim != 3:
        raise ValueError("audio tensors must have shape (batch, time) or (batch, channels, time)")
    if stereo_policy == "mean":
        return audio.mean(dim=1)
    if stereo_policy == "first":
        return audio[:, 0]
    if stereo_policy == "channels":
        return audio.reshape(audio.shape[0] * audio.shape[1], audio.shape[2])
    raise ValueError("stereo_policy must be 'mean', 'first', or 'channels'")


def _power_spectrogram(
    audio: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int | None,
    stereo_policy: str,
) -> torch.Tensor:
    wav = _flatten_audio_channels(audio, stereo_policy)
    win_length = int(win_length or n_fft)
    window = _cached_hann_window(win_length, wav)
    spec = torch.stft(
        wav,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    return spec.abs().square()


def _smooth_power(power: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return power
    if kernel_size % 2 == 0:
        raise ValueError("smooth_kernel_size must be odd")
    pad = kernel_size // 2
    padded = F.pad(power[:, None], (pad, pad, pad, pad), mode="replicate")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)[:, 0]


def smoothed_linear_power_spectrogram(
    audio: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int | None = None,
    filterbank_bins: int | None = None,
    smooth_kernel_size: int = 3,
    stereo_policy: str = "mean",
) -> torch.Tensor:
    """Return Flow2GAN's S(x): smoothed power STFT projected by a linear filterbank."""
    power = _power_spectrogram(audio, n_fft, hop_length, win_length, stereo_policy)
    smooth = _smooth_power(power, smooth_kernel_size)
    fb = _linear_filterbank(smooth.shape[-2], int(filterbank_bins or smooth.shape[-2]), smooth)
    return torch.einsum("mf,bft->bmt", fb, smooth)


def spectral_energy_inverse_weight(
    reference: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int | None = None,
    filterbank_bins: int | None = None,
    smooth_kernel_size: int = 3,
    stereo_policy: str = "mean",
    eps: float = 1e-5,
    clamp_min: float = 0.01,
    clamp_max: float = 100.0,
) -> torch.Tensor:
    """Flow2GAN inverse spectral-energy weights on the smoothed filterbank grid."""
    energy = smoothed_linear_power_spectrogram(
        reference,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        filterbank_bins=filterbank_bins,
        smooth_kernel_size=smooth_kernel_size,
        stereo_policy=stereo_policy,
    )
    return (energy + eps).rsqrt().clamp(clamp_min, clamp_max)


def spectral_energy_weighted_loss(
    error: torch.Tensor,
    reference: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int | None = None,
    filterbank_bins: int | None = None,
    smooth_kernel_size: int = 3,
    stereo_policy: str = "mean",
    eps: float = 1e-5,
) -> torch.Tensor:
    """Flow2GAN spectral term: mean S(error) / sqrt(S(reference) + eps)."""
    error_energy = smoothed_linear_power_spectrogram(
        error,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        filterbank_bins=filterbank_bins,
        smooth_kernel_size=smooth_kernel_size,
        stereo_policy=stereo_policy,
    )
    weights = spectral_energy_inverse_weight(
        reference,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        filterbank_bins=filterbank_bins,
        smooth_kernel_size=smooth_kernel_size,
        stereo_policy=stereo_policy,
        eps=eps,
    ).detach()
    return (error_energy * weights).mean()


def mr_stft_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    resolutions: list[dict] | None = None,
    log_weight: float = 0.0,
    stereo_policy: str = "mean",
    eps: float = 1e-7,
) -> torch.Tensor:
    """Multi-resolution STFT loss with spectral convergence and magnitude terms."""
    resolutions = resolutions or DEFAULT_MR_STFT_RESOLUTIONS
    pred = _flatten_audio_channels(prediction, stereo_policy)
    tgt = _flatten_audio_channels(target, stereo_policy)
    if pred.shape != tgt.shape:
        raise ValueError("prediction and target must have matching shape after stereo_policy")
    losses = []
    for cfg in resolutions:
        n_fft = int(cfg["n_fft"])
        hop_length = int(cfg["hop_length"])
        win_length = int(cfg.get("win_length", n_fft))
        window = _cached_hann_window(win_length, pred)
        pred_mag = torch.stft(
            pred,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        ).abs()
        tgt_mag = torch.stft(
            tgt,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        ).abs()
        sc = torch.linalg.vector_norm(tgt_mag - pred_mag) / torch.linalg.vector_norm(tgt_mag).clamp_min(eps)
        mag = F.l1_loss(pred_mag, tgt_mag)
        if log_weight:
            mag = mag + float(log_weight) * F.l1_loss(torch.log(pred_mag + eps), torch.log(tgt_mag + eps))
        losses.append(sc + mag)
    return torch.stack(losses).mean()
