from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_MR_STFT_RESOLUTIONS = [
    {"n_fft": 1024, "hop_length": 256, "win_length": 1024},
    {"n_fft": 512, "hop_length": 128, "win_length": 512},
    {"n_fft": 256, "hop_length": 64, "win_length": 256},
]

_WINDOW_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def _cached_hann_window(win_length: int, like: torch.Tensor) -> torch.Tensor:
    key = (int(win_length), str(like.device), like.dtype)
    window = _WINDOW_CACHE.get(key)
    if window is None:
        window = torch.hann_window(int(win_length), device=like.device, dtype=like.dtype)
        _WINDOW_CACHE[key] = window
    return window


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
