from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio

from backbone.io import STFTConfig, as_waveform, waveform_to_stft

DEFAULT_MR_STFT_RESOLUTIONS = [
    {"n_fft": 1024, "hop_length": 256, "win_length": 1024},
    {"n_fft": 512, "hop_length": 128, "win_length": 512},
    {"n_fft": 256, "hop_length": 64, "win_length": 256},
]

# WaveFM's redesigned resolutions (2503.16689 §3): more uniform window sizes than Parallel WaveGAN.
WAVEFM_RESOLUTIONS = [
    {"n_fft": 1024, "hop_length": 128, "win_length": 512},
    {"n_fft": 2048, "hop_length": 256, "win_length": 1024},
    {"n_fft": 512, "hop_length": 64, "win_length": 256},
]

# Magnitude edge/structure filters on the spectrogram (WaveFM); applied with weights (4, 4, 2).
_GRAD_KERNELS = {
    "t_grad": ([[-1.0, 1.0], [-2.0, 2.0], [-1.0, 1.0]], 0.25),
    "f_grad": ([[-1.0, -2.0, -1.0], [1.0, 2.0, 1.0]], 0.25),
    "laplacian": ([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]], 0.125),
}

_WINDOW_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
_KERNEL_CACHE: dict[tuple[str, torch.dtype], list[torch.Tensor]] = {}
_MEL_CACHE: dict[tuple[int, int, int, str, torch.dtype], torch.Tensor] = {}


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


def _stft(x: torch.Tensor, n_fft: int, hop_length: int, win_length: int) -> torch.Tensor:
    """Complex STFT [.., F, T] of a [.., T] real signal, shared by the spectral losses."""
    window = _cached_hann_window(win_length, x)
    return torch.stft(
        x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True
    )


def _grad_kernels(like: torch.Tensor) -> list[torch.Tensor]:
    key = (str(like.device), like.dtype)
    kernels = _KERNEL_CACHE.get(key)
    if kernels is None:
        kernels = []
        for taps, scale in _GRAD_KERNELS.values():
            k = torch.tensor(taps, device=like.device, dtype=like.dtype) * scale
            kernels.append(k[None, None])
        _KERNEL_CACHE[key] = kernels
    return kernels


def _mel_fbank(n_freqs: int, n_mels: int, sample_rate: int, like: torch.Tensor) -> torch.Tensor:
    key = (n_freqs, n_mels, int(sample_rate), str(like.device), like.dtype)
    fbank = _MEL_CACHE.get(key)
    if fbank is None:
        fbank = torchaudio.functional.melscale_fbanks(
            n_freqs, f_min=0.0, f_max=sample_rate / 2.0, n_mels=n_mels, sample_rate=sample_rate
        ).to(device=like.device, dtype=like.dtype)
        _MEL_CACHE[key] = fbank
    return fbank


def _resolved(resolutions: list[dict] | None) -> list[dict]:
    return resolutions or WAVEFM_RESOLUTIONS


def phase_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    resolutions: list[dict] | None = None,
    mag_floor: float = 1e-6,
    stereo_policy: str = "mean",
) -> torch.Tensor:
    """Anti-wrapping phase loss (WaveFM Eq. 22-23): mean |atan2(sin ΔP, cos ΔP)| over cells where
    both |STFT|² ≥ mag_floor (phase is uninformative where either magnitude is ~0)."""
    pred = _flatten_audio_channels(prediction, stereo_policy)
    tgt = _flatten_audio_channels(target, stereo_policy)
    losses = []
    for cfg in _resolved(resolutions):
        n_fft, hop, win = int(cfg["n_fft"]), int(cfg["hop_length"]), int(cfg.get("win_length", cfg["n_fft"]))
        ps, ts = _stft(pred, n_fft, hop, win), _stft(tgt, n_fft, hop, win)
        delta = torch.angle(ps) - torch.angle(ts)
        wrapped = torch.atan2(torch.sin(delta), torch.cos(delta)).abs()
        mask = ((ps.abs().pow(2) >= mag_floor) & (ts.abs().pow(2) >= mag_floor)).to(wrapped.dtype)
        losses.append((wrapped * mask).sum() / mask.sum().clamp_min(1.0))
    return torch.stack(losses).mean()


def log_magnitude_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    resolutions: list[dict] | None = None,
    stereo_policy: str = "mean",
    eps: float = 1e-7,
) -> torch.Tensor:
    """Multi-resolution log-STFT-magnitude L1."""
    pred = _flatten_audio_channels(prediction, stereo_policy)
    tgt = _flatten_audio_channels(target, stereo_policy)
    losses = []
    for cfg in _resolved(resolutions):
        n_fft, hop, win = int(cfg["n_fft"]), int(cfg["hop_length"]), int(cfg.get("win_length", cfg["n_fft"]))
        pm = _stft(pred, n_fft, hop, win).abs()
        tm = _stft(tgt, n_fft, hop, win).abs()
        losses.append(F.l1_loss(torch.log(pm + eps), torch.log(tm + eps)))
    return torch.stack(losses).mean()


def spectral_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    resolutions: list[dict] | None = None,
    weights: tuple[float, float, float] = (4.0, 4.0, 2.0),
    stereo_policy: str = "mean",
) -> torch.Tensor:
    """MSE between predicted and target magnitude spectrograms after time-gradient, frequency-gradient,
    and Laplacian filters (WaveFM): match the spectrogram's edges/structure, not just its level."""
    pred = _flatten_audio_channels(prediction, stereo_policy)
    tgt = _flatten_audio_channels(target, stereo_policy)
    losses = []
    for cfg in _resolved(resolutions):
        n_fft, hop, win = int(cfg["n_fft"]), int(cfg["hop_length"]), int(cfg.get("win_length", cfg["n_fft"]))
        pm = _stft(pred, n_fft, hop, win).abs().unsqueeze(1)
        tm = _stft(tgt, n_fft, hop, win).abs().unsqueeze(1)
        res = pm.new_zeros(())
        for weight, kernel in zip(weights, _grad_kernels(pm), strict=True):
            res = res + weight * F.mse_loss(F.conv2d(pm, kernel), F.conv2d(tm, kernel))
        losses.append(res)
    return torch.stack(losses).mean()


def mel_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    stereo_policy: str = "mean",
    eps: float = 1e-5,
) -> torch.Tensor:
    """L1 between log-mel spectrograms (HiFi-GAN-style mel reconstruction)."""
    pred = _flatten_audio_channels(prediction, stereo_policy)
    tgt = _flatten_audio_channels(target, stereo_policy)
    pm = _stft(pred, n_fft, hop_length, n_fft).abs()
    tm = _stft(tgt, n_fft, hop_length, n_fft).abs()
    fbank = _mel_fbank(pm.shape[-2], n_mels, sample_rate, pm)
    pmel = torch.log(torch.einsum("fm,bft->bmt", fbank, pm) + eps)
    tmel = torch.log(torch.einsum("fm,bft->bmt", fbank, tm) + eps)
    return F.l1_loss(pmel, tmel)


def complex_stft_loss(
    pred_spec: torch.Tensor,
    target: torch.Tensor,
    stft: STFTConfig | dict,
    energy_weight: bool = True,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Complex L1 between the backbone's raw predicted spectrogram and STFT(target) at the model's
    resolution. Hooks the pre-iSTFT output, so it penalises the inconsistent (incoherent-phase)
    component the internal iSTFT would otherwise project away. Flow2GAN energy-inverse weighting
    (1/√(S+ε), clamped) keeps quiet bins from being drowned by high-energy ones."""
    cfg = stft if isinstance(stft, STFTConfig) else STFTConfig.from_dict(stft)
    tgt_spec = waveform_to_stft(as_waveform(target), cfg).to(pred_spec.dtype)
    diff = pred_spec - tgt_spec
    l1 = diff.real.abs() + diff.imag.abs()
    if energy_weight:
        l1 = l1 * (tgt_spec.abs().pow(2) + eps).rsqrt().clamp(0.01, 100.0)
    return l1.mean()


def wavefm_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int,
    resolutions: list[dict] | None = None,
    mag_floor: float = 1e-6,
    grad_weights: tuple[float, float, float] = (4.0, 4.0, 2.0),
    mel_weight: float = 0.02,
    mel_kwargs: dict | None = None,
    stereo_policy: str = "mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """WaveFM auxiliary bundle as one scalar: anti-wrapping phase + log-magnitude + magnitude
    gradient/Laplacian + λ·mel-L1 (2503.16689). Returns (total, per-component terms for logging)."""
    resolutions = _resolved(resolutions)
    pha = phase_loss(prediction, target, resolutions, mag_floor, stereo_policy)
    lmag = log_magnitude_loss(prediction, target, resolutions, stereo_policy)
    grad = spectral_gradient_loss(prediction, target, resolutions, grad_weights, stereo_policy)
    mel = mel_l1_loss(prediction, target, sample_rate, stereo_policy=stereo_policy, **(mel_kwargs or {}))
    total = pha + lmag + grad + mel_weight * mel
    terms = {"phase": pha.detach(), "log_mag": lmag.detach(), "spectral_grad": grad.detach(), "mel": mel.detach()}
    return total, terms
