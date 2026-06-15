from __future__ import annotations

import torch

from backbone.io import STFTConfig, waveform_to_stft
from losses.audio import (
    complex_stft_loss,
    log_magnitude_loss,
    mel_l1_loss,
    mr_stft_loss,
    phase_loss,
    spectral_gradient_loss,
    wavefm_loss,
)

_RES = [
    {"n_fft": 256, "hop_length": 64, "win_length": 256},
    {"n_fft": 128, "hop_length": 32, "win_length": 128},
]


def _tone(batch: int = 2, length: int = 2048, freq: float = 220.0) -> torch.Tensor:
    t = torch.arange(length, dtype=torch.float32) / 8000.0
    return torch.sin(2.0 * torch.pi * freq * t).expand(batch, 1, length).contiguous()


def test_mr_stft_loss_supports_resolutions_log_term_and_stereo_channels():
    pred = torch.randn(2, 2, 64, requires_grad=True)
    target = torch.randn(2, 2, 64)
    loss = mr_stft_loss(
        pred,
        target,
        resolutions=[
            {"n_fft": 16, "hop_length": 8, "win_length": 16},
            {"n_fft": 8, "hop_length": 4, "win_length": 8},
        ],
        log_weight=0.25,
        stereo_policy="channels",
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_phase_loss_zero_for_identical_and_positive_otherwise():
    target = _tone()
    assert phase_loss(target.clone(), target, resolutions=_RES) < 1e-6
    other = _tone(freq=330.0).requires_grad_(True)
    loss = phase_loss(other, target, resolutions=_RES)
    loss.backward()
    assert loss > 0
    assert torch.isfinite(other.grad).all()


def test_log_magnitude_loss_zero_for_identical_and_backprops():
    target = _tone()
    assert log_magnitude_loss(target.clone(), target, resolutions=_RES) < 1e-5
    pred = torch.randn(2, 1, 2048, requires_grad=True)
    loss = log_magnitude_loss(pred, target, resolutions=_RES)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()


def test_spectral_gradient_loss_zero_for_identical_and_backprops():
    target = _tone()
    assert spectral_gradient_loss(target.clone(), target, resolutions=_RES) < 1e-5
    pred = torch.randn(2, 1, 2048, requires_grad=True)
    loss = spectral_gradient_loss(pred, target, resolutions=_RES)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()


def test_mel_l1_loss_zero_for_identical_and_backprops():
    target = _tone()
    assert mel_l1_loss(target.clone(), target, sample_rate=8000, n_fft=256, hop_length=64, n_mels=20) < 1e-4
    pred = torch.randn(2, 1, 2048, requires_grad=True)
    loss = mel_l1_loss(pred, target, sample_rate=8000, n_fft=256, hop_length=64, n_mels=20)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()


def test_complex_stft_loss_zero_when_spec_matches_target():
    cfg = STFTConfig(n_fft=256, hop_length=64, win_length=256)
    target = _tone()
    spec = waveform_to_stft(target, cfg).requires_grad_(True)
    assert complex_stft_loss(spec.detach(), target, cfg) < 1e-6
    # a mismatched spec gives a positive, differentiable loss
    loss = complex_stft_loss(spec, target * 0.5, cfg)
    loss.backward()
    assert loss > 0 and torch.isfinite(spec.grad).all()


def test_wavefm_loss_bundles_components_and_backprops():
    target = _tone()
    pred = torch.randn(2, 1, 2048, requires_grad=True)
    total, terms = wavefm_loss(pred, target, sample_rate=8000, resolutions=_RES, mel_kwargs={"n_fft": 256, "hop_length": 64, "n_mels": 20})
    total.backward()
    assert torch.isfinite(total) and total > 0
    assert {"phase", "log_mag", "spectral_grad", "mel"} <= set(terms)
    assert torch.isfinite(pred.grad).all()
    # identical prediction drives the bundle to ~0
    near_zero, _ = wavefm_loss(target.clone(), target, sample_rate=8000, resolutions=_RES, mel_kwargs={"n_fft": 256, "hop_length": 64, "n_mels": 20})
    assert near_zero < 1e-3
