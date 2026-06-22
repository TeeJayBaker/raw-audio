from __future__ import annotations

import torch

from backbone.io import STFTConfig, waveform_to_stft
from losses.spec import complex_stft_loss, consistency_residual

_CFG = STFTConfig(n_fft=256, hop_length=64, win_length=256)


def _tone(batch: int = 2, length: int = 2048, freq: float = 220.0) -> torch.Tensor:
    t = torch.arange(length, dtype=torch.float32) / 8000.0
    return torch.sin(2.0 * torch.pi * freq * t).expand(batch, 1, length).contiguous()


def test_complex_stft_loss_zero_when_spec_matches_target():
    target = _tone()
    spec = waveform_to_stft(target, _CFG).requires_grad_(True)
    assert complex_stft_loss(spec.detach(), target, _CFG) < 1e-6
    # a mismatched spec gives a positive, differentiable loss
    loss = complex_stft_loss(spec, target * 0.5, _CFG)
    loss.backward()
    assert loss > 0 and torch.isfinite(spec.grad).all()


def test_consistency_residual_near_zero_for_valid_stft():
    length = 2048
    spec = waveform_to_stft(_tone(length=length), _CFG)
    # a real signal's STFT lies on the consistent manifold -> residual ~ 0
    assert consistency_residual(spec, _CFG, length=length) < 1e-3


def test_consistency_residual_positive_for_inconsistent_spec():
    spec = waveform_to_stft(_tone(length=2048), _CFG)
    noisy = spec + 0.5 * spec.abs().mean() * torch.randn_like(spec)
    # random off-manifold perturbation -> the iSTFT projects part of it away -> positive residual
    assert consistency_residual(noisy, _CFG, length=2048) > 0.05


def test_consistency_residual_scale_invariant_and_backprops():
    length = 2048
    spec = (1.0 + 0.3 * torch.randn(2, 1, _CFG.freq_bins, length // _CFG.hop_length + 1)).to(
        torch.complex64
    )
    spec.requires_grad_(True)
    base = consistency_residual(spec, _CFG, length=length)
    scaled = consistency_residual(7.0 * spec.detach(), _CFG, length=length)
    assert torch.allclose(base.detach(), scaled, atol=1e-5)  # relative metric -> scale-free
    base.backward()
    assert torch.isfinite(spec.grad).all()
