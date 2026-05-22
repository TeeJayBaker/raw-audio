from __future__ import annotations

import torch

from flow.fm import linear_interpolant
from losses.audio import (
    FMLoss,
    mr_stft_loss,
    smoothed_linear_power_spectrogram,
    spectral_energy_inverse_weight,
    spectral_energy_weighted_loss,
)


def test_spectral_energy_inverse_weight_uses_filterbank_and_clamp():
    audio = torch.zeros(2, 2, 64)
    weights = spectral_energy_inverse_weight(
        audio,
        n_fft=16,
        hop_length=8,
        win_length=16,
        filterbank_bins=5,
        smooth_kernel_size=3,
        stereo_policy="channels",
    )
    assert weights.shape[:2] == (4, 5)
    assert torch.all(weights == 100.0)


def test_smoothed_linear_power_spectrogram_has_audio_gradients():
    audio = torch.randn(2, 1, 64, requires_grad=True)
    power = smoothed_linear_power_spectrogram(
        audio,
        n_fft=16,
        hop_length=8,
        win_length=16,
        filterbank_bins=6,
    )
    power.mean().backward()
    assert audio.grad is not None
    assert torch.isfinite(audio.grad).all()


def test_spectral_energy_weighted_loss_has_prediction_gradients():
    pred = torch.randn(2, 1, 64, requires_grad=True)
    target = torch.randn(2, 1, 64)
    loss = spectral_energy_weighted_loss(
        pred - target,
        target,
        n_fft=16,
        hop_length=8,
        win_length=16,
        filterbank_bins=6,
    )
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


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


def test_fm_loss_spectral_term_is_dedicated_not_waveform_mse_weighting():
    x1 = torch.randn(2, 1, 64)
    flow_batch = linear_interpolant(x1)
    pred = torch.randn_like(x1, requires_grad=True)
    spectral_cfg = {"n_fft": 16, "hop_length": 8, "win_length": 16, "filterbank_bins": 6}

    base = FMLoss(spectral_energy_weight=0.0)(pred, flow_batch)
    weighted = FMLoss(
        spectral_energy_weight=0.5,
        spectral_energy_resolution=spectral_cfg,
    )(pred, flow_batch)

    assert torch.allclose(base.terms["x_mse"], weighted.terms["x_mse"])
    assert "spectral_energy_weighted" in weighted.terms
    weighted.total.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
