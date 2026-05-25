from __future__ import annotations

import torch

from losses.audio import mr_stft_loss


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
