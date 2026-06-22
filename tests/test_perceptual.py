"""Tests for the CDPAM perceptual loss and its differentiable distance head.

CDPAM serves two roles: a learned perceptual distance (the LPIPS-analog aux loss) and a
frozen embedder for FD-loss. Both must carry gradients into the generated waveform. These load
the real bundled checkpoint and skip gracefully if the optional ``cdpam`` backend is missing.
"""

from __future__ import annotations

import pytest
import torch

INPUT_SR = 48000


@pytest.fixture(scope="module")
def cdpam_emb():
    pytest.importorskip("cdpam")
    from emb.cdpam import CDPAMEmbedding

    try:
        return CDPAMEmbedding(device="cpu", input_sample_rate=INPUT_SR)
    except Exception as exc:  # noqa: BLE001 - checkpoint/network failures should skip, not fail
        pytest.skip(f"could not build CDPAM embedder (checkpoint?): {exc}")


def _audio(batch: int = 2, seconds: int = 1) -> torch.Tensor:
    return torch.randn(batch, 1, INPUT_SR * seconds)


def test_distance_is_per_sample_nonnegative_and_differentiable(cdpam_emb):
    pred = _audio().requires_grad_(True)
    target = _audio()
    dist = cdpam_emb.distance(pred, target)
    assert dist.shape == (2,)
    assert torch.isfinite(dist).all() and (dist >= 0).all()
    dist.sum().backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_distance_zero_for_identical_inputs(cdpam_emb):
    audio = _audio()
    assert cdpam_emb.distance(audio, audio).abs().max() < 1e-4


def test_cdpam_loss_is_scalar_and_differentiable(cdpam_emb):
    from losses.perceptual import cdpam_loss

    pred = _audio().requires_grad_(True)
    target = _audio()
    loss = cdpam_loss(cdpam_emb, pred, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss) and loss >= 0
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_factory_builds_cdpam(cdpam_emb):
    from emb.factory import build_embedding

    emb = build_embedding({"type": "cdpam", "input_sample_rate": INPUT_SR}, device="cpu")
    assert emb is not None and emb.name == "cdpam" and emb.embedding_dim == 512
