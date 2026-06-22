"""Gradient-flow tests for the pretrained embedding wrappers used by FD-loss.

Each wrapper's ``embed`` must be differentiable end-to-end (audio -> features) so the Frechet
loss can backprop into generated audio. These load real checkpoints; they ``importorskip`` the
optional backend and skip gracefully when the checkpoint cannot be downloaded (offline CI).
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

INPUT_SR = 48000
DURATION_S = 3


def _assert_differentiable(emb, *, clip_level: bool = True) -> None:
    audio = torch.randn(2, 1, INPUT_SR * DURATION_S, requires_grad=True)
    feats = emb.embed(audio, sample_rate=INPUT_SR)
    assert feats.ndim == 2 and feats.shape[1] == emb.embedding_dim
    if clip_level:
        assert feats.shape[0] == 2
    assert torch.isfinite(feats).all()
    feats.sum().backward()
    assert audio.grad is not None
    assert torch.isfinite(audio.grad).all()
    assert audio.grad.abs().sum() > 0


def _build_or_skip(factory):
    try:
        return factory()
    except Exception as exc:  # noqa: BLE001 - network/checkpoint failures should skip, not fail
        pytest.skip(f"could not build embedder (checkpoint/network?): {exc}")


def test_pann_differentiable():
    pytest.importorskip("panns_inference")
    from emb.pann import PANNEmbedding

    _assert_differentiable(_build_or_skip(lambda: PANNEmbedding(device="cpu", input_sample_rate=INPUT_SR)))


def test_encodec_differentiable():
    pytest.importorskip("transformers")
    from emb.encodec import EnCodecEmbedding

    _assert_differentiable(
        _build_or_skip(lambda: EnCodecEmbedding(device="cpu", input_sample_rate=INPUT_SR)),
        clip_level=False,
    )


def test_mert_differentiable():
    pytest.importorskip("transformers")
    from emb.mert import MERTEmbedding

    _assert_differentiable(_build_or_skip(lambda: MERTEmbedding(device="cpu", input_sample_rate=INPUT_SR)))


def test_audiomae_differentiable():
    pytest.importorskip("timm")
    from emb.audiomae import AudioMAEEmbedding

    _assert_differentiable(_build_or_skip(lambda: AudioMAEEmbedding(device="cpu", input_sample_rate=INPUT_SR)))


def test_clap_differentiable():
    if importlib.util.find_spec("laion_clap") is None:
        pytest.skip("laion_clap is not installed")
    from emb.clap import CLAPEmbedding

    _assert_differentiable(_build_or_skip(lambda: CLAPEmbedding(device="cpu", input_sample_rate=INPUT_SR)))


def test_vggish_differentiable():
    from emb.vggish import VGGishEmbedding

    _assert_differentiable(_build_or_skip(lambda: VGGishEmbedding(device="cpu", input_sample_rate=INPUT_SR)))


def test_cdpam_differentiable():
    pytest.importorskip("cdpam")
    from emb.cdpam import CDPAMEmbedding

    _assert_differentiable(_build_or_skip(lambda: CDPAMEmbedding(device="cpu", input_sample_rate=INPUT_SR)))
