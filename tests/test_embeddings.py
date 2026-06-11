from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from emb.clap import CLAPEmbedding
from emb.encodec import EnCodecEmbedding
from emb.vggish import VGGishEmbedding


class _RecordingEncoder(torch.nn.Module):
    def forward(self, audio):
        self.audio = audio
        return audio


@pytest.mark.parametrize("channels", [1, 2])
def test_encodec_48khz_uses_stereo_input(channels):
    embedding = EnCodecEmbedding.__new__(EnCodecEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding.device = torch.device("cpu")
    embedding.input_sample_rate = 48000
    embedding.resampler = None
    embedding.model = torch.nn.Module()
    embedding.model.encoder = _RecordingEncoder()

    audio = torch.randn(2, channels, 16)
    frames = embedding.embed(audio, sample_rate=48000)

    expected = audio.repeat(1, 2, 1) if channels == 1 else audio
    assert torch.equal(embedding.model.encoder.audio, expected)
    assert frames.shape == (2 * 16, 2)


def test_vggish_actual_model_smoke_if_cached():
    hub_dir = Path(torch.hub.get_dir())
    if not any(hub_dir.glob("harritaylor_torchvggish*")):
        pytest.skip("torchvggish is not present in the local torch hub cache")

    try:
        embedding = VGGishEmbedding(device="cpu", input_sample_rate=48000)
    except ImportError as exc:
        pytest.skip(str(exc))

    out = embedding(torch.zeros(1, 1, 48000), sample_rate=48000)

    assert out.shape == (1, embedding.embedding_dim)
    assert torch.isfinite(out).all()


def test_clap_actual_model_smoke_if_checkpoint_is_configured():
    checkpoint_path = os.environ.get("RAW_AUDIO_CLAP_CHECKPOINT")
    if not checkpoint_path:
        pytest.skip("set RAW_AUDIO_CLAP_CHECKPOINT to run the real CLAP smoke test")
    if not Path(checkpoint_path).exists():
        pytest.skip(f"CLAP checkpoint not found: {checkpoint_path}")

    try:
        embedding = CLAPEmbedding(
            device="cpu",
            checkpoint_path=checkpoint_path,
            input_sample_rate=48000,
        )
    except ImportError as exc:
        pytest.skip(str(exc))

    out = embedding(torch.zeros(1, 1, 48000), sample_rate=48000)

    assert out.shape == (1, embedding.embedding_dim)
    assert torch.isfinite(out).all()
    assert not any("text" in name for name, _ in embedding.named_parameters())
