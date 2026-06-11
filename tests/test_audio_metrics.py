from __future__ import annotations

import pytest
import torch

from eval.audio_metrics import (
    density_coverage,
    embedding_cosine_score,
    frechet_audio_distance,
    kernel_audio_distance,
    monge_audio_distance,
    vendi_score,
)


def test_embedding_cosine_accepts_frozen_backend():
    class MeanBackend:
        def __init__(self):
            self.training = True
            self.was_eval = False

        def eval(self):
            self.was_eval = True
            self.training = False

        def train(self):
            self.training = True

        def embed_audio(self, audio, sample_rate: int):
            assert sample_rate == 16000
            return torch.stack([audio.mean(dim=-1), audio.std(dim=-1)], dim=-1)

    backend = MeanBackend()
    audio = torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
    score = embedding_cosine_score(audio, audio, embedding_backend=backend, sample_rate=16000)
    assert score == pytest.approx(1.0)
    assert backend.was_eval
    assert backend.training


def test_fd_and_mind_are_near_zero_for_identical_embeddings():
    embeddings = torch.tensor(
        [
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    generator = torch.Generator().manual_seed(0)

    fad = frechet_audio_distance(embeddings, embeddings)["fad"]
    mind = monge_audio_distance(embeddings, embeddings, projections=32, generator=generator)["mind"]

    assert fad.item() == pytest.approx(0.0, abs=1e-5)
    assert mind.item() == pytest.approx(0.0, abs=1e-7)


def test_fd_and_mind_increase_for_shifted_distribution():
    real = torch.tensor(
        [
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    fake = real + torch.tensor([2.0, -1.0])

    generator = torch.Generator().manual_seed(123)
    fad = frechet_audio_distance(real, fake)["fad"]
    mind = monge_audio_distance(real, fake, projections=64, generator=generator)["mind"]

    assert fad.item() > 4.9
    assert mind.item() > 0.1


def test_kad_matches_unbiased_rbf_mmd():
    real = torch.tensor([[0.0], [2.0], [5.0]])
    fake = torch.tensor([[1.0], [3.0], [4.0]])
    bandwidth = 2.0
    gamma = 1.0 / (2.0 * bandwidth**2 + 1e-8)

    def kernel(x, y):
        return torch.exp(-gamma * torch.cdist(x, y).square())

    k_real = kernel(real, real)
    k_fake = kernel(fake, fake)
    expected = 100.0 * (
        (k_real.sum() - k_real.diagonal().sum()) / 6
        + (k_fake.sum() - k_fake.diagonal().sum()) / 6
        - 2.0 * kernel(real, fake).mean()
    )
    result = kernel_audio_distance(real, fake, bandwidth=bandwidth)

    assert result["kad"].item() == pytest.approx(expected.item(), abs=1e-6)
    assert result["bandwidth"].item() == pytest.approx(bandwidth)


def test_kad_uses_reference_median_bandwidth_and_detects_shift():
    real = torch.tensor([[0.0], [1.0], [3.0], [6.0]])
    same = kernel_audio_distance(real, real)
    shifted = kernel_audio_distance(real, real + 20.0)

    assert same["bandwidth"].item() == pytest.approx(torch.pdist(real).median().item())
    assert shifted["kad"].item() > same["kad"].item()


def test_kad_requires_two_embeddings_per_set():
    with pytest.raises(ValueError, match="at least two"):
        kernel_audio_distance(torch.zeros(1, 2), torch.zeros(2, 2))


def test_vendi_score_detects_collapse_vs_orthogonal_diversity():
    diverse = torch.eye(4)
    collapsed = torch.ones(4, 3)

    assert vendi_score(diverse).item() == pytest.approx(4.0, abs=1e-5)
    assert vendi_score(collapsed).item() == pytest.approx(1.0, abs=1e-5)


def test_density_coverage_identity_and_mode_drop():
    real = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ]
    )
    same = density_coverage(real, real, k=1)
    collapsed = density_coverage(real, real[:1].repeat(4, 1), k=1)

    assert same["coverage"].item() == pytest.approx(1.0)
    assert torch.isfinite(same["density"])
    assert same["density"].item() > 0.0
    assert collapsed["coverage"].item() < same["coverage"].item()
    assert torch.isfinite(collapsed["density"])


def test_metrics_validate_embedding_shapes():
    real = torch.randn(4, 3)
    fake = torch.randn(4, 2)

    with pytest.raises(ValueError, match="same embedding dimension"):
        frechet_audio_distance(real, fake)
    with pytest.raises(ValueError, match="shape"):
        vendi_score(torch.randn(4, 2, 1))
