from __future__ import annotations

import torch

from emb.random import RandomProjEmbedding
from eval.audio_metrics import _covariance, frechet_audio_distance, frechet_from_moments
from losses.dist import FrechetLoss, compute_real_moments

SR = 8000


def _embedder(dim: int = 16, seed: int = 0) -> RandomProjEmbedding:
    return RandomProjEmbedding(
        embedding_dim=dim, n_fft=64, hop_length=16, input_sample_rate=SR, seed=seed, device="cpu"
    )


def _moments(embedder, audio):
    return compute_real_moments(embedder, [{"audio": audio}], sample_rate=SR)


def test_real_moments_include_the_final_short_batch():
    embedder = _embedder(dim=8)
    audio = torch.randn(5, 1, 256)
    batches = [{"audio": audio[:3]}, {"audio": audio[3:]}]

    expected = _moments(embedder, audio)
    actual = compute_real_moments(embedder, batches, sample_rate=SR)

    assert torch.allclose(actual[0], expected[0])
    assert torch.allclose(actual[1], expected[1])


def test_frechet_from_moments_matches_embedding_path():
    torch.manual_seed(0)
    real, fake = torch.randn(32, 16), torch.randn(32, 16)
    reference = frechet_audio_distance(real, fake)["fad"]
    got = frechet_from_moments(real.mean(0), _covariance(real), fake.mean(0), _covariance(fake))["fad"]
    assert torch.allclose(got, reference, atol=1e-5)


def test_fd_loss_backprops_to_generated_audio():
    torch.manual_seed(0)
    embedder = _embedder()
    real = torch.randn(8, 1, 256)
    loss_fn = FrechetLoss([embedder], [_moments(embedder, real)], mode="ema", sample_rate=SR)

    fake = torch.randn(8, 1, 256, requires_grad=True)
    loss, terms = loss_fn(fake)
    loss.backward()

    assert torch.isfinite(loss)
    assert "fd/random" in terms
    assert fake.grad is not None and torch.isfinite(fake.grad).all()
    assert fake.grad.abs().sum() > 0


def test_queue_estimator_is_finite_and_differentiable():
    torch.manual_seed(0)
    embedder = _embedder()
    real = torch.randn(8, 1, 256)
    loss_fn = FrechetLoss([embedder], [_moments(embedder, real)], mode="queue", queue_size=64, sample_rate=SR)

    fake = torch.randn(8, 1, 256, requires_grad=True)
    loss, _ = loss_fn(fake)
    loss.backward()
    assert torch.isfinite(loss) and fake.grad is not None and torch.isfinite(fake.grad).all()


def test_fd_is_much_smaller_when_distributions_match():
    torch.manual_seed(0)
    embedder = _embedder()
    real = torch.randn(16, 1, 256)
    moments = _moments(embedder, real)

    matched = FrechetLoss([embedder], [moments], mode="ema", sample_rate=SR)(real)[1]["fd/random"]
    mismatched = FrechetLoss([embedder], [moments], mode="ema", sample_rate=SR)(
        3.0 * torch.randn(16, 1, 256) + 5.0
    )[1]["fd/random"]

    # Embedding the reference population itself gives a near-zero FD, far below a shifted batch.
    assert matched < 0.05
    assert matched < 0.05 * mismatched


def test_multi_embedder_aggregation_is_unit_scale():
    torch.manual_seed(0)
    embedders = [_embedder(dim=16, seed=1), _embedder(dim=8, seed=2)]
    embedders[0].name, embedders[1].name = "phi_a", "phi_b"  # distinct φ in practice (CLAP, MATPAC, …)
    real = torch.randn(8, 1, 256)
    moments = [_moments(emb, real) for emb in embedders]
    loss_fn = FrechetLoss(embedders, moments, mode="ema", c=1e-3, sample_rate=SR)

    loss, terms = loss_fn(torch.randn(8, 1, 256))

    # Each self-normalised term sits below 1, so the aggregate stays O(num embedders) regardless of
    # the two backbones' feature scales.
    assert set(terms) == {"fd/phi_a", "fd/phi_b"}
    assert torch.isfinite(loss) and 0.0 < float(loss) < len(embedders)
