from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

EmbeddingBackend = Callable[..., torch.Tensor]


def _as_embeddings(
    inputs: torch.Tensor,
    embedding_backend: EmbeddingBackend | None = None,
    **backend_kwargs: object,
) -> torch.Tensor:
    if embedding_backend is None:
        embeddings = inputs
    else:
        was_training = getattr(embedding_backend, "training", None)
        if hasattr(embedding_backend, "eval"):
            embedding_backend.eval()
        with torch.no_grad():
            if hasattr(embedding_backend, "embed_audio"):
                embeddings = embedding_backend.embed_audio(inputs, **backend_kwargs)
            else:
                embeddings = embedding_backend(inputs, **backend_kwargs)
        if was_training is True and hasattr(embedding_backend, "train"):
            embedding_backend.train()

    if embeddings.ndim != 2:
        raise ValueError("audio metrics expect embeddings with shape [N, D]")
    if embeddings.shape[0] == 0:
        raise ValueError("audio metrics require at least one embedding")
    if not torch.is_floating_point(embeddings):
        embeddings = embeddings.float()
    return embeddings


def _validate_pair(real_embeddings: torch.Tensor, fake_embeddings: torch.Tensor) -> None:
    if real_embeddings.ndim != 2 or fake_embeddings.ndim != 2:
        raise ValueError("real and fake embeddings must have shape [N, D]")
    if real_embeddings.shape[1] != fake_embeddings.shape[1]:
        raise ValueError("real and fake embeddings must have the same embedding dimension")
    if real_embeddings.shape[0] == 0 or fake_embeddings.shape[0] == 0:
        raise ValueError("real and fake embeddings must be non-empty")


def embedding_cosine_score(
    real_embeddings: torch.Tensor,
    fake_embeddings: torch.Tensor,
    embedding_backend: EmbeddingBackend | None = None,
    **backend_kwargs: object,
) -> torch.Tensor:
    """Mean paired cosine similarity for frozen audio embeddings.

    Inputs may be precomputed [N, D] embeddings, or audio tensors when a lightweight
    backend/callable returning [N, D] embeddings is supplied.
    """
    real_embeddings = _as_embeddings(real_embeddings, embedding_backend, **backend_kwargs)
    fake_embeddings = _as_embeddings(fake_embeddings, embedding_backend, **backend_kwargs)
    if real_embeddings.shape != fake_embeddings.shape:
        raise ValueError("real_embeddings and fake_embeddings must have the same shape")
    return F.cosine_similarity(real_embeddings, fake_embeddings, dim=-1).mean()


def _covariance(embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.shape[0] < 2:
        return embeddings.new_zeros((embeddings.shape[1], embeddings.shape[1]))
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    return centered.T @ centered / (embeddings.shape[0] - 1)


def _symmetric_matrix_sqrt(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    matrix = (matrix + matrix.T) * 0.5
    evals, evecs = torch.linalg.eigh(matrix)
    evals = torch.where(evals > eps, evals, evals.new_zeros(()))
    return (evecs * evals.sqrt().unsqueeze(0)) @ evecs.T


def frechet_from_moments(
    mu_real: torch.Tensor,
    cov_real: torch.Tensor,
    mu_fake: torch.Tensor,
    cov_fake: torch.Tensor,
    eps: float = 1e-6,
    cov_real_sqrt: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Gaussian 2-Wasserstein distance from precomputed moments. Differentiable in every argument.

    Follows the FD-loss reference (arXiv 2604.28190): the cross term ``Tr((Σ_r Σ_g)^½)`` is the sum of
    square roots of the eigenvalues of ``Σ_r^½ Σ_g Σ_r^½`` via ``eigvalsh`` — no eigenvectors, so the
    backward is stable on rank-deficient covariances (unlike an ``eigh`` matrix-sqrt). ``cov_real_sqrt``
    lets the caller pass a precomputed ``Σ_r^½`` (the real side is fixed in the loss).
    """
    out_dtype = mu_real.dtype
    # The eigendecomposition is ill-conditioned on high-dim, near-rank-deficient encoder
    # covariances (CLAP/MATPAC); run it in float64 regardless of input dtype, cast results back.
    dtype = torch.float64
    mu_real, cov_real = mu_real.to(dtype), cov_real.to(dtype)
    mu_fake, cov_fake = mu_fake.to(dtype), cov_fake.to(dtype)
    sqrt_cov_real = cov_real_sqrt.to(dtype) if cov_real_sqrt is not None else _symmetric_matrix_sqrt(cov_real, eps)

    mean_term = (mu_real - mu_fake).pow(2).sum()
    middle = sqrt_cov_real @ cov_fake @ sqrt_cov_real
    evals = torch.linalg.eigvalsh((middle + middle.T) * 0.5)
    # Floor the (numerically near-zero) eigenvalues so √· has a finite gradient on rank-deficient
    # covariances; negligible on well-conditioned ones.
    tr_covmean = evals.clamp_min(1e-12).sqrt().sum()
    cov_term = torch.trace(cov_real) + torch.trace(cov_fake) - 2.0 * tr_covmean
    return {
        "fad": (mean_term + cov_term).clamp_min(0.0).to(out_dtype),
        "mean": mean_term.to(out_dtype),
        "covariance": cov_term.clamp_min(0.0).to(out_dtype),
    }


def frechet_audio_distance(
    real_embeddings: torch.Tensor,
    fake_embeddings: torch.Tensor,
    embedding_backend: EmbeddingBackend | None = None,
    eps: float = 1e-6,
    **backend_kwargs: object,
) -> dict[str, torch.Tensor]:
    """FAD/FD-style Gaussian 2-W distance in audio embedding space."""
    real_embeddings = _as_embeddings(real_embeddings, embedding_backend, **backend_kwargs)
    fake_embeddings = _as_embeddings(fake_embeddings, embedding_backend, **backend_kwargs)
    _validate_pair(real_embeddings, fake_embeddings)

    result = frechet_from_moments(
        real_embeddings.mean(dim=0),
        _covariance(real_embeddings),
        fake_embeddings.mean(dim=0),
        _covariance(fake_embeddings),
        eps,
    )
    result["n_real"] = torch.tensor(real_embeddings.shape[0], device=real_embeddings.device)
    result["n_fake"] = torch.tensor(fake_embeddings.shape[0], device=fake_embeddings.device)
    return result


def monge_audio_distance(
    real_embeddings: torch.Tensor,
    fake_embeddings: torch.Tensor,
    projections: int = 256,
    embedding_backend: EmbeddingBackend | None = None,
    generator: torch.Generator | None = None,
    squared: bool = True,
    **backend_kwargs: object,
) -> dict[str, torch.Tensor]:
    """MIND/Monge-style sliced 1-D Wasserstein over audio embeddings."""
    if projections <= 0:
        raise ValueError("projections must be positive")
    real_embeddings = _as_embeddings(real_embeddings, embedding_backend, **backend_kwargs)
    fake_embeddings = _as_embeddings(fake_embeddings, embedding_backend, **backend_kwargs)
    _validate_pair(real_embeddings, fake_embeddings)

    directions = torch.randn(
        real_embeddings.shape[1],
        projections,
        device=real_embeddings.device,
        dtype=real_embeddings.dtype,
        generator=generator,
    )
    directions = F.normalize(directions, dim=0)
    real_projected = (real_embeddings @ directions).sort(dim=0).values
    fake_projected = (fake_embeddings @ directions).sort(dim=0).values

    if real_projected.shape[0] != fake_projected.shape[0]:
        n_quantiles = max(real_projected.shape[0], fake_projected.shape[0])
        real_projected = _interpolate_sorted_quantiles(real_projected, n_quantiles)
        fake_projected = _interpolate_sorted_quantiles(fake_projected, n_quantiles)

    delta = real_projected - fake_projected
    per_projection = delta.pow(2).mean(dim=0) if squared else delta.abs().mean(dim=0)
    # Authors scale by 3·D so MIND lives in roughly the same range as FAD.
    scale = 3.0 * real_embeddings.shape[1]
    return {
        "mind": scale * per_projection.mean(),
        "per_projection": scale * per_projection,
        "projections": torch.tensor(projections, device=real_embeddings.device),
    }


def _interpolate_sorted_quantiles(sorted_values: torch.Tensor, n_quantiles: int) -> torch.Tensor:
    if sorted_values.shape[0] == n_quantiles:
        return sorted_values
    if sorted_values.shape[0] == 1:
        return sorted_values.expand(n_quantiles, -1)

    positions = torch.linspace(
        0,
        sorted_values.shape[0] - 1,
        n_quantiles,
        device=sorted_values.device,
        dtype=sorted_values.dtype,
    )
    lower = positions.floor().long()
    upper = positions.ceil().long()
    weight = (positions - lower).unsqueeze(1)
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def vendi_score(
    embeddings: torch.Tensor,
    embedding_backend: EmbeddingBackend | None = None,
    eps: float = 1e-12,
    **backend_kwargs: object,
) -> torch.Tensor:
    """Reference-free diversity from the spectrum of a cosine Gram matrix."""
    embeddings = _as_embeddings(embeddings, embedding_backend, **backend_kwargs)
    normalized = F.normalize(embeddings, dim=-1)
    kernel = normalized @ normalized.T
    kernel = (kernel + kernel.T) * 0.5
    eigenvalues = torch.linalg.eigvalsh(kernel / embeddings.shape[0]).clamp_min(0.0)
    eigenvalues = eigenvalues / eigenvalues.sum().clamp_min(eps)
    entropy = -(eigenvalues * eigenvalues.clamp_min(eps).log()).sum()
    return entropy.exp()


def density_coverage(
    real_embeddings: torch.Tensor,
    fake_embeddings: torch.Tensor,
    k: int = 5,
    embedding_backend: EmbeddingBackend | None = None,
    **backend_kwargs: object,
) -> dict[str, torch.Tensor]:
    """Density and Coverage using the real-set k-NN manifold."""
    if k <= 0:
        raise ValueError("k must be positive")
    real_embeddings = _as_embeddings(real_embeddings, embedding_backend, **backend_kwargs)
    fake_embeddings = _as_embeddings(fake_embeddings, embedding_backend, **backend_kwargs)
    _validate_pair(real_embeddings, fake_embeddings)
    if real_embeddings.shape[0] < 2:
        raise ValueError("density/coverage requires at least two real embeddings")

    effective_k = min(k, real_embeddings.shape[0] - 1)
    real_to_real = torch.cdist(real_embeddings, real_embeddings)
    real_to_real.fill_diagonal_(torch.inf)
    radii = real_to_real.kthvalue(effective_k, dim=1).values
    real_to_fake = torch.cdist(real_embeddings, fake_embeddings)
    in_manifold = real_to_fake <= radii.unsqueeze(1)

    density = in_manifold.sum(dim=0).to(real_embeddings.dtype).mean() / effective_k
    coverage = in_manifold.any(dim=1).to(real_embeddings.dtype).mean()
    return {
        "density": density,
        "coverage": coverage,
        "k": torch.tensor(effective_k, device=real_embeddings.device),
    }
