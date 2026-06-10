"""Distributional losses over frozen audio embeddings.

FD-loss (Representation Fréchet Loss, arXiv 2604.28190): minimise the Fréchet distance
between embedding distributions of generated vs real audio. The trick is to decouple the
large population used to *estimate* the generated moments (a feature queue or an EMA of the
moments) from the small batch that *carries gradients* — only the current batch is
differentiable, the rest of the population is detached. A sliced-Wasserstein (MIND) loss can
join this module later, sharing :class:`MomentEstimator`.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from eval.audio_metrics import _symmetric_matrix_sqrt, frechet_from_moments


class MomentEstimator(nn.Module):
    """First/second-moment estimator with a decoupled, mostly-detached population.

    ``update_and_moments`` returns ``(mu, cov)`` over the population while routing gradients
    only through the current batch. ``queue`` keeps a FIFO of recent detached features; ``ema``
    keeps running moments with decay ``beta``. Population state only updates in training mode,
    so validation passes leave it untouched.
    """

    def __init__(self, dim: int, mode: str = "ema", beta: float = 0.999, queue_size: int = 50000):
        super().__init__()
        if mode not in {"ema", "queue"}:
            raise ValueError("mode must be 'ema' or 'queue'")
        self.mode = mode
        self.beta = float(beta)
        self.queue_size = int(queue_size)
        if mode == "ema":
            self.register_buffer("m1", torch.zeros(dim))
            self.register_buffer("m2", torch.zeros(dim, dim))
            self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        else:
            self.register_buffer("queue", torch.zeros(0, dim))

    def update_and_moments(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._ema(feats) if self.mode == "ema" else self._queue(feats)

    def _ema(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_m1 = feats.mean(dim=0)
        batch_m2 = feats.T @ feats / feats.shape[0]
        if not bool(self.initialized):
            mu, m2 = batch_m1, batch_m2
            if self.training:
                self.m1.copy_(batch_m1.detach())
                self.m2.copy_(batch_m2.detach())
                self.initialized.fill_(True)
        else:
            # The (1-beta) weight on the grad-carrying batch matches the EMA the buffers track,
            # so the loss sees the population moments with a gradient toward the current samples.
            mu = (1.0 - self.beta) * batch_m1 + self.beta * self.m1
            m2 = (1.0 - self.beta) * batch_m2 + self.beta * self.m2
            if self.training:
                self.m1.mul_(self.beta).add_(batch_m1.detach(), alpha=1.0 - self.beta)
                self.m2.mul_(self.beta).add_(batch_m2.detach(), alpha=1.0 - self.beta)
        return mu, m2 - torch.outer(mu, mu)

    def _queue(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        history = self.queue.to(device=feats.device, dtype=feats.dtype)
        pool = torch.cat([history, feats], dim=0)  # history detached, current batch carries grad
        mu = pool.mean(dim=0)
        centered = pool - mu
        cov = centered.T @ centered / (pool.shape[0] - 1)
        if self.training:
            self.queue = torch.cat([history, feats.detach()], dim=0)[-self.queue_size :]
        return mu, cov


class FrechetLoss(nn.Module):
    """Multi-embedder FD-loss with self-normalising aggregation.

    For each frozen embedder φ_i the per-embedder term is ``FD_i / sg(FD_i + c)`` so every
    backbone contributes at unit scale regardless of its feature magnitude; the total is the
    weighted sum (equal weights by default). Real moments ``(μ_r, Σ_r)`` are fixed buffers
    computed once via :func:`compute_real_moments`.
    """

    def __init__(
        self,
        embedders: Iterable[nn.Module],
        real_moments: Iterable[tuple[torch.Tensor, torch.Tensor]],
        *,
        mode: str = "ema",
        beta: float = 0.999,
        queue_size: int = 50000,
        c: float = 1e-2,
        weights: Iterable[float] | None = None,
        sample_rate: int = 48000,
        eps: float = 1e-6,
    ):
        super().__init__()
        embedders = list(embedders)
        real_moments = list(real_moments)
        if len(embedders) != len(real_moments):
            raise ValueError("need one (mu_r, cov_r) pair per embedder")
        if not embedders:
            raise ValueError("FrechetLoss needs at least one embedder")
        self.embedders = nn.ModuleList(embedders)
        self.embedders.eval()  # frozen φ: deterministic features, never train-mode dropout/SpecAugment
        self.names = [getattr(emb, "name", f"emb{i}") for i, emb in enumerate(embedders)]
        self.weights = [1.0] * len(embedders) if weights is None else [float(w) for w in weights]
        self.sample_rate = int(sample_rate)
        self.c = float(c)
        self.eps = float(eps)
        self.estimators = nn.ModuleList(
            MomentEstimator(emb.embedding_dim, mode=mode, beta=beta, queue_size=queue_size)
            for emb in embedders
        )
        for i, (mu_r, cov_r) in enumerate(real_moments):
            cov_r = cov_r.detach()
            self.register_buffer(f"mu_real_{i}", mu_r.detach().clone())
            self.register_buffer(f"cov_real_{i}", cov_r.clone())
            # Σ_r^½ is fixed (real side), so precompute it once with eigh rather than every step.
            self.register_buffer(f"cov_real_sqrt_{i}", _symmetric_matrix_sqrt(cov_r.double(), eps).to(cov_r.dtype))

    def train(self, mode: bool = True):
        # Frozen feature extractors must stay in eval (deterministic φ, no dropout/SpecAugment) even
        # when the training loop flips the loss module to train mode; only the estimators may train.
        super().train(mode)
        self.embedders.eval()
        return self

    def forward(
        self, fake_audio: torch.Tensor, audio_lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = fake_audio.new_zeros(())
        terms: dict[str, torch.Tensor] = {}
        for i, (emb, est, name, weight) in enumerate(
            zip(self.embedders, self.estimators, self.names, self.weights, strict=True)
        ):
            feats = emb.embed(fake_audio, sample_rate=self.sample_rate, audio_lengths=audio_lengths).float()
            mu_g, cov_g = est.update_and_moments(feats)
            mu_r = getattr(self, f"mu_real_{i}").to(mu_g)
            cov_r = getattr(self, f"cov_real_{i}").to(cov_g)
            cov_r_sqrt = getattr(self, f"cov_real_sqrt_{i}").to(cov_g)
            fd = frechet_from_moments(mu_r, cov_r, mu_g, cov_g, self.eps, cov_real_sqrt=cov_r_sqrt)["fad"]
            total = total + weight * fd / (fd.detach() + self.c)
            terms[f"fd/{name}"] = fd.detach()
        return total, terms


@torch.no_grad()
def compute_real_moments(
    embedder: nn.Module,
    loader: Iterable,
    sample_rate: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate fixed reference moments ``(μ_r, Σ_r)`` over the complete loader."""
    dim = int(embedder.embedding_dim)
    sum1 = torch.zeros(dim, dtype=torch.float64)
    sum2 = torch.zeros(dim, dim, dtype=torch.float64)
    count = 0
    for batch in loader:
        audio = batch["audio"]
        lengths = batch.get("audio_lengths")
        feats = embedder(audio, sample_rate=sample_rate, audio_lengths=lengths).double().cpu()
        sum1 += feats.sum(dim=0)
        sum2 += feats.T @ feats
        count += feats.shape[0]
    if count == 0:
        raise ValueError("compute_real_moments saw no clips")
    mu = sum1 / count
    cov = (sum2 - count * torch.outer(mu, mu)) / (count - 1)
    return mu.float(), cov.float()
