from __future__ import annotations

from typing import Any

import torch

from ema import ema_swapped
from eval.audio_metrics import (
    density_coverage,
    embedding_cosine_score,
    frechet_audio_distance,
    monge_audio_distance,
)


def embedding_metric_cfg(cfg) -> dict[str, Any]:
    """Resolve the eval.metrics.embedding_validation block (with FAD/MIND/cosine fallbacks)."""
    eval_cfg = cfg.get("eval", {})
    metrics_cfg = eval_cfg.get("metrics", {}) if eval_cfg is not None else {}
    embedding_cfg = dict(metrics_cfg.get("embedding_validation", {}) or {})
    if not embedding_cfg:
        distance = "none"
        if bool(metrics_cfg.get("fad", {}).get("enabled", False)):
            distance = "fad"
        if bool(metrics_cfg.get("mind", {}).get("enabled", False)):
            distance = "mind" if distance == "none" else "both"
        embedding_cfg = {
            "enabled": distance != "none"
            or bool(metrics_cfg.get("embedding_cosine", {}).get("enabled", False)),
            "backend": "clap",
            "distance": distance,
            "cosine": bool(metrics_cfg.get("embedding_cosine", {}).get("enabled", False)),
        }
    embedding_cfg.setdefault("backend", "clap")
    return embedding_cfg


def _cache_keys(batch) -> list[str] | None:
    paths = batch.get("path") if isinstance(batch, dict) else None
    return None if paths is None else [str(path) for path in paths]


def _embed_with_cache(backend, batch, audio, sample_rate, audio_lengths, cache) -> torch.Tensor | None:
    if backend is None:
        return None
    keys = _cache_keys(batch)
    if cache is None or keys is None:
        return backend(audio, sample_rate=sample_rate, audio_lengths=audio_lengths)
    cached = [cache.get(key) for key in keys]
    if all(value is not None for value in cached):
        return torch.stack([v.to(device=audio.device, dtype=audio.dtype) for v in cached], dim=0)
    embeddings = backend(audio, sample_rate=sample_rate, audio_lengths=audio_lengths)
    if embeddings is not None:
        for key, embedding in zip(keys, embeddings.detach().cpu(), strict=True):
            cache[key] = embedding
    return embeddings


def _finalize_embedding_metrics(
    real_metric: list[torch.Tensor],
    fake_metric: list[torch.Tensor],
    real_cond: list[torch.Tensor],
    fake_cond: list[torch.Tensor],
    embedding_cfg: dict[str, Any],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    distance = str(embedding_cfg.get("distance", "none")).lower()
    if distance not in {"none", "fad", "mind", "both"}:
        raise ValueError("eval.metrics.embedding_validation.distance must be none, fad, mind, or both")
    if bool(embedding_cfg.get("cosine", False)):
        if real_cond and fake_cond:
            metrics["val_embedding_cosine"] = float(
                embedding_cosine_score(torch.cat(real_cond), torch.cat(fake_cond)).detach().cpu()
            )
        else:
            metrics["val_embedding_cosine_skipped"] = 1.0
    if distance in {"fad", "both"}:
        if real_metric and fake_metric:
            fad = frechet_audio_distance(torch.cat(real_metric), torch.cat(fake_metric))
            metrics["val_fad"] = float(fad["fad"].detach().cpu())
        else:
            metrics["val_fad_skipped"] = 1.0
    if distance in {"mind", "both"}:
        if real_metric and fake_metric:
            mind = monge_audio_distance(
                torch.cat(real_metric),
                torch.cat(fake_metric),
                projections=int(embedding_cfg.get("mind_projections", 256)),
            )
            metrics["val_mind"] = float(mind["mind"].detach().cpu())
        else:
            metrics["val_mind_skipped"] = 1.0
    dc_cfg = dict(embedding_cfg.get("density_coverage", {}) or {})
    if bool(dc_cfg.get("enabled", False)):
        if real_metric and fake_metric:
            dc = density_coverage(
                torch.cat(real_metric),
                torch.cat(fake_metric),
                k=int(dc_cfg.get("k", 5)),
            )
            metrics["val_density"] = float(dc["density"].detach().cpu())
            metrics["val_coverage"] = float(dc["coverage"].detach().cpu())
        else:
            metrics["val_density_coverage_skipped"] = 1.0
    return metrics


@torch.no_grad()
def validate_metrics(trainer) -> dict[str, float]:
    """Average validation loss terms over a few batches, plus optional FAD/MIND/cosine."""
    loader = trainer.val_loader
    if loader is None:
        return {}
    cfg = trainer.cfg
    embedding_cfg = embedding_metric_cfg(cfg)
    enabled = bool(embedding_cfg.get("enabled", False))
    distance = str(embedding_cfg.get("distance", "none")).lower()
    wants_dc = bool((embedding_cfg.get("density_coverage", {}) or {}).get("enabled", False))
    needs_backend = distance in {"fad", "mind", "both"} or wants_dc
    wants_cosine = bool(embedding_cfg.get("cosine", False))
    cache = trainer.real_embedding_cache if enabled and bool(embedding_cfg.get("cache_real", True)) else None
    max_batches = int(cfg.train.get("val_batches", 1))

    sums: dict[str, float] = {}
    real_metric: list[torch.Tensor] = []
    fake_metric: list[torch.Tensor] = []
    real_cond: list[torch.Tensor] = []
    fake_cond: list[torch.Tensor] = []
    batches = 0

    was_training = trainer.model.training
    trainer.model.eval()
    try:
        with ema_swapped(trainer.ema, trainer.model):
            for batch in loader:
                audio = batch["audio"].to(trainer.device)
                audio_lengths = batch["audio_lengths"].to(trainer.device)
                sample_rate = int(batch["sample_rate"])
                cond = trainer.condition(audio, sample_rate, audio_lengths)
                loss, terms = trainer.training_step(audio, cond)
                sums["val_total"] = sums.get("val_total", 0.0) + float(loss.detach().cpu())
                for name, value in terms.items():
                    sums[f"val_{name}"] = sums.get(f"val_{name}", 0.0) + float(value.detach().cpu())
                if enabled and (needs_backend or wants_cosine):
                    fake = trainer.sample(tuple(audio.shape), cond=cond).clamp(-1.0, 1.0)
                    if needs_backend:
                        if trainer.metric_backend is None:
                            raise ValueError("FAD/MIND validation requires a metric embedding backend")
                        real = _embed_with_cache(
                            trainer.metric_backend, batch, audio, sample_rate, audio_lengths, cache
                        )
                        fake_emb = trainer.metric_backend(
                            fake, sample_rate=sample_rate, audio_lengths=audio_lengths
                        )
                        if real is not None and fake_emb is not None:
                            real_metric.append(real.detach())
                            fake_metric.append(fake_emb.detach())
                    if wants_cosine and trainer.conditioner is not None and cond is not None:
                        fc = trainer.condition(fake, sample_rate, audio_lengths)
                        if fc is not None:
                            real_cond.append(cond.detach())
                            fake_cond.append(fc.detach())
                batches += 1
                if batches >= max_batches:
                    break
    finally:
        trainer.model.train(was_training)

    metrics = {key: value / max(1, batches) for key, value in sums.items()}
    if enabled:
        metrics.update(
            _finalize_embedding_metrics(real_metric, fake_metric, real_cond, fake_cond, embedding_cfg)
        )
        if cache is not None:
            metrics["val_real_embedding_cache_size"] = float(len(cache))
    return metrics


@torch.no_grad()
def generate_examples(trainer) -> list[torch.Tensor] | None:
    """Sample each fixed example at its own native length (EMA weights if available)."""
    if not trainer.example_audio:
        return None
    was_training = trainer.model.training
    trainer.model.eval()
    try:
        with ema_swapped(trainer.ema, trainer.model):
            generated = [
                trainer.sample(noise.shape, cond=cond, noise=noise.to(trainer.device))[0].detach().cpu()
                for noise, cond in zip(trainer.example_noise, trainer.example_cond, strict=True)
            ]
    finally:
        trainer.model.train(was_training)
    return generated
