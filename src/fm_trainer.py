from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from backbone.factory import build_backbone
from data.audio_dataset import (
    AudioDirectoryDataset,
    BucketBatchSampler,
    collate_audio_batch,
    subset_durations,
)
from data.augmentations import AugmentedDataset, build_waveform_augmenter
from ema import EMA, ema_swapped
from emb.factory import build_embedding, build_embedding_backend
from eval.audio_metrics import embedding_cosine_score, frechet_audio_distance, monge_audio_distance
from flow.fm import linear_interpolant, output_to_v, sample_fm
from losses.audio import FMLoss
from utils.loggers import (
    build_audio_monitor,
    init_wandb,
    log_audio_monitor,
    wandb_val_metrics,
)


def _as_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)


def build_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader | None]:
    data_cfg = _as_dict(cfg.data)
    pool_multiplier = int(data_cfg.pop("bucket_pool_multiplier", 100))
    augment_cfg = data_cfg.pop("augmentations", None)
    dataset = AudioDirectoryDataset(**data_cfg)
    val_fraction = float(cfg.train.get("val_fraction", 0.0))
    if val_fraction > 0.0 and len(dataset) > 1:
        val_size = max(1, int(round(len(dataset) * val_fraction)))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
    else:
        train_set, val_set = dataset, None
    loader_cfg = _as_dict(cfg.train.dataloader)
    batch_size = int(loader_cfg.pop("batch_size"))
    drop_last = bool(loader_cfg.pop("drop_last", True))
    train_sampler = BucketBatchSampler(
        subset_durations(train_set),
        batch_size=batch_size,
        pool_multiplier=pool_multiplier,
        shuffle=True,
        drop_last=drop_last,
    )
    if len(train_sampler) == 0:
        raise ValueError(
            "Training dataloader is empty. Reduce train.dataloader.batch_size, "
            "disable drop_last, or provide more audio files."
        )
    augmenter = build_waveform_augmenter(augment_cfg, dataset.sample_rate)
    if augmenter is not None:
        train_set = AugmentedDataset(train_set, augmenter)
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        collate_fn=collate_audio_batch,
        **loader_cfg,
    )
    val_loader = None
    if val_set is not None:
        val_sampler = BucketBatchSampler(
            subset_durations(val_set),
            batch_size=batch_size,
            pool_multiplier=pool_multiplier,
            shuffle=False,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_set,
            batch_sampler=val_sampler,
            collate_fn=collate_audio_batch,
            **loader_cfg,
        )
    return train_loader, val_loader


def _condition(
    conditioner, audio: torch.Tensor, sample_rate: int, audio_lengths: torch.Tensor
) -> torch.Tensor | None:
    if conditioner is None:
        return None
    return conditioner(audio, sample_rate=sample_rate, audio_lengths=audio_lengths)


def _batch_cache_keys(batch) -> list[str] | None:
    paths = batch.get("path") if isinstance(batch, dict) else None
    if paths is None:
        return None
    return [str(path) for path in paths]


def _embed_with_cache(
    embedding_backend,
    batch,
    audio: torch.Tensor,
    sample_rate: int,
    audio_lengths: torch.Tensor,
    cache: dict[str, torch.Tensor] | None,
) -> torch.Tensor | None:
    if embedding_backend is None:
        return None
    keys = _batch_cache_keys(batch)
    if cache is None or keys is None:
        return embedding_backend(audio, sample_rate=sample_rate, audio_lengths=audio_lengths)
    cached = [cache.get(key) for key in keys]
    if all(value is not None for value in cached):
        return torch.stack(
            [value.to(device=audio.device, dtype=audio.dtype) for value in cached], dim=0
        )
    embeddings = embedding_backend(audio, sample_rate=sample_rate, audio_lengths=audio_lengths)
    if embeddings is not None:
        for key, embedding in zip(keys, embeddings.detach().cpu(), strict=True):
            cache[key] = embedding
    return embeddings


def _conditioner_cfg(cfg: DictConfig, device: torch.device) -> dict[str, Any]:
    conditioner_cfg = _as_dict(cfg.get("conditioner", {"type": "none"}))
    if conditioner_cfg.get("type") == "matpac":
        conditioner_cfg["device"] = str(device)
    return conditioner_cfg


def _embedding_metric_cfg(cfg: DictConfig) -> dict[str, Any]:
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


def train_fm(cfg: DictConfig) -> None:
    device = torch.device(cfg.train.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = Path(cfg.train.get("run_dir", "runs/fm-baseline")).expanduser()
    sample_dir = run_dir / "samples"
    ckpt_dir = run_dir / "checkpoints"
    sample_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    wandb_logger = init_wandb(cfg, run_dir)

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_backbone(cfg.backbone).to(device)
    conditioner = build_embedding(_conditioner_cfg(cfg, device), device=device)
    if conditioner is not None:
        conditioner = conditioner.to(device).eval()
    audio_monitor = build_audio_monitor(val_loader, conditioner, cfg, device) if wandb_logger else None
    reference_length = _reference_length(audio_monitor, val_loader, train_loader)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    amp_enabled = bool(cfg.train.get("amp", True)) and device.type == "cuda"
    ema = (
        EMA(model, decay=float(cfg.train.get("ema_decay", 0.999)))
        if cfg.train.get("ema_decay")
        else None
    )
    loss_fn = FMLoss(**_as_dict(cfg.loss), eps=float(cfg.flow.get("eps", 1e-5)))
    embedding_cfg = _embedding_metric_cfg(cfg)
    metric_embedding_backend = build_embedding_backend(embedding_cfg, device=device)
    real_metric_embedding_cache: dict[str, torch.Tensor] = {}

    max_steps = int(cfg.train.max_steps)
    log_every = int(cfg.train.get("log_every", 10))
    sample_every = int(cfg.train.get("sample_every", 0))
    ckpt_every = int(cfg.train.get("ckpt_every", 500))
    grad_clip = float(cfg.train.get("grad_clip", 0.0))
    cond_dropout_prob = float(cfg.train.get("cond_dropout_prob", 0.0))
    step = 0
    model.train()
    progress = tqdm(total=max_steps, desc="fm")
    try:
        while step < max_steps:
            for batch in train_loader:
                audio = batch["audio"].to(device)
                audio_lengths = batch["audio_lengths"].to(device)
                sample_rate = int(batch["sample_rate"])
                with torch.no_grad():
                    cond = _condition(conditioner, audio, sample_rate, audio_lengths)
                if cond is not None and cond_dropout_prob > 0.0:
                    keep = (
                        torch.rand(cond.shape[0], device=cond.device) >= cond_dropout_prob
                    ).view(-1, *([1] * (cond.ndim - 1)))
                    cond = cond * keep
                flow_batch = linear_interpolant(audio, eps=float(cfg.flow.get("eps", 1e-5)))
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
                ):
                    pred = model(flow_batch.x_t, t=flow_batch.t, cond=cond, length=audio.shape[-1])
                    loss_out = loss_fn(pred, flow_batch)
                loss_out.total.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if ema is not None:
                    ema.update(model)
                step += 1
                progress.update(1)
                if step % log_every == 0:
                    terms = {
                        name: float(value.detach().cpu()) for name, value in loss_out.terms.items()
                    }
                    loss_value = float(loss_out.total.detach().cpu())
                    progress.set_postfix(loss=loss_value, **terms)
                    if wandb_logger is not None:
                        wandb_logger.log(
                            {"train/loss_total": loss_value}
                            | {f"train/{name}": value for name, value in terms.items()},
                            step=step,
                        )
                if sample_every and step % sample_every == 0:
                    _save_sample(
                        model,
                        conditioner,
                        cfg,
                        device,
                        sample_dir / f"step_{step:08d}.wav",
                        reference_length=reference_length,
                        ema=ema,
                    )
                if step % ckpt_every == 0 or step == max_steps:
                    _save_checkpoint(
                        model, optimizer, ema, cfg,
                        ckpt_dir / f"step_{step:08d}.pt", step,
                    )
                if val_loader is not None and step % int(cfg.train.get("val_every", 500)) == 0:
                    metrics = _validate(
                        model,
                        conditioner,
                        val_loader,
                        loss_fn,
                        cfg,
                        device,
                        ema=ema,
                        metric_embedding_backend=metric_embedding_backend,
                        real_embedding_cache=real_metric_embedding_cache,
                    )
                    progress.set_postfix(**metrics)
                    if wandb_logger is not None:
                        wandb_logger.log(wandb_val_metrics(metrics), step=step)
                    if audio_monitor is not None:
                        log_audio_monitor(
                            model,
                            cfg,
                            device,
                            sample_dir,
                            step,
                            audio_monitor,
                            wandb_logger,
                            ema=ema,
                        )
                if step >= max_steps:
                    break
    finally:
        progress.close()
        if wandb_logger is not None:
            wandb_logger.finish()


def _reference_length(audio_monitor, val_loader, train_loader) -> int:
    """Pick a reference audio length for periodic samples (matches a real batch)."""
    if audio_monitor is not None:
        return int(audio_monitor.audio.shape[-1])
    loader = val_loader if val_loader is not None else train_loader
    sample_batch = next(iter(loader))
    return int(sample_batch["audio"].shape[-1])


def _save_sample(
    model,
    conditioner,
    cfg,
    device,
    path: Path,
    reference_length: int,
    ema: EMA | None = None,
) -> None:
    import soundfile as sf

    was_training = model.training
    model.eval()
    shape = (
        int(cfg.sampling.get("batch_size", 1)),
        int(cfg.data.get("channels", 1)),
        int(reference_length),
    )
    cond = None
    if conditioner is not None:
        cond = torch.zeros(shape[0], int(getattr(conditioner, "embedding_dim", 1)), device=device)
    try:
        with ema_swapped(ema, model):
            audio = (
                sample_fm(
                    model,
                    shape=shape,
                    cond=cond,
                    steps=int(cfg.sampling.get("steps", 1)),
                    eps=float(cfg.flow.get("eps", 1e-5)),
                    device=device,
                    method=str(cfg.sampling.get("method", "euler")),
                    rms_lift=bool(cfg.data.get("rms_lift", False)),
                    lift_scale=float(cfg.data.get("lift_scale", 3.0)),
                )
                .detach()
                .cpu()
            )
    finally:
        model.train(was_training)
    peak = float(audio.abs().amax())
    audio = audio.clamp(-1.0, 1.0)
    sf.write(path, audio[0].transpose(0, 1).numpy(), int(cfg.data.sample_rate))
    path.with_suffix(".peak.txt").write_text(f"{peak:.6f}\n")


def _save_checkpoint(model, optimizer, ema, cfg, path: Path, step: int) -> None:
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


@torch.no_grad()
def _validate(
    model,
    conditioner,
    loader,
    loss_fn,
    cfg,
    device,
    ema: EMA | None = None,
    metric_embedding_backend=None,
    real_embedding_cache: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    if ema is not None:
        ema.apply_to(model)
    sums: dict[str, float] = {}
    real_metric_embeddings = []
    fake_metric_embeddings = []
    real_condition_embeddings = []
    fake_condition_embeddings = []
    batches = 0
    max_batches = int(cfg.train.get("val_batches", 1))
    embedding_cfg = _embedding_metric_cfg(cfg)
    embedding_enabled = bool(embedding_cfg.get("enabled", False))
    cache_real = embedding_enabled and bool(embedding_cfg.get("cache_real", True))
    active_cache = real_embedding_cache if cache_real else None
    try:
        for batch in loader:
            audio = batch["audio"].to(device)
            audio_lengths = batch["audio_lengths"].to(device)
            sample_rate = int(batch["sample_rate"])
            cond = _condition(conditioner, audio, sample_rate, audio_lengths)
            flow_batch = linear_interpolant(audio, eps=float(cfg.flow.get("eps", 1e-5)))
            pred = model(flow_batch.x_t, t=flow_batch.t, cond=cond, length=audio.shape[-1])
            loss_out = loss_fn(pred, flow_batch)
            v_hat = output_to_v(pred, flow_batch.x_t, flow_batch.t, eps=float(cfg.flow.get("eps", 1e-5)))
            batch_metrics = {
                "val_total": float(loss_out.total.cpu()),
                "val_x_mse": float(F.mse_loss(pred, flow_batch.x1).cpu()),
                "val_v_mse": float(F.mse_loss(v_hat, flow_batch.v).cpu()),
            }
            batch_metrics.update(
                {f"val_{k}": float(v.detach().cpu()) for k, v in loss_out.terms.items()}
            )
            for key, value in batch_metrics.items():
                sums[key] = sums.get(key, 0.0) + value
            distance = str(embedding_cfg.get("distance", "none")).lower()
            needs_metric_backend = distance in {"fad", "mind", "both"}
            wants_condition_cosine = bool(embedding_cfg.get("cosine", False))
            if embedding_enabled and needs_metric_backend and metric_embedding_backend is None:
                raise ValueError("Embedding validation requires a CLAP/VGGish metric embedding backend")
            if embedding_enabled and (needs_metric_backend or wants_condition_cosine):
                fake_audio = sample_fm(
                    model,
                    shape=tuple(audio.shape),
                    cond=cond,
                    steps=int(embedding_cfg.get("sample_steps", cfg.sampling.get("steps", 1))),
                    eps=float(cfg.flow.get("eps", 1e-5)),
                    device=device,
                    method=str(cfg.sampling.get("method", "euler")),
                ).clamp(-1.0, 1.0)
                fake_lengths = torch.full_like(audio_lengths, audio.shape[-1])
                if needs_metric_backend:
                    real_metric = _embed_with_cache(
                        metric_embedding_backend, batch, audio, sample_rate, audio_lengths, active_cache
                    )
                    fake_metric = metric_embedding_backend(
                        fake_audio, sample_rate=sample_rate, audio_lengths=fake_lengths
                    )
                    if real_metric is not None and fake_metric is not None:
                        real_metric_embeddings.append(real_metric.detach())
                        fake_metric_embeddings.append(fake_metric.detach())
                if wants_condition_cosine and conditioner is not None and cond is not None:
                    fake_cond = _condition(conditioner, fake_audio, sample_rate, fake_lengths)
                    if fake_cond is not None:
                        real_condition_embeddings.append(cond.detach())
                        fake_condition_embeddings.append(fake_cond.detach())
            batches += 1
            if batches >= max_batches:
                break
        metrics = {key: value / max(1, batches) for key, value in sums.items()}
        if embedding_enabled:
            metrics.update(
                _finalize_embedding_metrics(
                    real_metric_embeddings,
                    fake_metric_embeddings,
                    real_condition_embeddings,
                    fake_condition_embeddings,
                    embedding_cfg,
                )
            )
            metrics["val_real_embedding_cache_size"] = float(len(active_cache or {}))
    finally:
        if ema is not None:
            ema.restore(model)
        model.train(was_training)
    return metrics


def _finalize_embedding_metrics(
    real_metric_embeddings: list[torch.Tensor],
    fake_metric_embeddings: list[torch.Tensor],
    real_condition_embeddings: list[torch.Tensor],
    fake_condition_embeddings: list[torch.Tensor],
    embedding_cfg: dict[str, Any],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    distance = str(embedding_cfg.get("distance", "none")).lower()
    if bool(embedding_cfg.get("cosine", False)):
        if real_condition_embeddings and fake_condition_embeddings:
            real_cond = torch.cat(real_condition_embeddings, dim=0)
            fake_cond = torch.cat(fake_condition_embeddings, dim=0)
            metrics["val_matpac_embedding_cosine"] = float(
                embedding_cosine_score(real_cond, fake_cond).detach().cpu()
            )
        else:
            metrics["val_matpac_embedding_cosine_skipped"] = 1.0
    if distance in {"fad", "both"}:
        if real_metric_embeddings and fake_metric_embeddings:
            real_metric = torch.cat(real_metric_embeddings, dim=0)
            fake_metric = torch.cat(fake_metric_embeddings, dim=0)
            fad = frechet_audio_distance(real_metric, fake_metric)
            metrics["val_fad"] = float(fad["fad"].detach().cpu())
        else:
            metrics["val_fad_skipped"] = 1.0
    if distance in {"mind", "both"}:
        if real_metric_embeddings and fake_metric_embeddings:
            real_metric = torch.cat(real_metric_embeddings, dim=0)
            fake_metric = torch.cat(fake_metric_embeddings, dim=0)
            mind = monge_audio_distance(
                real_metric,
                fake_metric,
                projections=int(embedding_cfg.get("mind_projections", 256)),
            )
            metrics["val_mind"] = float(mind["mind"].detach().cpu())
        else:
            metrics["val_mind_skipped"] = 1.0
    if distance not in {"none", "fad", "mind", "both"}:
        raise ValueError(
            "eval.metrics.embedding_validation.distance must be none, fad, mind, or both"
        )
    return metrics
