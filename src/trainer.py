from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
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
from ema import EMA
from emb.factory import build_embedding, build_embedding_backend
from flow.fm import EPS, RectifiedFlow
from loggers import init_wandb, save_wavs, wandb_cfg, wandb_val_metrics
from losses.audio import mr_stft_loss
from validation import embedding_metric_cfg, generate_examples, validate_metrics


def _as_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)


def build_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader | None]:
    """Variable-length audio loaders with bucketed batching and optional augmentation."""
    data_cfg = _as_dict(cfg.data)
    pool_multiplier = int(data_cfg.pop("bucket_pool_multiplier", 100))
    augment_cfg = data_cfg.pop("augmentations", None)
    for key in ("rms_lift", "rms_target", "lift_scale"):
        data_cfg.pop(key, None)  # applied at the model boundary by the trainer, not the dataset
    dataset = AudioDirectoryDataset(**data_cfg)
    val_fraction = float(cfg.train.get("val_fraction", 0.0))
    if val_fraction > 0.0 and len(dataset) > 1:
        val_size = max(1, int(round(len(dataset) * val_fraction)))
        train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])
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
        train_set, batch_sampler=train_sampler, collate_fn=collate_audio_batch, **loader_cfg
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
            val_set, batch_sampler=val_sampler, collate_fn=collate_audio_batch, **loader_cfg
        )
    return train_loader, val_loader


class BaseTrainer:
    """Owns all shared training infrastructure and the loop.

    Paradigm-specific behaviour lives in three overridable hooks: ``build_method``
    (the generative method object), ``training_step`` (batch -> loss), and ``sample``.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(
            cfg.train.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.run_dir = Path(cfg.train.get("run_dir", "runs/fm-baseline")).expanduser()
        self.sample_dir = self.run_dir / "samples"
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, self.run_dir / "config.yaml")
        self.sample_rate = int(cfg.data.sample_rate)
        # WavFlow amplitude lift (a data property): forward in training_step, inverse passed to the sampler.
        self.rms_lift = bool(cfg.data.get("rms_lift", False))
        self.rms_target = float(cfg.data.get("rms_target", 0.33))
        self.lift_scale = float(cfg.data.get("lift_scale", 3.0))

        self.logger = init_wandb(cfg, self.run_dir)
        self.train_loader, self.val_loader = build_dataloaders(cfg)
        self.model = build_backbone(cfg.backbone).to(self.device)
        self.conditioner = self._build_conditioner()
        self.method = self.build_method()

        self.optimizer = instantiate(cfg.optimizer, params=self.model.parameters())
        self.max_steps = int(cfg.train.max_steps)
        self.scheduler = self._build_scheduler()
        self.amp_enabled = bool(cfg.train.get("amp", True)) and self.device.type == "cuda"
        self.ema = EMA(self.model, decay=float(cfg.train.ema_decay)) if cfg.train.get("ema_decay") else None
        self.metric_backend = build_embedding_backend(embedding_metric_cfg(cfg), device=self.device)
        self.real_embedding_cache: dict[str, torch.Tensor] = {}

        self.log_every = int(cfg.train.get("log_every", 10))
        self.sample_every = int(cfg.train.get("sample_every", 0))
        self.ckpt_every = int(cfg.train.get("ckpt_every", 500))
        self.val_every = int(cfg.train.get("val_every", 500))
        self.grad_clip = float(cfg.train.get("grad_clip", 0.0))
        self.grad_accum = max(1, int(cfg.train.get("grad_accum_steps", 1)))
        self.cond_dropout_prob = float(cfg.train.get("cond_dropout_prob", 0.0))
        self.step = 0
        self.progress: tqdm | None = None

        self._build_examples()
        self._maybe_resume()

    # ---- construction helpers -------------------------------------------------
    def _build_conditioner(self):
        conditioner_cfg = _as_dict(self.cfg.get("conditioner", {"type": "none"}))
        if conditioner_cfg.get("type") == "matpac":
            conditioner_cfg["device"] = str(self.device)
        conditioner = build_embedding(conditioner_cfg, device=self.device)
        return conditioner.to(self.device).eval() if conditioner is not None else None

    def _build_scheduler(self):
        warmup = int(self.cfg.train.get("warmup_steps", 0))
        min_ratio = float(self.cfg.train.get("min_lr_ratio", 0.0))
        total = self.max_steps

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total - warmup)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    @torch.no_grad()
    def _build_examples(self) -> None:
        """Capture fixed random real examples at their native lengths + seeded noise."""
        self.example_audio: list[torch.Tensor] = []
        self.example_cond: list[torch.Tensor | None] = []
        self.example_noise: list[torch.Tensor] = []
        self._reference_logged = False
        count = int(wandb_cfg(self.cfg).get("audio_examples", 4))
        if count <= 0:
            return
        dataset = (self.val_loader or self.train_loader).dataset
        generator = torch.Generator().manual_seed(int(wandb_cfg(self.cfg).get("audio_seed", 0)))
        for index in torch.randperm(len(dataset), generator=generator)[:count].tolist():
            item = dataset[index]
            audio = item["audio"]  # [C, T] at native length, no batch padding
            lengths = item["audio_lengths"].view(1).to(self.device)
            cond = self.condition(audio.unsqueeze(0).to(self.device), self.sample_rate, lengths)
            self.example_audio.append(audio)
            self.example_cond.append(None if cond is None else cond.detach())
            self.example_noise.append(torch.randn((1, *audio.shape), generator=generator))

    # ---- checkpointing --------------------------------------------------------
    def save_checkpoint(self) -> None:
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "ema": self.ema.state_dict() if self.ema is not None else None,
                "cfg": OmegaConf.to_container(self.cfg, resolve=True),
            },
            self.ckpt_dir / f"step_{self.step:08d}.pt",
        )

    def _maybe_resume(self) -> None:
        resume = self.cfg.train.get("resume", None)
        if not resume:
            return
        if str(resume) == "auto":
            checkpoints = sorted(self.ckpt_dir.glob("step_*.pt"))
            if not checkpoints:
                return
            path = checkpoints[-1]
        else:
            path = Path(resume).expanduser()
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if self.ema is not None and state.get("ema") is not None:
            self.ema.load_state_dict(state["ema"])
        self.step = int(state.get("step", 0))
        tqdm.write(f"Resumed from {path} at step {self.step}")

    # ---- shared utilities -----------------------------------------------------
    def condition(self, audio, sample_rate: int, audio_lengths) -> torch.Tensor | None:
        if self.conditioner is None:
            return None
        with torch.no_grad():
            return self.conditioner(audio, sample_rate=sample_rate, audio_lengths=audio_lengths)

    def _cfg_dropout(self, cond: torch.Tensor | None) -> torch.Tensor | None:
        if cond is None or self.cond_dropout_prob <= 0.0:
            return cond
        keep = (torch.rand(cond.shape[0], device=cond.device) >= self.cond_dropout_prob)
        return cond * keep.view(-1, *([1] * (cond.ndim - 1)))

    # ---- paradigm hooks -------------------------------------------------------
    def build_method(self):
        raise NotImplementedError

    def training_step(self, audio: torch.Tensor, cond: torch.Tensor | None):
        raise NotImplementedError

    def sample(self, shape, cond=None, noise=None) -> torch.Tensor:
        raise NotImplementedError

    def validate(self) -> dict[str, float]:
        return validate_metrics(self)

    # ---- training loop --------------------------------------------------------
    def run(self) -> None:
        self.model.train()
        self.progress = tqdm(initial=self.step, total=self.max_steps, desc=type(self).__name__)
        try:
            micro = 0
            accum_loss = 0.0
            accum_terms: dict[str, torch.Tensor] = {}
            while self.step < self.max_steps:
                for batch in self.train_loader:
                    audio = batch["audio"].to(self.device)
                    audio_lengths = batch["audio_lengths"].to(self.device)
                    sample_rate = int(batch["sample_rate"])
                    cond = self._cfg_dropout(self.condition(audio, sample_rate, audio_lengths))

                    if micro == 0:
                        self.optimizer.zero_grad(set_to_none=True)
                    loss, terms = self.training_step(audio, cond)
                    # Scale by 1/grad_accum so the accumulated gradient is the mean over the
                    # grad_accum micro-batches (== the gradient of one batch_size*grad_accum batch).
                    (loss / self.grad_accum).backward()
                    accum_loss += loss.detach()
                    for name, value in terms.items():
                        accum_terms[name] = accum_terms.get(name, 0.0) + value.detach()
                    micro += 1
                    if micro < self.grad_accum:
                        continue

                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()
                    if self.ema is not None:
                        self.ema.update(self.model)

                    self.step += 1
                    self.progress.update(1)
                    self._periodic(
                        accum_loss / self.grad_accum,
                        {name: value / self.grad_accum for name, value in accum_terms.items()},
                    )
                    micro = 0
                    accum_loss = 0.0
                    accum_terms = {}
                    if self.step >= self.max_steps:
                        break
        finally:
            self.progress.close()
            if self.logger is not None:
                self.logger.finish()

    def _periodic(self, loss: torch.Tensor, terms: dict[str, torch.Tensor]) -> None:
        step = self.step
        if self.log_every and step % self.log_every == 0:
            loss_value = float(loss.detach().cpu())
            term_values = {name: float(value.detach().cpu()) for name, value in terms.items()}
            self.progress.set_postfix(loss=loss_value, **term_values)
            if self.logger is not None:
                values = {"train/loss_total": loss_value, "train/lr": self.scheduler.get_last_lr()[0]}
                values |= {f"train/{name}": value for name, value in term_values.items()}
                self.logger.log(values, step=step)
        if self.sample_every and step % self.sample_every == 0:
            self._log_audio_examples()
        if (self.ckpt_every and step % self.ckpt_every == 0) or step == self.max_steps:
            self.save_checkpoint()
        if self.val_loader is not None and self.val_every and step % self.val_every == 0:
            metrics = self.validate()
            if metrics:
                self.progress.set_postfix(**metrics)
                if self.logger is not None:
                    self.logger.log(wandb_val_metrics(metrics), step=step)

    def _log_audio_examples(self) -> None:
        generated = generate_examples(self)
        if generated is None:
            return
        if not self._reference_logged:
            paths = save_wavs(self.example_audio, self.sample_rate, self.sample_dir, "reference_{index:03d}.wav")
            if self.logger is not None:
                self.logger.audio(
                    "audio/reference", paths, [f"reference {i}" for i in range(len(paths))], step=self.step
                )
            self._reference_logged = True
        paths = save_wavs(
            generated, self.sample_rate, self.sample_dir, f"step_{self.step:08d}_generated_{{index:03d}}.wav"
        )
        if self.logger is not None:
            self.logger.audio(
                "audio/generated",
                paths,
                [f"generated {i} step {self.step}" for i in range(len(paths))],
                step=self.step,
            )


class RFTrainer(BaseTrainer):
    """Rectified-flow / flow-matching training: velocity loss + optional MR-STFT aux."""

    def build_method(self) -> RectifiedFlow:
        return RectifiedFlow()

    def _sample_t(self, audio: torch.Tensor) -> torch.Tensor:
        """Draw training timesteps. 'logit_normal' biases toward the middle of [0, 1]."""
        flow_cfg = self.cfg.get("flow", {}) or {}
        batch, device, dtype = audio.shape[0], audio.device, audio.dtype
        if str(flow_cfg.get("t_distribution", "logit_normal")) == "uniform":
            t = torch.rand(batch, device=device, dtype=dtype)
        else:
            logits = torch.randn(batch, device=device, dtype=dtype)
            t = (logits * float(flow_cfg.get("logit_std", 1.0)) + float(flow_cfg.get("logit_mean", 0.0))).sigmoid()
        return t.clamp(EPS, 1.0 - EPS)

    def training_step(self, audio: torch.Tensor, cond: torch.Tensor | None):
        loss_cfg = self.cfg.loss
        if self.rms_lift:
            # WavFlow (§3.2) amplitude lift of the target; inverse ÷lift_scale in RectifiedFlow.sample.
            # tanh (not WavFlow's hard clamp) soft-saturates rare high-crest peaks instead of flat-topping;
            # near-linear at r_*=0.1 so the body is just unit-RMS-normalised (lifted RMS ≈ rms_target*lift_scale).
            rms = audio.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
            audio = self.lift_scale * torch.tanh((self.rms_target / rms) * audio)
        x_t, t, x1 = self.method.train_tuple(audio, t=self._sample_t(audio))
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp_enabled):
            pred = self.model(x_t, t=t, cond=cond, length=audio.shape[-1])
            total, terms = self.method.loss(
                pred,
                x1,
                x_t,
                t,
                space=str(loss_cfg.get("loss_space", "v")),
                loss_type=str(loss_cfg.get("primary", "mse")),
            )
            mr_stft_weight = float(loss_cfg.get("mr_stft_weight", 0.0))
            if mr_stft_weight > 0.0:
                aux = mr_stft_loss(pred, x1, log_weight=float(loss_cfg.get("mr_stft_log_weight", 0.0)))
                total = total + mr_stft_weight * aux
                terms = {**terms, "mr_stft": aux}
        return total, terms

    def sample(self, shape, cond=None, noise=None) -> torch.Tensor:
        audio = self.method.sample(
            self.model,
            shape,
            cond=cond,
            noise=noise,
            steps=int(self.cfg.sampling.get("steps", 1)),
            method=str(self.cfg.sampling.get("method", "euler")),
            guidance_scale=float(self.cfg.sampling.get("guidance_scale", 1.0)),
            lift_scale=self.lift_scale if self.rms_lift else 1.0,
        )
        peak = audio.abs().amax(dim=tuple(range(1, audio.ndim)), keepdim=True).clamp_min(1e-8)
        return audio / peak
