from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from ema import EMA, ema_swapped
from flow.fm import sample_fm


@dataclass
class WandbLogger:
    module: Any
    run: Any

    def log(self, values: dict[str, Any], step: int) -> None:
        if values:
            self.run.log(values, step=step)

    def audio(self, key: str, paths: list[Path], captions: list[str], step: int) -> None:
        if paths:
            self.log(
                {
                    key: [
                        self.module.Audio(str(path), caption=caption)
                        for path, caption in zip(paths, captions, strict=True)
                    ]
                },
                step=step,
            )

    def audio_arrays(
        self,
        key: str,
        audio: torch.Tensor,
        sample_rate: int,
        captions: list[str],
        step: int,
    ) -> None:
        if audio.numel():
            audio = audio.detach().cpu().clamp(-1.0, 1.0)
            self.log(
                {
                    key: [
                        self.module.Audio(
                            example.transpose(0, 1).numpy(),
                            sample_rate=sample_rate,
                            caption=caption,
                        )
                        for example, caption in zip(audio, captions, strict=True)
                    ]
                },
                step=step,
            )

    def finish(self) -> None:
        self.run.finish()


@dataclass
class AudioMonitor:
    audio: torch.Tensor
    audio_lengths: torch.Tensor
    cond: torch.Tensor | None
    noise: torch.Tensor
    sample_rate: int
    references_logged: bool = False


def wandb_cfg(cfg: DictConfig) -> dict[str, Any]:
    train_cfg = cfg.get("train", {})
    return dict(train_cfg.get("wandb", {}) or {})


def init_wandb(cfg: DictConfig, run_dir: Path) -> WandbLogger | None:
    cfg_wandb = wandb_cfg(cfg)
    if not bool(cfg_wandb.get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("train.wandb.enabled=true requires the 'wandb' package.") from exc
    init_kwargs = {
        "project": cfg_wandb.get("project") or "raw-audio",
        "dir": str(run_dir),
        "mode": cfg_wandb.get("mode", "online"),
    }
    for key in ("entity", "name", "group"):
        if cfg_wandb.get(key):
            init_kwargs[key] = cfg_wandb[key]
    if cfg_wandb.get("tags"):
        init_kwargs["tags"] = list(cfg_wandb["tags"])
    if bool(cfg_wandb.get("log_config", True)):
        init_kwargs["config"] = OmegaConf.to_container(cfg, resolve=True)
    return WandbLogger(module=wandb, run=wandb.init(**init_kwargs))


@torch.no_grad()
def build_audio_monitor(
    val_loader: DataLoader | None,
    conditioner,
    cfg: DictConfig,
    device: torch.device,
) -> AudioMonitor | None:
    cfg_wandb = wandb_cfg(cfg)
    count = int(cfg_wandb.get("audio_examples", 0))
    if count <= 0:
        return None
    if val_loader is None:
        tqdm.write("wandb audio monitor skipped: no validation loader is configured")
        return None
    collected_audio = []
    collected_lengths = []
    sample_rate: int | None = None
    for batch in val_loader:
        audio = batch["audio"]
        lengths = batch["audio_lengths"]
        sample_rate = int(batch["sample_rate"])
        remaining = count - sum(part.shape[0] for part in collected_audio)
        collected_audio.append(audio[:remaining].detach().cpu())
        collected_lengths.append(lengths[:remaining].detach().cpu())
        if sum(part.shape[0] for part in collected_audio) >= count:
            break
    if not collected_audio or sample_rate is None:
        tqdm.write("wandb audio monitor skipped: validation loader produced no batches")
        return None
    audio = torch.cat(collected_audio, dim=0)[:count]
    audio_lengths = torch.cat(collected_lengths, dim=0)[:count]
    cond = None
    if conditioner is not None:
        conditioner_audio = audio.to(device)
        conditioner_lengths = audio_lengths.to(device)
        cond = conditioner(
            conditioner_audio, sample_rate=sample_rate, audio_lengths=conditioner_lengths
        )
        cond = None if cond is None else cond.detach().cpu()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg_wandb.get("audio_seed", 0)))
    noise = torch.randn(audio.shape, generator=generator)
    return AudioMonitor(
        audio=audio,
        audio_lengths=audio_lengths,
        cond=cond,
        noise=noise,
        sample_rate=sample_rate,
    )


def wandb_val_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {f"val/{key.removeprefix('val_')}": value for key, value in metrics.items()}


def _write_audio_batch(
    audio: torch.Tensor,
    sample_rate: int,
    sample_dir: Path,
    pattern: str,
) -> list[Path]:
    import soundfile as sf

    sample_dir.mkdir(parents=True, exist_ok=True)
    audio = audio.detach().cpu().clamp(-1.0, 1.0)
    paths = []
    for index, example in enumerate(audio):
        path = sample_dir / pattern.format(index=index)
        sf.write(path, example.transpose(0, 1).numpy(), sample_rate)
        paths.append(path)
    return paths


@torch.no_grad()
def log_audio_monitor(
    model,
    cfg,
    device,
    sample_dir: Path,
    step: int,
    monitor: AudioMonitor,
    logger: WandbLogger,
    ema: EMA | None = None,
) -> None:
    cfg_wandb = wandb_cfg(cfg)
    log_reference_once = bool(cfg_wandb.get("log_reference_once", True))
    if not log_reference_once or not monitor.references_logged:
        captions = [f"reference {index}" for index in range(monitor.audio.shape[0])]
        if bool(cfg_wandb.get("save_reference_local", True)):
            reference_paths = _write_audio_batch(
                monitor.audio,
                monitor.sample_rate,
                sample_dir,
                "reference_{index:03d}.wav",
            )
            logger.audio("audio/reference", reference_paths, captions, step=step)
        else:
            logger.audio_arrays(
                "audio/reference", monitor.audio, monitor.sample_rate, captions, step=step
            )
        monitor.references_logged = True

    was_training = model.training
    model.eval()
    cond = None if monitor.cond is None else monitor.cond.to(device)
    try:
        with ema_swapped(ema, model):
            generated = sample_fm(
                model,
                shape=tuple(monitor.audio.shape),
                cond=cond,
                steps=int(cfg.sampling.get("steps", 1)),
                eps=float(cfg.flow.get("eps", 1e-5)),
                device=device,
                method=str(cfg.sampling.get("method", "euler")),
                noise=monitor.noise,
                rms_lift=bool(cfg.data.get("rms_lift", False)),
                lift_scale=float(cfg.data.get("lift_scale", 3.0)),
            )
    finally:
        model.train(was_training)

    generated_paths = _write_audio_batch(
        generated,
        monitor.sample_rate,
        sample_dir,
        f"step_{step:08d}_generated_{{index:03d}}.wav",
    )
    logger.audio(
        "audio/generated",
        generated_paths,
        [f"generated {index} step {step}" for index in range(len(generated_paths))],
        step=step,
    )
