from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf


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

    def finish(self) -> None:
        self.run.finish()


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


def wandb_val_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {f"val/{key.removeprefix('val_')}": value for key, value in metrics.items()}


def save_wavs(
    audio: torch.Tensor,
    sample_rate: int,
    sample_dir: Path,
    pattern: str,
) -> list[Path]:
    """Write a [B, C, T] batch to ``sample_dir`` as wavs named by ``pattern.format(index=i)``."""
    import soundfile as sf

    sample_dir.mkdir(parents=True, exist_ok=True)
    audio = audio.detach().cpu().clamp(-1.0, 1.0)
    paths = []
    for index, example in enumerate(audio):
        path = sample_dir / pattern.format(index=index)
        sf.write(path, example.transpose(0, 1).numpy(), sample_rate)
        paths.append(path)
    return paths
