from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def _config_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.exists():
        return path
    if path.suffix != ".yaml":
        path = Path("configs/backbone") / f"{name_or_path}.yaml"
    if path.exists():
        return path
    raise FileNotFoundError(f"Backbone config not found: {name_or_path}")


def load_backbone_config(cfg: str | Path | dict[str, Any] | DictConfig) -> DictConfig:
    if isinstance(cfg, DictConfig):
        return cfg
    if isinstance(cfg, dict):
        return OmegaConf.create(cfg)
    return OmegaConf.load(_config_path(str(cfg)))


def build_backbone(cfg: str | Path | dict[str, Any] | DictConfig):
    return instantiate(load_backbone_config(cfg))
