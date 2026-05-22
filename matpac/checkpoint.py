from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from matpac.matpac import MATPAC


def load_conditioner_from_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    device: str = "cuda",
) -> torch.nn.Module:
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    config_path = Path(config_path).expanduser() if config_path is not None else checkpoint_path.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.model.model if "model" in cfg and "model" in cfg.model else cfg.model
    params = {key: value for key, value in OmegaConf.to_container(model_cfg, resolve=True).items() if key != "_target_"}
    model = MATPAC(**params)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model_state = {key[len("model.") :]: value for key, value in state_dict.items() if key.startswith("model.")}
    if not model_state:
        model_state = state_dict
    model.load_state_dict(model_state, strict=True)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model
