from __future__ import annotations

from typing import Any

import torch
from torch import nn

from emb.clap import CLAPEmbedding
from emb.matpac import MATPACEmbedding
from emb.null import NullEmbedding
from emb.vggish import VGGishEmbedding


def build_embedding(cfg: dict[str, Any] | None, device: str | torch.device = "cpu") -> nn.Module | None:
    cfg = dict(cfg or {})
    kind = str(cfg.pop("type", cfg.pop("backend", "none"))).lower()
    if kind in {"none", "null"} and not cfg:
        return None
    cfg.setdefault("device", str(device))
    if kind == "none":
        return None
    if kind == "null":
        cfg.pop("device", None)
        return NullEmbedding(**cfg)
    if kind == "matpac":
        return MATPACEmbedding(**cfg)
    if kind == "clap":
        return CLAPEmbedding(**cfg)
    if kind == "vggish":
        return VGGishEmbedding(**cfg)
    raise ValueError(f"Unknown embedding type/backend: {kind}")


def build_embedding_backend(cfg: dict[str, Any], device: str | torch.device = "cpu") -> nn.Module | None:
    cfg = dict(cfg or {})
    if not bool(cfg.pop("enabled", False)):
        return None
    backend_cfg = dict(cfg.pop("backend_cfg", {}) or {})
    backend_cfg["backend"] = cfg.pop("backend", "clap")
    backend_cfg.setdefault("device", str(device))
    return build_embedding(backend_cfg, device=device)
