from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import nn

from emb.clap import CLAPEmbedding
from emb.matpac import MATPACEmbedding
from emb.null import NullEmbedding
from emb.vggish import VGGishEmbedding

_EMBEDDINGS: dict[str, type[nn.Module]] = {
    "clap": CLAPEmbedding,
    "matpac": MATPACEmbedding,
    "null": NullEmbedding,
    "vggish": VGGishEmbedding,
}


def build_embedding(cfg: dict[str, Any] | None, device: str | torch.device = "cpu") -> nn.Module | None:
    cfg = dict(cfg or {})
    kind = str(cfg.pop("type", cfg.pop("backend", "none"))).lower()
    if kind == "none":
        return None
    if kind not in _EMBEDDINGS:
        raise ValueError(f"Unknown embedding type/backend: {kind}")
    target = _EMBEDDINGS[kind]
    if kind != "null":
        cfg.setdefault("device", str(device))
    # Hydra deep-merges configs, so derived experiments can carry keys meant for a
    # different backend (e.g. matpac fields left over when overriding type=null).
    # Keep only the kwargs this backend's constructor accepts.
    accepted = set(inspect.signature(target).parameters)
    return target(**{key: value for key, value in cfg.items() if key in accepted})


def build_embedding_backend(cfg: dict[str, Any], device: str | torch.device = "cpu") -> nn.Module | None:
    cfg = dict(cfg or {})
    if not bool(cfg.pop("enabled", False)):
        return None
    backend_cfg = dict(cfg.pop("backend_cfg", {}) or {})
    backend_cfg["backend"] = cfg.pop("backend", "clap")
    backend_cfg.setdefault("device", str(device))
    return build_embedding(backend_cfg, device=device)
