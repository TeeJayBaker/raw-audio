from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import nn

from emb.audiomae import AudioMAEEmbedding
from emb.cdpam import CDPAMEmbedding
from emb.clap import CLAPEmbedding
from emb.encodec import EnCodecEmbedding
from emb.matpac import MATPACEmbedding
from emb.mert import MERTEmbedding
from emb.null import NullEmbedding
from emb.pann import PANNEmbedding
from emb.random import RandomProjEmbedding
from emb.vggish import VGGishEmbedding

# Embedder protocol: each wrapper exposes ``embed(audio, sample_rate, audio_lengths) -> [N, D]``
# (gradient-carrying, frozen params), where N is usually clips but may be frames for local
# representations such as EnCodec. ``forward`` = ``no_grad(embed)`` for metric/conditioner paths.
_EMBEDDINGS: dict[str, type[nn.Module]] = {
    "audiomae": AudioMAEEmbedding,
    "cdpam": CDPAMEmbedding,
    "clap": CLAPEmbedding,
    "encodec": EnCodecEmbedding,
    "matpac": MATPACEmbedding,
    "mert": MERTEmbedding,
    "null": NullEmbedding,
    "pann": PANNEmbedding,
    "random": RandomProjEmbedding,
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


def build_embeddings(
    cfg_list: list[dict[str, Any]] | None, device: str | torch.device = "cpu"
) -> list[nn.Module]:
    """Build a list of embedder wrappers (e.g. the FD-loss φ stack) from a list of config dicts."""
    embedders = [build_embedding(cfg, device=device) for cfg in (cfg_list or [])]
    return [embedder for embedder in embedders if embedder is not None]


def build_embedding_backend(cfg: dict[str, Any], device: str | torch.device = "cpu") -> nn.Module | None:
    cfg = dict(cfg or {})
    if not bool(cfg.pop("enabled", False)):
        return None
    backend_cfg = dict(cfg.pop("backend_cfg", {}) or {})
    backend_cfg["backend"] = cfg.pop("backend", "clap")
    backend_cfg.setdefault("device", str(device))
    return build_embedding(backend_cfg, device=device)
