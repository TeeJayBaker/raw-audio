"""Optimizers. Vendored Muon (``muon.py``, verbatim from KellerJordan/Muon) plus the glue that
builds it from a flat parameter iterable, so the trainer's Hydra ``instantiate`` path can use it."""
from __future__ import annotations

from optim.muon import (
    SingleDeviceMuon,
    SingleDeviceMuonWithAuxAdam,
    adam_update,
    muon_update,
    zeropower_via_newtonschulz5,
)

__all__ = [
    "SingleDeviceMuon",
    "SingleDeviceMuonWithAuxAdam",
    "adam_update",
    "build_muon",
    "muon_update",
    "zeropower_via_newtonschulz5",
]


def build_muon(
    params,
    lr: float = 0.02,
    momentum: float = 0.95,
    weight_decay: float = 0.0,
    adamw_lr: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-10,
) -> SingleDeviceMuonWithAuxAdam:
    """Build the vendored ``SingleDeviceMuonWithAuxAdam`` from a flat param iterable.

    Splits by dimensionality — ``ndim>=2`` weight matrices go to the Muon group, ``ndim<2`` gains
    and biases to the AdamW-aux group — so the trainer's
    ``instantiate(cfg.optimizer, params=model.parameters())`` call constructs Muon with no trainer
    change. ``lr`` is the Muon-group LR (units of spectral norm per update, ~0.02, ≫ Adam's);
    ``adamw_lr`` the aux-group LR. Per Keller Jordan's guidance Muon is for hidden weight matrices
    only — this backbone has no discrete embedding table and an ``identity`` head, so the ndim split
    is the right grouping here; revisit if either changes. Muon is a from-scratch optimizer: pair it
    with dropping ``train.init_from``, not a warm-started fine-tune.

    Hydra: ``optimizer._target_=optim.build_muon`` (set ``optimizer.lr`` ~0.02 explicitly).
    """
    params = [p for p in params if p.requires_grad]
    muon = [p for p in params if p.ndim >= 2]
    aux = [p for p in params if p.ndim < 2]
    groups: list[dict] = []
    if muon:
        groups.append(
            {"params": muon, "use_muon": True, "lr": lr, "momentum": momentum, "weight_decay": weight_decay}
        )
    if aux:
        groups.append(
            {"params": aux, "use_muon": False, "lr": adamw_lr, "betas": tuple(betas), "eps": eps,
             "weight_decay": weight_decay}
        )
    return SingleDeviceMuonWithAuxAdam(groups)
