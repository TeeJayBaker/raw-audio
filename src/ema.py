from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    def update(self, model: nn.Module) -> None:
        for name, param in self._parameters(model):
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
            else:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in self._parameters(model):
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in self._parameters(model):
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        self.shadow = {name: value.clone() for name, value in state["shadow"].items()}

    def _parameters(self, model: nn.Module) -> Iterator[tuple[str, nn.Parameter]]:
        return ((name, param) for name, param in model.named_parameters() if param.requires_grad)


@contextmanager
def ema_swapped(ema: EMA | None, model: nn.Module):
    """Swap EMA weights into `model` for the duration of the block; always restore."""
    if ema is None:
        yield
        return
    ema.apply_to(model)
    try:
        yield
    finally:
        ema.restore(model)
