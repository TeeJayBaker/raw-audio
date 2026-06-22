from __future__ import annotations

from typing import Any

from flow.fm import RectifiedFlow
from flow.mf import MeanFlow

# Generative-method protocol: a stateless object exposing ``train_tuple``/``loss`` and a
# differentiable ``generate(model, shape, cond, noise, steps, ...) -> audio`` (with a no-grad
# ``sample`` wrapper). FD post-training reconstructs the method from a checkpoint's cfg via this
# factory, so it inherits the base model's generation without naming a concrete method.
_METHODS: dict[str, type] = {
    "rectified_flow": RectifiedFlow,
    "mean_flow": MeanFlow,
}


def build_method(cfg: Any) -> RectifiedFlow:
    """Build the generative method from cfg. Dispatches on ``cfg.flow.method`` (default rectified_flow)."""
    flow_cfg = cfg.get("flow", {}) if hasattr(cfg, "get") else getattr(cfg, "flow", {})
    flow_cfg = flow_cfg or {}
    kind = str(flow_cfg.get("method", "rectified_flow")).lower()
    if kind not in _METHODS:
        raise ValueError(f"Unknown flow.method: {kind}")
    space = str(flow_cfg.get("space", "waveform")).lower()
    if space not in {"waveform", "spec"}:
        raise ValueError(f"flow.space must be 'waveform' or 'spec', got {space!r}")
    return _METHODS[kind](spec_space=space == "spec", spec_scale=float(flow_cfg.get("spec_scale", 1.0)))
