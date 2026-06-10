from __future__ import annotations

from typing import Any

from flow.fm import RectifiedFlow

# Generative-method protocol: a stateless object exposing ``train_tuple``/``loss`` and a
# differentiable ``generate(model, shape, cond, noise, steps, ...) -> audio`` (with a no-grad
# ``sample`` wrapper). FD post-training reconstructs the method from a checkpoint's cfg via this
# factory, so it inherits the base model's generation without naming a concrete method.
_METHODS: dict[str, type] = {
    "rectified_flow": RectifiedFlow,
}


def build_method(cfg: Any) -> RectifiedFlow:
    """Build the generative method from cfg. Dispatches on ``cfg.flow.method`` (default rectified_flow)."""
    flow_cfg = cfg.get("flow", {}) if hasattr(cfg, "get") else getattr(cfg, "flow", {})
    kind = str((flow_cfg or {}).get("method", "rectified_flow")).lower()
    if kind not in _METHODS:
        raise ValueError(f"Unknown flow.method: {kind}")
    return _METHODS[kind]()
