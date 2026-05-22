from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FlowBatch:
    x_t: torch.Tensor
    t: torch.Tensor
    x0: torch.Tensor
    x1: torch.Tensor
    v: torch.Tensor


def sample_time(batch_size: int, device: torch.device, eps: float = 1e-5) -> torch.Tensor:
    return torch.rand(batch_size, device=device).clamp(eps, 1.0 - eps)


def linear_interpolant(
    x1: torch.Tensor,
    noise: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> FlowBatch:
    x0 = torch.randn_like(x1) if noise is None else noise
    if x0.shape != x1.shape:
        raise ValueError(f"noise shape {tuple(x0.shape)} must match x1 {tuple(x1.shape)}")
    t = sample_time(x1.shape[0], x1.device, eps=eps) if t is None else t.to(device=x1.device)
    t_view = _broadcast_time(t, x1)
    x_t = (1.0 - t_view) * x0 + t_view * x1
    v = x1 - x0
    return FlowBatch(x_t=x_t, t=t, x0=x0, x1=x1, v=v)


def output_to_v(x_pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return (x_pred - x_t) / (1.0 - _broadcast_time(t, x_t)).clamp_min(eps)


@torch.no_grad()
def sample_fm(
    model,
    shape: tuple[int, int, int],
    cond: torch.Tensor | None = None,
    steps: int = 1,
    eps: float = 1e-5,
    device: str | torch.device = "cpu",
    method: str = "euler",
) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if method not in {"euler", "heun"}:
        raise ValueError(f"method must be 'euler' or 'heun', got {method!r}")

    x = torch.randn(shape, device=device)
    batch = shape[0]
    length = shape[-1]
    grid = torch.linspace(eps, 1.0 - eps, steps + 1, device=device)

    for i in range(steps):
        t = grid[i].expand(batch)
        dt = grid[i + 1] - grid[i]
        v = output_to_v(model(x, t=t, cond=cond, length=length), x, t, eps=eps)

        if method == "euler":
            x = x + dt * v
            continue

        x_pred = x + dt * v
        t_next = grid[i + 1].expand(batch)
        v_next = output_to_v(model(x_pred, t=t_next, cond=cond, length=length), x_pred, t_next, eps=eps)
        x = x + 0.5 * dt * (v + v_next)

    return x


def _broadcast_time(t: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if t.ndim == reference.ndim:
        return t
    if t.ndim != 1:
        raise ValueError(f"Expected time [B] or broadcastable tensor, got {tuple(t.shape)}")
    return t.view((t.shape[0],) + (1,) * (reference.ndim - 1))
