from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-5


class RectifiedFlow:
    """Rectified-flow math: linear interpolant, velocity target, loss, ODE sampling.

    Stateless. The caller supplies the timesteps ``t`` (the trainer owns the timestep
    distribution); ``noise`` defaults to the flow's Gaussian source ``x0 ~ N(0, I)``.
    ``sample`` divides its output by ``lift_scale`` to undo the trainer's amplitude lift
    (default 1.0 = no-op); the caller passes the data config's value.
    """

    def train_tuple(
        self,
        x1: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = torch.randn_like(x1) if noise is None else noise.to(device=x1.device, dtype=x1.dtype)
        if x0.shape != x1.shape:
            raise ValueError(f"noise shape {tuple(x0.shape)} must match x1 {tuple(x1.shape)}")
        t = t.to(device=x1.device, dtype=x1.dtype)
        t_view = self._time_like(t, x1)
        x_t = (1.0 - t_view) * x0 + t_view * x1
        return x_t, t, x1

    def target_to_v(self, target: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (target - x_t) / (1.0 - self._time_like(t, x_t)).clamp_min(EPS)

    def loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        space: str = "v",
        loss_type: str = "mse",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if space not in {"x", "v"}:
            raise ValueError("space must be 'x' or 'v'")
        if loss_type not in {"mse", "l1"}:
            raise ValueError("loss_type must be 'mse' or 'l1'")

        if space == "v":
            pred = self.target_to_v(pred, x_t, t)
            target = self.target_to_v(target, x_t, t)

        loss = F.mse_loss(pred, target) if loss_type == "mse" else F.l1_loss(pred, target)
        return loss, {f"{space}_{loss_type}": loss}

    @torch.no_grad()
    def sample(
        self,
        model,
        shape: tuple[int, ...],
        cond: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        steps: int = 1,
        method: str = "euler",
        guidance_scale: float = 1.0,
        lift_scale: float = 1.0,
    ) -> torch.Tensor:
        """No-grad ODE sampling (inference / logging). See :meth:`generate` for the grad-carrying path."""
        return self.generate(model, shape, cond, noise, steps, method, guidance_scale, lift_scale)

    def generate(
        self,
        model,
        shape: tuple[int, ...],
        cond: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        steps: int = 1,
        method: str = "euler",
        guidance_scale: float = 1.0,
        lift_scale: float = 1.0,
    ) -> torch.Tensor:
        """ODE integration noise → audio, differentiable w.r.t. ``model`` (used by FD-loss)."""
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if method not in {"euler", "heun"}:
            raise ValueError(f"method must be 'euler' or 'heun', got {method!r}")

        device = next(model.parameters()).device
        x = torch.randn(shape, device=device) if noise is None else noise.to(device=device)
        if tuple(x.shape) != tuple(shape):
            raise ValueError(f"noise shape {tuple(x.shape)} must match sample shape {tuple(shape)}")

        if cond is not None:
            cond = cond.to(device=device, dtype=x.dtype)
        batch = shape[0]
        length = shape[-1]
        grid = torch.linspace(EPS, 1.0 - EPS, steps + 1, device=device, dtype=x.dtype)
        for i in range(steps):
            t = grid[i].expand(batch)
            dt = grid[i + 1] - grid[i]
            v = self._model_v(model, x, t, cond, length, guidance_scale)

            if method == "euler" or i == steps - 1:
                x = x + dt * v
                continue

            x_next = x + dt * v
            t_next = grid[i + 1].expand(batch)
            v_next = self._model_v(model, x_next, t_next, cond, length, guidance_scale)
            x = x + 0.5 * dt * (v + v_next)
        # ``lift_scale`` undoes the trainer's WavFlow amplitude lift; 1.0 (default) is a no-op.
        return x / lift_scale

    def _model_v(
        self,
        model,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None,
        length: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        x_pred = model(x, t=t, cond=cond, length=length)
        if cond is not None and guidance_scale != 1.0:
            x_null = model(x, t=t, cond=None, length=length)
            x_pred = x_null + float(guidance_scale) * (x_pred - x_null)
        return self.target_to_v(x_pred, x, t)

    @staticmethod
    def _time_like(t: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if t.ndim == reference.ndim:
            return t
        if t.ndim != 1:
            raise ValueError(f"Expected time [B] or broadcastable tensor, got {tuple(t.shape)}")
        return t.view((t.shape[0],) + (1,) * (reference.ndim - 1))
