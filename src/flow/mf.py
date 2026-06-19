from __future__ import annotations

import torch

from flow.fm import EPS, RectifiedFlow


class MeanFlow(RectifiedFlow):
    """MeanFlow on an x-prediction backbone."""

    def u_and_dudt(
        self,
        model,
        x_t: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        tangent: torch.Tensor,
        cond: torch.Tensor | None,
        length: int,
        omega: torch.Tensor | None = None,
        t_lo: torch.Tensor | None = None,
        t_hi: torch.Tensor | None = None,
        mode: str = "dde",
        dde_eps: float = 5e-3,
        return_spec: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if mode not in {"dde", "jvp"}:
            raise ValueError("flow.mf.dudt must be 'dde' or 'jvp'")

        # torch.func.jvp requires every output to be a tensor, so the spec is only part of
        # the (primal, tangent) pytree when requested — never a None leaf.
        def u_fn(z, t_, h_):
            kwargs = dict(t=t_, h=h_, cond=cond, omega=omega, t_lo=t_lo, t_hi=t_hi, length=length)
            if return_spec:
                x_pred, x_spec = model(z, return_spec=True, **kwargs)
                return self.target_to_v(x_pred, z, t_), x_pred, x_spec
            x_pred = model(z, **kwargs)
            return self.target_to_v(x_pred, z, t_), x_pred

        if mode == "jvp":
            from backbone.blocks import Attention

            Attention.set_triton_jvp(True)
            try:
                primals, tangents = torch.func.jvp(
                    u_fn,
                    (x_t, t, h),
                    (tangent, torch.ones_like(t), -torch.ones_like(t)),
                )
            finally:
                Attention.set_triton_jvp(False)
            (u, x_pred, x_spec) = primals if return_spec else (*primals, None)
            return u, tangents[0].detach(), x_pred, x_spec

        (u, x_pred, x_spec) = u_fn(x_t, t, h) if return_spec else (*u_fn(x_t, t, h), None)

        # The central difference divides by 2*dde_eps, so the two probe forwards must run in
        # fp32: under the trainer's bf16 autocast they cancel catastrophically (~85% noise on a
        # real model), corrupting the MF target and slowly driving the 1-NFE field to noise.
        cond_f = None if cond is None else cond.float()
        omega_f = None if omega is None else omega.float()
        t_lo_f = None if t_lo is None else t_lo.float()
        t_hi_f = None if t_hi is None else t_hi.float()

        def u_probe(z, t_, h_):
            x_pred_ = model(z, t=t_, h=h_, cond=cond_f, omega=omega_f, t_lo=t_lo_f, t_hi=t_hi_f, length=length)
            return self.target_to_v(x_pred_, z, t_)

        x_t_f, t_f, h_f, tangent_f = x_t.float(), t.float(), h.float(), tangent.float()
        eps_t = torch.full_like(t_f, dde_eps)
        eps_x = self._time_like(eps_t, x_t_f)
        with torch.no_grad(), torch.autocast(device_type=x_t.device.type, enabled=False):
            u_plus = u_probe(x_t_f + eps_x * tangent_f, t_f + eps_t, h_f - eps_t)
            u_minus = u_probe(x_t_f - eps_x * tangent_f, t_f - eps_t, h_f + eps_t)
            dudt = (u_plus - u_minus) / (2.0 * dde_eps)
        return u, dudt.to(u.dtype), x_pred, x_spec

    @staticmethod
    def mf_loss(
        V: torch.Tensor,
        v_tgt: torch.Tensor,
        p: float = 1.0,
        c: float = 1e-3,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        delta = V.float() - v_tgt.float()
        per_sample = delta.pow(2).mean(dim=tuple(range(1, delta.ndim)))
        weight = (per_sample.detach() + c).pow(-p)
        loss = (weight * per_sample).mean()
        return loss, {"mf_mse": per_sample.mean().detach()}

    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        cond=None,
        noise=None,
        steps=1,
        method="euler",
        guidance_scale=1.0,
        guidance_t_lo=None,
        guidance_t_hi=None,
        lift_scale=1.0,
    ):
        return self.generate(
            model,
            shape,
            cond,
            noise,
            steps,
            method,
            guidance_scale,
            guidance_t_lo,
            guidance_t_hi,
            lift_scale,
        )

    def generate(
        self,
        model,
        shape: tuple[int, ...],
        cond: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        steps: int = 1,
        method: str = "euler",
        guidance_scale: float = 1.0,
        guidance_t_lo: float | None = None,
        guidance_t_hi: float | None = None,
        lift_scale: float = 1.0,
    ) -> torch.Tensor:
        del method
        if steps < 1:
            raise ValueError("steps must be >= 1")
        device = self._model_device(model, noise)
        x = torch.randn(shape, device=device) if noise is None else noise.to(device=device)
        if tuple(x.shape) != tuple(shape):
            raise ValueError(f"noise shape {tuple(x.shape)} must match sample shape {tuple(shape)}")
        if cond is not None:
            cond = cond.to(device=device, dtype=x.dtype)
        batch, length = shape[0], shape[-1]

        def _col(value):
            if value is None:
                return None
            return torch.full((batch,), float(value), device=device, dtype=x.dtype)

        omega = None if float(guidance_scale) == 1.0 else _col(guidance_scale)
        t_lo, t_hi = _col(guidance_t_lo), _col(guidance_t_hi)
        grid = torch.linspace(EPS, 1.0 - EPS, steps + 1, device=device, dtype=x.dtype)
        for i in range(steps):
            t = grid[i].expand(batch)
            dt = grid[i + 1] - grid[i]
            h = dt.expand(batch)
            x_pred = model(
                x,
                t=t,
                h=h,
                cond=cond,
                omega=omega,
                t_lo=t_lo,
                t_hi=t_hi,
                length=length,
            )
            x = x + dt * self.target_to_v(x_pred, x, t)
        return x / lift_scale

    @staticmethod
    def _model_device(model, noise: torch.Tensor | None) -> torch.device:
        parameter = next(model.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(model.buffers(), None)
        if buffer is not None:
            return buffer.device
        return noise.device if noise is not None else torch.device("cpu")
