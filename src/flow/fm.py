from __future__ import annotations

import torch
import torch.nn.functional as F

from backbone.io import istft_channels, stft_channels

EPS = 1e-5
VELOCITY_CLIP = 0.05  # pMF floors the velocity denominator |1-t| at 0.05, bounding u/v near the data end


class RectifiedFlow:
    """Rectified-flow math: linear interpolant, velocity target, loss, ODE sampling.

    The caller supplies the timesteps ``t`` (the trainer owns the timestep distribution); ``noise``
    defaults to the flow's Gaussian source ``x0 ~ N(0, I)``. ``generate`` divides its output by
    ``wav_scale`` to undo the trainer's waveform amplitude lift (default 1.0 = no-op).

    Flow space (``spec_space``): the flow variable is the waveform (default) or the channelised
    STFT. In spec space the STFT/iSTFT move OUT of the per-eval bracket to the data edges
    (``to_flow``/``from_flow``) and the backbone runs spectrogram->spectrogram directly; ``spec_scale``
    matches the spectrogram std to the N(0, I) noise (the spec analogue of ``wav_scale``).
    ``spec_space=False, spec_scale=1.0`` is the waveform path, bit-for-bit.
    """

    def __init__(self, spec_space: bool = False, spec_scale: float = 1.0):
        self.spec_space = bool(spec_space)
        self.spec_scale = float(spec_scale)

    def to_flow(self, audio: torch.Tensor, model) -> torch.Tensor:
        """Lifted waveform -> flow variable (identity in waveform space)."""
        return stft_channels(audio, model.stft) * self.spec_scale if self.spec_space else audio

    def from_flow(self, x: torch.Tensor, model, length: int) -> torch.Tensor:
        """Flow variable -> lifted waveform (identity in waveform space)."""
        if not self.spec_space:
            return x
        return istft_channels(x / self.spec_scale, model.stft, model.out_channels, length)

    def _flow_shape(self, model, shape: tuple[int, ...]) -> tuple[int, ...]:
        if not self.spec_space:
            return tuple(shape)
        frames = shape[-1] // model.stft.hop_length + 1
        return (shape[0], 2 * model.out_channels * model.stft.freq_bins, frames)

    def _predict(self, model, x: torch.Tensor, length: int | None = None, with_aux: bool = True, **model_kwargs):
        """Backbone eval, returning ``(pred, aux, audio, spec)``: ``pred`` = the u-head prediction in
        flow space (waveform, or channelised STFT in spec space) — what the loss / sampler / du-dt
        consume, so no call site re-picks; ``aux`` = v-head flow prediction or ``None`` (``with_aux``
        gates it); ``audio`` = u-head waveform (always iSTFT'd, cheap) for waveform aux + inference;
        ``spec`` = u-head channelised STFT for the complex aux (descaled by ``spec_scale``)."""
        length = x.shape[-1] if length is None else length
        spec_in = x if self.spec_space else stft_channels(x, model.stft)
        out = model(spec_in, return_aux=True, **model_kwargs) if with_aux else model(spec_in, **model_kwargs)
        u_chan, v_chan = out if isinstance(out, tuple) else (out, None)
        audio = istft_channels(u_chan / self.spec_scale, model.stft, model.out_channels, length)
        if v_chan is None:
            aux = None
        elif self.spec_space:
            aux = v_chan
        else:
            aux = istft_channels(v_chan / self.spec_scale, model.stft, model.out_channels, length)
        return (u_chan if self.spec_space else audio), aux, audio, u_chan

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
        denom = 1.0 - self._time_like(t, x_t)
        sign = torch.where(denom < 0, -torch.ones_like(denom), torch.ones_like(denom))
        denom = sign * denom.abs().clamp_min(VELOCITY_CLIP)
        return (target - x_t) / denom

    def v_to_target(self, v: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x_t + (1.0 - self._time_like(t, x_t)) * v

    @torch.no_grad()
    def guided_velocity_target(
        self,
        model,
        x1: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None,
        omega: torch.Tensor,
        length: int,
        interval_lo,
        interval_hi,
        model_t_lo: torch.Tensor | None = None,
        model_t_hi: torch.Tensor | None = None,
    ) -> torch.Tensor:
        v_cond = self.target_to_v(x1, x_t, t)
        if cond is None:
            return v_cond
        kwargs = {
            "t": t,
            "omega": omega,
            "t_lo": model_t_lo,
            "t_hi": model_t_hi,
            "length": length,
        }
        v_c = self.target_to_v(self._predict(model, x_t, with_aux=False, cond=cond, **kwargs)[0], x_t, t)
        v_u = self.target_to_v(
            self._predict(model, x_t, with_aux=False, cond=torch.zeros_like(cond), **kwargs)[0], x_t, t
        )
        coeff = 1.0 - 1.0 / self._time_like(omega, v_cond)
        mask = self._time_like(
            ((t >= interval_lo) & (t <= interval_hi)).to(v_cond.dtype),
            v_cond,
        )
        return v_cond + coeff * mask * (v_c - v_u)

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
        wav_scale: float = 1.0,
        return_spec: bool = False,
    ) -> torch.Tensor:
        """No-grad ODE sampling (inference / logging). See :meth:`generate` for the grad-carrying path."""
        return self.generate(model, shape, cond, noise, steps, method, guidance_scale, wav_scale, return_spec)

    def generate(
        self,
        model,
        shape: tuple[int, ...],
        cond: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        steps: int = 1,
        method: str = "euler",
        guidance_scale: float = 1.0,
        wav_scale: float = 1.0,
        return_spec: bool = False,
    ) -> torch.Tensor:
        """ODE integration noise → audio, differentiable w.r.t. ``model`` (used by FD-loss).

        Integration runs in the flow space; ``noise``/``shape`` are waveform-shaped and the spec
        flow's noise/output crossings are handled by ``_flow_shape``/``from_flow``. ``return_spec``
        additionally returns the model's raw channelised STFT at the final step (for the
        consistency-residual metric)."""
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if method not in {"euler", "heun"}:
            raise ValueError(f"method must be 'euler' or 'heun', got {method!r}")

        device = next(model.parameters()).device
        length = shape[-1]
        flow_shape = self._flow_shape(model, shape)
        x = torch.randn(flow_shape, device=device) if noise is None else noise.to(device=device)
        if tuple(x.shape) != tuple(flow_shape):
            raise ValueError(f"noise shape {tuple(x.shape)} must match flow shape {tuple(flow_shape)}")

        if cond is not None:
            cond = cond.to(device=device, dtype=x.dtype)
        batch = shape[0]
        grid = torch.linspace(EPS, 1.0 - EPS, steps + 1, device=device, dtype=x.dtype)
        spec_chan = None
        for i in range(steps):
            t = grid[i].expand(batch)
            dt = grid[i + 1] - grid[i]
            v = self._model_v(model, x, t, cond, length, guidance_scale)
            if return_spec and i == steps - 1:
                spec_chan = self._predict(model, x, t=t, cond=cond, length=length, with_aux=False)[3]

            if method == "euler" or i == steps - 1:
                x = x + dt * v
                continue

            x_next = x + dt * v
            t_next = grid[i + 1].expand(batch)
            v_next = self._model_v(model, x_next, t_next, cond, length, guidance_scale)
            x = x + 0.5 * dt * (v + v_next)
        # ``from_flow`` crosses spec-flow back to waveform; ``wav_scale`` undoes the WavFlow lift.
        wav = self.from_flow(x, model, length) / wav_scale
        return (wav, spec_chan) if return_spec else wav

    def _model_v(
        self,
        model,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None,
        length: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        x_pred = self._predict(model, x, t=t, cond=cond, length=length, with_aux=False)[0]
        if cond is not None and guidance_scale != 1.0:
            x_null = self._predict(model, x, t=t, cond=None, length=length, with_aux=False)[0]
            x_pred = x_null + float(guidance_scale) * (x_pred - x_null)
        return self.target_to_v(x_pred, x, t)

    @staticmethod
    def _time_like(t: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if t.ndim == reference.ndim:
            return t
        if t.ndim != 1:
            raise ValueError(f"Expected time [B] or broadcastable tensor, got {tuple(t.shape)}")
        return t.view((t.shape[0],) + (1,) * (reference.ndim - 1))
