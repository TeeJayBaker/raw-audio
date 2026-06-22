from __future__ import annotations

import torch

from backbone.blocks import Attention
from backbone.transformer import Transformer
from flow.fm import EPS, VELOCITY_CLIP, RectifiedFlow
from flow.mf import MeanFlow


class WaveformMeanFlow(MeanFlow):
    """MeanFlow whose _predict skips the STFT bracket, so the dde/jvp/CFG math can be
    exercised with analytic waveform-space toy models. The real STFT crossing is covered
    by test_mean_flow_dde_jvp_agree_on_stft_transformer (uses the real MeanFlow)."""

    def _predict(self, model, x, length=None, with_aux=True, **model_kwargs):
        out = model(x, **model_kwargs)  # waveform-space toy: no STFT, no v-head
        return out, None, out, out  # (pred, aux, audio, spec)


class WaveformRF(RectifiedFlow):
    """RectifiedFlow with the STFT bracket disabled (waveform-space toy models)."""

    def _predict(self, model, x, length=None, with_aux=True, **model_kwargs):
        out = model(x, **model_kwargs)
        return out, None, out


def _tiny_stft_transformer() -> Transformer:
    return Transformer(
        channels=1,
        sample_rate=8000,
        stft={"n_fft": 64, "hop_length": 16, "win_length": 64},
        block={"dim": 32, "depth": 1, "heads": 2},
        conditioning={"cond_dim": 16, "embed_dim": 16, "gap_embed": True, "time_scale": 1.0},
    )


class ConstantVelocityModel(torch.nn.Module):
    def __init__(self, c: torch.Tensor):
        super().__init__()
        self.register_buffer("c", c)

    def forward(self, x, t=None, h=None, cond=None, omega=None, t_lo=None, t_hi=None, length=None):
        del h, cond, omega, t_lo, t_hi, length
        return x + (1.0 - t.view(-1, 1, 1)) * self.c


class CurvedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)

    def forward(self, x, t=None, h=None, cond=None, omega=None, t_lo=None, t_hi=None, length=None):
        del cond, omega, t_lo, t_hi, length
        t = t.view(-1, 1, 1)
        h = torch.zeros_like(t) if h is None else h.view(-1, 1, 1)
        mix = torch.tanh(self.proj(x))
        return x * torch.cos(t + h) + (1.0 - t) * mix * (1.0 + h)


class CondShiftModel(torch.nn.Module):
    def forward(self, x, t=None, h=None, cond=None, omega=None, t_lo=None, t_hi=None, length=None):
        del h, omega, t_lo, t_hi, length
        return x + (1.0 - t.view(-1, 1, 1)) * (
            1.0 + cond.mean(dim=1).view(-1, 1, 1)
        )


def _pair(batch=4, length=8):
    torch.manual_seed(0)
    c = torch.randn(1, 1, length)
    x0 = torch.randn(batch, 1, length)
    return c, x0, x0 + c


def test_constant_field_satisfies_mf_identity():
    flow = WaveformMeanFlow()
    c, x0, x1 = _pair()
    model = ConstantVelocityModel(c)
    t = torch.tensor([0.1, 0.4, 0.6, 0.8])
    h = torch.tensor([0.3, 0.2, 0.3, 0.1])
    x_t, t, _ = flow.train_tuple(x1, t=t, noise=x0)
    v_tgt = flow.target_to_v(x1, x_t, t)
    for mode in ("dde", "jvp"):
        u, dudt, _, _ = flow.u_and_dudt(
            model,
            x_t,
            t,
            h,
            v_tgt,
            cond=None,
            length=8,
            mode=mode,
            dde_eps=5e-3,
        )
        V = u - flow._time_like(h, u) * dudt.detach()
        assert torch.allclose(V, v_tgt.expand_as(V), atol=1e-4), mode
        loss, terms = flow.mf_loss(V, v_tgt.expand_as(V), p=1.0, c=1e-3)
        assert loss < 1e-6
        assert "mf_mse" in terms


def test_dde_matches_jvp_on_curved_model():
    flow = WaveformMeanFlow()
    torch.manual_seed(1)
    model = CurvedModel()
    x_t = torch.randn(4, 1, 8)
    t = torch.tensor([0.2, 0.4, 0.6, 0.8])
    h = torch.tensor([0.1, 0.2, 0.15, 0.1])
    tangent = torch.randn_like(x_t)
    _, dudt_dde, _, _ = flow.u_and_dudt(
        model, x_t, t, h, tangent, None, 8, mode="dde", dde_eps=1e-3
    )
    _, dudt_jvp, _, _ = flow.u_and_dudt(
        model, x_t, t, h, tangent, None, 8, mode="jvp", dde_eps=1e-3
    )
    assert torch.allclose(dudt_dde, dudt_jvp, atol=1e-2, rtol=1e-2)


def test_attention_jvp_flag_enables_forward_ad_and_resets():
    # The fused SDPA kernels lack forward-AD; the flag must route to a forward-AD-capable
    # path (Triton on CUDA, MATH backend on CPU) and always reset, even on error.
    attn = Attention(dim=16, heads=2)
    x = torch.randn(1, 8, 16)
    tx = torch.randn_like(x)
    Attention.set_triton_jvp(True)
    try:
        out, dout = torch.func.jvp(attn, (x,), (tx,))
    finally:
        Attention.set_triton_jvp(False)
    assert out.shape == x.shape and dout.shape == x.shape
    assert torch.isfinite(dout).all() and dout.abs().sum() > 0
    assert not Attention._use_triton_jvp


def test_mean_flow_dde_jvp_agree_on_stft_transformer():
    # End-to-end A/B on the real backbone: jvp (MATH-backend forward-AD on CPU) must match the dde
    # central difference, and the returned u-head audio / channelised spec must agree in both.
    torch.manual_seed(0)
    model = _tiny_stft_transformer()
    flow = MeanFlow()
    x_t = torch.randn(2, 1, 256)
    t = torch.tensor([0.3, 0.6])
    h = torch.tensor([0.12, 0.18])
    tangent = torch.randn_like(x_t)
    args = (model, x_t, t, h, tangent, None, 256)
    u_d, dudt_d, audio_d, spec_d = flow.u_and_dudt(*args, mode="dde", dde_eps=1e-3)
    u_j, dudt_j, audio_j, spec_j = flow.u_and_dudt(*args, mode="jvp")

    assert torch.allclose(u_d, u_j, atol=1e-4) and torch.allclose(audio_d, audio_j, atol=1e-4)
    assert torch.allclose(dudt_d, dudt_j, atol=2e-2, rtol=2e-2)
    assert audio_d.shape == x_t.shape  # u-head waveform
    assert not spec_d.is_complex() and spec_d.shape == spec_j.shape  # channelised STFT
    assert spec_d.shape[1] == 2 * model.out_channels * model.stft.freq_bins
    assert torch.allclose(spec_d, spec_j, atol=1e-4)
    assert not Attention._use_triton_jvp


def test_target_to_v_clips_denominator_at_velocity_clip():
    # pMF floors |1-t| at VELOCITY_CLIP, bounding the velocity near (and past) the data endpoint
    # while preserving its sign across t=1.
    flow = MeanFlow()
    x_t = torch.randn(2, 1, 8)
    t = torch.tensor([0.99, 1.01])  # |1-t| = 0.01 < clip, on both sides of the endpoint
    target = torch.randn_like(x_t)
    denom = torch.tensor([VELOCITY_CLIP, -VELOCITY_CLIP]).view(2, 1, 1)  # floored, sign preserved
    assert torch.allclose(flow.target_to_v(target, x_t, t), (target - x_t) / denom, atol=1e-5)


def test_u_carries_grad_and_dudt_does_not():
    flow = WaveformMeanFlow()
    model = CurvedModel()
    x_t = torch.randn(2, 1, 8)
    t = torch.tensor([0.3, 0.5])
    h = torch.tensor([0.2, 0.2])
    u, dudt, _, _ = flow.u_and_dudt(
        model,
        x_t,
        t,
        h,
        torch.randn_like(x_t),
        None,
        8,
        mode="dde",
        dde_eps=5e-3,
    )
    assert u.requires_grad and not dudt.requires_grad


def test_one_step_generate_returns_endpoint_prediction():
    flow = WaveformMeanFlow()
    c, _, _ = _pair()
    model = ConstantVelocityModel(c)
    noise = torch.randn(2, 1, 8)
    sample = flow.sample(model, shape=(2, 1, 8), noise=noise, steps=1)
    expected = noise + (1.0 - 2 * EPS) * c
    assert torch.allclose(sample, expected, atol=1e-5)


def test_generate_runs_with_omega_and_interval():
    flow = WaveformMeanFlow()
    model = ConstantVelocityModel(torch.zeros(1, 1, 8))
    out = flow.sample(
        model,
        shape=(2, 1, 8),
        steps=1,
        guidance_scale=3.0,
        guidance_t_lo=0.1,
        guidance_t_hi=0.8,
    )
    assert out.shape == (2, 1, 8)


def test_guided_target_formula_fixed_interval():
    flow = WaveformRF()
    model = CondShiftModel()
    x1 = torch.randn(2, 1, 8)
    x_t, t, _ = flow.train_tuple(x1, t=torch.tensor([0.3, 0.9]))
    cond = torch.full((2, 4), 2.0)
    omega = torch.tensor([3.0, 3.0])
    v_cond = flow.target_to_v(x1, x_t, t)
    v_g = flow.guided_velocity_target(
        model, x1, x_t, t, cond, omega, 8, 0.0, 0.8, None, None
    )
    assert torch.allclose(v_g[0], v_cond[0] + (1 - 1 / 3) * 2.0, atol=1e-5)
    assert torch.allclose(v_g[1], v_cond[1], atol=1e-5)


def test_guided_target_conditioned_interval_per_row():
    flow = WaveformRF()
    model = CondShiftModel()
    x1 = torch.randn(2, 1, 8)
    x_t, t, _ = flow.train_tuple(x1, t=torch.tensor([0.3, 0.7]))
    cond = torch.full((2, 4), 2.0)
    omega = torch.tensor([3.0, 3.0])
    t_lo = torch.tensor([0.0, 0.0])
    t_hi = torch.tensor([0.5, 0.5])
    v_cond = flow.target_to_v(x1, x_t, t)
    v_g = flow.guided_velocity_target(
        model, x1, x_t, t, cond, omega, 8, t_lo, t_hi, t_lo, t_hi
    )
    assert torch.allclose(v_g[0], v_cond[0] + (1 - 1 / 3) * 2.0, atol=1e-5)
    assert torch.allclose(v_g[1], v_cond[1], atol=1e-5)


def test_guided_target_omega_one_and_unconditional_are_v_cond():
    flow = WaveformRF()
    x1 = torch.randn(2, 1, 8)
    x_t, t, _ = flow.train_tuple(x1, t=torch.rand(2))
    v_cond = flow.target_to_v(x1, x_t, t)
    g1 = flow.guided_velocity_target(
        CondShiftModel(),
        x1,
        x_t,
        t,
        torch.full((2, 4), 2.0),
        torch.ones(2),
        8,
        0.0,
        1.0,
        None,
        None,
    )
    assert torch.allclose(g1, v_cond, atol=1e-6)
    g2 = flow.guided_velocity_target(
        CondShiftModel(),
        x1,
        x_t,
        t,
        None,
        torch.full((2,), 3.0),
        8,
        0.0,
        1.0,
        None,
        None,
    )
    assert torch.allclose(g2, v_cond)


class _LinearBoundaryModel(torch.nn.Module):
    """x-pred whose velocity is a Linear of z, so under bf16 autocast the boundary forward is
    rounded to bf16 and a naive central difference for du/dt cancels catastrophically."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)

    def forward(self, x, t=None, h=None, cond=None, omega=None, t_lo=None, t_hi=None, length=None):
        del cond, omega, t_lo, t_hi, length
        t = t.view(-1, 1, 1)
        h = torch.zeros_like(t) if h is None else h.view(-1, 1, 1)
        return x + (1.0 - t) * (self.proj(x) + h)


def test_dde_dudt_runs_fp32_under_bf16_autocast():
    # du/dt is a central difference / (2*eps); under bf16 autocast the two probe forwards cancel
    # catastrophically. u_and_dudt must force the probes to fp32 so du/dt is invariant to the
    # outer autocast — the bug that drove the MF run's 1-NFE output to noise.
    flow = WaveformMeanFlow()
    torch.manual_seed(0)
    model = _LinearBoundaryModel()
    x_t = torch.randn(4, 1, 8)
    t = torch.tensor([0.2, 0.4, 0.6, 0.8])
    h = torch.tensor([0.1, 0.2, 0.15, 0.1])
    tangent = torch.randn_like(x_t)
    args = (model, x_t, t, h, tangent, None, 8)
    _, dudt_fp32, _, _ = flow.u_and_dudt(*args, mode="dde", dde_eps=5e-3)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _, dudt_auto, _, _ = flow.u_and_dudt(*args, mode="dde", dde_eps=5e-3)
    rel = (dudt_auto.float() - dudt_fp32).norm() / dudt_fp32.norm().clamp_min(1e-9)
    assert rel < 0.02, f"dde du/dt not fp32-stable under bf16 autocast: rel={rel:.3f}"
