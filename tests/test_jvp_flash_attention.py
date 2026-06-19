"""Correctness tests for the Triton JVP Flash Attention kernels.

Ported from ../flow-one-shot. GPU + Triton only; skipped otherwise. The kernel is the
production forward-AD attention path used by MeanFlow's `flow.mf.dudt=jvp` on CUDA
(`MeanFlow.u_and_dudt` flips `Attention.set_triton_jvp` around `torch.func.jvp`).
"""

import pytest
import torch
from torch.func import jvp

pytest.importorskip("triton")

from backbone.jvp_flash_attention import (  # noqa: E402
    _triton_fwd,
    _triton_jvp,
    flash_attn_jvp,
    fused_attn_fwd_jvp,
    reference_attn_jvp,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda"
# Use IEEE precision for exact correctness checks; production uses TF32
FP = "ieee"


def _rand(shape, device=DEVICE, requires_grad=False):
    return torch.randn(shape, device=device, dtype=torch.float32, requires_grad=requires_grad)


# ─── Forward-only correctness ──────────────────────────────────────


@pytest.mark.parametrize("BH,N,D", [(4, 64, 64), (12, 128, 64), (1, 37, 64), (96, 256, 64)])
def test_forward_matches_sdpa(BH, N, D):
    """Triton forward should match SDPA MATH backend output."""
    q, k, v = _rand((BH, N, D)), _rand((BH, N, D)), _rand((BH, N, D))
    scale = D**-0.5

    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):
        ref_o = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), scale=scale,
        ).squeeze(1)

    tri_o, lse = _triton_fwd(q, k, v, scale, fp_precision=FP)
    torch.testing.assert_close(tri_o, ref_o, atol=1e-4, rtol=1e-4)


# ─── Two-pass JVP correctness ──────────────────────────────────────


@pytest.mark.parametrize("BH,N,D", [(4, 64, 64), (12, 128, 64), (1, 37, 64), (96, 256, 64)])
def test_two_pass_jvp_matches_reference(BH, N, D):
    """Two-pass Triton JVP should match MATH backend JVP."""
    q, k, v = _rand((BH, N, D)), _rand((BH, N, D)), _rand((BH, N, D))
    dq, dk, dv = _rand((BH, N, D)), _rand((BH, N, D)), _rand((BH, N, D))
    scale = D**-0.5

    ref_o, ref_do = reference_attn_jvp(q, k, v, dq, dk, dv, scale)
    tri_o, lse = _triton_fwd(q, k, v, scale, fp_precision=FP)
    tri_do = _triton_jvp(q, k, v, tri_o, lse, dq, dk, dv, scale, fp_precision=FP)

    torch.testing.assert_close(tri_o, ref_o, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(tri_do, ref_do, atol=1e-3, rtol=1e-3)


# ─── Fused JVP correctness ─────────────────────────────────────────


@pytest.mark.parametrize("BH,N,D", [(4, 64, 64), (12, 128, 64), (1, 37, 64), (96, 256, 64)])
def test_fused_jvp_matches_reference(BH, N, D):
    """Fused single-pass Triton JVP should match MATH backend JVP."""
    q, k, v = _rand((BH, N, D)), _rand((BH, N, D)), _rand((BH, N, D))
    dq, dk, dv = _rand((BH, N, D)), _rand((BH, N, D)), _rand((BH, N, D))
    scale = D**-0.5

    ref_o, ref_do = reference_attn_jvp(q, k, v, dq, dk, dv, scale)
    tri_o, tri_do = fused_attn_fwd_jvp(
        q.contiguous(), k.contiguous(), v.contiguous(),
        dq.contiguous(), dk.contiguous(), dv.contiguous(),
        scale, fp_precision=FP,
    )

    torch.testing.assert_close(tri_o, ref_o, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(tri_do, ref_do, atol=1e-3, rtol=1e-3)


# ─── Autograd Function + torch.func.jvp integration ────────────────


@pytest.mark.parametrize("B,H,N,D", [(2, 4, 64, 64), (1, 12, 128, 64), (4, 12, 100, 64)])
def test_flash_attn_jvp_via_func_jvp(B, H, N, D):
    """flash_attn_jvp works correctly inside torch.func.jvp.

    Uses TF32 (production default) so tolerances are wider.
    """
    q, k, v = _rand((B, H, N, D)), _rand((B, H, N, D)), _rand((B, H, N, D))
    dq, dk, dv = _rand((B, H, N, D)), _rand((B, H, N, D)), _rand((B, H, N, D))
    scale = D**-0.5

    def triton_attn(q_, k_, v_):
        return flash_attn_jvp(q_, k_, v_, scale=scale)

    tri_o, tri_do = jvp(triton_attn, (q, k, v), (dq, dk, dv))

    from torch.nn.attention import SDPBackend, sdpa_kernel

    def ref_attn(q_, k_, v_):
        return torch.nn.functional.scaled_dot_product_attention(q_, k_, v_, scale=scale)

    with sdpa_kernel(SDPBackend.MATH):
        ref_o, ref_do = jvp(ref_attn, (q, k, v), (dq, dk, dv))

    # TF32 tolerances (production precision)
    torch.testing.assert_close(tri_o, ref_o, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(tri_do, ref_do, atol=0.02, rtol=0.02)


# ─── Backward correctness ──────────────────────────────────────────


@pytest.mark.parametrize("BH,N,D", [(4, 64, 64), (12, 128, 64)])
def test_backward_matches_sdpa(BH, N, D):
    """Our naive backward should produce correct gradients."""
    q = _rand((BH, N, D), requires_grad=True)
    k = _rand((BH, N, D), requires_grad=True)
    v = _rand((BH, N, D), requires_grad=True)
    scale = D**-0.5

    from torch.nn.attention import SDPBackend, sdpa_kernel

    from backbone.jvp_flash_attention import _FlashAttnJVPFn

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    with sdpa_kernel(SDPBackend.MATH):
        o_ref = torch.nn.functional.scaled_dot_product_attention(
            q_ref.unsqueeze(1), k_ref.unsqueeze(1), v_ref.unsqueeze(1), scale=scale,
        ).squeeze(1)
    loss_ref = o_ref.sum()
    loss_ref.backward()

    o_tri, _lse = _FlashAttnJVPFn.apply(q, k, v, scale)
    loss_tri = o_tri.sum()
    loss_tri.backward()

    # TF32 tolerances for backward (Triton forward uses TF32 by default)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=5e-3, rtol=5e-3)


# ─── Edge cases ────────────────────────────────────────────────────


def test_single_token():
    """N=1 edge case."""
    q, k, v = _rand((1, 1, 64)), _rand((1, 1, 64)), _rand((1, 1, 64))
    dq, dk, dv = _rand((1, 1, 64)), _rand((1, 1, 64)), _rand((1, 1, 64))
    scale = 64**-0.5

    ref_o, ref_do = reference_attn_jvp(q, k, v, dq, dk, dv, scale)
    tri_o, lse = _triton_fwd(q, k, v, scale, fp_precision=FP)
    tri_do = _triton_jvp(q, k, v, tri_o, lse, dq, dk, dv, scale, fp_precision=FP)

    torch.testing.assert_close(tri_o, ref_o, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(tri_do, ref_do, atol=1e-3, rtol=1e-3)


def test_non_power_of_two_seq_len():
    """N not a power of 2 (e.g. 100)."""
    q, k, v = _rand((8, 100, 64)), _rand((8, 100, 64)), _rand((8, 100, 64))
    dq, dk, dv = _rand((8, 100, 64)), _rand((8, 100, 64)), _rand((8, 100, 64))
    scale = 64**-0.5

    ref_o, ref_do = reference_attn_jvp(q, k, v, dq, dk, dv, scale)
    tri_o, tri_do = fused_attn_fwd_jvp(q, k, v, dq, dk, dv, scale, fp_precision=FP)

    torch.testing.assert_close(tri_o, ref_o, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(tri_do, ref_do, atol=1e-3, rtol=1e-3)
