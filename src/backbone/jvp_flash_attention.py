"""Triton Flash Attention with forward-mode AD (JVP) support.

Three implementations for benchmarking:
  1. Two-pass: flash forward (saves LSE) + JVP recompute pass
  2. Fused: single-pass forward + JVP (fewer HBM reads)
  3. Reference: PyTorch MATH backend SDPA (O(N²) baseline)

Usage — drop-in replacement for F.scaled_dot_product_attention::

    # In your Attention module, replace:
    #   x = F.scaled_dot_product_attention(q, k, v)
    # with:
    x = flash_attn_jvp(q, k, v)

torch.func.jvp will automatically use the efficient Triton JVP kernel
instead of falling back to the O(N²) MATH backend.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._C._functorch import get_unwrapped, is_functorch_wrapped_tensor


def _unwrap_functorch(t: torch.Tensor) -> torch.Tensor:
    """Recursively unwrap functorch TensorWrappers so Triton can access storage."""
    while is_functorch_wrapped_tensor(t):
        t = get_unwrapped(t)
    return t


# ─── Forward Flash Attention Kernel ─────────────────────────────────

@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_b, stride_n, stride_d,
    stride_lb,
    N, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    FP_PRECISION: tl.constexpr = "ieee",
):
    """Standard flash attention forward. Saves LSE for the JVP pass."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N

    base = pid_bh * stride_b
    base_l = pid_bh * stride_lb

    # Load Q block [BLOCK_M, HEAD_DIM]
    q = tl.load(
        Q + base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
        mask=mask_m[:, None], other=0.0,
    ).to(tl.float32)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        col_ptrs = offs_n[:, None] * stride_n + offs_d[None, :] * stride_d

        k = tl.load(K + base + col_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
        v = tl.load(V + base + col_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # S = Q @ K^T * scale
        s = tl.dot(q, tl.trans(k), input_precision=FP_PRECISION) * scale
        s = tl.where(mask_n[None, :], s, float("-inf"))

        # Online softmax
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        o_i = o_i * alpha[:, None] + tl.dot(p, v, input_precision=FP_PRECISION)
        m_i = m_new

    # Normalise
    o_i = o_i / l_i[:, None]
    lse_i = m_i + tl.log(l_i)

    # Store O and LSE
    tl.store(
        O + base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
        o_i, mask=mask_m[:, None],
    )
    tl.store(LSE + base_l + offs_m, lse_i, mask=mask_m)


# ─── JVP-Only Pass Kernel ──────────────────────────────────────────

@triton.jit
def _jvp_kernel(
    Q, K, V, O, LSE, dQ, dK, dV, dO,
    stride_b, stride_n, stride_d,
    stride_lb,
    N, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    FP_PRECISION: tl.constexpr = "ieee",
):
    """Given forward results (O, LSE), compute dO from tangents dQ, dK, dV.

    Math:
        P_ij   = exp(S_ij - LSE_i)          (recover softmax from LSE)
        dS_ij  = (dQ @ K^T + Q @ dK^T) * s  (score tangent)
        D_i    = Σ_j P_ij * dS_ij           (softmax correction term)
        dO_i   = Σ_j [(P*dS) @ V + P @ dV] - D_i * O_i
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N

    base = pid_bh * stride_b
    base_l = pid_bh * stride_lb

    # Row-persistent loads: Q, dQ, O, LSE
    row_ptrs = offs_m[:, None] * stride_n + offs_d[None, :] * stride_d
    q = tl.load(Q + base + row_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    dq = tl.load(dQ + base + row_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    o = tl.load(O + base + row_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    lse = tl.load(LSE + base_l + offs_m, mask=mask_m, other=float("-inf")).to(tl.float32)

    do_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    D_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        col_ptrs = offs_n[:, None] * stride_n + offs_d[None, :] * stride_d

        k = tl.load(K + base + col_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
        v = tl.load(V + base + col_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
        dk = tl.load(dK + base + col_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
        dv = tl.load(dV + base + col_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # Recompute attention weights from LSE
        s = tl.dot(q, tl.trans(k), input_precision=FP_PRECISION) * scale
        s = tl.where(mask_n[None, :], s, float("-inf"))
        p = tl.exp(s - lse[:, None])

        # Score tangent
        ds = (tl.dot(dq, tl.trans(k), input_precision=FP_PRECISION)
              + tl.dot(q, tl.trans(dk), input_precision=FP_PRECISION)) * scale
        ds = tl.where(mask_n[None, :], ds, 0.0)

        # Accumulate
        p_ds = p * ds
        D_acc += tl.sum(p_ds, axis=1)
        do_acc += (tl.dot(p_ds, v, input_precision=FP_PRECISION)
                   + tl.dot(p, dv, input_precision=FP_PRECISION))

    # Softmax correction: dO -= D * O
    do_acc -= D_acc[:, None] * o

    tl.store(
        dO + base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
        do_acc, mask=mask_m[:, None],
    )


# ─── Fused Forward + JVP Kernel (optimised) ────────────────────────

_fused_configs = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_stages=s, num_warps=w)
    for bm in [64, 128]
    for bn in [16, 32, 64]
    for s in [3, 4, 7]
    for w in [4, 8]
]


# key omits HEAD_DIM (a keyword constexpr): triton 3.1.0's autotuner resolves key names to
# positional-arg indices, and HEAD_DIM's index overruns the positional list. N alone suffices
# (HEAD_DIM is fixed at 64 here). The fused kernel is benchmarking-only; production uses two-pass.
@triton.autotune(configs=_fused_configs, key=["N"])
@triton.jit
def _fused_fwd_jvp_kernel(
    Q, K, V, dQ, dK, dV, O, dO,
    stride_b, stride_n, stride_d,
    N, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    FP_PRECISION: tl.constexpr = "ieee",
):
    """Single pass forward + JVP with online softmax.

    Optimisations vs naive version (adopted from NVlabs/rcm):
      - exp2/log2 (native HW instructions, avoids extra ln2 multiply)
      - K loaded as [D, BLOCK_N] (pre-transposed, avoids tl.trans)
      - 3-arg tl.dot(a, b, acc) for fused accumulation
      - Rescale *before* dot to separate alpha multiply from matmul
      - tl.multiple_of alignment hint on inner loop counter
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N
    base = pid_bh * stride_b

    # Row-persistent loads: Q, dQ [BLOCK_M, HEAD_DIM]
    row_ptrs = offs_m[:, None] * stride_n + offs_d[None, :] * stride_d
    q = tl.load(Q + base + row_ptrs, mask=mask_m[:, None], other=0.0)
    dq = tl.load(dQ + base + row_ptrs, mask=mask_m[:, None], other=0.0)

    # Scale for base-2 softmax: score * scale / ln(2)
    qk_scale_log2: tl.constexpr = 1.4426950408889634  # 1/ln(2)
    scale_log2 = scale * qk_scale_log2

    # Accumulators
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)   # row max (log2)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # row sum (exp2)
    o_i = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)       # O~
    a_i = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)       # P~ @ dV
    b_i = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)       # H~ @ V
    r_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # rowsum(H~)

    for start_n in range(0, N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        # K, dK loaded in transposed layout [HEAD_DIM, BLOCK_N]
        kt_ptrs = offs_n[None, :] * stride_n + offs_d[:, None] * stride_d
        k = tl.load(K + base + kt_ptrs, mask=mask_n[None, :], other=0.0)
        dk = tl.load(dK + base + kt_ptrs, mask=mask_n[None, :], other=0.0)

        # Scores in log2 domain
        s = tl.dot(q, k, input_precision=FP_PRECISION).to(tl.float32) * scale_log2
        s = tl.where(mask_n[None, :], s, float("-inf"))

        # Online softmax (base-2)
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Rescale all accumulators BEFORE dot (fuses better)
        o_i *= alpha[:, None]
        a_i *= alpha[:, None]
        b_i *= alpha[:, None]
        r_i *= alpha

        # V, dV [BLOCK_N, HEAD_DIM]
        col_ptrs = offs_n[:, None] * stride_n + offs_d[None, :] * stride_d
        v = tl.load(V + base + col_ptrs, mask=mask_n[:, None], other=0.0)
        dv = tl.load(dV + base + col_ptrs, mask=mask_n[:, None], other=0.0)

        # Score tangent: dS = (dQ @ K^T + Q @ dK^T) * scale
        ds = (tl.dot(dq, k, input_precision=FP_PRECISION).to(tl.float32)
              + tl.dot(q, dk, input_precision=FP_PRECISION).to(tl.float32)) * scale
        ds = tl.where(mask_n[None, :], ds, 0.0)

        # H~ = P~ * dS (unnormalised softmax-JVP product)
        h = p * ds
        r_i += tl.sum(h, axis=1)

        # Fused accumulate: 3-arg tl.dot(a, b, acc) → acc += a @ b
        p_v = p.to(v.dtype)
        h_v = h.to(v.dtype)
        o_i = tl.dot(p_v, v, o_i, input_precision=FP_PRECISION)
        a_i = tl.dot(p_v, dv, a_i, input_precision=FP_PRECISION)
        b_i = tl.dot(h_v, v, b_i, input_precision=FP_PRECISION)

        m_i = m_new

    # Normalise: O = O~/l,  dO = (A~ + B~)/l − (r~/l)·O
    inv_l = 1.0 / l_i
    o_i *= inv_l[:, None]
    a_i *= inv_l[:, None]
    b_i *= inv_l[:, None]
    do_i = a_i + b_i - (r_i * inv_l)[:, None] * o_i

    # Store
    tl.store(O + base + row_ptrs, o_i, mask=mask_m[:, None])
    tl.store(dO + base + row_ptrs, do_i, mask=mask_m[:, None])


# ─── Python Launchers ──────────────────────────────────────────────

def _triton_fwd(q, k, v, scale, BLOCK_M=64, BLOCK_N=64, fp_precision="tf32"):
    """Flash attention forward → (O, LSE).  All tensors [BH, N, D]."""
    BH, N, D = q.shape
    o = torch.empty(BH, N, D, dtype=torch.float32, device=q.device)
    lse = torch.empty(BH, N, dtype=torch.float32, device=q.device)

    # Unwrap functorch wrappers (if called inside torch.func.jvp)
    q, k, v = _unwrap_functorch(q), _unwrap_functorch(k), _unwrap_functorch(v)
    o, lse = _unwrap_functorch(o), _unwrap_functorch(lse)

    grid = (triton.cdiv(N, BLOCK_M), BH)
    _fwd_kernel[grid](
        q, k, v, o, lse,
        q.stride(0), q.stride(1), q.stride(2),
        lse.stride(0),
        N, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        FP_PRECISION=fp_precision,
        num_warps=4, num_stages=2,
    )
    return o, lse


def _triton_jvp(q, k, v, o, lse, dq, dk, dv, scale, BLOCK_M=64, BLOCK_N=64,
                fp_precision="tf32"):
    """JVP pass using saved (O, LSE) → dO.  All tensors [BH, N, D]."""
    BH, N, D = q.shape
    do = torch.empty(BH, N, D, dtype=torch.float32, device=q.device)

    # Unwrap functorch wrappers (created when called inside torch.func.jvp)
    q, k, v = _unwrap_functorch(q), _unwrap_functorch(k), _unwrap_functorch(v)
    o, lse = _unwrap_functorch(o), _unwrap_functorch(lse)
    dq, dk, dv = _unwrap_functorch(dq), _unwrap_functorch(dk), _unwrap_functorch(dv)
    do = _unwrap_functorch(do)

    grid = (triton.cdiv(N, BLOCK_M), BH)
    _jvp_kernel[grid](
        q, k, v, o, lse, dq, dk, dv, do,
        q.stride(0), q.stride(1), q.stride(2),
        lse.stride(0),
        N, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        FP_PRECISION=fp_precision,
        num_warps=4, num_stages=1,
    )
    return do


def fused_attn_fwd_jvp(q, k, v, dq, dk, dv, scale=None, fp_precision="tf32"):
    """Fused forward + JVP in a single pass → (O, dO).

    Standalone function (not via torch.func.jvp).
    Input tensors must be contiguous float32 [BH, N, D].
    """
    BH, N, D = q.shape
    if scale is None:
        scale = D**-0.5
    o = torch.empty(BH, N, D, dtype=torch.float32, device=q.device)
    do = torch.empty(BH, N, D, dtype=torch.float32, device=q.device)

    def grid(args):
        return (triton.cdiv(N, args["BLOCK_M"]), BH)

    _fused_fwd_jvp_kernel[grid](
        q, k, v, dq, dk, dv, o, do,
        q.stride(0), q.stride(1), q.stride(2),
        N, scale,
        HEAD_DIM=D,
        FP_PRECISION=fp_precision,
    )
    return o, do


# ─── Autograd Function (two-pass, for torch.func.jvp) ─────────────

class _FlashAttnJVPFn(torch.autograd.Function):
    """Custom autograd Function with a Triton-accelerated JVP rule.

    Forward:  standard flash attention (Triton) — saves (Q,K,V,O,LSE).
    JVP:      Triton kernel recomputes P from LSE, fuses dS → dO.
    Backward: naive materialised attention (fine for N ≤ 512).
    """

    @staticmethod
    def forward(q, k, v, scale):
        o, lse = _triton_fwd(q, k, v, scale)
        return o, lse

    @staticmethod
    def setup_context(ctx, inputs, output):
        q, k, v, scale = inputs
        o, lse = output
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.save_for_forward(q, k, v, o, lse)
        ctx.scale = scale

    @staticmethod
    def jvp(ctx, dq, dk, dv, _dscale):
        q, k, v, o, lse = ctx.saved_tensors
        # Unwrap functorch TensorWrappers so Triton can access raw storage
        q, k, v = _unwrap_functorch(q), _unwrap_functorch(k), _unwrap_functorch(v)
        o, lse = _unwrap_functorch(o), _unwrap_functorch(lse)
        dq, dk, dv = _unwrap_functorch(dq), _unwrap_functorch(dk), _unwrap_functorch(dv)
        do = _triton_jvp(q, k, v, o, lse, dq, dk, dv, ctx.scale)
        return do, torch.zeros(lse.shape, device=lse.device, dtype=lse.dtype)

    @staticmethod
    def backward(ctx, grad_o, _grad_lse):
        q, k, v, o, lse = ctx.saved_tensors
        scale = ctx.scale

        # Recompute attention weights from saved LSE
        s = torch.bmm(q, k.transpose(-2, -1)) * scale
        p = torch.exp(s - lse.unsqueeze(-1))

        # Standard attention backward
        grad_v = torch.bmm(p.transpose(-2, -1), grad_o)
        dp = torch.bmm(grad_o, v.transpose(-2, -1))
        d = (grad_o * o).sum(dim=-1, keepdim=True)
        ds = p * (dp - d)
        grad_q = torch.bmm(ds, k) * scale
        grad_k = torch.bmm(ds.transpose(-2, -1), q) * scale

        return grad_q, grad_k, grad_v, None


# ─── Public API ─────────────────────────────────────────────────────

def flash_attn_jvp(q, k, v, scale=None):
    """Drop-in replacement for ``F.scaled_dot_product_attention``.

    Works with ``torch.func.jvp`` — the JVP uses a Triton flash kernel
    instead of falling back to the O(N²) MATH backend.

    Args:
        q, k, v: ``[B, H, N, D]`` float32 tensors.
        scale:   Optional scale (default ``1/√D``).

    Returns:
        ``[B, H, N, D]`` attention output.
    """
    B, H, N, D = q.shape
    if scale is None:
        scale = D**-0.5

    q_flat = q.reshape(B * H, N, D).contiguous()
    k_flat = k.reshape(B * H, N, D).contiguous()
    v_flat = v.reshape(B * H, N, D).contiguous()

    o_flat, _lse = _FlashAttnJVPFn.apply(q_flat, k_flat, v_flat, scale)
    return o_flat.reshape(B, H, N, D)


# ─── Reference (MATH backend, for testing) ─────────────────────────

def reference_attn_jvp(q, k, v, dq, dk, dv, scale=None):
    """Compute O and dO using PyTorch MATH backend + torch.func.jvp.

    Inputs are 3-D ``[BH, N, D]`` float32 tensors.
    Returns ``(O, dO)`` each ``[BH, N, D]``.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if scale is None:
        scale = q.shape[-1] ** -0.5

    def _attn(q_, k_, v_):
        # Unsqueeze to 4-D so SDPA treats dim-1 as heads
        return F.scaled_dot_product_attention(
            q_.unsqueeze(1), k_.unsqueeze(1), v_.unsqueeze(1), scale=scale,
        ).squeeze(1)

    with sdpa_kernel(SDPBackend.MATH):
        (o, do) = torch.func.jvp(_attn, (q, k, v), (dq, dk, dv))

    return o, do
