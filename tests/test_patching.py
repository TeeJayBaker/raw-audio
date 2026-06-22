from __future__ import annotations

import math

import pytest
import torch

from backbone.blocks import apply_rope, rope_tables
from backbone.patching import ColumnPatcher, SquarePatcher, build_patcher


# Reference implementations copied from the validated probe / the original blocks._rope.
def _rope_1d_ref(x):
    dim = x.shape[-1]
    pos = torch.arange(x.shape[-2], dtype=torch.float32)
    freq = torch.arange(0, dim, 2, dtype=torch.float32) / dim
    inv = 1.0 / (10000**freq)
    angles = pos[:, None] * inv[None, :]
    cos = angles.cos()[None, None].to(x.dtype)
    sin = angles.sin()[None, None].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


def _rope_at_ref(x, pos):
    d = x.shape[-1]
    freq = torch.arange(0, d, 2, dtype=torch.float32) / d
    inv = 1.0 / (10000**freq)
    ang = pos.float()[:, None] * inv[None, :]
    cos = ang.cos()[None, None].to(x.dtype)
    sin = ang.sin()[None, None].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


def _axial_rope_ref(x, pos_f, pos_t):
    h = x.shape[-1] // 2
    return torch.cat([_rope_at_ref(x[..., :h], pos_f), _rope_at_ref(x[..., h:], pos_t)], dim=-1)


def _channelised(b, channels, freq_bins, frames):
    # [B, 2*C*F, frames] real/imag-channelised spectrogram, like complex_to_channels output.
    return torch.randn(b, 2 * channels * freq_bins, frames)


# --- geometry --------------------------------------------------------------

def test_column_geometry():
    p = ColumnPatcher(freq_bins=1025)
    assert p.npf == 1
    assert p.feat(1) == 2 * 1 * 1025
    assert p.feat(2) == 2 * 2 * 1025


def test_square_geometry():
    p = SquarePatcher(freq_bins=1025, patch_f=512, patch_t=8)
    assert p.npf == 2  # (1025 - 1) // 512
    assert p.feat(1) == 2 * 1 * 512 * 8
    assert p.feat(2) == 2 * 2 * 512 * 8


def test_square_rejects_indivisible_patch_f():
    with pytest.raises(ValueError, match="patch_f"):
        SquarePatcher(freq_bins=1025, patch_f=300, patch_t=8)


# --- round trips (pure reshape, no params) ---------------------------------

@pytest.mark.parametrize("frames", [16, 19])
def test_column_round_trip_is_exact(frames):
    p = ColumnPatcher(freq_bins=9)
    x = _channelised(2, 1, 9, frames)
    tokens = p.patch(x)
    assert tokens.shape == (2, p.feat(1), frames)  # one token per frame
    recon = p.unpatch(tokens, frames)
    assert torch.equal(recon, x)


@pytest.mark.parametrize("frames", [8, 19])
def test_square_round_trip_reconstructs_except_nyquist(frames):
    freq_bins, patch_f, patch_t = 9, 4, 2
    p = SquarePatcher(freq_bins=freq_bins, patch_f=patch_f, patch_t=patch_t)
    x = _channelised(2, 1, freq_bins, frames)
    tokens = p.patch(x)
    npt = math.ceil(frames / patch_t)
    assert tokens.shape == (2, p.feat(1), p.npf * npt)
    recon = p.unpatch(tokens, frames)
    # Nyquist (last freq bin of every real/imag channel) is dropped on patch, returned as zero.
    expected = x.view(2, 2, freq_bins, frames).clone()
    expected[:, :, freq_bins - 1, :] = 0.0
    assert torch.allclose(recon, expected.reshape(2, 2 * freq_bins, frames), atol=1e-6)


def test_square_token_count_matches_formula():
    p = SquarePatcher(freq_bins=1025, patch_f=512, patch_t=8)
    frames = 38
    tokens = p.patch(_channelised(1, 1, 1025, frames))
    assert tokens.shape[-1] == p.npf * math.ceil(frames / 8) == 10


def test_square_multichannel_round_trip():
    freq_bins, patch_f, patch_t = 9, 4, 2
    p = SquarePatcher(freq_bins=freq_bins, patch_f=patch_f, patch_t=patch_t)
    x = _channelised(2, 3, freq_bins, 10)
    tokens = p.patch(x)
    assert tokens.shape[1] == p.feat(3)
    recon = p.unpatch(tokens, 10)
    expected = x.view(2, 6, freq_bins, 10).clone()
    expected[:, :, freq_bins - 1, :] = 0.0
    assert torch.allclose(recon, expected.reshape(2, 6 * freq_bins, 10), atol=1e-6)


# --- factory ---------------------------------------------------------------

def test_build_patcher_selects_scheme():
    assert isinstance(build_patcher({"scheme": "column"}, freq_bins=1025), ColumnPatcher)
    sq = build_patcher({"scheme": "square", "patch_f": 512, "patch_t": 8}, freq_bins=1025)
    assert isinstance(sq, SquarePatcher)
    assert (sq.patch_f, sq.patch_t) == (512, 8)


def test_build_patcher_defaults_to_column():
    assert isinstance(build_patcher(None, freq_bins=1025), ColumnPatcher)


def test_build_patcher_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="scheme"):
        build_patcher({"scheme": "hexagons"}, freq_bins=1025)


# --- axial RoPE ------------------------------------------------------------

def test_rope_1d_matches_original_blocks_rope():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 7, 16)  # [b, heads, n, head_dim]
    cos, sin = rope_tables(npf=1, npt=7, head_dim=16, device=x.device)
    assert torch.allclose(apply_rope(x, cos, sin), _rope_1d_ref(x), atol=1e-6)


def test_rope_axial_matches_probe_axial_rope():
    torch.manual_seed(0)
    npf, npt, head_dim = 2, 5, 16
    x = torch.randn(2, 4, npf * npt, head_dim)
    pos_f = torch.arange(npf).repeat_interleave(npt)  # freq-major raster
    pos_t = torch.arange(npt).repeat(npf)
    cos, sin = rope_tables(npf=npf, npt=npt, head_dim=head_dim, device=x.device)
    assert torch.allclose(apply_rope(x, cos, sin), _axial_rope_ref(x, pos_f, pos_t), atol=1e-6)


def test_rope_requires_head_dim_divisible_by_four_for_axial():
    with pytest.raises(ValueError, match="head_dim"):
        rope_tables(npf=2, npt=3, head_dim=6, device=torch.device("cpu"))


def test_rope_1d_is_relative_position_invariant():
    # A constant query/key vector at every position: the attention logit between two
    # tokens depends only on their separation, so logit[i,j] == logit[i+1,j+1].
    u = torch.randn(1, 1, 1, 16).expand(1, 1, 8, 16).contiguous()
    cos, sin = rope_tables(npf=1, npt=8, head_dim=16, device=u.device)
    q = apply_rope(u, cos, sin)
    logits = (q @ q.transpose(-1, -2))[0, 0]
    assert torch.allclose(logits.diagonal(0), logits.diagonal(0).mean().expand(8), atol=1e-5)
    assert torch.allclose(logits.diagonal(1), logits.diagonal(1).mean().expand(7), atol=1e-5)
