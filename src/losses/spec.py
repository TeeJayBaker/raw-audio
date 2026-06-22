"""Phase-aware spectral losses/metrics on the model's raw complex spectrogram.

Both quantities here score the backbone's pre-iSTFT output (the channelised complex STFT), where
phase information is still present -- unlike the magnitude/log-mag/mel losses in :mod:`losses.audio`,
which are phase-blind. ``complex_stft_loss`` is *referenced* (compares to STFT(target)) and conflates
wrong phase with inconsistent phase; ``consistency_residual`` is *reference-free* and isolates only
the inconsistency. They are complementary, not substitutes.
"""
from __future__ import annotations

import torch

from backbone.io import STFTConfig, as_waveform, stft_to_waveform, waveform_to_stft


def complex_stft_loss(
    pred_spec: torch.Tensor,
    target: torch.Tensor,
    stft: STFTConfig | dict,
    energy_weight: bool = True,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Complex L1 between the backbone's raw predicted spectrogram and STFT(target) at the model's
    resolution. Hooks the pre-iSTFT output, so it penalises the inconsistent (incoherent-phase)
    component the internal iSTFT would otherwise project away. Flow2GAN energy-inverse weighting
    (1/√(S+ε), clamped) keeps quiet bins from being drowned by high-energy ones."""
    cfg = stft if isinstance(stft, STFTConfig) else STFTConfig.from_dict(stft)
    tgt_spec = waveform_to_stft(as_waveform(target), cfg).to(pred_spec.dtype)
    diff = pred_spec - tgt_spec
    l1 = diff.real.abs() + diff.imag.abs()
    if energy_weight:
        l1 = l1 * (tgt_spec.abs().pow(2) + eps).rsqrt().clamp(0.01, 100.0)
    return l1.mean()


def consistency_residual(
    spec: torch.Tensor,
    stft: STFTConfig | dict,
    length: int | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Reference-free relative complex-L1 distance of a complex STFT ``S`` from the consistent
    (real-signal) manifold: ``‖S − STFT(iSTFT(S))‖₁ / ‖S‖₁``. Exactly 0 for any consistent STFT;
    grows with cross-frame phase incoherence -- the direct fingerprint of the phasey / hop-rate AM
    artifact. Differentiable (RFWave-style overlap/consistency regularizer) and cheap under
    ``no_grad`` (the most diagnostic phase-aware eval metric).

    Score the model's RAW pre-iSTFT spectrogram, never the output waveform: a real waveform's STFT
    is trivially consistent (residual ≈ 0), so re-projecting it is uninformative. ``spec`` is the
    complex STFT ``[B, C, F, T]``; ``length`` is the original signal length (pins the iSTFT so the
    re-STFT frame count matches ``spec``)."""
    cfg = stft if isinstance(stft, STFTConfig) else STFTConfig.from_dict(stft)
    reproj = waveform_to_stft(stft_to_waveform(spec, cfg, length=length), cfg)
    return (spec - reproj).abs().sum() / spec.abs().sum().clamp_min(eps)
