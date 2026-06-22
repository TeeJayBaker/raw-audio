"""Perceptual auxiliary losses over frozen learned audio metrics.

Unlike the stateless spectral losses in :mod:`losses.audio`, a perceptual loss carries a loaded
frozen network, so it takes the embedder as its first argument; the trainer holds the one
instance (as it does for the FD-loss / conditioner) and passes it in.
"""

from __future__ import annotations

import torch
from torch import nn


def cdpam_loss(embedder: nn.Module, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean CDPAM perceptual distance between prediction and target waveforms (the audio LPIPS-analog,
    as used for pMF). Both are waveforms at the embedder's ``input_sample_rate``; the wrapper owns
    the resample-to-22.05 kHz, mono reduction, and int16 amplitude lift. Feeding lifted-domain
    audio (matching ``mr_stft``) is fine: prediction and target share the domain, so the relative
    perceptual distance is well-defined despite CDPAM not being strictly scale-invariant."""
    return embedder.distance(prediction, target).mean()
