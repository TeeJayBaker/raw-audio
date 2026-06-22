from __future__ import annotations

import torch
from torch.nn import functional as F

"""Pure-reshape spectrogram patchers (no learned weights).

A patcher rearranges a channelised spectrogram ``[B, 2*C*F, frames]`` (the
``complex_to_channels`` layout: ``C`` real bands then ``C`` imag bands, each of
``F`` freq bins) into transformer tokens ``[B, feat, N]`` and back. The learned
``in_proj``/``out_proj`` that map ``feat <-> dim`` live in the transformer; this
module only moves numbers around, so its forward is trivially scriptable.
"""


class ColumnPatcher:
    """One token per STFT frame: the full-frequency column. Identity reshape."""

    npf = 1

    def __init__(self, freq_bins: int):
        self.freq_bins = int(freq_bins)

    def feat(self, channels: int) -> int:
        return 2 * channels * self.freq_bins

    def patch(self, x: torch.Tensor) -> torch.Tensor:
        return x  # [B, 2*C*F, frames] is already [B, feat, N] with N = frames

    def unpatch(self, y: torch.Tensor, frames: int) -> torch.Tensor:
        return y[..., :frames]


class SquarePatcher:
    """2-D ViT tiling: ``patch_f`` freq bands x ``patch_t`` time patches, with the
    real/imag channels carried into the per-tile feature. The Nyquist bin is dropped
    so ``freq_bins - 1`` divides evenly, and reattached as zero on ``unpatch``."""

    def __init__(self, freq_bins: int, patch_f: int, patch_t: int):
        self.freq_bins = int(freq_bins)
        self.patch_f = int(patch_f)
        self.patch_t = int(patch_t)
        self.fbins = self.freq_bins - 1  # drop Nyquist
        if self.fbins % self.patch_f:
            raise ValueError(
                f"patch_f={self.patch_f} must divide freq_bins-1={self.fbins} (Nyquist dropped)"
            )
        self.npf = self.fbins // self.patch_f

    def feat(self, channels: int) -> int:
        return 2 * channels * self.patch_f * self.patch_t

    def patch(self, x: torch.Tensor) -> torch.Tensor:
        b, cf2, frames = x.shape
        nc2 = cf2 // self.freq_bins  # 2 * channels (real/imag bands)
        v = x.view(b, nc2, self.freq_bins, frames)[:, :, : self.fbins, :]  # drop Nyquist
        npt = (frames + self.patch_t - 1) // self.patch_t
        v = F.pad(v, (0, npt * self.patch_t - frames))  # pad time up to a whole tile
        v = v.view(b, nc2, self.npf, self.patch_f, npt, self.patch_t)
        tokens = v.permute(0, 2, 4, 1, 3, 5).reshape(b, self.npf * npt, nc2 * self.patch_f * self.patch_t)
        return tokens.transpose(1, 2)  # [B, feat, N]

    def unpatch(self, y: torch.Tensor, frames: int) -> torch.Tensor:
        b, feat, n = y.shape
        nc2 = feat // (self.patch_f * self.patch_t)
        npt = n // self.npf
        v = y.transpose(1, 2).view(b, self.npf, npt, nc2, self.patch_f, self.patch_t)
        v = v.permute(0, 3, 1, 4, 2, 5).reshape(b, nc2, self.fbins, npt * self.patch_t)
        v = v[:, :, :, :frames]
        nyquist = torch.zeros_like(v[:, :, :1, :])
        return torch.cat([v, nyquist], dim=2).reshape(b, nc2 * self.freq_bins, frames)


def build_patcher(patching_cfg: dict | None, freq_bins: int):
    cfg = dict(patching_cfg or {})
    scheme = cfg.get("scheme", "column")
    if scheme == "column":
        return ColumnPatcher(freq_bins)
    if scheme == "square":
        return SquarePatcher(freq_bins, patch_f=int(cfg["patch_f"]), patch_t=int(cfg["patch_t"]))
    raise ValueError(f"Unknown patching scheme: {scheme!r}")
