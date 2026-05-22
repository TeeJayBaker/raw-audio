from __future__ import annotations

import torch
from torch import nn


class NullEmbedding(nn.Module):
    """Deterministic zero embedding for unconditional runs and tests."""

    def __init__(self, embedding_dim: int = 4):
        super().__init__()
        self.embedding_dim = int(embedding_dim)

    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del sample_rate, audio_lengths
        return audio.new_zeros(audio.shape[0], self.embedding_dim)
