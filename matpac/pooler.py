"""Attentive Pooler for MATPAC++ v2.

Replaces the CLS token with a learned weighted sum of patch embeddings.
Guarantees that the global embedding is a function of patches only,
ensuring redundancy between global and patch representations.
"""

import torch
import torch.nn as nn


class AttentivePooler(nn.Module):
    """Attention-weighted mean pooler.

    Computes a global embedding as a learned weighted sum of patch embeddings.
    The output is guaranteed to be derivable from patches alone.

    Args:
        hidden_size: Patch embedding dimension
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score_proj = nn.Linear(hidden_size, 1)

    def forward(
        self,
        patches: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool patch embeddings into a single global embedding.

        Args:
            patches: [B, N, D] patch embeddings
            mask: [B, N] attention mask (1=valid, 0=pad)

        Returns:
            global_embedding: [B, D] weighted sum of patches
        """
        scores = self.score_proj(patches).squeeze(-1)  # [B, N]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)  # [B, N]
        return (weights.unsqueeze(-1) * patches).sum(dim=1)  # [B, D]
