from __future__ import annotations

import torch
from torch import nn


class VGGishEmbedding(nn.Module):
    """VGGish audio embedding wrapper for FAD/MIND metrics."""

    def __init__(self, device: str = "cuda"):
        super().__init__()
        try:
            self.model = torch.hub.load("harritaylor/torchvggish", "vggish", verbose=False)
        except Exception as exc:
            raise ImportError("VGGish metrics require torch hub access to harritaylor/torchvggish.") from exc
        self.model.preprocess = True
        self.model = self.model.to(device).eval()
        self.device = torch.device(device)
        self.embedding_dim = 128
        self.eval()

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del audio_lengths
        if sample_rate != 16000:
            raise ValueError("VGGishEmbedding expects 16 kHz audio.")
        audio = audio.detach().to(self.device)
        if audio.ndim == 3:
            audio = audio.mean(dim=1)
        return self.model(audio.float())
