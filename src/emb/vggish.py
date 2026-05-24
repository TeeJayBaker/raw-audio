from __future__ import annotations

import torch
import torchaudio
from torch import nn


class VGGishEmbedding(nn.Module):
    """VGGish audio embedding wrapper for FAD/MIND metrics."""

    sample_rate = 16000

    def __init__(
        self,
        device: str = "cuda",
        input_sample_rate: int = 48000,
    ):
        super().__init__()
        try:
            self.model = torch.hub.load("harritaylor/torchvggish", "vggish", verbose=False)
        except Exception as exc:
            raise ImportError("VGGish metrics require torch hub access to harritaylor/torchvggish.") from exc
        self.model.preprocess = True
        self.model = self.model.to(device).eval()
        self.device = torch.device(device)
        self.input_sample_rate = int(input_sample_rate)
        self.resampler = (
            None
            if self.input_sample_rate == self.sample_rate
            else torchaudio.transforms.Resample(self.input_sample_rate, self.sample_rate).to(self.device)
        )
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
        if int(sample_rate) != self.input_sample_rate:
            raise ValueError(
                f"VGGishEmbedding was initialized for {self.input_sample_rate} Hz input, "
                f"but got {sample_rate} Hz."
            )
        audio = audio.detach().to(self.device).float()
        if audio.ndim == 3:
            audio = audio.mean(dim=1)
        if self.resampler is not None:
            audio = self.resampler(audio)
        return self.model(audio)
