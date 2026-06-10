from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RandomProjEmbedding(nn.Module):
    """Frozen random-projection embedder: differentiable STFT-magnitude → mean-pool → linear.

    A cheap, fully differentiable φ for exercising FD-loss (tests, CPU smoke runs) without a
    heavy pretrained encoder. Deterministic given ``seed``; parameter-free (only buffers).
    """

    name = "random"

    def __init__(
        self,
        embedding_dim: int = 64,
        n_fft: int = 512,
        hop_length: int = 256,
        input_sample_rate: int = 48000,
        seed: int = 0,
        device: str = "cpu",
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.input_sample_rate = int(input_sample_rate)
        self.sample_rate = int(input_sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        generator = torch.Generator().manual_seed(int(seed))
        proj = torch.randn(self.n_fft // 2 + 1, self.embedding_dim, generator=generator)
        self.register_buffer("proj", proj)
        self.register_buffer("window", torch.hann_window(self.n_fft))
        self.device = torch.device(device)
        self.to(self.device)
        self.eval()

    def embed(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del sample_rate, audio_lengths
        audio = audio.to(self.device).float()
        if audio.ndim == 3:
            audio = audio.mean(dim=1)
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        magnitude = spec.abs().mean(dim=-1)  # [B, n_fft//2 + 1]
        return F.normalize(magnitude @ self.proj, dim=-1)

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.embed(audio, sample_rate=sample_rate, audio_lengths=audio_lengths)
