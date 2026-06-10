from __future__ import annotations

import torch
import torchaudio
from torch import nn


class EnCodecEmbedding(nn.Module):
    """EnCodec (24 kHz mono) embedding wrapper.

    Uses the continuous encoder latent *before* quantization (``model.encoder``), a pure conv
    stack that is differentiable. ``model.encode`` is avoided as its quantization is not.
    Mean-pools the 128-d latent over time.
    """

    sample_rate = 24000
    name = "encodec"

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "facebook/encodec_24khz",
        input_sample_rate: int = 48000,
    ):
        super().__init__()
        try:
            from transformers import EncodecModel
        except ImportError as exc:
            raise ImportError("EnCodec metrics require optional dependency 'transformers'.") from exc
        self.device = torch.device(device)
        self.input_sample_rate = int(input_sample_rate)
        self.resampler = (
            None
            if self.input_sample_rate == self.sample_rate
            else torchaudio.transforms.Resample(self.input_sample_rate, self.sample_rate).to(self.device)
        )
        self.model = EncodecModel.from_pretrained(model_name).to(self.device)
        self.embedding_dim = int(self.model.config.hidden_size)
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def embed(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del audio_lengths
        if int(sample_rate) != self.input_sample_rate:
            raise ValueError(
                f"EnCodecEmbedding was initialized for {self.input_sample_rate} Hz input, but got {sample_rate} Hz."
            )
        audio = audio.to(self.device).float()
        if audio.ndim == 3:
            audio = audio.mean(dim=1)
        if self.resampler is not None:
            audio = self.resampler(audio)
        latents = self.model.encoder(audio.unsqueeze(1))  # [B, 128, T']
        return latents.mean(dim=-1)

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.embed(audio, sample_rate=sample_rate, audio_lengths=audio_lengths)
