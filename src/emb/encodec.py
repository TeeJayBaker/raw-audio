from __future__ import annotations

import torch
import torchaudio
from torch import nn
from torch.utils.checkpoint import checkpoint


class EnCodecEmbedding(nn.Module):
    """EnCodec (48 kHz stereo) embedding wrapper.

    Uses the continuous encoder latent *before* quantization (``model.encoder``), a pure conv
    stack that is differentiable. ``model.encode`` is avoided as its quantization is not. Each
    latent frame is returned as one 128-d population sample for frame-level FD.
    """

    sample_rate = 48000
    name = "encodec"

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "facebook/encodec_48khz",
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

    def train(self, mode: bool = True):
        # Stays frozen/eval, except the encoder LSTMs: cudnn refuses RNN backward for eval-mode
        # modules, and EnCodec's LSTM is dropout-free so train mode is numerically identical.
        del mode
        super().train(False)
        for module in self.model.modules():
            if isinstance(module, nn.LSTM):
                module.train(True)
        return self

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
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        if audio.ndim != 3 or audio.shape[1] not in (1, 2):
            raise ValueError("EnCodec audio must have shape [B, T] or [B, 1|2, T].")
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        if self.resampler is not None:
            audio = self.resampler(audio)
        # The encoder graph at 48 kHz is by far the largest critic allocation (early conv layers
        # at sample rate); checkpoint per layer so only layer-boundary tensors are retained.
        hidden = audio
        if torch.is_grad_enabled() and audio.requires_grad:
            for layer in self.model.encoder.layers:
                hidden = checkpoint(layer, hidden, use_reentrant=False)
        else:
            hidden = self.model.encoder(hidden)
        return hidden.transpose(1, 2).reshape(-1, hidden.shape[1])  # [B*T', 128]

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.embed(audio, sample_rate=sample_rate, audio_lengths=audio_lengths)
