from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
from torch import nn

from matpac.wrapper import MATpacWrapper


class MATPACEmbedding(nn.Module):
    """Frozen MATPAC embedding wrapper."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        use_teacher: bool = False,
        encode_batch_size: int = 0,
        compile_encoder: bool = False,
        input_sample_rate: int = 48000,
    ):
        super().__init__()
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"MATPAC checkpoint not found: {checkpoint_path}. "
                "Set conditioner.checkpoint_path to a local checkpoint; its sibling config.yaml is used automatically."
            )
        self.wrapper = MATpacWrapper(
            checkpoint_path=str(checkpoint_path),
            device=device,
            use_teacher=use_teacher,
            encode_batch_size=encode_batch_size,
            compile_encoder=compile_encoder,
        )
        self.device = torch.device(device)
        self.input_sample_rate = int(input_sample_rate)
        self.sample_rate = int(self.wrapper.SAMPLE_RATE)
        self.resampler = (
            None
            if self.input_sample_rate == self.sample_rate
            else torchaudio.transforms.Resample(self.input_sample_rate, self.sample_rate).to(self.device)
        )
        self.embedding_dim = int(self.wrapper.embedding_dim)
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if int(sample_rate) != self.input_sample_rate:
            raise ValueError(
                f"MATPACEmbedding was initialized for {self.input_sample_rate} Hz input, "
                f"but got {sample_rate} Hz."
            )
        audio = audio.detach().to(self.device).float()
        if audio_lengths is not None:
            audio_lengths = audio_lengths.to(self.device)
        if self.resampler is not None:
            audio = self.resampler(audio)
            if audio_lengths is not None:
                audio_lengths = (audio_lengths.float() * (self.sample_rate / self.input_sample_rate)).long()
        return self.wrapper.encode_audio(audio, sample_rate=self.sample_rate, audio_lengths=audio_lengths)

    @torch.no_grad()
    def encode_frames(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if int(sample_rate) != self.input_sample_rate:
            raise ValueError(
                f"MATPACEmbedding was initialized for {self.input_sample_rate} Hz input, "
                f"but got {sample_rate} Hz."
            )
        audio = audio.detach().to(self.device).float()
        if audio_lengths is not None:
            audio_lengths = audio_lengths.to(self.device)
        if self.resampler is not None:
            audio = self.resampler(audio)
            if audio_lengths is not None:
                audio_lengths = (audio_lengths.float() * (self.sample_rate / self.input_sample_rate)).long()
        return self.wrapper.encode_audio_frames(audio, sample_rate=self.sample_rate, audio_lengths=audio_lengths)
