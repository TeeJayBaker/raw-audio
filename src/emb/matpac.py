from __future__ import annotations

from pathlib import Path

import torch
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
        project_to_dim: int | None = None,
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
        self.embedding_dim = int(self.wrapper.embedding_dim)
        self.projector = nn.Identity()
        if project_to_dim is not None and int(project_to_dim) != self.embedding_dim:
            self.projector = nn.Linear(self.embedding_dim, int(project_to_dim), bias=False)
            self.embedding_dim = int(project_to_dim)
        for param in self.wrapper.parameters():
            param.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.wrapper.encode_audio(audio, sample_rate=sample_rate, audio_lengths=audio_lengths)
        return self.projector(embeddings)

    @torch.no_grad()
    def encode_frames(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.wrapper.encode_audio_frames(audio, sample_rate=sample_rate, audio_lengths=audio_lengths)
