from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CLAPEmbedding(nn.Module):
    """LAION-CLAP audio embedding wrapper for distribution metrics."""

    def __init__(
        self,
        device: str = "cuda",
        checkpoint_path: str | None = None,
        enable_fusion: bool = False,
        encode_batch_size: int = 16,
    ):
        super().__init__()
        try:
            import laion_clap
        except ImportError as exc:
            raise ImportError("CLAP metrics require optional dependency 'laion_clap'.") from exc
        self.device = torch.device(device)
        self.encode_batch_size = int(encode_batch_size)
        self.model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel="HTSAT-base", device=str(device))
        self.model.load_ckpt(ckpt=checkpoint_path, verbose=False)
        self.embedding_dim = 512
        self.eval()

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del sample_rate, audio_lengths
        audio = audio.detach().to(self.device)
        if audio.ndim == 3:
            audio = audio.mean(dim=1)
        chunks = []
        for i in range(0, audio.shape[0], self.encode_batch_size):
            chunk = audio[i : i + self.encode_batch_size].float().cpu()
            chunks.append(self.model.get_audio_embedding_from_data(x=chunk, use_tensor=True))
        return F.normalize(torch.cat(chunks, dim=0), p=2, dim=-1)
