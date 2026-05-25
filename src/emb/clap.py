from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn


class CLAPEmbedding(nn.Module):
    """LAION-CLAP audio embedding wrapper for distribution metrics."""

    sample_rate = 48000

    def __init__(
        self,
        device: str = "cuda",
        checkpoint_path: str | None = None,
        enable_fusion: bool = False,
        amodel: str = "HTSAT-base",
        encode_batch_size: int = 16,
        input_sample_rate: int = 48000,
    ):
        super().__init__()
        try:
            import laion_clap
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("CLAP metrics require optional dependency 'laion_clap'.") from exc
        self.device = torch.device(device)
        self.encode_batch_size = int(encode_batch_size)
        self.input_sample_rate = int(input_sample_rate)
        self.resampler = (
            None
            if self.input_sample_rate == self.sample_rate
            else torchaudio.transforms.Resample(self.input_sample_rate, self.sample_rate).to(self.device)
        )
        # Default to the music-trained HTSAT-base checkpoint; hf_hub_download caches under
        # ~/.cache/huggingface and returns the local path, so this is a no-op on repeat runs.
        if checkpoint_path is None:
            checkpoint_path = hf_hub_download("lukewys/laion_clap", "music_audioset_epoch_15_esc_90.14.pt")
        self.model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=amodel, device=str(device))
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
        del audio_lengths
        if int(sample_rate) != self.input_sample_rate:
            raise ValueError(
                f"CLAPEmbedding was initialized for {self.input_sample_rate} Hz input, "
                f"but got {sample_rate} Hz."
            )
        audio = audio.detach().to(self.device).float()
        if audio.ndim == 3:
            audio = audio.mean(dim=1)
        if self.resampler is not None:
            audio = self.resampler(audio)
        chunks = []
        for i in range(0, audio.shape[0], self.encode_batch_size):
            chunk = audio[i : i + self.encode_batch_size].cpu()
            chunks.append(self.model.get_audio_embedding_from_data(x=chunk, use_tensor=True))
        return F.normalize(torch.cat(chunks, dim=0), p=2, dim=-1)
