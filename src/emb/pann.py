from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
from torch import nn

# Pretrained Cnn14 AudioSet checkpoint (mAP=0.431), as used by panns_inference.
_CKPT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
_DEFAULT_CKPT = Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"


class PANNEmbedding(nn.Module):
    """PANN Cnn14 (AudioSet) embedding wrapper.

    The Cnn14 front-end is torchlibrosa (``torch.stft`` based), so ``embed`` is fully
    differentiable and batched. Returns the 2048-d penultimate embedding.
    """

    sample_rate = 32000
    name = "pann"

    def __init__(
        self,
        device: str = "cuda",
        checkpoint_path: str | None = None,
        input_sample_rate: int = 48000,
    ):
        super().__init__()
        try:
            from panns_inference.models import Cnn14
        except ImportError as exc:
            raise ImportError("PANN metrics require optional dependency 'panns-inference'.") from exc
        self.device = torch.device(device)
        self.input_sample_rate = int(input_sample_rate)
        self.resampler = (
            None
            if self.input_sample_rate == self.sample_rate
            else torchaudio.transforms.Resample(self.input_sample_rate, self.sample_rate).to(self.device)
        )
        ckpt_path = Path(checkpoint_path).expanduser() if checkpoint_path else _DEFAULT_CKPT
        if not ckpt_path.exists():
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(_CKPT_URL, str(ckpt_path))
        self.model = Cnn14(
            sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527
        )
        self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"])
        self.model.to(self.device)
        self.embedding_dim = 2048
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
                f"PANNEmbedding was initialized for {self.input_sample_rate} Hz input, but got {sample_rate} Hz."
            )
        audio = audio.to(self.device).float()
        if audio.ndim == 3:
            audio = audio.mean(dim=1)
        if self.resampler is not None:
            audio = self.resampler(audio)
        # Cnn14 applies five (2,2) time-pools, so the log-mel needs >= 2**5 = 32 frames
        # (n_frames ~= 1 + T // hop_size); fewer collapses the feature map to size 0. Zero-pad
        # short clips up to that floor, matching PANN's own pad_or_truncate — Cnn14 was trained on
        # zero-padded clips, so zero- (not repeat-) padding stays in-distribution.
        min_samples = 33 * 320  # 33 mel frames (32 + 1 margin) at hop_size=320
        if audio.shape[-1] < min_samples:
            audio = torch.nn.functional.pad(audio, (0, min_samples - audio.shape[-1]))
        return self.model(audio)["embedding"]

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.embed(audio, sample_rate=sample_rate, audio_lengths=audio_lengths)
