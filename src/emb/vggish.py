from __future__ import annotations

import torch
import torchaudio
from torch import nn

# VGGish (AudioSet) log-mel + framing params, from vggish_params.py.
_SR = 16000
_N_FFT = 512
_WIN = 400  # 0.025 s STFT window
_HOP = 160  # 0.010 s STFT hop
_N_MELS = 64
_FMIN, _FMAX = 125.0, 7500.0
_LOG_OFFSET = 0.01
_FRAMES_PER_EXAMPLE = 96  # 0.96 s example, non-overlapping
_WEIGHTS_URL = "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth"


def _make_features() -> nn.Sequential:
    layers: list[nn.Module] = []
    in_ch = 1
    for v in (64, "M", 128, "M", 256, 256, "M", 512, 512, "M"):
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers += [nn.Conv2d(in_ch, v, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            in_ch = v
    return nn.Sequential(*layers)


class _VGG(nn.Module):
    """The VGGish conv net (architecture matches harritaylor/torchvggish weights)."""

    def __init__(self):
        super().__init__()
        self.features = _make_features()
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [N, 1, 96, 64]
        x = self.features(x)
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous().view(x.size(0), -1)
        return self.embeddings(x)


class VGGishEmbedding(nn.Module):
    """VGGish (AudioSet) embedding wrapper for FAD/MIND metrics.

    The upstream vggish_input frontend is numpy/resampy (non-differentiable); this reproduces its
    64-bin log-mel and 96-frame examples in torch so ``embed`` carries gradients. Per-clip
    embeddings average the 0.96 s examples. Clips under one example are zero-padded to 96 frames
    (VGGish's own framing drops, never pads); longer clips drop the incomplete trailing frames.
    """

    sample_rate = _SR
    name = "vggish"

    def __init__(
        self,
        device: str = "cuda",
        input_sample_rate: int = 48000,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.input_sample_rate = int(input_sample_rate)
        self.resampler = (
            None
            if self.input_sample_rate == self.sample_rate
            else torchaudio.transforms.Resample(self.input_sample_rate, self.sample_rate).to(self.device)
        )
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=_SR,
            n_fft=_N_FFT,
            win_length=_WIN,
            hop_length=_HOP,
            f_min=_FMIN,
            f_max=_FMAX,
            n_mels=_N_MELS,
            power=1.0,
            center=False,
            mel_scale="htk",
            norm=None,
        ).to(self.device)
        self.model = _VGG()
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(_WEIGHTS_URL, progress=False, map_location="cpu"))
        self.model.to(self.device)
        self.embedding_dim = 128
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
                f"VGGishEmbedding was initialized for {self.input_sample_rate} Hz input, "
                f"but got {sample_rate} Hz."
            )
        audio = audio.to(self.device).float()
        if audio.ndim == 3:
            audio = audio.mean(dim=1)
        if self.resampler is not None:
            audio = self.resampler(audio)
        logmel = torch.log(self.melspec(audio) + _LOG_OFFSET).transpose(1, 2)  # [B, n_frames, 64]
        n_frames = logmel.shape[1]
        if n_frames < _FRAMES_PER_EXAMPLE:
            logmel = torch.nn.functional.pad(logmel, (0, 0, 0, _FRAMES_PER_EXAMPLE - n_frames))
        n_ex = logmel.shape[1] // _FRAMES_PER_EXAMPLE
        examples = logmel[:, : n_ex * _FRAMES_PER_EXAMPLE].reshape(-1, 1, _FRAMES_PER_EXAMPLE, _N_MELS)
        return self.model(examples).view(logmel.shape[0], n_ex, self.embedding_dim).mean(dim=1)

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.embed(audio, sample_rate=sample_rate, audio_lengths=audio_lengths)
