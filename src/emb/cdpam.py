from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

# CDPAM ships at 22.05 kHz mono and was trained on int16-magnitude floats (its load_audio scales
# by 32768), so waveforms in [-1, 1] are lifted to that range before the encoder.
_SR = 22050
_INT16_SCALE = 32768.0


class CDPAMEmbedding(nn.Module):
    """CDPAM (contrastive deep perceptual audio metric) wrapper.

    Serves two roles. As an FD-loss embedder, ``embed`` returns the 512-d normalized acoustics
    vector from CDPAM's ``base_encoder`` (clip-level: the encoder time-averages). As a perceptual
    loss, ``distance`` runs the learned ``model_dist`` head over a pair of those embeddings — the
    audio LPIPS-analog. CDPAM 0.0.6's own ``forward`` is already torch- and grad-friendly; we
    reproduce its three lines against the submodules so both paths carry gradients into the audio.
    """

    sample_rate = _SR
    name = "cdpam"

    def __init__(
        self,
        device: str = "cuda",
        input_sample_rate: int = 48000,
    ):
        super().__init__()
        try:
            import cdpam
        except ImportError as exc:
            raise ImportError("CDPAM metrics require optional dependency 'cdpam'.") from exc
        self.device = torch.device(device)
        self.input_sample_rate = int(input_sample_rate)
        self.resampler = (
            None
            if self.input_sample_rate == self.sample_rate
            else torchaudio.transforms.Resample(self.input_sample_rate, self.sample_rate).to(self.device)
        )
        # CDPAM(...) loads FINnet + bundled weights and sets it to eval; we keep its model only.
        self.model = cdpam.CDPAM(dev=str(self.device)).model
        self.embedding_dim = 512
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def _prep(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if int(sample_rate) != self.input_sample_rate:
            raise ValueError(
                f"CDPAMEmbedding was initialized for {self.input_sample_rate} Hz input, but got {sample_rate} Hz."
            )
        audio = audio.to(self.device).float()
        if audio.ndim == 3:
            audio = audio.mean(dim=1)
        if self.resampler is not None:
            audio = self.resampler(audio)
        return audio * _INT16_SCALE

    def _acoustics(self, prepped: torch.Tensor) -> torch.Tensor:
        _, acoustics, _ = self.model.base_encoder(prepped.unsqueeze(1))
        return F.normalize(acoustics, dim=1)  # [N, 512]

    def embed(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del audio_lengths
        return self._acoustics(self._prep(audio, sample_rate))

    def distance(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        sample_rate: int | None = None,
    ) -> torch.Tensor:
        """CDPAM's learned perceptual distance over a (prediction, target) waveform pair -> [N]."""
        sample_rate = self.input_sample_rate if sample_rate is None else sample_rate
        a1 = self._acoustics(self._prep(prediction, sample_rate))
        a2 = self._acoustics(self._prep(target, sample_rate))
        return self.model.model_dist(a1, a2)

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.embed(audio, sample_rate=sample_rate, audio_lengths=audio_lengths)
