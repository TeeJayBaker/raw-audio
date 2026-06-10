from __future__ import annotations

import torch
import torchaudio
from torch import nn

# AudioSet fbank standardization stats (AudioMAE / AST).
_NORM_MEAN = -4.2677393
_NORM_STD = 4.5689974
_TARGET_FRAMES = 1024


class AudioMAEEmbedding(nn.Module):
    """AudioMAE (ViT-B, AudioSet-2M) embedding wrapper.

    Reproduces AudioMAE's kaldi-fbank frontend with ``torchaudio.compliance.kaldi.fbank``
    (pure-torch, differentiable; per-sample so looped over the batch) feeding the timm ViT.
    Mean-pools the patch tokens (CLS dropped) to a 768-d vector.
    """

    sample_rate = 16000
    name = "audiomae"

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "hf-hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m",
        input_sample_rate: int = 48000,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("AudioMAE metrics require optional dependency 'timm'.") from exc
        self.device = torch.device(device)
        self.input_sample_rate = int(input_sample_rate)
        self.resampler = (
            None
            if self.input_sample_rate == self.sample_rate
            else torchaudio.transforms.Resample(self.input_sample_rate, self.sample_rate).to(self.device)
        )
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).to(self.device)
        self.embedding_dim = 768
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def _fbank(self, wav: torch.Tensor) -> torch.Tensor:
        wav = wav - wav.mean()
        feat = torchaudio.compliance.kaldi.fbank(
            wav.unsqueeze(0),
            htk_compat=True,
            sample_frequency=self.sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        )
        frames = feat.shape[0]
        if frames < _TARGET_FRAMES:
            feat = torch.nn.functional.pad(feat, (0, 0, 0, _TARGET_FRAMES - frames))
        else:
            feat = feat[:_TARGET_FRAMES]
        return (feat - _NORM_MEAN) / (_NORM_STD * 2)

    def embed(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del audio_lengths
        if int(sample_rate) != self.input_sample_rate:
            raise ValueError(
                f"AudioMAEEmbedding was initialized for {self.input_sample_rate} Hz input, but got {sample_rate} Hz."
            )
        audio = audio.to(self.device).float()
        if audio.ndim == 3:
            audio = audio.mean(dim=1)
        if self.resampler is not None:
            audio = self.resampler(audio)
        mels = torch.stack([self._fbank(audio[i]) for i in range(audio.shape[0])]).unsqueeze(1)
        feats = self.model.forward_features(mels)  # [B, 1 + patches, 768]
        return feats[:, 1:, :].mean(dim=1)

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.embed(audio, sample_rate=sample_rate, audio_lengths=audio_lengths)
