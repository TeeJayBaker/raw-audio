from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
from torch import nn


def _ensure_local_model(model_name: str) -> str:
    """Return a local snapshot dir for ``model_name`` with a safetensors checkpoint.

    MERT ships only a pickle ``pytorch_model.bin``; transformers refuses to ``torch.load`` it on
    torch < 2.6. We load it ourselves (weights_only) once and write ``model.safetensors`` so
    ``from_pretrained`` uses the safe path. nnAudio's CQT extractor is unused by the forward
    embedding path, so the "requires nnAudio" warning is benign.
    """
    from huggingface_hub import snapshot_download

    local = Path(snapshot_download(model_name))
    safetensors = local / "model.safetensors"
    if not safetensors.exists():
        from safetensors.torch import save_file

        state = torch.load(local / "pytorch_model.bin", map_location="cpu", weights_only=True)
        save_file({key: value.contiguous() for key, value in state.items()}, safetensors)
    return str(local)


class MERTEmbedding(nn.Module):
    """MERT (music) embedding wrapper.

    Bypasses the numpy ``Wav2Vec2FeatureExtractor`` (only per-sequence standardization) so the
    waveform path stays differentiable. Mean-pools the last hidden state over time.
    """

    sample_rate = 24000
    name = "mert"

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "m-a-p/MERT-v1-95M",
        input_sample_rate: int = 48000,
    ):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("MERT metrics require optional dependency 'transformers'.") from exc
        self.device = torch.device(device)
        self.input_sample_rate = int(input_sample_rate)
        self.resampler = (
            None
            if self.input_sample_rate == self.sample_rate
            else torchaudio.transforms.Resample(self.input_sample_rate, self.sample_rate).to(self.device)
        )
        self.model = AutoModel.from_pretrained(_ensure_local_model(model_name), trust_remote_code=True).to(self.device)
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
                f"MERTEmbedding was initialized for {self.input_sample_rate} Hz input, but got {sample_rate} Hz."
            )
        audio = audio.to(self.device).float()
        if audio.ndim == 3:
            audio = audio.mean(dim=1)
        if self.resampler is not None:
            audio = self.resampler(audio)
        audio = (audio - audio.mean(dim=-1, keepdim=True)) / (
            audio.var(dim=-1, keepdim=True, unbiased=False) + 1e-7
        ).sqrt()
        outputs = self.model(input_values=audio, output_hidden_states=False)
        return outputs.last_hidden_state.mean(dim=1)

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.embed(audio, sample_rate=sample_rate, audio_lengths=audio_lengths)
