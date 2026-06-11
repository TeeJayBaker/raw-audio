from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn


class CLAPEmbedding(nn.Module):
    """LAION-CLAP audio embedding wrapper for distribution metrics."""

    sample_rate = 48000
    name = "clap"

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
        spec = importlib.util.find_spec("laion_clap")
        if spec is None or spec.origin is None:
            raise ImportError("CLAP metrics require optional dependency 'laion_clap'.")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("CLAP metrics require optional dependency 'laion_clap'.") from exc
        package_dir = Path(spec.origin).parent
        self.device = torch.device(device)
        self.encode_batch_size = int(encode_batch_size)
        self.input_sample_rate = int(input_sample_rate)
        self.resampler = (
            None
            if self.input_sample_rate == self.sample_rate
            else torchaudio.transforms.Resample(self.input_sample_rate, self.sample_rate).to(self.device)
        )
        if checkpoint_path is None:
            repo_id = "lukewys/laion_clap"
            filename = "music_audioset_epoch_15_esc_90.14.pt"
            try:
                checkpoint_path = hf_hub_download(repo_id, filename, local_files_only=True)
            except FileNotFoundError:
                checkpoint_path = hf_hub_download(repo_id, filename)

        if str(package_dir) not in sys.path:
            sys.path.insert(0, str(package_dir))
        from clap_module import create_model

        self.model, model_cfg = create_model(
            amodel,
            "transformer",
            device=self.device,
            enable_fusion=enable_fusion,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        audio_state = {
            name.removeprefix("module."): value
            for name, value in state.items()
            if name.removeprefix("module.").startswith(("audio_branch.", "audio_projection."))
        }
        self.model.load_state_dict(audio_state, strict=False)
        for name in (
            "text_branch",
            "text_transform",
            "text_projection",
            "token_embedding",
            "positional_embedding",
            "ln_final",
        ):
            delattr(self.model, name)
        self.max_audio_samples = int(model_cfg["audio_cfg"]["clip_samples"])
        self.embedding_dim = 512
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
                f"CLAPEmbedding was initialized for {self.input_sample_rate} Hz input, "
                f"but got {sample_rate} Hz."
            )
        audio = audio.to(self.device).float()
        if audio.ndim == 3:
            audio = audio.mean(dim=1)
        if self.resampler is not None:
            audio = self.resampler(audio)
        chunks = []
        for i in range(0, audio.shape[0], self.encode_batch_size):
            chunk = audio[i : i + self.encode_batch_size]
            audio_input = []
            for waveform in chunk:
                if waveform.numel() > self.max_audio_samples:
                    overflow = waveform.numel() - self.max_audio_samples
                    start = int(torch.randint(overflow + 1, ()).item())
                    waveform = waveform[start : start + self.max_audio_samples]
                elif waveform.numel() < self.max_audio_samples:
                    repeats = self.max_audio_samples // max(waveform.numel(), 1)
                    waveform = F.pad(
                        waveform.repeat(repeats),
                        (0, self.max_audio_samples - waveform.numel() * repeats),
                    )
                audio_input.append({"longer": torch.tensor([False]), "waveform": waveform})
            chunks.append(self.model.get_audio_embedding(audio_input))
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.embed(audio, sample_rate=sample_rate, audio_lengths=audio_lengths)
