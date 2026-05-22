from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

SUPPORTED_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}


@dataclass(frozen=True)
class AudioExample:
    audio: torch.Tensor
    audio_lengths: torch.Tensor
    path: str
    sample_rate: int


def discover_audio_files(root: str | Path, exts: set[str] | None = None) -> list[Path]:
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Audio root not found: {root}")
    extensions = {ext.lower() for ext in (exts or SUPPORTED_AUDIO_EXTS)}
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
    if not files:
        raise FileNotFoundError(f"No audio files found under {root}")
    return files


def _to_channels(audio: torch.Tensor, channels: int) -> torch.Tensor:
    if audio.ndim != 2:
        raise ValueError(f"Expected audio [C, T], got {tuple(audio.shape)}")
    if channels == 1:
        return audio.mean(dim=0, keepdim=True)
    if audio.shape[0] == channels:
        return audio
    if audio.shape[0] == 1:
        return audio.expand(channels, -1)
    return audio[:channels]


class AudioDirectoryDataset(Dataset[dict[str, Any]]):
    """Recursive fixed-length audio dataset for waveform FM training."""

    def __init__(
        self,
        root: str | Path,
        sample_rate: int = 48000,
        clip_seconds: float = 1.0,
        channels: int = 1,
        random_crop: bool = True,
        normalize: str = "peak",
        exts: list[str] | None = None,
    ):
        self.root = Path(root).expanduser()
        self.sample_rate = int(sample_rate)
        self.clip_samples = int(round(float(clip_seconds) * self.sample_rate))
        if self.clip_samples <= 0:
            raise ValueError("clip_seconds must produce at least one sample")
        self.channels = int(channels)
        self.random_crop = bool(random_crop)
        self.normalize = normalize
        self.files = discover_audio_files(self.root, set(exts) if exts else None)
        self._resamplers: dict[tuple[int, int], torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.files)

    def _resample(self, audio: torch.Tensor, source_rate: int) -> torch.Tensor:
        if source_rate == self.sample_rate:
            return audio
        key = (int(source_rate), self.sample_rate)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(key[0], key[1])
        return self._resamplers[key](audio)

    def _crop_or_pad(self, audio: torch.Tensor) -> tuple[torch.Tensor, int]:
        valid_length = min(audio.shape[-1], self.clip_samples)
        if audio.shape[-1] == self.clip_samples:
            return audio, valid_length
        if audio.shape[-1] < self.clip_samples:
            return F.pad(audio, (0, self.clip_samples - audio.shape[-1])), valid_length
        max_start = audio.shape[-1] - self.clip_samples
        start = random.randint(0, max_start) if self.random_crop else max_start // 2
        return audio[:, start : start + self.clip_samples], valid_length

    def _normalize(self, audio: torch.Tensor) -> torch.Tensor:
        if self.normalize == "none":
            return audio
        if self.normalize != "peak":
            raise ValueError(f"Unknown normalize mode: {self.normalize}")
        peak = audio.abs().amax().clamp_min(1e-8)
        return audio / peak

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]
        audio, source_rate = self._load(path)
        audio = self._resample(audio, source_rate)
        audio = _to_channels(audio, self.channels)
        audio = self._normalize(audio)
        audio, valid_length = self._crop_or_pad(audio)
        return {
            "audio": audio,
            "audio_lengths": torch.tensor(valid_length, dtype=torch.long),
            "path": str(path),
            "sample_rate": self.sample_rate,
        }

    def _load(self, path: Path) -> tuple[torch.Tensor, int]:
        try:
            return torchaudio.load(path)
        except ImportError:
            data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
            return torch.from_numpy(data).transpose(0, 1), int(sample_rate)


def collate_audio_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "audio": torch.stack([item["audio"] for item in batch], dim=0),
        "audio_lengths": torch.stack([item["audio_lengths"] for item in batch], dim=0),
        "path": [item["path"] for item in batch],
        "sample_rate": batch[0]["sample_rate"],
    }
