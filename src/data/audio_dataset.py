from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, Subset

SUPPORTED_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}


def discover_audio_files(root: str | Path, exts: set[str] | None = None) -> list[Path]:
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Audio root not found: {root}")
    extensions = {ext.lower() for ext in (exts or SUPPORTED_AUDIO_EXTS)}
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
    if not files:
        raise FileNotFoundError(f"No audio files found under {root}")
    return files


def _to_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    if audio.ndim != 2:
        raise ValueError(f"Expected audio [T, C], got {tuple(audio.shape)}")
    if channels == 1:
        return audio.mean(axis=1, keepdims=True)
    if audio.shape[1] == channels:
        return audio
    if audio.shape[1] == 1:
        return np.repeat(audio, channels, axis=1)
    return audio[:, :channels]


def audio_item(
    audio: np.ndarray,
    valid_length: int,
    path: str,
    sample_rate: int,
) -> dict[str, Any]:
    return {
        "audio": torch.from_numpy(audio).transpose(0, 1).contiguous(),
        "audio_lengths": torch.tensor(valid_length, dtype=torch.long),
        "path": path,
        "sample_rate": sample_rate,
    }


class AudioDirectoryDataset(Dataset[dict[str, Any]]):
    """Variable-length audio dataset that always crops from sample 0.

    One-shots are characterised by their attack transient, so the crop window
    is anchored to the start of the file. Files shorter than ``min_seconds``
    are right-padded; files longer than ``max_seconds`` are truncated to
    ``max_seconds``; everything in between is left at its native length.
    Per-file durations are pre-scanned from headers so a ``BucketBatchSampler``
    can group similar-length items into the same batch.
    """

    def __init__(
        self,
        root: str | Path,
        sample_rate: int = 48000,
        min_seconds: float = 0.05,
        max_seconds: float = 4.0,
        channels: int = 1,
        exts: list[str] | None = None,
    ):
        self.root = Path(root).expanduser()
        self.sample_rate = int(sample_rate)
        self.min_samples = int(round(float(min_seconds) * self.sample_rate))
        self.max_samples = int(round(float(max_seconds) * self.sample_rate))
        if self.max_samples <= 0:
            raise ValueError("max_seconds must produce at least one sample")
        if self.min_samples > self.max_samples:
            raise ValueError("min_seconds must not exceed max_seconds")
        self.channels = int(channels)
        self.files = discover_audio_files(self.root, set(exts) if exts else None)
        self.durations = self._scan_durations()

    def __len__(self) -> int:
        return len(self.files)

    def _scan_durations(self) -> list[int]:
        t0 = time.monotonic()
        durations: list[int] = []
        for path in self.files:
            try:
                info = sf.info(str(path))
                dur = int(round(info.frames * self.sample_rate / info.samplerate))
            except Exception:
                dur = self.max_samples
            durations.append(max(self.min_samples, min(dur, self.max_samples)))
        print(f"Scanned {len(durations)} file durations in {time.monotonic() - t0:.1f}s")
        return durations

    def _crop_or_pad_start(self, audio: np.ndarray) -> tuple[np.ndarray, int]:
        length = audio.shape[0]
        if length >= self.max_samples:
            return audio[: self.max_samples], self.max_samples
        if length < self.min_samples:
            return np.pad(audio, ((0, self.min_samples - length), (0, 0))), length
        return audio, length

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        peak = max(float(np.abs(audio).max()), 1e-8)
        return audio / peak

    def __getitem__(self, index: int) -> dict[str, Any]:
        audio, valid_length, path = self.get_audio(index)
        return audio_item(audio, valid_length, str(path), self.sample_rate)

    def get_audio(self, index: int) -> tuple[np.ndarray, int, Path]:
        path = self.files[index]
        audio = self._load(path)
        audio = _to_channels(audio, self.channels)
        audio = self._normalize(audio)
        audio, valid_length = self._crop_or_pad_start(audio)
        return audio, valid_length, path

    def _load(self, path: Path) -> np.ndarray:
        audio, source_rate = sf.read(path, always_2d=True, dtype="float32")
        if int(source_rate) != self.sample_rate:
            audio = soxr.resample(audio, source_rate, self.sample_rate, quality="HQ")
        return audio


def subset_durations(ds: Dataset | Subset) -> list[int]:
    """Return per-item durations whether ``ds`` is a raw dataset or a Subset."""
    if isinstance(ds, Subset):
        parent = ds.dataset.durations  # type: ignore[attr-defined]
        return [parent[i] for i in ds.indices]
    return ds.durations  # type: ignore[attr-defined]


class BucketBatchSampler(Sampler[list[int]]):
    """Pool-sort-chunk sampler for minimal-padding variable-length batching.

    Shuffle indices, take pools of ``pool_multiplier * batch_size``, sort each
    pool by duration, chunk into batches of ``batch_size``, and shuffle batch
    order. Items within a batch share similar lengths so the dynamic collate
    wastes little on padding.
    """

    def __init__(
        self,
        durations: list[int],
        batch_size: int,
        pool_multiplier: int = 100,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.durations = list(durations)
        self.batch_size = int(batch_size)
        self.pool_multiplier = int(pool_multiplier)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        n = len(self.durations)
        if self.shuffle:
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))

        pool_size = max(self.pool_multiplier * self.batch_size, self.batch_size)
        batches: list[list[int]] = []
        for pool_start in range(0, n, pool_size):
            pool = indices[pool_start : pool_start + pool_size]
            pool.sort(key=lambda i: self.durations[i])
            for chunk_start in range(0, len(pool), self.batch_size):
                chunk = pool[chunk_start : chunk_start + self.batch_size]
                if len(chunk) == self.batch_size or not self.drop_last:
                    batches.append(chunk)

        if self.shuffle:
            order = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in order]
        yield from batches

    def __len__(self) -> int:
        n = len(self.durations)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


def collate_audio_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    audios = [item["audio"] for item in batch]
    target = max(a.shape[-1] for a in audios)
    padded = [
        F.pad(a, (0, target - a.shape[-1])) if a.shape[-1] < target else a for a in audios
    ]
    return {
        "audio": torch.stack(padded, dim=0),
        "audio_lengths": torch.stack([item["audio_lengths"] for item in batch], dim=0),
        "path": [item["path"] for item in batch],
        "sample_rate": batch[0]["sample_rate"],
    }
