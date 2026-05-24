from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import soxr
from torch.utils.data import Dataset, Subset

from data.audio_dataset import audio_item


def apply_gain(audio: np.ndarray, db: float) -> np.ndarray:
    return audio * np.float32(10.0 ** (float(db) / 20.0))


def apply_start_pad(
    audio: np.ndarray, valid_length: int, pad_samples: int
) -> tuple[np.ndarray, int]:
    pad_samples = int(pad_samples)
    if pad_samples <= 0:
        return audio, int(valid_length)
    target = audio.shape[0]
    shifted = np.pad(audio, ((pad_samples, 0), (0, 0)))[:target]
    return shifted, min(target, int(valid_length) + pad_samples)


def apply_pitch_shift(
    audio: np.ndarray,
    valid_length: int,
    semitones: int,
    sample_rate: int,
) -> tuple[np.ndarray, int]:
    semitones = int(semitones)
    if semitones == 0:
        return audio, int(valid_length)

    target = audio.shape[0]
    factor = 2.0 ** (semitones / 12.0)
    shifted = soxr.resample(
        audio,
        in_rate=float(sample_rate) * factor,
        out_rate=float(sample_rate),
        quality="HQ",
    )
    return _fit_length(shifted, target), min(
        target, max(1, int(round(int(valid_length) / factor)))
    )


@dataclass(frozen=True)
class WaveformAugmenter:
    sample_rate: int
    pitch_prob: float = 0.0
    pitch_semitones: tuple[int, int] = (-2, 2)
    start_pad_prob: float = 0.0
    start_pad_max_samples: int = 0
    gain_prob: float = 0.0
    gain_db: tuple[float, float] = (-3.0, 3.0)

    def __call__(self, audio: np.ndarray, valid_length: int) -> tuple[np.ndarray, int]:
        if _chance(self.pitch_prob):
            audio, valid_length = apply_pitch_shift(
                audio,
                valid_length,
                random.randint(*self.pitch_semitones),
                self.sample_rate,
            )
        if _chance(self.start_pad_prob) and self.start_pad_max_samples > 0:
            audio, valid_length = apply_start_pad(
                audio,
                valid_length,
                random.randint(0, self.start_pad_max_samples),
            )
        if _chance(self.gain_prob):
            audio = apply_gain(audio, random.uniform(*self.gain_db))
        return audio, valid_length


class AugmentedDataset(Dataset[dict[str, Any]]):
    def __init__(self, dataset: Dataset, augmenter: WaveformAugmenter):
        self.dataset = dataset
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        audio, valid_length, path, sample_rate = _source_audio(self.dataset, index)
        audio, valid_length = self.augmenter(audio, valid_length)
        return audio_item(audio, valid_length, path, sample_rate)


def build_waveform_augmenter(
    cfg: dict[str, Any] | None, sample_rate: int
) -> WaveformAugmenter | None:
    if not cfg or not bool(cfg.get("enabled", False)):
        return None

    pitch = cfg.get("pitch_shift", {}) or {}
    start_pad = cfg.get("start_pad", {}) or {}
    gain = cfg.get("gain", {}) or {}

    return WaveformAugmenter(
        sample_rate=int(sample_rate),
        pitch_prob=float(pitch.get("prob", 0.0)),
        pitch_semitones=_int_range(pitch.get("semitones", [-2, 2])),
        start_pad_prob=float(start_pad.get("prob", 0.0)),
        start_pad_max_samples=int(
            round(float(start_pad.get("max_ms", 0.0)) * int(sample_rate) / 1000.0)
        ),
        gain_prob=float(gain.get("prob", 0.0)),
        gain_db=_float_range(gain.get("db", [-3.0, 3.0])),
    )


def _source_audio(dataset: Dataset, index: int) -> tuple[np.ndarray, int, str, int]:
    if isinstance(dataset, Subset) and hasattr(dataset.dataset, "get_audio"):
        audio, valid_length, path = dataset.dataset.get_audio(dataset.indices[index])
        return audio, valid_length, str(path), int(dataset.dataset.sample_rate)
    if hasattr(dataset, "get_audio"):
        audio, valid_length, path = dataset.get_audio(index)
        return audio, valid_length, str(path), int(dataset.sample_rate)

    item = dataset[index]
    audio = item["audio"].detach().cpu().transpose(0, 1).numpy()
    return audio, int(item["audio_lengths"].item()), str(item["path"]), int(item["sample_rate"])


def _fit_length(audio: np.ndarray, target: int) -> np.ndarray:
    if audio.shape[0] >= target:
        return audio[:target]
    return np.pad(audio, ((0, target - audio.shape[0]), (0, 0)))


def _chance(prob: float) -> bool:
    return prob >= 1.0 or (prob > 0.0 and random.random() < prob)


def _int_range(value: Any) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError("integer augmentation ranges must have two values")
    low, high = int(value[0]), int(value[1])
    if low > high:
        raise ValueError("augmentation range lower bound must not exceed upper bound")
    return low, high


def _float_range(value: Any) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError("float augmentation ranges must have two values")
    low, high = float(value[0]), float(value[1])
    if low > high:
        raise ValueError("augmentation range lower bound must not exceed upper bound")
    return low, high
