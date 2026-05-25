from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf

from data.audio_dataset import (
    AudioDirectoryDataset,
    BucketBatchSampler,
    collate_audio_batch,
)
from data.augmentations import (
    AugmentedDataset,
    apply_gain,
    apply_pitch_shift,
    apply_start_pad,
    build_waveform_augmenter,
)
from ema import EMA
from emb.factory import build_embedding
from emb.matpac import MATPACEmbedding
from emb.null import NullEmbedding
from matpac.matpac import MATPAC


def test_audio_directory_dataset_start_crop_and_pad(tmp_path: Path):
    long_signal = torch.linspace(-1.0, 1.0, 4000)
    sf.write(tmp_path / "long.wav", long_signal.numpy(), 16000)
    sf.write(tmp_path / "short.wav", torch.ones(40).numpy(), 16000)
    # sample_rate matches source rate so cropping is byte-exact (no resampler smoothing).
    dataset = AudioDirectoryDataset(tmp_path, sample_rate=16000, min_seconds=0.005, max_seconds=0.1)
    items = {Path(d["path"]).name: d for d in (dataset[0], dataset[1])}
    long_item = items["long.wav"]
    short_item = items["short.wav"]
    # Long file (4000 src samples) is start-cropped to max_samples=1600 with the original head intact.
    assert long_item["audio"].shape == (1, 1600)
    assert long_item["audio_lengths"].item() == 1600
    # WAV PCM-16 round-trip introduces ~1/2^15 quantisation noise.
    assert torch.allclose(long_item["audio"][0], long_signal[:1600], atol=1e-4)
    # Short file is right-padded to min_samples=80, valid length preserved.
    assert short_item["audio"].shape == (1, 80)
    assert short_item["audio_lengths"].item() == 40
    assert torch.allclose(short_item["audio"][0, :40], torch.ones(40), atol=1e-4)
    assert torch.allclose(short_item["audio"][0, 40:], torch.zeros(40))
    # Collate pads short batch entry up to long entry's length.
    batch = collate_audio_batch([long_item, short_item])
    assert batch["audio"].shape == (2, 1, 1600)


def test_audio_directory_dataset_resamples_with_soxr(tmp_path: Path):
    signal = torch.sin(torch.linspace(0.0, 4.0, 400))
    sf.write(tmp_path / "tone.wav", signal.numpy(), 8000)

    dataset = AudioDirectoryDataset(
        tmp_path,
        sample_rate=16000,
        min_seconds=0.001,
        max_seconds=1.0,
    )
    item = dataset[0]

    assert item["sample_rate"] == 16000
    assert item["audio"].shape == (1, 800)
    assert item["audio_lengths"].item() == 800
    assert torch.isfinite(item["audio"]).all()


def test_bucket_sampler_groups_similar_lengths(tmp_path: Path):
    for i, samples in enumerate([100, 400, 1500, 1600]):
        sf.write(tmp_path / f"file_{i}.wav", torch.zeros(samples).numpy(), 16000)
    dataset = AudioDirectoryDataset(
        tmp_path, sample_rate=16000, min_seconds=0.001, max_seconds=0.2
    )
    sampler = BucketBatchSampler(
        dataset.durations, batch_size=2, pool_multiplier=10, shuffle=True, seed=0
    )
    assert len(sampler) == 2
    batches = list(sampler)
    assert all(len(b) == 2 for b in batches)
    # Every batch's two items are adjacent in the sorted-by-duration order.
    sorted_indices = sorted(range(len(dataset)), key=lambda i: dataset.durations[i])
    rank = {idx: pos for pos, idx in enumerate(sorted_indices)}
    for b in batches:
        positions = sorted(rank[i] for i in b)
        assert positions[1] - positions[0] == 1


def test_audio_gain_augmentation_uses_db_scale():
    audio = np.ones((4, 1), dtype=np.float32)

    gained = apply_gain(audio, 6.020599913)

    assert np.allclose(gained, np.full_like(audio, 2.0), atol=1e-5)


def test_audio_start_pad_preserves_shape_and_updates_length():
    audio = np.array([[1.0], [2.0], [3.0], [0.0], [0.0]], dtype=np.float32)

    padded, length = apply_start_pad(audio, valid_length=3, pad_samples=2)

    assert padded.shape == audio.shape
    assert length == 5
    assert np.allclose(padded, np.array([[0.0], [0.0], [1.0], [2.0], [3.0]]))


def test_audio_pitch_shift_uses_integer_resampling_and_preserves_shape():
    audio = np.sin(np.linspace(0.0, 6.0, 64, dtype=np.float32))[:, None]

    shifted, length = apply_pitch_shift(audio, valid_length=64, semitones=2, sample_rate=16000)

    assert shifted.shape == audio.shape
    assert np.isfinite(shifted).all()
    assert length == round(64 / (2.0 ** (2 / 12)))


def test_dataloader_augmentation_applies_to_wrapped_dataset_only(tmp_path: Path):
    for index in range(2):
        sf.write(tmp_path / f"file_{index}.wav", torch.full((64,), 0.25).numpy(), 16000)
    dataset = AudioDirectoryDataset(
        tmp_path,
        sample_rate=16000,
        min_seconds=0.004,
        max_seconds=0.004,
        channels=1,
    )
    augmenter = build_waveform_augmenter(
        {
            "enabled": True,
            "pitch_shift": {"prob": 0.0, "semitones": [0, 0]},
            "start_pad": {"prob": 0.0, "max_ms": 0.0},
            "gain": {"prob": 1.0, "db": [6.020599913, 6.020599913]},
        },
        sample_rate=16000,
    )

    augmented = AugmentedDataset(dataset, augmenter)
    train_item = augmented[0]
    val_item = dataset[0]

    assert torch.allclose(train_item["audio"], val_item["audio"] * 2.0)


def test_null_conditioner_shape():
    cond = NullEmbedding(embedding_dim=7)
    out = cond(torch.randn(3, 1, 16), sample_rate=48000, audio_lengths=torch.tensor([16, 16, 16]))
    assert out.shape == (3, 7)


def test_matpac_conditioner_missing_checkpoint_fails(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="MATPAC checkpoint"):
        MATPACEmbedding(tmp_path / "missing.ckpt")


def test_matpac_conditioner_loads_local_checkpoint_config(tmp_path: Path):
    params = {
        "sample_rate": 8000,
        "n_mels": 16,
        "n_fft": 64,
        "hop_length": 16,
        "f_min": 20.0,
        "f_max": 4000.0,
        "patch_size": 8,
        "norm_mean": 0.0,
        "norm_std": 1.0,
        "center": False,
        "hidden_size": 32,
        "encoder_depth": 1,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "use_rope": False,
        "rope_2d": False,
        "predictor_depth": 1,
        "predictor_dim": 16,
        "predictor_num_heads": 4,
        "num_hypotheses": 1,
        "classifier_hidden_dim": 32,
        "classifier_bottleneck_dim": 8,
        "num_classes": 16,
        "cls_classifier_hidden_dim": 32,
        "cls_classifier_bottleneck_dim": 8,
        "cls_num_classes": 16,
        "use_cls_token": True,
    }
    cfg = OmegaConf.create({"model": {"model": {"_target_": "matpac.matpac.MATPAC"} | params}})
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    model = MATPAC(**params)
    torch.save(
        {"state_dict": {f"model.{key}": value for key, value in model.state_dict().items()}},
        tmp_path / "tiny.ckpt",
    )

    conditioner = MATPACEmbedding(
        tmp_path / "tiny.ckpt",
        device="cpu",
        compile_encoder=False,
        input_sample_rate=8000,
    )
    out = conditioner(
        torch.randn(2, 1, 128), sample_rate=8000, audio_lengths=torch.tensor([128, 96])
    )

    assert out.shape == (2, 32)
    assert torch.isfinite(out).all()

    with pytest.raises(ValueError, match="initialized for 8000 Hz input"):
        conditioner(torch.randn(2, 1, 128), sample_rate=16000)

    resampling_conditioner = MATPACEmbedding(
        tmp_path / "tiny.ckpt",
        device="cpu",
        compile_encoder=False,
        input_sample_rate=16000,
    )
    resampled_out = resampling_conditioner(
        torch.randn(2, 1, 256),
        sample_rate=16000,
        audio_lengths=torch.tensor([256, 192]),
    )

    assert resampled_out.shape == (2, 32)
    assert torch.isfinite(resampled_out).all()


def test_embedding_factory_builds_null_embedding():
    embedding = build_embedding({"type": "null", "embedding_dim": 3})
    assert embedding(torch.randn(2, 1, 8)).shape == (2, 3)


def test_ema_apply_restore():
    model = torch.nn.Linear(2, 2)
    ema = EMA(model, decay=0.5)
    original = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(2.0)
    changed = model.weight.detach().clone()
    ema.update(model)
    ema.apply_to(model)
    assert not torch.allclose(model.weight, changed)
    ema.restore(model)
    assert torch.allclose(model.weight, changed)
    assert not torch.allclose(model.weight, original)
