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
    apply_gain,
    apply_pitch_shift,
    apply_start_pad,
)
from ema import EMA
from emb.factory import build_embedding
from emb.matpac import MATPACEmbedding
from emb.null import NullEmbedding
from flow.fm import linear_interpolant, output_to_v, sample_fm
from fm_trainer import _validate, build_dataloaders, train_fm
from losses.audio import FMLoss
from matpac.matpac import MATPAC
from utils.loggers import AudioMonitor, WandbLogger, build_audio_monitor, log_audio_monitor


def test_audio_directory_dataset_start_crop_and_pad(tmp_path: Path):
    long_signal = torch.linspace(-1.0, 1.0, 4000)
    sf.write(tmp_path / "long.wav", long_signal.numpy(), 16000)
    sf.write(tmp_path / "short.wav", torch.ones(40).numpy(), 16000)
    # sample_rate matches source rate so cropping is byte-exact (no resampler smoothing).
    dataset = AudioDirectoryDataset(
        tmp_path, sample_rate=16000, min_seconds=0.005, max_seconds=0.1, normalize="none"
    )
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
        normalize="none",
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


def test_dataloader_applies_augmentation_to_train_only(tmp_path: Path):
    for index in range(2):
        sf.write(tmp_path / f"file_{index}.wav", torch.full((64,), 0.25).numpy(), 16000)
    cfg = OmegaConf.create(
        {
            "data": {
                "root": str(tmp_path),
                "sample_rate": 16000,
                "min_seconds": 0.004,
                "max_seconds": 0.004,
                "channels": 1,
                "normalize": "none",
                "augmentations": {
                    "enabled": True,
                    "pitch_shift": {"prob": 0.0, "semitones": [0, 0]},
                    "start_pad": {"prob": 0.0, "max_ms": 0.0},
                    "gain": {"prob": 1.0, "db": [6.020599913, 6.020599913]},
                },
            },
            "train": {
                "val_fraction": 0.5,
                "dataloader": {
                    "batch_size": 1,
                    "num_workers": 0,
                    "pin_memory": False,
                    "drop_last": False,
                },
            },
        }
    )

    train_loader, val_loader = build_dataloaders(cfg)
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    assert train_batch["audio"].abs().mean() > 0.45
    assert val_batch["audio"].abs().mean() < 0.3


def test_hydra_trainer_target_is_importable():
    assert train_fm.__name__ == "train_fm"


def test_empty_training_loader_fails_fast(tmp_path: Path):
    path = tmp_path / "tone.wav"
    sf.write(path, torch.ones(64).numpy(), 16000)
    cfg = OmegaConf.create(
        {
            "data": {
                "root": str(tmp_path),
                "sample_rate": 16000,
                "min_seconds": 0.01,
                "max_seconds": 0.01,
                "channels": 1,
                "normalize": "none",
            },
            "train": {
                "val_fraction": 0.0,
                "dataloader": {
                    "batch_size": 4,
                    "num_workers": 0,
                    "pin_memory": False,
                    "drop_last": True,
                },
            },
        }
    )
    with pytest.raises(ValueError, match="Training dataloader is empty"):
        build_dataloaders(cfg)


def test_fm_interpolant_recovers_v_from_x_prediction():
    x1 = torch.randn(2, 1, 16)
    fb = linear_interpolant(x1, t=torch.tensor([0.25, 0.75]))
    assert fb.x_t.shape == x1.shape
    # Network outputs x_1 exactly; output_to_v must invert the interpolant to recover v.
    assert torch.allclose(output_to_v(fb.x1, fb.x_t, fb.t), fb.v, atol=1e-6)


@pytest.mark.parametrize("loss_space", ["x", "v"])
def test_fm_loss_backward(loss_space: str):
    x1 = torch.randn(2, 1, 16)
    fb = linear_interpolant(x1)
    pred = torch.randn_like(x1, requires_grad=True)
    loss = FMLoss(loss_space=loss_space)(pred, fb)
    loss.total.backward()
    assert pred.grad is not None
    assert torch.isfinite(loss.total)


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


def test_sampler_runs_with_toy_model():
    class Toy(torch.nn.Module):
        def forward(self, x, t=None, cond=None, length=None):
            del t, cond, length
            return torch.zeros_like(x)

    sample = sample_fm(Toy(), shape=(2, 1, 8), steps=2)
    assert sample.shape == (2, 1, 8)


def test_sampler_can_reuse_fixed_distinct_noise():
    class Toy(torch.nn.Module):
        def forward(self, x, t=None, cond=None, length=None):
            del t, cond, length
            return x

    noise = torch.randn(2, 1, 8)
    sample_a = sample_fm(Toy(), shape=(2, 1, 8), steps=2, noise=noise)
    sample_b = sample_fm(Toy(), shape=(2, 1, 8), steps=2, noise=noise)

    assert torch.allclose(sample_a, noise)
    assert torch.allclose(sample_b, noise)
    assert not torch.allclose(noise[0], noise[1])


def test_sampler_rejects_wrong_noise_shape():
    class Toy(torch.nn.Module):
        def forward(self, x, t=None, cond=None, length=None):
            del t, cond, length
            return x

    with pytest.raises(ValueError, match="noise shape"):
        sample_fm(Toy(), shape=(2, 1, 8), noise=torch.randn(1, 1, 8))


@pytest.mark.parametrize("loss_space", ["x", "v"])
def test_validation_reports_x_and_v_mse(loss_space: str):
    class Toy(torch.nn.Module):
        def forward(self, x, t=None, cond=None, length=None):
            del t, cond, length
            return torch.zeros_like(x)

    batch = {
        "audio": torch.randn(2, 1, 16),
        "audio_lengths": torch.tensor([16, 16]),
        "sample_rate": 16000,
    }
    cfg = OmegaConf.create({"flow": {"eps": 1e-5}, "train": {"val_batches": 1}})
    metrics = _validate(
        Toy(), None, [batch], FMLoss(loss_space=loss_space), cfg, torch.device("cpu")
    )
    assert "val_x_mse" in metrics
    assert "val_v_mse" in metrics
    assert torch.isfinite(torch.tensor([metrics["val_x_mse"], metrics["val_v_mse"]])).all()


@pytest.mark.parametrize(
    ("distance", "expected_key"),
    [
        ("fad", "val_fad"),
        ("mind", "val_mind"),
        ("both", "val_fad"),
    ],
)
def test_validation_can_report_embedding_distance_metrics(distance: str, expected_key: str):
    class Toy(torch.nn.Module):
        def forward(self, x, t=None, cond=None, length=None):
            del t, cond, length
            return torch.zeros_like(x)

    class MeanStdConditioner(torch.nn.Module):
        embedding_dim = 2

        def forward(self, audio, sample_rate=16000, audio_lengths=None):
            del sample_rate, audio_lengths
            flat = audio.flatten(1)
            return torch.stack([flat.mean(dim=1), flat.std(dim=1) + 10.0], dim=1)

    class MetricBackend(torch.nn.Module):
        def forward(self, audio, sample_rate=16000, audio_lengths=None):
            del sample_rate, audio_lengths
            flat = audio.flatten(1)
            return torch.stack([flat.mean(dim=1), flat.std(dim=1), flat.abs().mean(dim=1)], dim=1)

    batch = {
        "audio": torch.randn(4, 1, 16),
        "audio_lengths": torch.tensor([16, 16, 16, 16]),
        "path": ["a.wav", "b.wav", "c.wav", "d.wav"],
        "sample_rate": 16000,
    }
    cfg = OmegaConf.create(
        {
            "flow": {"eps": 1e-5},
            "train": {"val_batches": 1},
            "sampling": {"steps": 1},
            "eval": {
                "metrics": {
                    "embedding_validation": {
                        "enabled": True,
                        "distance": distance,
                        "sample_steps": 1,
                        "mind_projections": 8,
                        "cosine": True,
                    }
                }
            },
        }
    )
    metrics = _validate(
        Toy(),
        MeanStdConditioner(),
        [batch],
        FMLoss(),
        cfg,
        torch.device("cpu"),
        metric_embedding_backend=MetricBackend(),
    )
    assert expected_key in metrics
    assert "val_matpac_embedding_cosine" in metrics
    if distance == "both":
        assert "val_mind" in metrics
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()


def test_validation_caches_real_metric_embeddings_by_path_when_crop_is_deterministic():
    class Toy(torch.nn.Module):
        def forward(self, x, t=None, cond=None, length=None):
            del t, cond, length
            return torch.zeros_like(x)

    class CountingConditioner(torch.nn.Module):
        embedding_dim = 2

        def forward(self, audio, sample_rate=16000, audio_lengths=None):
            del sample_rate, audio_lengths
            flat = audio.flatten(1)
            return torch.stack([flat.mean(dim=1), flat.std(dim=1)], dim=1)

    class CountingMetricBackend(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, audio, sample_rate=16000, audio_lengths=None):
            del sample_rate, audio_lengths
            self.calls += 1
            flat = audio.flatten(1)
            return torch.stack([flat.mean(dim=1), flat.std(dim=1)], dim=1)

    batch = {
        "audio": torch.randn(2, 1, 16),
        "audio_lengths": torch.tensor([16, 16]),
        "path": ["a.wav", "b.wav"],
        "sample_rate": 16000,
    }
    cfg = OmegaConf.create(
        {
            "flow": {"eps": 1e-5},
            "train": {"val_batches": 1},
            "sampling": {"steps": 1},
            "eval": {
                "metrics": {
                    "embedding_validation": {
                        "enabled": True,
                        "distance": "fad",
                        "sample_steps": 1,
                        "cache_real": True,
                    }
                }
            },
        }
    )
    conditioner = CountingConditioner()
    metric_backend = CountingMetricBackend()
    cache = {}

    _validate(
        Toy(),
        conditioner,
        [batch],
        FMLoss(),
        cfg,
        torch.device("cpu"),
        metric_embedding_backend=metric_backend,
        real_embedding_cache=cache,
    )
    first_calls = metric_backend.calls
    metrics = _validate(
        Toy(),
        conditioner,
        [batch],
        FMLoss(),
        cfg,
        torch.device("cpu"),
        metric_embedding_backend=metric_backend,
        real_embedding_cache=cache,
    )

    assert len(cache) == 2
    assert metrics["val_real_embedding_cache_size"] == 2.0
    assert metric_backend.calls == first_calls + 1


def test_audio_monitor_uses_validation_audio_and_caches_conditioning():
    class CountingConditioner(torch.nn.Module):
        embedding_dim = 2

        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, audio, sample_rate=16000, audio_lengths=None):
            del sample_rate, audio_lengths
            self.calls += 1
            flat = audio.flatten(1)
            return torch.stack([flat.mean(dim=1), flat.std(dim=1)], dim=1)

    batch = {
        "audio": torch.arange(4 * 1 * 8, dtype=torch.float32).view(4, 1, 8),
        "audio_lengths": torch.tensor([8, 8, 8, 8]),
        "sample_rate": 16000,
    }
    cfg = OmegaConf.create({"train": {"wandb": {"audio_examples": 3, "audio_seed": 123}}})
    conditioner = CountingConditioner()

    monitor = build_audio_monitor([batch], conditioner, cfg, torch.device("cpu"))

    assert monitor is not None
    assert monitor.audio.shape == (3, 1, 8)
    assert monitor.cond is not None
    assert monitor.cond.shape == (3, 2)
    assert conditioner.calls == 1
    assert not torch.allclose(monitor.noise[0], monitor.noise[1])


def test_audio_monitor_logging_logs_reference_once_and_generated_each_time(tmp_path: Path):
    class IdentitySampler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_cond = None

        def forward(self, x, t=None, cond=None, length=None):
            del t, length
            self.seen_cond = cond.detach().cpu() if cond is not None else None
            return x

    class MockWandb:
        class Audio:
            def __init__(self, data_or_path, sample_rate=None, caption=None):
                self.data_or_path = data_or_path
                self.sample_rate = sample_rate
                self.caption = caption

    class MockRun:
        def __init__(self):
            self.logged = []

        def log(self, values, step=None):
            self.logged.append((step, values))

        def finish(self):
            pass

    run = MockRun()
    logger = WandbLogger(MockWandb, run)
    monitor = AudioMonitor(
        audio=torch.zeros(2, 1, 8),
        audio_lengths=torch.tensor([8, 8]),
        cond=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        noise=torch.randn(2, 1, 8),
        sample_rate=16000,
    )
    cfg = OmegaConf.create(
        {
            "train": {
                "wandb": {
                    "log_reference_once": True,
                    "save_reference_local": True,
                }
            },
            "sampling": {"steps": 1, "method": "euler"},
            "flow": {"eps": 1e-5},
        }
    )
    model = IdentitySampler()

    log_audio_monitor(model, cfg, torch.device("cpu"), tmp_path, 1, monitor, logger)
    log_audio_monitor(model, cfg, torch.device("cpu"), tmp_path, 2, monitor, logger)

    logged_keys = [next(iter(values.keys())) for _, values in run.logged]
    assert logged_keys == ["audio/reference", "audio/generated", "audio/generated"]
    assert (tmp_path / "reference_000.wav").exists()
    assert (tmp_path / "step_00000001_generated_000.wav").exists()
    assert (tmp_path / "step_00000002_generated_000.wav").exists()
    assert torch.allclose(model.seen_cond, monitor.cond)
