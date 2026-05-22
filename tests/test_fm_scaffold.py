from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf

from data.audio_dataset import AudioDirectoryDataset, collate_audio_batch
from ema import EMA
from emb.factory import build_embedding
from emb.matpac import MATPACEmbedding
from emb.null import NullEmbedding
from flow.fm import linear_interpolant, output_to_v, sample_fm
from fm_trainer import _validate, build_dataloaders, train_fm
from losses.audio import FMLoss
from matpac.matpac import MATPAC


def test_audio_directory_dataset_fixed_shape(tmp_path: Path):
    path = tmp_path / "tone.wav"
    sf.write(path, torch.ones(64).numpy(), 16000)
    dataset = AudioDirectoryDataset(
        tmp_path, sample_rate=8000, clip_seconds=0.01, random_crop=False
    )
    item = dataset[0]
    assert item["audio"].shape == (1, 80)
    assert item["audio_lengths"].item() == 32
    batch = collate_audio_batch([item, item])
    assert batch["audio"].shape == (2, 1, 80)


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
                "clip_seconds": 0.01,
                "channels": 1,
                "random_crop": False,
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

    conditioner = MATPACEmbedding(tmp_path / "tiny.ckpt", device="cpu", compile_encoder=False)
    out = conditioner(
        torch.randn(2, 1, 128), sample_rate=8000, audio_lengths=torch.tensor([128, 96])
    )

    assert out.shape == (2, 32)
    assert torch.isfinite(out).all()


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
            "data": {"random_crop": False},
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
            "data": {"random_crop": False},
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
