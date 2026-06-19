from __future__ import annotations

import math
from pathlib import Path

import hydra
import numpy as np
import soundfile as sf
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from trainer import MFTrainer, RFTrainer

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _compose(overrides: list[str]):
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = hydra.compose(config_name="experiment/fm_wavenext_smoke", overrides=overrides)
    return cfg


def _compose_mf(overrides: list[str]):
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return hydra.compose(config_name="experiment/mf_smoke", overrides=overrides)


def _write_dataset(root: Path, sample_rate: int = 8000, n: int = 4, seconds: float = 0.04) -> None:
    root.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, seconds, int(sample_rate * seconds), dtype=np.float32)
    for i in range(n):
        signal = 0.5 * np.sin(2.0 * math.pi * (110.0 + 20.0 * i) * t)
        sf.write(root / f"tone_{i}.wav", signal, sample_rate)


def test_rftrainer_smoke_trains_checkpoints_validates_and_samples(tmp_path: Path):
    data_root = tmp_path / "data"
    run_dir = tmp_path / "run"
    _write_dataset(data_root)
    cfg = _compose(
        [
            f"data.root={data_root}",
            f"train.run_dir={run_dir}",
            "train.wandb.enabled=false",
        ]
    )

    trainer = RFTrainer(cfg)
    trainer.run()

    # Training reached max_steps and wrote a checkpoint for it.
    assert trainer.step == 2
    assert (run_dir / "checkpoints" / "step_00000002.pt").exists()

    # Audio examples were generated and saved to disk.
    generated = list((run_dir / "samples").glob("step_*_generated_*.wav"))
    assert generated, "expected generated sample wavs"

    # Validation produces finite loss terms.
    metrics = trainer.validate()
    assert metrics and "val_total" in metrics
    assert all(math.isfinite(value) for value in metrics.values())


def test_rftrainer_resumes_from_latest_checkpoint(tmp_path: Path):
    data_root = tmp_path / "data"
    run_dir = tmp_path / "run"
    _write_dataset(data_root)

    first = RFTrainer(_compose([f"data.root={data_root}", f"train.run_dir={run_dir}", "train.wandb.enabled=false"]))
    first.run()
    assert first.step == 2

    resumed = RFTrainer(
        _compose(
            [
                f"data.root={data_root}",
                f"train.run_dir={run_dir}",
                "train.wandb.enabled=false",
                "train.resume=auto",
                "train.max_steps=4",
            ]
        )
    )
    assert resumed.step == 2  # picked up the latest checkpoint before training
    resumed.run()
    assert resumed.step == 4
    assert (run_dir / "checkpoints" / "step_00000004.pt").exists()


def _stft_trainer_overrides(data_root: Path, run_dir: Path) -> list[str]:
    return [
        f"data.root={data_root}",
        f"train.run_dir={run_dir}",
        "train.wandb.enabled=false",
        "train.device=cpu",
        "train.amp=false",
        "data.sample_rate=8000",
        "data.min_seconds=0.3",
        "data.max_seconds=0.3",
        "data.channels=1",
        "data.augmentations.enabled=false",
        "backbone.sample_rate=8000",
        "backbone.stft.n_fft=64",
        "backbone.stft.hop_length=16",
        "backbone.stft.win_length=64",
        "backbone.block.dim=32",
        "backbone.block.depth=1",
        "backbone.block.heads=2",
        "backbone.conditioning.cond_dim=16",
        "backbone.conditioning.embed_dim=16",
        "eval.metrics.embedding_validation.enabled=false",
        "train.dataloader.batch_size=2",
        "train.dataloader.num_workers=0",
        "train.dataloader.drop_last=false",
        "loss.wavefm_weight=1.0",
        "loss.complex_stft_weight=1.0",
    ]


def test_rftrainer_applies_wavefm_and_complex_aux_losses(tmp_path: Path):
    data_root = tmp_path / "data"
    _write_dataset(data_root, sample_rate=8000, seconds=0.3)
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = hydra.compose(
            config_name="experiment/fm_oneshots_mars_stft",
            overrides=_stft_trainer_overrides(data_root, tmp_path / "run"),
        )
    cfg.conditioner = OmegaConf.create({"type": "null", "embedding_dim": 16})

    trainer = RFTrainer(cfg)
    batch = next(iter(trainer.train_loader))
    audio = batch["audio"].to(trainer.device)
    cond = trainer.condition(audio, trainer.sample_rate, batch["audio_lengths"].to(trainer.device))
    loss, terms = trainer.training_step(audio, cond)
    loss.backward()

    assert {"wavefm", "complex_stft"} <= set(terms)
    assert math.isfinite(float(loss)) and float(loss) > 0


def test_mftrainer_rf_anchor_uses_adaptive_weighting(tmp_path: Path):
    data_root = tmp_path / "data"
    _write_dataset(data_root, seconds=0.032)
    trainer = MFTrainer(
        _compose_mf(
            [
                f"data.root={data_root}",
                f"train.run_dir={tmp_path / 'run'}",
                "train.wandb.enabled=false",
                "train.val_fraction=0",
            ]
        )
    )
    batch = next(iter(trainer.train_loader))
    audio = batch["audio"].to(trainer.device)
    cond = trainer.condition(
        audio,
        trainer.sample_rate,
        batch["audio_lengths"].to(trainer.device),
    )
    loss, terms = trainer._rf_step(audio, cond, adaptive=True)
    loss.backward()

    assert set(terms) == {"rf_mse"}
    assert math.isfinite(float(loss))


def test_aux_gate_keeps_only_rows_above_t_min():
    trainer = object.__new__(RFTrainer)
    pred = torch.randn(4, 1, 32)
    target = torch.randn_like(pred)
    t = torch.tensor([0.1, 0.2, 0.7, 0.0])

    # gated: only t >= aux_t_min (near data) survive — shared by every x-space aux loss
    trainer.cfg = OmegaConf.create({"loss": {"aux_t_min": 0.2}})
    gated = trainer._aux_gate(t, pred, target)
    assert gated is not None
    p, q = gated
    assert p.shape[0] == 2 and torch.equal(p, pred[t >= 0.2]) and torch.equal(q, target[t >= 0.2])

    # ungated by default (aux_t_min unset): tensors returned unchanged
    trainer.cfg = OmegaConf.create({"loss": {}})
    ungated = trainer._aux_gate(t, pred, target)
    assert ungated[0] is pred and ungated[1] is target

    # gate empties the batch -> None
    trainer.cfg = OmegaConf.create({"loss": {"aux_t_min": 0.99}})
    assert trainer._aux_gate(t, pred, target) is None
