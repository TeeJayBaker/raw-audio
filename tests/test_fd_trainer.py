from __future__ import annotations

import math
from pathlib import Path

import hydra
import numpy as np
import soundfile as sf
import torch
from hydra.core.global_hydra import GlobalHydra

import trainer as trainer_module
from data.audio_dataset import AudioDirectoryDataset
from trainer import FDTrainer, RFTrainer, _dataset_checksum

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _compose(config_name: str, overrides: list[str]):
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return hydra.compose(config_name=config_name, overrides=overrides)


def _write_dataset(root: Path, sample_rate: int = 8000, n: int = 4) -> None:
    root.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 0.032, int(sample_rate * 0.032), dtype=np.float32)
    for i in range(n):
        signal = 0.5 * np.sin(2.0 * math.pi * (110.0 + 20.0 * i) * t)
        sf.write(root / f"tone_{i}.wav", signal, sample_rate)


def test_dataset_checksum_uses_path_and_file_count(tmp_path: Path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write_dataset(first_root, n=2)
    _write_dataset(second_root, n=2)

    first = AudioDirectoryDataset(first_root, sample_rate=8000)
    second = AudioDirectoryDataset(second_root, sample_rate=8000)
    assert _dataset_checksum(first) != _dataset_checksum(second)

    _write_dataset(first_root, n=3)
    first_with_extra_file = AudioDirectoryDataset(first_root, sample_rate=8000)
    assert _dataset_checksum(first) != _dataset_checksum(first_with_extra_file)


def test_fdtrainer_smoke_finetunes_from_checkpoint(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "data"
    _write_dataset(data_root)
    cache_dir = tmp_path / "fd_moments"
    monkeypatch.setattr(trainer_module, "FD_MOMENTS_DIR", cache_dir)

    # 1. Pretrain a tiny RF generator so FD post-training has a checkpoint to inherit from.
    base_run = tmp_path / "base"
    base = RFTrainer(
        _compose(
            "experiment/fm_smoke",
            [f"data.root={data_root}", f"train.run_dir={base_run}", "train.wandb.enabled=false"],
        )
    )
    base.run()
    checkpoint = base_run / "checkpoints" / "step_00000002.pt"
    assert checkpoint.exists()

    # 2. FD-loss fine-tune, inheriting backbone/flow/sampling/conditioner from the checkpoint cfg.
    fd_run = tmp_path / "fd"
    trainer = FDTrainer(
        _compose(
            "experiment/fd_finetune",
            [
                f"data.root={data_root}",
                "data.sample_rate=8000",
                "data.min_seconds=0.01",
                "data.augmentations.enabled=true",
                f"train.run_dir={fd_run}",
                "train.device=cpu",
                "train.amp=false",
                f"train.init_from={checkpoint}",
                "train.max_steps=2",
                "train.grad_accum_steps=2",
                "train.warmup_steps=0",
                "train.ema_decay=0.9",
                "train.val_fraction=0.5",
                "train.log_every=1",
                "train.val_every=1",
                "train.sample_every=1",
                "train.ckpt_every=1",
                "train.wandb.enabled=false",
                "train.dataloader.batch_size=2",
                "train.dataloader.drop_last=false",
                "train.dataloader.num_workers=0",
                "train.dataloader.pin_memory=false",
                "eval.metrics.embedding_validation.enabled=false",
                "fd.warm_start_samples=4",
                "fd.beta=0.81",
                "fd.embedders=[{type:random,embedding_dim:8,n_fft:64,hop_length:16,input_sample_rate:8000}]",
            ],
        )
    )
    cache_path = cache_dir / f"{_dataset_checksum(trainer.dataset)}-random.pt"
    assert cache_path.exists()
    cached = torch.load(cache_path, map_location="cpu", weights_only=True)
    expected = trainer_module.compute_real_moments(
        trainer.fd_loss.embedders[0],
        torch.utils.data.DataLoader(
            trainer.dataset,
            batch_size=2,
            shuffle=False,
            drop_last=False,
            collate_fn=trainer_module.collate_audio_batch,
        ),
        sample_rate=8000,
    )
    assert torch.allclose(cached["mu"], expected[0])
    assert torch.allclose(cached["cov"], expected[1])
    assert math.isclose(trainer.fd_loss.estimators[0].beta, 0.9)

    mtime = cache_path.stat().st_mtime_ns
    trainer._real_moments(trainer.fd_loss.embedders)
    assert cache_path.stat().st_mtime_ns == mtime

    optimizer_steps = []

    def record_step(*args, **kwargs):
        optimizer_steps.append(trainer.fd_warm_start_seen)

    trainer.optimizer.register_step_pre_hook(record_step)
    trainer.run()

    assert trainer.fd_warm_start_seen == 4
    assert all(bool(est.initialized) for est in trainer.fd_loss.estimators)
    assert optimizer_steps and all(seen >= 4 for seen in optimizer_steps)
    assert trainer.step == 2
    assert (fd_run / "checkpoints" / "step_00000002.pt").exists()

    metrics = trainer.validate()
    assert metrics and "val_total" in metrics
    assert "val_v_loss" in metrics
    assert all(math.isfinite(value) for value in metrics.values())
