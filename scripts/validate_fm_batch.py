from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hydra
import soundfile as sf
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from backbone.factory import build_backbone  # noqa: E402
from data.audio_dataset import AudioDirectoryDataset, collate_audio_batch  # noqa: E402
from emb.factory import build_embedding  # noqa: E402
from flow.fm import RectifiedFlow  # noqa: E402
from losses.audio import mr_stft_loss  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/fm_baseline.yaml")
    parser.add_argument("--use-null-conditioner", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Hydra-style config overrides")
    args = parser.parse_args()
    config_path = Path(args.config)
    configs_root = Path("configs").resolve()
    if config_path.suffix == ".yaml" and configs_root in config_path.resolve().parents:
        config_name = config_path.resolve().relative_to(configs_root).with_suffix("").as_posix()
        with hydra.initialize_config_dir(version_base=None, config_dir=str(configs_root)):
            cfg = hydra.compose(config_name=config_name, overrides=args.overrides)
    elif config_path == Path("configs/experiment/fm_baseline.yaml"):
        with hydra.initialize_config_dir(version_base=None, config_dir=str(configs_root)):
            cfg = hydra.compose(config_name="experiment/fm_baseline", overrides=args.overrides)
    else:
        if args.overrides:
            raise ValueError("Hydra overrides are only supported for configs under configs/")
        cfg = OmegaConf.load(config_path)
    if args.use_null_conditioner:
        cfg.conditioner = {"type": "null", "embedding_dim": cfg.backbone.conditioning.cond_dim}
    root = Path(cfg.data.root)
    root.mkdir(parents=True, exist_ok=True)
    if not any(root.rglob("*.wav")):
        sf.write(
            root / "validation.wav",
            torch.zeros(int(cfg.data.sample_rate * cfg.data.max_seconds)).numpy(),
            int(cfg.data.sample_rate),
        )
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    data_cfg.pop("bucket_pool_multiplier", None)
    data_cfg.pop("augmentations", None)
    dataset = AudioDirectoryDataset(**data_cfg)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_audio_batch)
    batch = next(iter(loader))
    model = build_backbone(cfg.backbone)
    conditioner = build_embedding(OmegaConf.to_container(cfg.conditioner, resolve=True))
    cond = None
    if conditioner is not None:
        cond = conditioner(batch["audio"], sample_rate=int(batch["sample_rate"]), audio_lengths=batch["audio_lengths"])
    flow = RectifiedFlow()
    x_t, t, x1 = flow.train_tuple(batch["audio"], t=torch.rand(batch["audio"].shape[0]))
    pred = flow._predict(model, x_t, t=t, cond=cond, length=batch["audio"].shape[-1], with_aux=False)[0]
    loss_cfg = OmegaConf.to_container(cfg.loss, resolve=True)
    total, _ = flow.loss(
        pred,
        x1,
        x_t,
        t,
        space=str(loss_cfg.get("loss_space", "v")),
        loss_type=str(loss_cfg.get("primary", "mse")),
    )
    mr_stft_weight = float(loss_cfg.get("mr_stft_weight", 0.0))
    if mr_stft_weight > 0.0:
        total = total + mr_stft_weight * mr_stft_loss(
            pred, x1, log_weight=float(loss_cfg.get("mr_stft_log_weight", 0.0))
        )
    total.backward()
    print(f"ok batch={tuple(batch['audio'].shape)} pred={tuple(pred.shape)} loss={float(total.detach()):.6f}")


if __name__ == "__main__":
    main()
