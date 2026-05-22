from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import soundfile as sf
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from backbone.factory import build_backbone
from data.audio_dataset import AudioDirectoryDataset, collate_audio_batch
from emb.factory import build_embedding
from flow.fm import linear_interpolant
from losses.audio import FMLoss


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
            torch.zeros(int(cfg.data.sample_rate * cfg.data.clip_seconds)).numpy(),
            int(cfg.data.sample_rate),
        )
    dataset = AudioDirectoryDataset(**OmegaConf.to_container(cfg.data, resolve=True))
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_audio_batch)
    batch = next(iter(loader))
    model = build_backbone(cfg.backbone)
    conditioner = build_embedding(OmegaConf.to_container(cfg.conditioner, resolve=True))
    cond = None
    if conditioner is not None:
        cond = conditioner(batch["audio"], sample_rate=int(batch["sample_rate"]), audio_lengths=batch["audio_lengths"])
    flow_batch = linear_interpolant(
        batch["audio"],
        prediction_target=str(cfg.flow.prediction_target),
        eps=float(cfg.flow.get("eps", 1e-5)),
    )
    pred = model(flow_batch.x_t, t=flow_batch.t, cond=cond, length=batch["audio"].shape[-1])
    loss = FMLoss(**OmegaConf.to_container(cfg.loss, resolve=True), prediction_target=str(cfg.flow.prediction_target))(pred, flow_batch)
    loss.total.backward()
    print(f"ok batch={tuple(batch['audio'].shape)} pred={tuple(pred.shape)} loss={float(loss.total.detach()):.6f}")


if __name__ == "__main__":
    main()
