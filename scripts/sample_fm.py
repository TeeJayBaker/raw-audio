from __future__ import annotations

import argparse

import soundfile as sf
import torch
from omegaconf import OmegaConf

from backbone.factory import build_backbone
from emb.factory import build_embedding
from flow.fm import sample_fm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--out", default="sample.wav")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="sample length in seconds (defaults to data.max_seconds)",
    )
    parser.add_argument(
        "--no-ema", action="store_true", help="sample raw model weights even when EMA is saved"
    )
    args = parser.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    model = build_backbone(cfg.backbone)
    model.load_state_dict(ckpt["model"])
    if ckpt.get("ema") is not None and not args.no_ema:
        shadow = ckpt["ema"]["shadow"]
        model_state = model.state_dict()
        model_state.update({name: value for name, value in shadow.items() if name in model_state})
        model.load_state_dict(model_state)
    model.eval()
    conditioner = build_embedding(
        {"type": "null", "embedding_dim": cfg.backbone.conditioning.cond_dim}
    )
    seconds = float(args.seconds if args.seconds is not None else cfg.data.max_seconds)
    shape = (
        1,
        int(cfg.data.channels),
        int(round(seconds * int(cfg.data.sample_rate))),
    )
    cond = conditioner(
        torch.zeros(shape),
        sample_rate=int(cfg.data.sample_rate),
        audio_lengths=torch.tensor([shape[-1]]),
    )
    audio = sample_fm(
        model,
        shape=shape,
        cond=cond,
        steps=args.steps or int(cfg.sampling.steps),
        prediction_target=str(cfg.flow.prediction_target),
        eps=float(cfg.flow.get("eps", 1e-5)),
        rms_lift=bool(cfg.data.get("rms_lift", False)),
        lift_scale=float(cfg.data.get("lift_scale", 3.0)),
    )
    audio = audio.clamp(-1.0, 1.0)
    sf.write(args.out, audio[0].cpu().transpose(0, 1).numpy(), int(cfg.data.sample_rate))


if __name__ == "__main__":
    main()
