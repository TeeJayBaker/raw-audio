"""CFG scale x NFE x guidance-interval sweep on a trained RF checkpoint (no src/ edits).

Inference-only. Loads a checkpoint (EMA weights), embeds N reference clips with MATPAC
(cond) + CLAP (metric backend), then for every (steps, w, [t_lo, t_hi)) combo samples
from the SAME per-batch noise and scores against the same reference embeddings:
KAD / MIND on CLAP, paired MATPAC cosine + retrieval, raw genRMS, true NFE per clip.

t convention follows src/flow/fm.py: t=0 noise -> t=1 data. Guidance fires on Euler
steps whose left endpoint lies in [t_lo, t_hi), applied in x-pred space exactly like
RectifiedFlow._model_v; outputs are per-item peak-normalised like RFTrainer.sample.

Training cond-dropout nulls by ZEROING the MATPAC vector before the cond MLP
(BaseTrainer._cfg_dropout), while RectifiedFlow._model_v passes cond=None, which skips
the cond MLP entirely -- a different null branch than the model was trained on.
--null picks which one the uncond pass uses (default: zeros, the trained one).

Reference files are a seeded draw from the full data root (the run's val split came
from an unseeded random_split and is irreproducible); constant across all combos, may
overlap train -- fine for comparing sampling configs, not for absolute numbers.

  python experiments/cfg_sampling/sweep.py --device cuda:0
  python experiments/cfg_sampling/sweep.py --device cpu --smoke
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, ".")
sys.path.insert(0, "src")
from backbone.factory import build_backbone  # noqa: E402
from data.audio_dataset import _to_channels, discover_audio_files  # noqa: E402
from emb.factory import build_embedding  # noqa: E402
from eval.audio_metrics import (  # noqa: E402
    embedding_cosine_score,
    kernel_audio_distance,
    monge_audio_distance,
)
from flow.fm import EPS, RectifiedFlow  # noqa: E402

DEFAULT_CKPT = "runs/fm-oneshots-mars-stft-b64-200k/checkpoints/step_00100000.pt"
OUT_DIR = Path("experiments/cfg_sampling/out")


def select_files(root, n, sample_rate, min_samples, max_samples, seed):
    files = list(discover_audio_files(root))
    random.Random(seed).shuffle(files)
    picked = []
    for path in files:
        try:
            info = sf.info(str(path))
        except RuntimeError:
            continue
        dur = int(info.frames * sample_rate / info.samplerate)
        if min_samples <= dur <= max_samples:
            picked.append(path)
            if len(picked) == n:
                break
    if len(picked) < n:
        raise ValueError(f"only {len(picked)}/{n} files in [{min_samples}, {max_samples}] samples")
    return picked


def load_clip(path, sample_rate, channels, max_samples):
    """Mirrors AudioDirectoryDataset.get_audio: resample, mono, peak-norm, crop from 0."""
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    if int(sr) != sample_rate:
        audio = soxr.resample(audio, sr, sample_rate, quality="HQ")
    audio = _to_channels(audio, channels)
    audio = audio / max(float(np.abs(audio).max()), 1e-8)
    audio = audio[:max_samples]
    return torch.from_numpy(audio).transpose(0, 1).contiguous()  # [C, T]


def make_batches(paths, sample_rate, channels, max_samples, batch_size, dev):
    clips = [load_clip(p, sample_rate, channels, max_samples) for p in paths]
    order = sorted(range(len(clips)), key=lambda i: clips[i].shape[-1])
    batches = []
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        target = max(clips[i].shape[-1] for i in idx)
        audio = torch.stack(
            [F.pad(clips[i], (0, target - clips[i].shape[-1])) for i in idx]
        ).to(dev)
        lengths = torch.tensor([clips[i].shape[-1] for i in idx], device=dev)
        names = [Path(paths[i]).stem for i in idx]
        batches.append({"audio": audio, "audio_lengths": lengths, "names": names})
    return batches


@torch.no_grad()
def sample_interval_cfg(model, flow, cond, null_cond, noise, length, steps, w, lo, hi):
    x = noise.clone()
    batch = x.shape[0]
    grid = torch.linspace(EPS, 1.0 - EPS, steps + 1, device=x.device, dtype=x.dtype)
    nfe = 0
    for i in range(steps):
        t = grid[i].expand(batch)
        x_pred = model(x, t=t, cond=cond, length=length)
        nfe += 1
        if w != 1.0 and lo <= float(grid[i]) < hi:
            x_null = model(x, t=t, cond=null_cond, length=length)
            nfe += 1
            x_pred = x_null + w * (x_pred - x_null)
        x = x + (grid[i + 1] - grid[i]) * flow.target_to_v(x_pred, x, t)
    return x, nfe / batch * batch / steps if steps else 0.0, nfe


def parse_intervals(spec):
    out = []
    for part in spec.split(","):
        lo, hi = part.split(":")
        out.append((float(lo), float(hi)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--min-samples", type=int, default=4800)
    ap.add_argument("--max-samples", type=int, default=96000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", default="1,2,4,8,16,32")
    ap.add_argument("--w", default="1.5,2.0,2.5,3.5,5.0")
    ap.add_argument("--intervals", default="0:0.8,0:0.6,0.2:1,0.2:0.8,0.4:1,0.6:1")
    ap.add_argument("--interval-steps", default="8,32")
    ap.add_argument("--interval-w", default="2.0,3.5")
    ap.add_argument("--null", choices=["zeros", "none"], default="zeros")
    ap.add_argument("--save-audio", type=int, default=0, help="save first N wavs per combo")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n, args.batch = 4, 2
        args.steps, args.w = "1,2", "2.0"
        args.intervals, args.interval_steps, args.interval_w = "0:0.6", "2", "2.0"

    dev = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    model = build_backbone(cfg.backbone)
    model.load_state_dict(ckpt["model"])
    if ckpt.get("ema") is not None:
        state = model.state_dict()
        state.update({k: v for k, v in ckpt["ema"]["shadow"].items() if k in state})
        model.load_state_dict(state)
    model.to(dev).eval()
    flow = RectifiedFlow()
    lift_scale = (
        float(cfg.data.get("lift_scale", 1.0)) if bool(cfg.data.get("rms_lift", False)) else 1.0
    )
    sr = int(cfg.data.sample_rate)

    conditioner = build_embedding(
        {**OmegaConf.to_container(cfg.conditioner), "device": str(dev)}, device=dev
    ).to(dev).eval()
    clap = build_embedding({"type": "clap", "device": str(dev)}, device=dev).eval()

    paths = select_files(cfg.data.root, args.n, sr, args.min_samples, args.max_samples, args.seed)
    batches = make_batches(paths, sr, int(cfg.data.channels), args.max_samples, args.batch, dev)
    with torch.no_grad():
        for bi, b in enumerate(batches):
            b["cond"] = conditioner(b["audio"], sample_rate=sr, audio_lengths=b["audio_lengths"])
            b["clap"] = clap(b["audio"], sample_rate=sr, audio_lengths=b["audio_lengths"])
            g = torch.Generator(device=dev).manual_seed(1000 + bi)
            b["noise"] = torch.randn(b["audio"].shape, device=dev, generator=g)
    real_clap = torch.cat([b["clap"] for b in batches])
    real_cond = torch.cat([b["cond"] for b in batches])
    real_rms = float(
        torch.cat([b["audio"].pow(2).mean(dim=(-2, -1)).sqrt() for b in batches]).mean()
    )

    steps_list = [int(s) for s in args.steps.split(",")]
    w_list = [float(w) for w in args.w.split(",") if float(w) != 1.0]
    combos = [(s, 1.0, (0.0, 1.0)) for s in steps_list]
    combos += [(s, w, (0.0, 1.0)) for s in steps_list for w in w_list]
    if args.intervals:
        combos += [
            (s, w, iv)
            for iv in parse_intervals(args.intervals)
            for s in (int(x) for x in args.interval_steps.split(","))
            for w in (float(x) for x in args.interval_w.split(","))
        ]

    tag = f"{Path(args.ckpt).parent.parent.name}_{Path(args.ckpt).stem}"
    print(
        f"=== {tag} | n={args.n} clips "
        f"[{args.min_samples}..{args.max_samples}] samples | null={args.null} | "
        f"real_rms={real_rms:.3f} | {len(combos)} combos ===",
        flush=True,
    )
    if args.save_audio:
        ref_dir = OUT_DIR / "wavs" / tag / "real"
        ref_dir.mkdir(parents=True, exist_ok=True)
        b0 = batches[0]
        for j in range(min(args.save_audio, b0["audio"].shape[0])):
            n_j = int(b0["audio_lengths"][j])
            sf.write(ref_dir / f"{j:02d}_{b0['names'][j]}.wav", b0["audio"][j, :, :n_j].cpu().numpy().T, sr)
    results = []
    for steps, w, (lo, hi) in combos:
        t0 = time.time()
        fake_clap, fake_cond, grms, nfe_total = [], [], [], 0
        for b in batches:
            null_cond = torch.zeros_like(b["cond"]) if args.null == "zeros" else None
            length = b["audio"].shape[-1]
            raw, _, nfe = sample_interval_cfg(
                model, flow, b["cond"], null_cond, b["noise"], length, steps, w, lo, hi
            )
            raw = raw / lift_scale
            grms.append(raw.pow(2).mean(dim=(-2, -1)).sqrt())
            peak = raw.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            fake = (raw / peak).clamp(-1.0, 1.0)
            fake_clap.append(clap(fake, sample_rate=sr, audio_lengths=b["audio_lengths"]))
            fake_cond.append(
                conditioner(fake, sample_rate=sr, audio_lengths=b["audio_lengths"])
            )
            if args.save_audio and b is batches[0]:
                combo_dir = OUT_DIR / "wavs" / tag / f"s{steps}_w{w}"
                combo_dir.mkdir(parents=True, exist_ok=True)
                for j in range(min(args.save_audio, fake.shape[0])):
                    n_j = int(b["audio_lengths"][j])
                    sf.write(combo_dir / f"{j:02d}_{b['names'][j]}.wav", fake[j, :, :n_j].cpu().numpy().T, sr)
            nfe_total += nfe
        fake_clap, fake_cond = torch.cat(fake_clap), torch.cat(fake_cond)
        kad = float(kernel_audio_distance(real_clap, fake_clap)["kad"])
        mind = float(monge_audio_distance(real_clap, fake_clap, projections=256)["mind"])
        cos = float(embedding_cosine_score(real_cond, fake_cond))
        sim = F.normalize(fake_cond, dim=1) @ F.normalize(real_cond, dim=1).T
        retr = float((sim.argmax(dim=1) == torch.arange(sim.shape[0], device=dev)).float().mean())
        grms = float(torch.cat(grms).mean())
        nfe_clip = nfe_total / args.n
        row = {
            "steps": steps, "w": w, "lo": lo, "hi": hi, "kad": kad, "mind": mind,
            "cos": cos, "retr": retr, "grms": grms, "nfe_per_clip": nfe_clip,
        }
        results.append(row)
        print(
            f"  steps={steps:>2} w={w:3.1f} int=[{lo:.1f},{hi:.1f}) | kad={kad:7.4f} "
            f"mind={mind:7.4f} cos={cos:.3f} retr={retr:.2f} grms={grms:.3f} "
            f"nfe/clip={nfe_clip:4.1f} | {time.time() - t0:.0f}s",
            flush=True,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"sweep_{tag}_{args.null}.json"
    out_path.write_text(json.dumps({"args": vars(args), "real_rms": real_rms, "rows": results}, indent=1))
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
