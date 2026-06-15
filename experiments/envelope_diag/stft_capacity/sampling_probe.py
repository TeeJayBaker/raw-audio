"""STFT-transformer SAMPLING-AXIS overfit probe (no src/ edits).

The capacity sweep showed a capacity-INDEPENDENT defect: free-running sampled
output collapses energy (genRMS ~0.25 vs target 0.30) at every backbone size, and a
large teacher-forced(0.96)->sampled(0.62) quality gap. That points at the flow/
training recipe, not the backbone. This probe fixes the backbone at d512_L12 / n=256
and sweeps the three training-side knobs that shape the velocity field's amplitude:

  rms_lift (WavFlow amplitude lift), loss_space (v vs x), t_distribution.

All sampling at CFG=1.0 (no guidance) so we isolate the training recipe, not CFG.
Quality metric peak-normalises BOTH signals before multi-res log-STFT L1, so lift and
no-lift runs are comparable on spectral SHAPE (loudness handled separately by genRMS).

Reuses experiments/stft_capacity/cache_n256.pt from the capacity sweep.

  python experiments/stft_capacity/sampling_probe.py --gpu 1 --steps 6000
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "src")
from backbone.transformer import Transformer  # noqa: E402
from emb.factory import build_embedding  # noqa: E402
from flow.fm import EPS, RectifiedFlow  # noqa: E402

SR = 48000
L = int(0.4 * SR)
MATPAC_CKPT = "/media/NAS/neutone/diff_one_shot/checkpoints/whole-violet-235/last-v1_clean.ckpt"
CACHE = "experiments/stft_capacity/cache_n256.pt"
STFT = {"n_fft": 2048, "hop_length": 512, "win_length": 2048}

# (label, rms_lift, loss_space, t_dist, logit_std)
CONFIGS = [
    ("base",        False, "v", "logit_normal", 1.0),  # reproduces capacity d512/n256
    ("lift",        True,  "v", "logit_normal", 1.0),  # WavFlow amplitude lift (#1 suspect)
    ("xspace",      False, "x", "logit_normal", 1.0),  # x-pred loss (no 1/(1-t)^2 v-weighting)
    ("lift_xspace", True,  "x", "logit_normal", 1.0),  # both
    ("t_uniform",   False, "v", "uniform",      1.0),  # uniform t instead of mid-biased
]
RMS_TARGET, LIFT_SCALE = 0.30, 3.0


def make_model(dev):
    torch.manual_seed(0)
    return Transformer(
        channels=1, out_channels=1, patching={"patch_size": 1},
        block={"dim": 512, "depth": 12, "heads": 8},
        conditioning={"cond_dim": 384, "time_scale": 100.0},
        head={"type": "identity"}, stft=STFT, sample_rate=SR, name="tf_stft",
    ).to(dev)


def _stft_logmag(x, n_fft, hop):
    w = torch.hann_window(n_fft, device=x.device)
    s = torch.stft(x.reshape(-1, x.shape[-1]), n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   window=w, center=True, return_complex=True)
    return torch.log(s.abs() + 1e-5)


def logstft_l1(a, b):
    """Peak-normalised (scale-invariant) multi-res log-mag STFT L1 between waveforms."""
    a = a / a.abs().max().clamp_min(1e-8)
    b = b / b.abs().max().clamp_min(1e-8)
    tot = 0.0
    for n_fft in (512, 1024, 2048):
        tot += float((_stft_logmag(a, n_fft, n_fft // 4) - _stft_logmag(b, n_fft, n_fft // 4)).abs().mean())
    return tot / 3.0


def liftfn(x, on):
    if not on:
        return x
    rms = x.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
    return LIFT_SCALE * torch.tanh((RMS_TARGET / rms) * x)


def sample_t(B, dist, std, dev):
    if dist == "uniform":
        return torch.rand(B, device=dev).clamp(EPS, 1 - EPS)
    return (torch.randn(B, device=dev) * std).sigmoid().clamp(EPS, 1 - EPS)


def run(label, lift, space, tdist, lstd, C, CONDS, conditioner, dev, steps, B):
    model = make_model(dev)
    flow = RectifiedFlow()
    Cn = torch.nn.functional.normalize(CONDS, dim=1)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.01)
    n = C.shape[0]
    ls = LIFT_SCALE if lift else 1.0
    t0 = time.time()
    model.train()
    for s in range(steps):
        idx = torch.randint(0, n, (B,), device=dev)
        x1 = liftfn(C[idx], lift)
        x_t, t, x1 = flow.train_tuple(x1, t=sample_t(B, tdist, lstd, dev))
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(x_t, t=t, cond=CONDS[idx], length=L)
            loss, _ = flow.loss(pred, x1, x_t, t, space=space, loss_type="mse")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    floss = float(loss.detach())

    model.eval()
    # teacher-forced corr (fitting, scale-invariant)
    tf = {}
    with torch.no_grad():
        for tv in (0.5, 0.8):
            cs = []
            for i in range(n):
                g = torch.Generator(device=dev).manual_seed(900 + i)
                x1i = liftfn(C[i:i + 1], lift)
                xt = (1 - tv) * torch.randn((1, 1, L), device=dev, generator=g) + tv * x1i
                xp = model(xt, t=torch.full((1,), tv, device=dev), cond=CONDS[i:i + 1], length=L)
                a, b = xp.flatten(), x1i.flatten()
                cs.append(float((a @ b) / (a.norm() * b.norm() + 1e-9)))
            tf[tv] = float(np.mean(cs))

    def embed(a1):
        with torch.no_grad():
            return conditioner(a1, sample_rate=SR, audio_lengths=torch.tensor([a1.shape[-1]], device=dev)).view(-1)

    own, l1s, grms, hits = [], [], [], 0
    traj = {2: [], 12: [], 24: []}
    grid = torch.linspace(EPS, 1 - EPS, 26, device=dev)
    with torch.no_grad():
        for i in range(n):
            g = torch.Generator(device=dev).manual_seed(50 + i)
            x = torch.randn((1, 1, L), device=dev, generator=g)
            for st in range(25):
                tt = grid[st].expand(1)
                x = x + (grid[st + 1] - grid[st]) * flow.target_to_v(model(x, t=tt, cond=CONDS[i:i + 1], length=L), x, tt)
                if st in traj:
                    traj[st].append(float((x / ls).pow(2).mean().sqrt()))
            o = (x / ls).clamp(-1, 1)
            l1s.append(logstft_l1(o, C[i:i + 1]))
            eg = torch.nn.functional.normalize(embed(o), dim=0)
            cos = Cn @ eg
            own.append(float(cos[i]))
            hits += int(cos.argmax() == i)
            grms.append(float((o ** 2).mean().sqrt()))
    tgt = RMS_TARGET if lift else float((C ** 2).mean(dim=(-2, -1)).sqrt().mean())
    dt = time.time() - t0
    print(f"  [{label:>11}] logSTFT_L1={np.mean(l1s):.3f}  tf_corr(.5/.8)={tf[0.5]:+.2f}/{tf[0.8]:+.2f}  "
          f"CLAP_own={np.mean(own):.3f}  retr={hits}/{n}  genRMS={np.mean(grms):.3f}(tgt{tgt:.2f})  "
          f"traj_rms[t.12/.52/1.0]={np.mean(traj[2]):.2f}/{np.mean(traj[12]):.2f}/{np.mean(traj[24]):.2f}  "
          f"loss={floss:.3f}  {dt:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--configs", type=str, default="all")
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    conditioner = build_embedding(
        {"type": "matpac", "checkpoint_path": MATPAC_CKPT, "device": str(dev),
         "use_teacher": False, "encode_batch_size": 0, "compile_encoder": False}, device=dev).to(dev).eval()
    d = torch.load(CACHE, map_location=dev)
    C, CONDS = d["C"].to(dev), d["CONDS"].to(dev)
    cfgs = CONFIGS if args.configs == "all" else [c for c in CONFIGS if c[0] in args.configs.split(",")]
    print(f"=== d512_L12 | n={C.shape[0]} | steps={args.steps} batch={args.batch} | CFG=1.0 ===", flush=True)
    for label, lift, space, tdist, lstd in cfgs:
        run(label, lift, space, tdist, lstd, C, CONDS, conditioner, dev, args.steps, args.batch)


if __name__ == "__main__":
    main()
