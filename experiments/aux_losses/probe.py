"""Auxiliary-loss overfit ablation (no src/ edits).

Question: do the three auxiliary losses the trainer can add on top of the primary
v-MSE actually improve the SAMPLED audio in the overfit regime, and which helps most?

  mr_stft   multi-resolution STFT (spectral-convergence + magnitude L1)  -- production weight 1.0
  wavefm    WaveFM bundle: anti-wrap phase + log-mag + spectral-grad/Laplacian + 0.02*mel-L1
  complex   energy-weighted complex L1 on the backbone's pre-iSTFT spectrogram

Reuses the real loss functions (losses.audio) and the column ProbeModel (the production
stft_transformer front-end) from experiments/stft_frontends/probe.py, untouched.

Design:
  - base model = column front-end, n=64 clips, 6000 steps (same harness as the front-end probe).
  - 4 configs: baseline (v-MSE only), +mr_stft, +wavefm, +complex -- each aux ADDED alone.
  - weights: mr_stft anchored at its production 1.0; wavefm/complex CALIBRATED so their loss
    contribution matches mr_stft@1.0 (magnitudes measured mid-training on the baseline run).
  - identical batch/noise sequence per config (reseeded) so the loss is the only variable.

CIRCULARITY NOTE: mr_stft / complex / wavefm all optimise spectral-L1-ish quantities, so the
sampled logSTFT-L1 metric is HOME-FIELD for them. The fair arbiters here are MATPAC own-cosine
+ retrieval (semantic, not trained on) and genRMS (energy). tf_corr confirms fitting is saturated.

  python experiments/aux_losses/probe.py --gpu 0 --nclips 64 --steps 6000
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
from stft_frontends.probe import (  # noqa: E402  (reuse harness, src/ untouched)
    L,
    SR,
    ProbeModel,
    get_data,
    logstft_l1,
)

from losses.audio import complex_stft_loss, mr_stft_loss, wavefm_loss  # noqa: E402
from flow.fm import EPS, RectifiedFlow  # noqa: E402

CALIB_LO, CALIB_HI = 500, 540  # steps over which to measure aux magnitudes on the baseline run
TRAIN_SEED = 1234


def _aux_value(kind, pred, pred_spec, x1, model):
    if kind == "mr":
        return mr_stft_loss(pred, x1)
    if kind == "wavefm":
        return wavefm_loss(pred, x1, sample_rate=model.sample_rate)[0]
    if kind == "complex":
        return complex_stft_loss(pred_spec, x1, model.stft)
    raise ValueError(kind)


def train_eval(label, aux_kind, weight, nclip, C, CONDS, conditioner, dev, steps, B, calibrate=False):
    model = ProbeModel(frontend="column").to(dev)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    flow = RectifiedFlow()
    Cn = torch.nn.functional.normalize(CONDS, dim=1)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.01)
    n = C.shape[0]
    torch.manual_seed(TRAIN_SEED)  # identical batch/noise stream across configs
    model.train()
    final_loss = float("nan")
    calib = {"mr": [], "wavefm": [], "complex": []}
    for s in range(steps):
        idx = torch.randint(0, n, (B,), device=dev)
        x1 = C[idx]
        t = torch.randn(B, device=dev).sigmoid().clamp(EPS, 1 - EPS)
        x_t, t, x1 = flow.train_tuple(x1, t=t)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred, pred_spec = model(x_t, t=t, cond=CONDS[idx], length=L, return_spec=True)
            loss, _ = flow.loss(pred, x1, x_t, t, space="v", loss_type="mse")
            if aux_kind is not None:
                loss = loss + weight * _aux_value(aux_kind, pred, pred_spec, x1, model)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        final_loss = float(loss.detach())
        if calibrate and CALIB_LO <= s < CALIB_HI:
            with torch.no_grad():
                for k in calib:
                    calib[k].append(float(_aux_value(k, pred.detach(), pred_spec.detach(), x1, model)))

    model.eval()
    tf_corr = {}
    with torch.no_grad():
        for tval in (0.3, 0.5, 0.8):
            corrs = []
            for i in range(n):
                g = torch.Generator(device=dev).manual_seed(900 + i)
                x0 = torch.randn((1, 1, L), device=dev, generator=g)
                xt = (1 - tval) * x0 + tval * C[i:i + 1]
                xp = model(xt, t=torch.full((1,), tval, device=dev), cond=CONDS[i:i + 1], length=L)
                a, b = xp.flatten(), C[i:i + 1].flatten()
                corrs.append(float((a @ b) / (a.norm() * b.norm() + 1e-9)))
            tf_corr[tval] = float(np.mean(corrs))

    def embed(a1):
        with torch.no_grad():
            return conditioner(a1, sample_rate=SR, audio_lengths=torch.tensor([a1.shape[-1]], device=dev)).view(-1)

    own, l1s, grms, hits = [], [], [], 0
    with torch.no_grad():
        for i in range(n):
            g = torch.Generator(device=dev).manual_seed(50 + i)
            noise = torch.randn((1, 1, L), device=dev, generator=g)
            o = flow.sample(model, (1, 1, L), cond=CONDS[i:i + 1], noise=noise, steps=25,
                            guidance_scale=1.0, lift_scale=1.0).clamp(-1, 1)
            l1s.append(logstft_l1(o, C[i:i + 1]))
            eg = torch.nn.functional.normalize(embed(o), dim=0)
            cos = Cn @ eg
            own.append(float(cos[i]))
            if int(cos.argmax()) == i:
                hits += 1
            grms.append(float((o ** 2).mean().sqrt()))
    tgt = float((C ** 2).mean(dim=(-2, -1)).sqrt().mean())
    wtxt = "    -" if aux_kind is None else f"{weight:.3f}"
    print(f"  [{label:>9} w={wtxt:>7}] "
          f"own={np.mean(own):.3f}  retr={hits}/{n}  genRMS={np.mean(grms):.3f}(tgt{tgt:.2f})  "
          f"tf_corr(.3/.5/.8)={tf_corr[0.3]:+.2f}/{tf_corr[0.5]:+.2f}/{tf_corr[0.8]:+.2f}  "
          f"[logSTFT_L1={np.mean(l1s):.3f}*]  loss={final_loss:.3f}", flush=True)
    res = {"label": label, "params": nparams, "weight": weight, "own": float(np.mean(own)),
           "retr": hits, "genrms": float(np.mean(grms)), "tf_corr_0.5": tf_corr[0.5],
           "logstft_l1": float(np.mean(l1s))}
    calib_means = {k: float(np.mean(v)) for k, v in calib.items()} if calibrate else None
    return res, calib_means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--nclips", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(0); np.random.seed(0)
    from emb.factory import build_embedding
    conditioner = build_embedding(
        {"type": "matpac", "checkpoint_path":
         "/media/NAS/neutone/diff_one_shot/checkpoints/whole-violet-235/last-v1_clean.ckpt",
         "device": str(dev), "use_teacher": False, "encode_batch_size": 0, "compile_encoder": False},
        device=dev).to(dev).eval()
    nclip = args.nclips
    C, CONDS = get_data(nclip, conditioner, dev)
    off = torch.nn.functional.normalize(CONDS, dim=1)
    offm = float((off @ off.T)[~torch.eye(nclip, dtype=bool, device=dev)].mean())
    print(f"\n=== aux-loss ablation | n={nclip} | cond pairwise cos={offm:.3f} | steps={args.steps} batch={args.batch} ===",
          flush=True)
    print("  (logSTFT_L1 marked * = trained-on for spectral losses; rank on own-cosine/retrieval/genRMS)", flush=True)

    results = []
    # baseline first -> measures aux magnitudes for calibration
    base, calib = train_eval("baseline", None, 0.0, nclip, C, CONDS, conditioner, dev,
                             args.steps, args.batch, calibrate=True)
    results.append(base)
    w_wavefm = calib["mr"] / max(calib["wavefm"], 1e-9)
    w_complex = calib["mr"] / max(calib["complex"], 1e-9)
    print(f"  calib (mid-train aux magnitudes): mr={calib['mr']:.3f} wavefm={calib['wavefm']:.3f} "
          f"complex={calib['complex']:.3f}  ->  w_wavefm={w_wavefm:.3f} w_complex={w_complex:.3f}", flush=True)

    for kind, w in [("mr", 1.0), ("wavefm", w_wavefm), ("complex", w_complex)]:
        r, _ = train_eval(f"+{kind}", kind, w, nclip, C, CONDS, conditioner, dev, args.steps, args.batch)
        results.append(r)

    print("\n=== summary (sorted by MATPAC own-cosine, higher=better; logSTFT_L1* trained-on) ===", flush=True)
    base_own = base["own"]
    for r in sorted(results, key=lambda r: -r["own"]):
        d = r["own"] - base_own
        print(f"  {r['label']:>9} w={r['weight']:.3f}  own={r['own']:.3f}({d:+.3f})  retr={r['retr']:>2}/{nclip}  "
              f"genRMS={r['genrms']:.3f}  tf_corr.5={r['tf_corr_0.5']:+.2f}  logSTFT_L1*={r['logstft_l1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
