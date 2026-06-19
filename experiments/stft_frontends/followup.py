"""Front-end follow-up: bottleneck optimum + a spread of spec-patching geometries.

Builds on experiments/stft_frontends/probe.py (reused untouched). Two questions:

  1. BOTTLENECK: n=64 was the best linear input bottleneck (2050->n->512). Is lower better?
     Sweep n in {16,32,48,64,128} to locate the optimum (below/at/above 64).

  2. SPEC PATCHING: a selection of 2D square-tile geometries spanning the freq<->time
     locality tradeoff, all within 1.5x the 38-token column baseline (<=57 tokens):
       patch_ff   64f x16t -> 17x3 = 51 tok   fine-freq / coarse-time
       patch_bal 128f x 8t ->  9x5 = 45 tok   balanced (= prior `square`)
       patch_ft  256f x 4t ->  5x10= 50 tok   coarse-freq / fine-time
       patch_strip 128f x fullT -> 9x1 = 9 tok  pure frequency strips, whole clip

  column (38 tok) is the control. band2 (76 tok) is included ONLY as an out-of-budget
  reference: its win is partly just 2x the tokens, so it is NOT the fair benchmark --
  patching options are judged against column and each other at comparable token budget.

Per-config identical batch/noise stream (train_seed) so the front-end is the only variable.

  python experiments/stft_frontends/followup.py --gpu 0 --nclips 64 --steps 6000
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
from stft_frontends.probe import MATPAC_CKPT, get_data, run  # noqa: E402

from emb.factory import build_embedding  # noqa: E402

TRAIN_SEED = 1234

# (label, kwargs, in_budget?)  -- in_budget = within 1.5x the 38-token baseline
CONFIGS = [
    ("column", dict(frontend="column"), True),                              # control / "the normal one"
    ("bneck16", dict(frontend="bottleneck", n_bottleneck=16), True),
    ("bneck32", dict(frontend="bottleneck", n_bottleneck=32), True),
    ("bneck48", dict(frontend="bottleneck", n_bottleneck=48), True),
    ("bneck64", dict(frontend="bottleneck", n_bottleneck=64), True),
    ("bneck128", dict(frontend="bottleneck", n_bottleneck=128), True),
    ("patch_ff_64x16", dict(frontend="square", patch_f=64, patch_t=16), True),
    ("patch_bal_128x8", dict(frontend="square", patch_f=128, patch_t=8), True),
    ("patch_ft_256x4", dict(frontend="square", patch_f=256, patch_t=4), True),
    ("patch_strip_128xT", dict(frontend="square", patch_f=128, patch_t=40), True),
    ("band2_ref", dict(frontend="band"), False),                            # out-of-budget reference (76 tok)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--nclips", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(0); np.random.seed(0)
    conditioner = build_embedding(
        {"type": "matpac", "checkpoint_path": MATPAC_CKPT, "device": str(dev),
         "use_teacher": False, "encode_batch_size": 0, "compile_encoder": False}, device=dev).to(dev).eval()
    nclip = args.nclips
    C, CONDS = get_data(nclip, conditioner, dev)
    off = torch.nn.functional.normalize(CONDS, dim=1)
    offm = float((off @ off.T)[~torch.eye(nclip, dtype=bool, device=dev)].mean())
    print(f"\n=== front-end follow-up | n={nclip} | cond cos={offm:.3f} | steps={args.steps} | seed={TRAIN_SEED} ===",
          flush=True)

    results = []
    budget = {}
    for label, kwargs, in_budget in CONFIGS:
        r = run(label, kwargs, nclip, C, CONDS, conditioner, dev, args.steps, args.batch, train_seed=TRAIN_SEED)
        budget[r["label"]] = in_budget
        results.append(r)

    col = next(r for r in results if r["label"] == "column")
    print("\n=== summary (sorted by sampled logSTFT_L1, lower=better; Δ vs column control) ===", flush=True)
    print(f"  {'(budget = <=57 tok, 1.5x the 38-tok column baseline)':<58}", flush=True)
    for r in sorted(results, key=lambda r: r["logstft_l1"]):
        flag = "" if budget[r["label"]] else "  <-- OUT-OF-BUDGET (token count inflates this)"
        d = r["logstft_l1"] - col["logstft_l1"]
        print(f"  {r['label']:>18} tok={r['ntok']:>3}  logSTFT_L1={r['logstft_l1']:.3f}({d:+.3f})  "
              f"own={r['own']:.3f}  retr={r['retr']:>2}/{nclip}  genRMS={r['genrms']:.3f}  "
              f"tf.5={r['tf_corr_0.5']:+.2f}{flag}", flush=True)

    print("\n=== bottleneck curve (logSTFT_L1 vs n; column = full-rank 2050->512) ===", flush=True)
    order = ["column", "bneck128", "bneck64", "bneck48", "bneck32", "bneck16"]
    for lab in order:
        r = next(x for x in results if x["label"] == lab)
        print(f"  {lab:>9}: logSTFT_L1={r['logstft_l1']:.3f}  own={r['own']:.3f}  retr={r['retr']}/{nclip}", flush=True)


if __name__ == "__main__":
    main()
