"""Front-end follow-up #6: the third Nyquist option — DROP it (lossy).

Completes the Nyquist-handling comparison on patch_512x8 (same n=64 / 6000-step / seed=1234
/ batch=24 harness, so it merges with followup5's anchor + packNyq):

  512x8_dropNyq   drop the Nyquist bin entirely -> 1024 bins = 2 clean bands, Nyquist reconstructed
                  as zero (lossy). 10 tok. The simplest handling; pack vs drop isolates whether
                  KEEPING the 24 kHz bin (via DC-imag fold) actually matters.

Compare against followup5: 512x8_anchor (pad, 15 tok) and 512x8_packNyq (lossless, 10 tok).

  python experiments/stft_frontends/followup6.py --gpu 1 --nclips 64 --steps 6000
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

CONFIGS = [
    ("512x8_dropNyq", dict(frontend="square", patch_f=512, patch_t=8, drop_nyquist=True)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
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
    print(f"\n=== follow-up #6 (drop Nyquist on 512x8) | n={nclip} | steps={args.steps} | seed={TRAIN_SEED} ===", flush=True)
    print("  compare vs fu5: anchor(pad,15tok)=2.413/.721 | packNyq(lossless,10tok)=<fu5> | +bn16 ref=2.321/.734", flush=True)
    for label, kwargs in CONFIGS:
        run(label, kwargs, nclip, C, CONDS, conditioner, dev, args.steps, args.batch, train_seed=TRAIN_SEED)


if __name__ == "__main__":
    main()
