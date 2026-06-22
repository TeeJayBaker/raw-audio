"""Front-end follow-up #5 (final): Nyquist packing + 2D RoPE on the 512x8 winner.

Same n=64 / 6000-step / seed=1234 / batch=24 harness. Two design changes, each isolated on
patch_512x8, vs the existing baseline (prior seed-1234 value: 512x8 = 2.413 / own 0.721; best
512x8+bn16 = 2.321):

  512x8_anchor    padded 1025->1536 = 3 bands (3rd is a near-empty Nyquist token), 1D-raster RoPE, 15 tok
  512x8_packNyq   packed rfft: Nyquist folded into DC imag -> 1024 bins = 2 CLEAN bands, 1D RoPE, 10 tok
                  (lossless; tests whether dropping the dead Nyquist token costs quality, and the RTF win)
  512x8_rope2d    same tokens as anchor (15 tok) but axial 2D RoPE (freq idx on half the head dim,
                  time idx on the other) replacing the 1D-raster RoPE. Identical init to anchor, so
                  this isolates the positional scheme.

  python experiments/stft_frontends/followup5.py --gpu 1 --nclips 64 --steps 6000
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
    ("512x8_anchor", dict(frontend="square", patch_f=512, patch_t=8)),
    ("512x8_packNyq", dict(frontend="square", patch_f=512, patch_t=8, pack_nyquist=True)),
    ("512x8_rope2d", dict(frontend="square", patch_f=512, patch_t=8, rope2d=True)),
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
    print(f"\n=== follow-up #5 (Nyquist pack + 2D RoPE on 512x8) | n={nclip} | steps={args.steps} | seed={TRAIN_SEED} ===",
          flush=True)
    print("  prior seed-1234 refs: 512x8 = 2.413 / own 0.721 / retr 46 (15 tok) | 512x8+bn16 = 2.321 / 0.734 / 50", flush=True)

    results = []
    for label, kwargs in CONFIGS:
        results.append(run(label, kwargs, nclip, C, CONDS, conditioner, dev, args.steps, args.batch, train_seed=TRAIN_SEED))

    print("\n=== summary (sorted by sampled logSTFT_L1; tok = RTF proxy) ===", flush=True)
    for r in sorted(results, key=lambda r: r["logstft_l1"]):
        print(f"  {r['label']:>16} tok={r['ntok']:>3}  logSTFT_L1={r['logstft_l1']:.3f}  "
              f"own={r['own']:.3f}  retr={r['retr']:>2}/{nclip}  genRMS={r['genrms']:.3f}  tf.5={r['tf_corr_0.5']:+.2f}", flush=True)


if __name__ == "__main__":
    main()
