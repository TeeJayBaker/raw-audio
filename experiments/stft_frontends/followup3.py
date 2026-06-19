"""Front-end follow-up #3 (final): cheap freq-banded patches + patch x bottleneck combo.

Same n=64 / 6000-step / seed=1234 harness (front-end is the only variable). Settled facts
going in: freq locality is essential (patch_512x4 3-band beats column at 30 tok); tp=4 is the
time sweet spot; bottleneck sweet spot n~4-8; both are sampling-side levers.

  1. CHEAP FREQ-BANDED FRONTIER (keep >=3 freq bands, cut time tokens):
       patch_512x8   3 freq x 5 time  = 15 tok   how cheap can 3 bands go (tp 4->8)?
       patch_256x8   5 freq x 5 time  = 25 tok   5 bands, coarse time
     re-anchored against patch_512x4 (30) and patch_256x4 (50) in the same run.

  2. PATCH x BOTTLENECK COMBO: add a JiT linear bottleneck to the best patch's projection.
       patch_512x4 + bn8 / bn4, patch_256x4 + bn8.
     Question: do the spectral gain (patching) and semantic gain (bottleneck) STACK?
     bneck8 (38 tok, bottleneck-only) included as the reference for the semantic axis.

  python experiments/stft_frontends/followup3.py --gpu 0 --nclips 64 --steps 6000
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
    ("column", dict(frontend="column")),
    ("bneck8", dict(frontend="bottleneck", n_bottleneck=8)),                      # bottleneck-only ref
    ("patch_512x4", dict(frontend="square", patch_f=512, patch_t=4)),             # best patch-only anchor
    ("patch_256x4", dict(frontend="square", patch_f=256, patch_t=4)),             # best spectral anchor
    ("patch_512x8", dict(frontend="square", patch_f=512, patch_t=8)),             # cheap: 3 bands, tp8
    ("patch_256x8", dict(frontend="square", patch_f=256, patch_t=8)),             # 5 bands, tp8
    ("patch_512x4_bn8", dict(frontend="square", patch_f=512, patch_t=4, n_bottleneck=8)),   # combo
    ("patch_512x4_bn4", dict(frontend="square", patch_f=512, patch_t=4, n_bottleneck=4)),   # combo, tighter
    ("patch_256x4_bn8", dict(frontend="square", patch_f=256, patch_t=4, n_bottleneck=8)),   # combo
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
    print(f"\n=== follow-up #3 | n={nclip} | cond cos={offm:.3f} | steps={args.steps} | seed={TRAIN_SEED} ===",
          flush=True)

    results = []
    for label, kwargs in CONFIGS:
        results.append(run(label, kwargs, nclip, C, CONDS, conditioner, dev, args.steps, args.batch, train_seed=TRAIN_SEED))

    col = next(r for r in results if r["label"] == "column")
    print("\n=== summary (sorted by sampled logSTFT_L1, lower=better; Δ vs column; tok = RTF proxy) ===", flush=True)
    for r in sorted(results, key=lambda r: r["logstft_l1"]):
        d = r["logstft_l1"] - col["logstft_l1"]
        print(f"  {r['label']:>16} tok={r['ntok']:>3}  logSTFT_L1={r['logstft_l1']:.3f}({d:+.3f})  "
              f"own={r['own']:.3f}  retr={r['retr']:>2}/{nclip}  genRMS={r['genrms']:.3f}  "
              f"tf.5={r['tf_corr_0.5']:+.2f}", flush=True)

    print("\n=== cheap freq-banded frontier (tok ↑ = slower) ===", flush=True)
    for lab in ["patch_512x8", "patch_256x8", "patch_512x4", "patch_256x4"]:
        r = next(x for x in results if x["label"] == lab)
        print(f"  {lab:>16} tok={r['ntok']:>3}  logSTFT_L1={r['logstft_l1']:.3f}  own={r['own']:.3f}  retr={r['retr']}/{nclip}", flush=True)

    print("\n=== patch x bottleneck combo (does it stack?) ===", flush=True)
    for lab in ["column", "bneck8", "patch_512x4", "patch_512x4_bn8", "patch_512x4_bn4", "patch_256x4", "patch_256x4_bn8"]:
        r = next(x for x in results if x["label"] == lab)
        print(f"  {lab:>16} tok={r['ntok']:>3}  logSTFT_L1={r['logstft_l1']:.3f}  own={r['own']:.3f}  retr={r['retr']}/{nclip}  genRMS={r['genrms']:.3f}", flush=True)


if __name__ == "__main__":
    main()
