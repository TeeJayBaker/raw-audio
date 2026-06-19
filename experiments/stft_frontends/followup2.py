"""Front-end follow-up #2: bottleneck floor + length-robust patch shapes for quality/RTF.

Builds on experiments/stft_frontends/{probe,followup}.py (reused untouched), same n=64 /
6000-step / seed=1234 harness so the front-end is the only variable.

  1. BOTTLENECK FLOOR: the linear input bottleneck was monotonic (tighter=better) down to
     n=16. Sweep n in {2,4,8,12,16} to find where it stops helping / breaks.

  2. PATCH SHAPES (quality vs RTF): token count is the RTF proxy (trunk attention is O(tok^2)).
     ALL patches use a FIXED patch_t (frames), so token count scales with clip length but the
     projection stays fixed -> length-robust. (The earlier `patch_strip` used patch_t=fullT,
     which is length-specific and invalid for variable-length inference; its length-robust
     analog is full-frequency x fixed-time-patch = the production `patch_size` knob.)
       patch_512x2     512f x 2t -> 3 x19 = 57 tok   quality push (finer time)
       patch_ft_256x4  256f x 4t -> 5 x10 = 50 tok   current best (anchor)
       patch_512x4     512f x 4t -> 3 x10 = 30 tok   coarser freq, keep fine time (~1/2 tokens)
       patch_full_t4  1025f x 4t -> 1 x10 = 10 tok   all-freq, column time-patched x4
       patch_full_t8  1025f x 8t -> 1 x 5 =  5 tok   all-freq, time-patched x8 (cheapest; strip analog)

  column (38 tok, full-rank 2050->512) is the control.

  python experiments/stft_frontends/followup2.py --gpu 0 --nclips 64 --steps 6000
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
    ("bneck2", dict(frontend="bottleneck", n_bottleneck=2)),
    ("bneck4", dict(frontend="bottleneck", n_bottleneck=4)),
    ("bneck8", dict(frontend="bottleneck", n_bottleneck=8)),
    ("bneck12", dict(frontend="bottleneck", n_bottleneck=12)),
    ("bneck16", dict(frontend="bottleneck", n_bottleneck=16)),
    ("patch_512x2", dict(frontend="square", patch_f=512, patch_t=2)),
    ("patch_ft_256x4", dict(frontend="square", patch_f=256, patch_t=4)),
    ("patch_512x4", dict(frontend="square", patch_f=512, patch_t=4)),
    ("patch_full_t4", dict(frontend="square", patch_f=1025, patch_t=4)),
    ("patch_full_t8", dict(frontend="square", patch_f=1025, patch_t=8)),
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
    print(f"\n=== follow-up #2 | n={nclip} | cond cos={offm:.3f} | steps={args.steps} | seed={TRAIN_SEED} ===",
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

    print("\n=== bottleneck floor (logSTFT_L1 vs n) ===", flush=True)
    for lab in ["column", "bneck16", "bneck12", "bneck8", "bneck4", "bneck2"]:
        r = next(x for x in results if x["label"] == lab)
        print(f"  {lab:>9}: logSTFT_L1={r['logstft_l1']:.3f}  own={r['own']:.3f}  retr={r['retr']}/{nclip}  genRMS={r['genrms']:.3f}", flush=True)

    print("\n=== patch quality/RTF frontier (tok ↑ = slower) ===", flush=True)
    patches = [r for r in results if r["label"].startswith("patch")]
    for r in sorted(patches, key=lambda r: r["ntok"]):
        print(f"  {r['label']:>16} tok={r['ntok']:>3}  logSTFT_L1={r['logstft_l1']:.3f}  own={r['own']:.3f}  retr={r['retr']}/{nclip}", flush=True)


if __name__ == "__main__":
    main()
