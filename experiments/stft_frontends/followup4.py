"""Front-end follow-up #4: fill the patch x bottleneck matrix across bottleneck sizes.

Same n=64 / 6000-step / seed=1234 / batch=24 harness as followup{2,3} so results merge
directly with the existing matrix. Only the GOOD patches are kept (512x8, 256x8, 512x4 --
the full-freq and 17-band patches were discarded as worse-than-column). Sweeps the JiT
input bottleneck on each patch projection across n in {4,8,16,32,64,128}.

NOTE ON BANDS: patch_f=512 pads freq 1025->1536 = 3 patches, but band 3 is just the lone
Nyquist bin 1024 + zeros, so it is effectively 2 real bands + remainder; patch_f=256 -> 4
real bands + remainder. (Token/RTF count unchanged by the bottleneck -- it only factorises
the in_proj, so each patch row keeps its tok: 512x8=15, 256x8=25, 512x4=30.)

Already have (seed 1234, NOT rerun here): 512x4_bn4, 512x4_bn8, 256x4_bn8, and the no-bn
patch baselines (512x8 2.413/own0.721, 256x8 2.372/0.699, 512x4 2.442/0.700). One no-bn
512x8 anchor IS rerun as a cross-GPU sanity check (GPU1 vs the GPU0 baselines).

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python experiments/stft_frontends/followup4.py --gpu 1 --nclips 64 --steps 6000
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

CONFIGS = [("p512x8_anchor", dict(frontend="square", patch_f=512, patch_t=8))]  # cross-GPU sanity
for bn in [4, 8, 16, 32, 64, 128]:
    CONFIGS.append((f"p512x8_bn{bn}", dict(frontend="square", patch_f=512, patch_t=8, n_bottleneck=bn)))
for bn in [4, 8, 16, 32, 64, 128]:
    CONFIGS.append((f"p256x8_bn{bn}", dict(frontend="square", patch_f=256, patch_t=8, n_bottleneck=bn)))
for bn in [16, 32, 64, 128]:  # 512x4 bn4/bn8 already done in followup3
    CONFIGS.append((f"p512x4_bn{bn}", dict(frontend="square", patch_f=512, patch_t=4, n_bottleneck=bn)))


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
    print(f"\n=== follow-up #4 (patch×bottleneck matrix) | n={nclip} | steps={args.steps} | batch={args.batch} | seed={TRAIN_SEED} ===",
          flush=True)
    print("  prior baselines (seed1234): 512x8 2.413/own0.721 | 256x8 2.372/0.699 | 512x4 2.442/0.700"
          " | 512x4_bn4 2.519 | 512x4_bn8 2.472 | 256x4_bn8 2.501", flush=True)

    results = []
    for label, kwargs in CONFIGS:
        results.append(run(label, kwargs, nclip, C, CONDS, conditioner, dev, args.steps, args.batch, train_seed=TRAIN_SEED))

    print("\n=== summary (sorted by sampled logSTFT_L1, lower=better; tok = RTF proxy) ===", flush=True)
    for r in sorted(results, key=lambda r: r["logstft_l1"]):
        print(f"  {r['label']:>16} tok={r['ntok']:>3}  logSTFT_L1={r['logstft_l1']:.3f}  "
              f"own={r['own']:.3f}  retr={r['retr']:>2}/{nclip}  genRMS={r['genrms']:.3f}  tf.5={r['tf_corr_0.5']:+.2f}", flush=True)

    print("\n=== per-patch bottleneck curve (logSTFT_L1 / own) ===", flush=True)
    for pf, pt, tok, base in [("512", "8", 15, "2.413/0.721"), ("256", "8", 25, "2.372/0.699"), ("512", "4", 30, "2.442/0.700")]:
        row = [f"none={base}"]
        for bn in [4, 8, 16, 32, 64, 128]:
            r = next((x for x in results if x["label"] == f"p{pf}x{pt}_bn{bn}"), None)
            if r is not None:
                row.append(f"bn{bn}={r['logstft_l1']:.3f}/{r['own']:.3f}")
        print(f"  patch_{pf}x{pt} (tok={tok}): " + "  ".join(row), flush=True)


if __name__ == "__main__":
    main()
