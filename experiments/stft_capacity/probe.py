"""STFT-transformer CAPACITY overfit probe (no src/ edits).

Question: is backbone size/capacity the limiter on quality for the STFT-domain
rectified-flow transformer, or is the ceiling elsewhere (sampling / velocity field)?

Method: overfit NCLIP clips with MATPAC conditioning at several backbone sizes and
clip counts, and separate two things:
  - FITTING capacity     : teacher-forced x1-pred corr to the true clip (given true x_t)
  - END-TO-END quality   : free-running sampled output vs the true clip, measured by
                           paired multi-resolution log-STFT L1 (lower=better), plus
                           CLAP own-cosine / retrieval / genRMS.

If bigger backbone monotonically lowers the SAMPLED log-STFT L1 -> capacity-limited
(supports scaling up). If teacher-forced is already good at every size but sampled
stays poor regardless of size -> it's a flow/sampling ceiling, not capacity.

Reuses the real backbone (backbone.transformer.Transformer) untouched.

  python experiments/stft_capacity/probe.py --gpu 1 --nclips 64,256 --steps 6000
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
import soxr
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "src")
from backbone.transformer import Transformer  # noqa: E402
from emb.factory import build_embedding  # noqa: E402
from flow.fm import EPS, RectifiedFlow  # noqa: E402

SR = 48000
L = int(0.4 * SR)
ROOT = "/media/storage/samples/samples_from_mars/one_shots"
MATPAC_CKPT = "/media/NAS/neutone/diff_one_shot/checkpoints/whole-violet-235/last-v1_clean.ckpt"
CACHE_DIR = "experiments/stft_capacity"
STFT = {"n_fft": 2048, "hop_length": 512, "win_length": 2048}

# (label, dim, depth, heads). real run = d512/L12/h8.
SIZES = [
    ("d256_L6", 256, 6, 4),
    ("d384_L8", 384, 8, 6),
    ("d512_L12", 512, 12, 8),
    ("d768_L12", 768, 12, 12),
]


def make_model(dim, depth, heads, dev):
    torch.manual_seed(0)
    return Transformer(
        channels=1, out_channels=1, patching={"patch_size": 1},
        block={"dim": dim, "depth": depth, "heads": heads},
        conditioning={"cond_dim": 384, "time_scale": 100.0},
        head={"type": "identity"}, stft=STFT, sample_rate=SR, name="tf_stft",
    ).to(dev)


def load_clips(nclip, dev):
    cands = []
    for dp, _, fns in os.walk(ROOT):
        for fn in fns:
            if fn.lower().endswith((".wav", ".flac", ".aif", ".aiff")):
                cands.append(os.path.join(dp, fn))
    random.seed(3)  # same seed as wavenext_cond/probe.py -> superset of those 8 clips
    random.shuffle(cands)
    clips = []
    for f in cands:
        if len(clips) >= nclip:
            break
        try:
            a, sr = sf.read(f, dtype="float32", always_2d=True)
        except Exception:
            continue
        a = a.mean(1)
        if sr != SR:
            a = soxr.resample(a, sr, SR, quality="HQ").astype(np.float32)
        if len(a) < 0.25 * SR:
            continue
        a = a[:L] if len(a) >= L else np.pad(a, (0, L - len(a)))
        pk = max(abs(a).max(), 1e-8)
        clips.append((a / pk).astype(np.float32))
    return torch.from_numpy(np.stack(clips)).unsqueeze(1).to(dev)


def get_data(nclip, conditioner, dev):
    cache = os.path.join(CACHE_DIR, f"cache_n{nclip}.pt")
    if os.path.exists(cache):
        d = torch.load(cache, map_location=dev)
        return d["C"].to(dev), d["CONDS"].to(dev)
    C = load_clips(nclip, dev)

    def embed(a1):
        with torch.no_grad():
            return conditioner(a1, sample_rate=SR, audio_lengths=torch.tensor([a1.shape[-1]], device=dev)).view(-1)

    CONDS = torch.stack([embed(C[i:i + 1]) for i in range(C.shape[0])], 0)
    torch.save({"C": C.cpu(), "CONDS": CONDS.cpu()}, cache)
    return C, CONDS


def _stft_logmag(x, n_fft, hop):
    w = torch.hann_window(n_fft, device=x.device)
    s = torch.stft(x.reshape(-1, x.shape[-1]), n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   window=w, center=True, return_complex=True)
    return torch.log(s.abs() + 1e-5)


def logstft_l1(a, b):
    """Multi-resolution paired log-magnitude STFT L1 between waveforms a,b [.,L]."""
    tot = 0.0
    for n_fft in (512, 1024, 2048):
        tot += float((_stft_logmag(a, n_fft, n_fft // 4) - _stft_logmag(b, n_fft, n_fft // 4)).abs().mean())
    return tot / 3.0


def run(label, dim, depth, heads, nclip, C, CONDS, conditioner, dev, steps, B):
    model = make_model(dim, depth, heads, dev)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    flow = RectifiedFlow()
    Cn = torch.nn.functional.normalize(CONDS, dim=1)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.01)
    n = C.shape[0]
    t0 = time.time()
    model.train()
    losshist = []
    for s in range(steps):
        idx = torch.randint(0, n, (B,), device=dev)
        x1 = C[idx]
        t = torch.randn(B, device=dev).sigmoid().clamp(EPS, 1 - EPS)
        x_t, t, x1 = flow.train_tuple(x1, t=t)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(x_t, t=t, cond=CONDS[idx], length=L)
            loss, _ = flow.loss(pred, x1, x_t, t, space="v", loss_type="mse")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % (steps // 5) == 0:
            losshist.append((s, float(loss.detach())))

    model.eval()
    # FITTING: teacher-forced x1-pred corr (given the TRUE x_t)
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

    # END-TO-END: sampled output vs true clip
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
    dt = time.time() - t0
    print(f"  [{label:>9} {nparams:5.1f}M | n={nclip:>3}] "
          f"logSTFT_L1={np.mean(l1s):.3f}  tf_corr(.3/.5/.8)={tf_corr[0.3]:+.2f}/{tf_corr[0.5]:+.2f}/{tf_corr[0.8]:+.2f}  "
          f"CLAP_own={np.mean(own):.3f}  retr={hits}/{n}  genRMS={np.mean(grms):.3f}(tgt{tgt:.2f})  "
          f"finalloss={losshist[-1][1]:.3f}  {dt:.0f}s", flush=True)
    return {"label": label, "params": nparams, "nclip": nclip, "logstft_l1": float(np.mean(l1s)),
            "tf_corr_0.5": tf_corr[0.5], "clap_own": float(np.mean(own)), "retr": hits, "genrms": float(np.mean(grms))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--nclips", type=str, default="64,256")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--sizes", type=str, default="all", help="comma list of size labels or 'all'")
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    os.makedirs(CACHE_DIR, exist_ok=True)
    sizes = SIZES if args.sizes == "all" else [s for s in SIZES if s[0] in args.sizes.split(",")]
    conditioner = build_embedding(
        {"type": "matpac", "checkpoint_path": MATPAC_CKPT, "device": str(dev),
         "use_teacher": False, "encode_batch_size": 0, "compile_encoder": False}, device=dev).to(dev).eval()
    for nclip in [int(x) for x in args.nclips.split(",")]:
        C, CONDS = get_data(nclip, conditioner, dev)
        off = torch.nn.functional.normalize(CONDS, dim=1)
        offm = float((off @ off.T)[~torch.eye(nclip, dtype=bool, device=dev)].mean())
        print(f"\n=== n={nclip} clips | cond pairwise cos mean={offm:.3f} | steps={args.steps} batch={args.batch} ===", flush=True)
        for label, dim, depth, heads in sizes:
            run(label, dim, depth, heads, nclip, C, CONDS, conditioner, dev, args.steps, args.batch)


if __name__ == "__main__":
    main()
