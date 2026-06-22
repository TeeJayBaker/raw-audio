"""aux_t_min gate sweep — does ungated x-space aux bias the flow? (no src/ edits)

Question: applying an x-space aux loss (MR-STFT) on the predicted clean signal x̂ vs x1 at
HIGH NOISE (low t) anchors x̂ to a single sample where the flow-matching optimum is the
conditional MEAN E[x1|x_t,cond] (blurry, low energy). That should bias the velocity field.
The production trainer guards this with a pMF-style gate `aux_t_min` (apply aux only on rows
with t >= aux_t_min; t=0 noise, t=1 data). This probe sweeps that gate.

KEY METHODOLOGY: at n=64 the cond is a near-unique lookup, so E[x1|x_t,cond] ≈ the single x1
at ALL t — the conditional-mean-vs-sample conflict VANISHES in-set, and an in-set-only sweep
shows ~nothing. The signal only appears on HELD-OUT conds (clips NOT in the training 64),
where cond→clip is no longer a memorised lookup. We therefore report in-set AND held-out.

Reuses the current-API scaffold (ProbeModel/euler_sample/metrics) from
experiments/stft_frontends/flow_space.py UNTOUCHED — column front-end, waveform-space flow,
the production stft_transformer trunk. Only adds: the MR-STFT@1.0 aux + its t-gate, a disjoint
held-out split, and the STFT-consistency residual ‖S−STFT(iSTFT(S))‖/‖S‖ on the model's raw
predicted spectrogram (the direct phase-coherence probe; current metrics are phase-blind).

  CUDA_VISIBLE_DEVICES=1 uv run python experiments/aux_losses/aux_tmin_sweep.py --gpu 0 --nclips 64 --steps 6000
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
from stft_frontends.flow_space import (  # noqa: E402  (reuse scaffold, src/ untouched)
    CACHE_DIR,
    L,
    MATPAC_CKPT,
    SR,
    ProbeModel,
    get_data,
    load_clips,
    logstft_l1,
)

from backbone.io import channels_to_complex, stft_to_waveform, waveform_to_stft  # noqa: E402
from emb.factory import build_embedding  # noqa: E402
from flow.fm import EPS, RectifiedFlow  # noqa: E402
from losses.audio import mr_stft_loss  # noqa: E402

TRAIN_SEED = 1234
MR_WEIGHT = 1.0

# (label, use_aux, aux_t_min) — repo axis t=0 noise … t=1 data; None = ungated (all t).
CONFIGS = [
    ("no-aux", False, None),
    ("ungated", True, None),
    ("tmin0.2", True, 0.2),
    ("tmin0.5", True, 0.5),
    ("tmin0.7", True, 0.7),
]


def _embed(conditioner, a1, dev):
    with torch.no_grad():
        return conditioner(a1, sample_rate=SR,
                           audio_lengths=torch.tensor([a1.shape[-1]], device=dev)).view(-1)


def held_out_data(nclip, conditioner, dev):
    """Disjoint held-out set: load 2*nclip clips (deterministic seed=3) and take the SECOND half.
    The first half is exactly the training set (same seed/filter), so the split can't overlap."""
    cache = os.path.join(CACHE_DIR, f"cache_tmin_held_n{nclip}.pt")
    if os.path.exists(cache):
        d = torch.load(cache, map_location=dev)
        return d["C"].to(dev), d["CONDS"].to(dev)
    allC = load_clips(2 * nclip, dev)
    if allC.shape[0] < 2 * nclip:
        raise RuntimeError(f"only {allC.shape[0]} clips available, need {2 * nclip} for a disjoint held-out set")
    C_held = allC[nclip:2 * nclip]
    CONDS_held = torch.stack([_embed(conditioner, C_held[i:i + 1], dev) for i in range(C_held.shape[0])], 0)
    torch.save({"C": C_held.cpu(), "CONDS": CONDS_held.cpu()}, cache)
    return C_held, CONDS_held


def consistency_residual(spec_chan, model, NF):
    """‖S − STFT(iSTFT(S))‖ / ‖S‖ on the model's raw channelised spec S (relative complex-L1).
    Zero for any consistent (real-signal) STFT; grows with cross-frame phase incoherence."""
    S = channels_to_complex(spec_chan.float(), 1, NF)
    re = waveform_to_stft(stft_to_waveform(S, model.stft, length=L), model.stft)
    return float((S - re).abs().sum() / S.abs().sum().clamp_min(1e-8))


def euler_sample_capture(model, flow, cond, noise, steps=25):
    """Inline 25-step Euler (mirrors flow_space.euler_sample / RectifiedFlow.generate, gs=1.0),
    additionally capturing the model's raw predicted spectrogram at the final step."""
    x = noise
    grid = torch.linspace(EPS, 1.0 - EPS, steps + 1, device=noise.device)
    last_spec = None
    for i in range(steps):
        t = grid[i].expand(1)
        wav_pred = model(x, t=t, cond=cond, length=L)
        if i == steps - 1:
            last_spec = model(x, t=t, cond=cond, length=L, return_spec=True)
        x = x + (grid[i + 1] - grid[i]) * flow.target_to_v(wav_pred, x, t)
    return x.clamp(-1, 1), last_spec


def train(use_aux, tmin, C, CONDS, dev, steps, B):
    model = ProbeModel("waveform", frontend="column").to(dev)
    flow = RectifiedFlow()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.01)
    n = C.shape[0]
    g = torch.Generator(device=dev).manual_seed(TRAIN_SEED)  # identical (idx,t) stream across configs
    model.train()
    final_loss = aux_seen = 0.0
    for _s in range(steps):
        idx = torch.randint(0, n, (B,), device=dev, generator=g)
        t = torch.randn(B, device=dev, generator=g).sigmoid().clamp(EPS, 1 - EPS)
        x_t, t, x1 = flow.train_tuple(C[idx], t=t)  # waveform space: flow var = waveform
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(x_t, t=t, cond=CONDS[idx], length=L)
            loss, _ = flow.loss(pred, x1, x_t, t, space="v", loss_type="mse")
            if use_aux:
                keep = torch.ones_like(t, dtype=torch.bool) if tmin is None else (t >= tmin)
                if keep.any():
                    loss = loss + MR_WEIGHT * mr_stft_loss(pred[keep].float(), x1[keep].float())
                    aux_seen += float(keep.float().mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        final_loss = float(loss.detach())
    return model, flow, final_loss, aux_seen / max(steps, 1)


@torch.no_grad()
def evaluate(model, flow, conditioner, C_eval, CONDS_eval, dev, NF, seed_base, neval):
    Cn = F.normalize(CONDS_eval, dim=1)
    n = C_eval.shape[0]
    m = n if neval <= 0 else min(neval, n)
    own, l1s, grms, resid, hits = [], [], [], [], 0
    for i in range(m):
        gi = torch.Generator(device=dev).manual_seed(seed_base + i)
        noise = torch.randn((1, 1, L), device=dev, generator=gi)
        o, spec = euler_sample_capture(model, flow, CONDS_eval[i:i + 1], noise)
        l1s.append(logstft_l1(o, C_eval[i:i + 1]))
        cos = Cn @ F.normalize(_embed(conditioner, o, dev), dim=0)
        own.append(float(cos[i]))
        if int(cos.argmax()) == i:
            hits += 1
        grms.append(float((o ** 2).mean().sqrt()))
        resid.append(consistency_residual(spec, model, NF))
    return {"own": float(np.mean(own)), "retr": hits, "m": m, "genrms": float(np.mean(grms)),
            "logstft": float(np.mean(l1s)), "resid": float(np.mean(resid))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)  # with CUDA_VISIBLE_DEVICES=1, GPU1 -> cuda:0
    ap.add_argument("--nclips", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--neval", type=int, default=0, help="eval clips per set (0=all); use small for smoke")
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(0); np.random.seed(0)
    os.makedirs(CACHE_DIR, exist_ok=True)

    conditioner = build_embedding(
        {"type": "matpac", "checkpoint_path": MATPAC_CKPT, "device": str(dev),
         "use_teacher": False, "encode_batch_size": 0, "compile_encoder": False}, device=dev).to(dev).eval()
    C, CONDS = get_data(args.nclips, conditioner, dev)
    Ch, CONDSh = held_out_data(args.nclips, conditioner, dev)
    NF = ProbeModel("waveform", frontend="column").stft.freq_bins

    cn, cnh = F.normalize(CONDS, dim=1), F.normalize(CONDSh, dim=1)
    pair = float((cn @ cn.T)[~torch.eye(args.nclips, dtype=bool, device=dev)].mean())
    leak = float((cn @ cnh.T).max())  # max train↔held cond cosine (≈1 would mean a near-duplicate)
    print(f"\n=== aux_t_min sweep | n={args.nclips} (+{Ch.shape[0]} held-out) | steps={args.steps} "
          f"batch={args.batch} | column waveform-flow | MR-STFT@{MR_WEIGHT} ===", flush=True)
    print(f"  cond pairwise cos (in-set)={pair:.3f}  |  max train↔held cos={leak:.3f}  (retr chance=1/{args.nclips})",
          flush=True)
    print("  conditional-mean conflict vanishes in-set @ n=64 → read the HELD-OUT columns for the gate effect\n",
          flush=True)

    rows = []
    for label, use_aux, tmin in CONFIGS:
        model, flow, fl, frac = train(use_aux, tmin, C, CONDS, dev, args.steps, args.batch)
        ins = evaluate(model, flow, conditioner, C, CONDS, dev, NF, 50, args.neval)
        hld = evaluate(model, flow, conditioner, Ch, CONDSh, dev, NF, 5000, args.neval)
        rows.append((label, frac, ins, hld))
        print(f"  [{label:>8}] aux_frac={frac:.2f} loss={fl:.3f}", flush=True)
        print(f"      in-set : L1={ins['logstft']:.3f} own={ins['own']:.3f} retr={ins['retr']}/{ins['m']} "
              f"genRMS={ins['genrms']:.3f} resid={ins['resid']:.3f}", flush=True)
        print(f"      heldout: L1={hld['logstft']:.3f} own={hld['own']:.3f} retr={hld['retr']}/{hld['m']} "
              f"genRMS={hld['genrms']:.3f} resid={hld['resid']:.3f}", flush=True)

    print("\n=== summary (HELD-OUT is the signal; in-set is the control) ===", flush=True)
    print(f"  {'config':>8} | {'in own':>7} {'in retr':>7} {'in gRMS':>7} {'in res':>7} | "
          f"{'ho own':>7} {'ho retr':>7} {'ho gRMS':>7} {'ho res':>7}", flush=True)
    for label, _frac, ins, hld in rows:
        print(f"  {label:>8} | {ins['own']:>7.3f} {ins['retr']:>5}/{ins['m']:<1} {ins['genrms']:>7.3f} "
              f"{ins['resid']:>7.3f} | {hld['own']:>7.3f} {hld['retr']:>5}/{hld['m']:<1} {hld['genrms']:>7.3f} "
              f"{hld['resid']:>7.3f}", flush=True)


if __name__ == "__main__":
    main()
