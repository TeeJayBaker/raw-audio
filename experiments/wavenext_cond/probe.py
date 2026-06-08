"""WaveNeXt/ConvNeXt conditioning overfit DIAGNOSIS probe (no src/ edits).

Trains a small ConvNeXt(wavenext-head) flow model to overfit NCLIP clips with
MATPAC conditioning, then measures whether the conditioning actually survives:

  - retrieval  : does MATPAC(generate | cond_i) point back at clip i (vs the others)?
  - own/other  : cosine of the generated-clip embedding to its own vs the wrong conds
  - genRMS     : generated amplitude (collapse-to-silence is the wavenext failure)
  - ||t_emb||  : raw norm of the time embedding (the suspected runaway culprit)

Each variant is a `ProbeConvNeXt` SUBCLASS that only changes how the timestep
embedding and (projected) conditioning are *combined* -- the real backbone code
is imported untouched. Run e.g.

  python experiments/wavenext_cond/probe.py --gpu 0 --variants raw_noproj,proj_add,sep_norm
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
from omegaconf import OmegaConf
from torch import nn

sys.path.insert(0, ".")  # repo root: top-level `matpac` package
sys.path.insert(0, "src")
from backbone.blocks import AdaLN, RMSNorm  # noqa: E402
from backbone.convnext import ConvNeXt  # noqa: E402
from backbone.io import as_waveform  # noqa: E402
from backbone.transformer import Transformer  # noqa: E402
from emb.factory import build_embedding  # noqa: E402
from flow.fm import EPS, RectifiedFlow  # noqa: E402

SR = 48000
L = int(0.4 * SR)
NCLIP = 8
ROOT = "/media/storage/samples/samples_from_mars/one_shots"
MATPAC_CKPT = "/media/NAS/neutone/diff_one_shot/checkpoints/whole-violet-235/last-v1_clean.ckpt"
CACHE = "experiments/wavenext_cond/cache.pt"


# --------------------------------------------------------------------------- model
class GatedConvNeXtBlock(nn.Module):
    """ConvNeXtBlock1d + a zero-init AdaLN gate on the residual branch (groups=3),
    mirroring the transformer block's per-sub-block multiplicative conditioning."""

    def __init__(self, channels, cond_dim, kernel_size=7, expansion=3, layer_scale=None):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.ada = AdaLN(cond_dim, channels, groups=3)  # scale, shift, gate
        hidden = channels * expansion
        self.pointwise = nn.Sequential(nn.Conv1d(channels, hidden, 1), nn.GELU(), nn.Conv1d(hidden, channels, 1))
        self.layer_scale = nn.Parameter(torch.ones(1, channels, 1) * layer_scale) if layer_scale is not None else None

    @classmethod
    def from_block(cls, old, cond_dim):
        channels = old.depthwise.in_channels
        ls = float(old.layer_scale.flatten()[0]) if old.layer_scale is not None else None
        return cls(channels, cond_dim, kernel_size=old.depthwise.kernel_size[0],
                   expansion=old.pointwise[0].out_channels // channels, layer_scale=ls)

    def forward(self, x, cond):
        h = self.depthwise(x)
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        scale, shift, gate = self.ada(cond)
        h = h * (1 + scale[..., None]) + shift[..., None]
        h = self.pointwise(h)
        if self.layer_scale is not None:
            h = h * self.layer_scale
        return x + gate[..., None] * h


class ProbeConvNeXt(ConvNeXt):
    """ConvNeXt whose conditioning *combination* is swappable. Reuses the real
    `time_embed`, `cond_embed` (the user's ConditioningEmbedder) and `branches`;
    only `forward`/`_combine` change. `use_proj` toggles the cond_embed MLP."""

    def __init__(self, *a, mode="add", t_norm=False, use_proj=True, gate=False, **kw):
        super().__init__(*a, **kw)
        self.mode = mode
        self.use_proj = use_proj
        self.t_normm = RMSNorm(self.cond_dim) if t_norm else None
        self.norm_t = RMSNorm(self.cond_dim) if mode == "sep_norm" else None
        self.norm_c = RMSNorm(self.cond_dim) if mode == "sep_norm" else None
        self.combine = nn.Linear(2 * self.cond_dim, self.cond_dim) if mode == "concat" else None
        if gate:  # swap each trunk block for a gated variant (the transformer's per-block control)
            for branch in self.branches:
                for j, old in enumerate(branch.trunk):
                    branch.trunk[j] = GatedConvNeXtBlock.from_block(old, self.cond_dim)

    def _combine(self, t_embed, cond):
        if t_embed is not None and self.t_normm is not None:
            t_embed = self.t_normm(t_embed)
        if cond is None:
            return t_embed
        if t_embed is None:
            return cond
        if self.mode == "sep_norm":
            return self.norm_t(t_embed) + self.norm_c(cond)
        if self.mode == "concat":
            return self.combine(torch.cat([t_embed, cond], dim=-1))
        return t_embed + cond  # add / proj_add

    def forward(self, x, t=None, cond=None, length=None):
        x = as_waveform(x)
        target = int(length or x.shape[-1])
        t_embed = self.time_embed(t) if t is not None else None
        if cond is not None and self.use_proj:
            cond = self.cond_embed(cond)
        c = self._combine(t_embed, cond)
        return sum(branch(x, c, target) for branch in self.branches).float()


VARIANTS = {
    # ConvNeXt/wavenext (kind=cx): conditioning-combination + optional gate
    "raw_noproj":   dict(kind="cx", mode="add",      use_proj=False),                # true original baseline
    "proj_add":     dict(kind="cx", mode="add",      use_proj=True),                 # user's ConditioningEmbedder alone
    "sep_noproj":   dict(kind="cx", mode="sep_norm", use_proj=False),                # RMSNorm both paths
    "sep_norm":     dict(kind="cx", mode="sep_norm", use_proj=True),                 # + learned cond proj
    "gate":         dict(kind="cx", mode="add",      use_proj=True,  gate=True),     # gate only (add cond)
    "sep_gate":     dict(kind="cx", mode="sep_norm", use_proj=True,  gate=True),     # gate + balanced cond
    "sep_gate_np":  dict(kind="cx", mode="sep_norm", use_proj=False, gate=True),     # gate + balanced, no proj
    # Discriminators: which part of the cx architecture breaks self-consistency?
    "cx_istft":     dict(kind="cx", mode="sep_norm", use_proj=False, gate=True,      # iSTFT head (realimag) not wavenext
                         head={"type": "istft", "parameterisation": "realimag"}),
    # Decoupling: does the iSTFT head fix amplitude on its own (raw conditioning)?
    "cx_istft_raw": dict(kind="cx", mode="add", use_proj=False, gate=False,
                         head={"type": "istft", "parameterisation": "realimag"}),
    # Does the WaveNeXt head collapse persist with LONGER training (not just undertraining)?
    "wnext_long":   dict(kind="cx", mode="sep_norm", use_proj=False, gate=True),
    # Real-scale (depth 8, width 512) confirmation of the working iSTFT recipe
    "cx_istft_big": dict(kind="cx", mode="sep_norm", use_proj=False, gate=True,
                         head={"type": "istft", "parameterisation": "realimag"},
                         block={"channels": 512, "depth": 8, "kernel_size": 7, "expansion": 3, "layer_scale": "1/depth"}),
    # Transformer reference (kind=tf): the backbone that already conditions
    "tf_p128":      dict(kind="tf", patch=128),
    "tf_p512":      dict(kind="tf", patch=512),
    "tf_stft":      dict(kind="tf", patch=1, stft={"n_fft": 2048, "hop_length": 512, "win_length": 2048},
                         head={"type": "identity"}),  # transformer in the STFT domain (stft_transformer)
}


def make_model(variant, dev):
    torch.manual_seed(0)
    cfg = dict(VARIANTS[variant])
    kind = cfg.pop("kind")
    if kind == "tf":
        patch = cfg.pop("patch", 128)
        stft = cfg.pop("stft", None)
        head = cfg.pop("head", {"type": "conv", "kernel_size": 13})
        return Transformer(
            channels=1, out_channels=1, patching={"patch_size": patch},
            block={"dim": cfg.pop("dim", 384), "depth": 6, "heads": 6},
            conditioning={"cond_dim": 384, "time_scale": 100.0},
            head=head, stft=stft, sample_rate=SR, name="tf",
        ).to(dev)
    head = cfg.pop("head", {"type": "wavenext"})
    block = cfg.pop("block", {"channels": 256, "depth": 6, "kernel_size": 7, "expansion": 3, "layer_scale": "1/depth"})
    return ProbeConvNeXt(
        channels=1, out_channels=1, branches={"mode": "single"},
        block=block,
        conditioning={"cond_dim": 384, "time_scale": 100.0}, head=head,
        stft={"n_fft": 2048, "hop_length": 512, "win_length": 2048}, sample_rate=SR, name="cx",
        **cfg,
    ).to(dev)


# ----------------------------------------------------------------------------- data
def load_clips(dev):
    cands = []
    for dp, _, fns in os.walk(ROOT):
        for fn in fns:
            if fn.lower().endswith((".wav", ".flac", ".aif", ".aiff")):
                cands.append(os.path.join(dp, fn))
    random.seed(3)
    random.shuffle(cands)
    clips = []
    for f in cands:
        if len(clips) >= NCLIP:
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


def get_conditioner(dev):
    cond_cfg = {"type": "matpac", "checkpoint_path": MATPAC_CKPT, "device": str(dev),
                "use_teacher": False, "encode_batch_size": 0, "compile_encoder": False}
    return build_embedding(cond_cfg, device=dev).to(dev).eval()


def get_data(dev, conditioner):
    if os.path.exists(CACHE):
        d = torch.load(CACHE, map_location=dev)
        return d["C"].to(dev), d["CONDS"].to(dev)
    C = load_clips(dev)

    def embed(a1):
        with torch.no_grad():
            return conditioner(a1, sample_rate=SR, audio_lengths=torch.tensor([a1.shape[-1]], device=dev)).view(-1)

    CONDS = torch.stack([embed(C[i:i + 1]) for i in range(C.shape[0])], 0)
    torch.save({"C": C.cpu(), "CONDS": CONDS.cpu()}, CACHE)
    return C, CONDS


# ------------------------------------------------------------------------------ run
def run(variant, C, CONDS, conditioner, dev, steps, B=8,
        lift=False, lift_scale=3.0, rms_target=0.33, lift_mode="tanh"):
    model = make_model(variant, dev)
    flow = RectifiedFlow()
    Cn = torch.nn.functional.normalize(CONDS, dim=1)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.01)
    n = C.shape[0]
    # WavFlow amplitude lift: per-clip RMS-normalise to rms_target, saturate (tanh|clamp), ×lift_scale;
    # ÷lift_scale at sampling. So every output is forced to RMS≈rms_target (per-clip loudness discarded).
    ls = lift_scale if lift else 1.0

    def liftfn(x):
        if not lift:
            return x
        rms = x.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
        z = (rms_target / rms) * x
        z = torch.tanh(z) if lift_mode == "tanh" else z.clamp(-1, 1)
        return lift_scale * z

    def raw_tnorm():
        with torch.no_grad():
            return float(model.time_embed(torch.tensor([0.1, 0.5, 0.9], device=dev)).norm(dim=-1).mean())

    model.train()
    traj = []
    t0 = time.time()
    for s in range(steps):
        idx = torch.randint(0, n, (B,), device=dev)
        x1 = liftfn(C[idx])
        t = torch.randn(B, device=dev).sigmoid().clamp(EPS, 1 - EPS)
        x_t, t, x1 = flow.train_tuple(x1, t=t)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(x_t, t=t, cond=CONDS[idx], length=L)
            loss, _ = flow.loss(pred, x1, x_t, t, space="v", loss_type="mse")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % 500 == 0:
            traj.append((s, float(loss.detach()), raw_tnorm()))

    model.eval()

    # Teacher-forced x1-prediction vs t: feed the TRUE x_t for each clip at fixed t
    # and measure predicted RMS + corr to the true clip. Separates a model-prediction
    # failure (low RMS even teacher-forced) from a sampler-trajectory collapse.
    tf_lines = []
    with torch.no_grad():
        for tval in (0.05, 0.2, 0.5, 0.8, 0.95):
            rmss, corrs = [], []
            for i in range(n):
                g = torch.Generator(device=dev).manual_seed(900 + i)
                x0 = torch.randn((1, 1, L), device=dev, generator=g)
                x1 = liftfn(C[i:i + 1])
                xt = (1 - tval) * x0 + tval * x1
                tt = torch.full((1,), tval, device=dev)
                xp = model(xt, t=tt, cond=CONDS[i:i + 1], length=L)
                rmss.append(float((xp / ls).clamp(-1, 1).pow(2).mean().sqrt()))  # report in original scale
                a, b = xp.flatten(), x1.flatten()
                corrs.append(float((a @ b) / (a.norm() * b.norm() + 1e-9)))
            tf_lines.append(f"t={tval}:rms{np.mean(rmss):.3f}/corr{np.mean(corrs):+.2f}")

    def embed(a1):
        with torch.no_grad():
            return conditioner(a1, sample_rate=SR, audio_lengths=torch.tensor([a1.shape[-1]], device=dev)).view(-1)

    own, wrong, grms, hits = [], [], [], 0
    with torch.no_grad():
        for i in range(n):
            g = torch.Generator(device=dev).manual_seed(50 + i)
            noise = torch.randn((1, 1, L), device=dev, generator=g)
            o = flow.sample(model, (1, 1, L), cond=CONDS[i:i + 1], noise=noise, steps=25,
                            guidance_scale=1.0, lift_scale=ls).clamp(-1, 1)
            eg = torch.nn.functional.normalize(embed(o), dim=0)
            cos = Cn @ eg
            own.append(float(cos[i]))
            wrong.append(float((cos.sum() - cos[i]) / (n - 1)))
            if int(cos.argmax()) == i:
                hits += 1
            grms.append(float((o ** 2).mean().sqrt()))
    dt = time.time() - t0
    gap = float(np.mean(own) - np.mean(wrong))
    # Under lift the by-design output level is rms_target (per-clip loudness is discarded); else the clips' RMS.
    tgt = rms_target if lift else float((C ** 2).mean(dim=(-2, -1)).sqrt().mean())
    hit = np.mean(grms) / tgt
    print(f"{variant:>13} | own={np.mean(own):.3f} other={np.mean(wrong):.3f} GAP={gap:+.3f} "
          f"| retr={hits}/{n} | genRMS={np.mean(grms):.3f} (tgt {tgt:.2f}, hit {hit:.0%}) | ||t_emb||={raw_tnorm():.1f} | {dt:.0f}s")
    print("        traj(step:loss/||t_emb||): " + "  ".join(f"{s}:{l:.2f}/{tn:.0f}" for s, l, tn in traj))
    print("        teacher-forced x1-pred (tgt rms~%.2f): " % tgt + "  ".join(tf_lines))
    # Free-running sampling trajectory: avg RMS of x along the ODE (where does energy collapse?)
    grid = torch.linspace(EPS, 1 - EPS, 26, device=dev)
    checks = [2, 6, 12, 19, 24]
    acc = {k: [] for k in checks}
    with torch.no_grad():
        for i in range(n):
            g = torch.Generator(device=dev).manual_seed(50 + i)
            x = torch.randn((1, 1, L), device=dev, generator=g)
            for s in range(25):
                tt = grid[s].expand(1)
                x = x + (grid[s + 1] - grid[s]) * flow.target_to_v(model(x, t=tt, cond=CONDS[i:i + 1], length=L), x, tt)
                if s in acc:
                    acc[s].append(float((x / ls).pow(2).mean().sqrt()))  # report in original scale
    print("        sampling traj rms(t): " + "  ".join(f"t{float(grid[s + 1]):.2f}:{np.mean(acc[s]):.3f}" for s in checks))
    # Sampler sweep on the SAME trained model: is the energy collapse an integration error?
    samp = []
    with torch.no_grad():
        for m, st in [("euler", 25), ("euler", 100), ("heun", 50)]:
            rr = []
            for i in range(n):
                g = torch.Generator(device=dev).manual_seed(50 + i)
                noise = torch.randn((1, 1, L), device=dev, generator=g)
                o = flow.sample(model, (1, 1, L), cond=CONDS[i:i + 1], noise=noise, steps=st,
                                method=m, guidance_scale=1.0, lift_scale=ls).clamp(-1, 1)
                rr.append(float((o ** 2).mean().sqrt()))
            samp.append(f"{m}{st}:{np.mean(rr):.3f}")
    print("        sampler genRMS sweep (tgt %.2f): " % tgt + "  ".join(samp))
    return {"variant": variant, "gap": gap, "retrieval": hits, "genRMS": float(np.mean(grms))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--variants", type=str, default=",".join(VARIANTS))
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lift", action="store_true", help="apply WavFlow amplitude lift")
    ap.add_argument("--lift_scale", type=float, default=3.0)
    ap.add_argument("--rms_target", type=float, default=0.33)
    ap.add_argument("--lift_mode", type=str, default="tanh", choices=["tanh", "clamp"])
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    conditioner = get_conditioner(dev)
    C, CONDS = get_data(dev, conditioner)
    Cn = torch.nn.functional.normalize(CONDS, dim=1)
    off = (Cn @ Cn.T)[~torch.eye(C.shape[0], dtype=bool, device=dev)]
    clip_rms = float((C ** 2).mean(dim=(-2, -1)).sqrt().mean())
    print(f"{C.shape[0]} clips | cond pairwise cos mean={float(off.mean()):.3f} "
          f"| chance retrieval=1/{C.shape[0]} | target genRMS~{clip_rms:.3f}\n")
    print(f"(lift={'ON' if args.lift else 'off'}"
          + (f": rms_target={args.rms_target} ×{args.lift_scale} {args.lift_mode}" if args.lift else "") + ")")
    for v in args.variants.split(","):
        run(v.strip(), C, CONDS, conditioner, dev, args.steps,
            lift=args.lift, lift_scale=args.lift_scale, rms_target=args.rms_target, lift_mode=args.lift_mode)


if __name__ == "__main__":
    main()
