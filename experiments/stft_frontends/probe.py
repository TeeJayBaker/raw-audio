"""STFT-transformer FRONT-END overfit probe (no src/ edits).

Question: does restructuring how the STFT enters the transformer change what it can
fit and what it can sample? Six front-ends share ONE identical trunk (12x d512/h8
TransformerBlock + MATPAC AdaLN cond, same as the real stft_transformer); the only
thing that varies is spectrogram -> tokens -> spectrogram:

  column      full-frequency column per frame, in_proj 2050->512   (= real backbone)   38 tok
  bneck{N}    column, JiT/pMF linear input bottleneck 2050->N->512, out full 512->2050  38 tok
  square      2D ViT tiles (128 freq x 8 time), feature ~= column                       45 tok
  band2       freq split into 2 equal bands, own in/out proj each, 2 tokens/frame        76 tok

Same overfit harness / metrics as experiments/stft_capacity/probe.py:
  FITTING       : teacher-forced x1-pred corr given the TRUE x_t (t=.3/.5/.8)
  END-TO-END    : free-running sampled output vs the true clip -> paired multi-res
                  log-STFT L1 (lower=better), MATPAC own-cosine / retrieval, genRMS.

Reuses the real TransformerBlock / conditioning / io helpers untouched.

  python experiments/stft_frontends/probe.py --gpu 0 --nclips 64 --steps 6000
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
from torch import nn

sys.path.insert(0, ".")
sys.path.insert(0, "src")
from backbone.blocks import TransformerBlock  # noqa: E402
from backbone.conditioning import (  # noqa: E402
    ConditioningCombiner,
    ConditioningEmbedding,
    TimeEmbedding,
)
from backbone.io import (  # noqa: E402
    STFTConfig,
    as_waveform,
    channels_to_complex,
    complex_to_channels,
    stft_to_waveform,
    waveform_to_stft,
)
from emb.factory import build_embedding  # noqa: E402
from flow.fm import EPS, RectifiedFlow  # noqa: E402

SR = 48000
L = int(0.4 * SR)
ROOT = "/media/storage/samples/samples_from_mars/one_shots"
MATPAC_CKPT = "/media/NAS/neutone/diff_one_shot/checkpoints/whole-violet-235/last-v1_clean.ckpt"
CACHE_DIR = "experiments/stft_frontends"
STFT = {"n_fft": 2048, "hop_length": 512, "win_length": 2048}
FREQ = 2048 // 2 + 1  # 1025

DIM, DEPTH, HEADS, COND_DIM = 512, 12, 8, 384

# square tiling: pad freq->FP_F*FP, time->TP_F*TP so the spectrogram divides evenly.
SQ_FP, SQ_TP = 128, 8  # freq / time patch sizes

# (label, kwargs for ProbeModel)
FRONTENDS = [
    ("column", dict(frontend="column")),
    ("bneck64", dict(frontend="bottleneck", n_bottleneck=64)),
    ("bneck128", dict(frontend="bottleneck", n_bottleneck=128)),
    ("bneck256", dict(frontend="bottleneck", n_bottleneck=256)),
    ("square", dict(frontend="square")),
    ("band2", dict(frontend="band")),
]


def _no_amp_complex(real, imag):
    with torch.amp.autocast(device_type=real.device.type, enabled=False):
        return torch.complex(real.float(), imag.float())


class ProbeModel(nn.Module):
    """Waveform-in / waveform-out STFT transformer with a swappable front-end.

    The trunk (time/cond embeds + `DEPTH` TransformerBlocks) is built first under a
    fixed seed so every front-end gets an identical trunk init; only the tokeniser and
    the in/out projections differ. A learned absolute positional embedding is added to
    every variant so the 1D (column/bneck) and 2D (square/band) token grids share a
    positional scheme.
    """

    def __init__(self, frontend: str, n_bottleneck: int | None = None):
        super().__init__()
        self.frontend = frontend
        self.stft = STFTConfig.from_dict(STFT)
        f = self.stft.freq_bins  # 1025
        # --- shared trunk (seeded so init matches across front-ends) ---
        torch.manual_seed(0)
        self.time_embed = TimeEmbedding(COND_DIM, time_scale=100.0)
        self.cond_embed = ConditioningEmbedding(COND_DIM, COND_DIM)
        self.cond_combine = ConditioningCombiner(COND_DIM)
        self.blocks = nn.ModuleList(
            [TransformerBlock(DIM, COND_DIM, heads=HEADS, mlp_ratio=8 / 3, rope=True, qk_norm=True)
             for _ in range(DEPTH)]
        )
        # --- front-end specific tokeniser / projections ---
        if frontend == "column":
            ntok = self._frames()
            self.in_proj = nn.Conv1d(2 * f, DIM, 1)
            self.out_proj = nn.Conv1d(DIM, 2 * f, 1)
        elif frontend == "bottleneck":
            ntok = self._frames()
            self.in_proj = nn.Sequential(nn.Conv1d(2 * f, n_bottleneck, 1, bias=False),
                                         nn.Conv1d(n_bottleneck, DIM, 1))
            self.out_proj = nn.Conv1d(DIM, 2 * f, 1)  # output full-rank (input-only bottleneck)
        elif frontend == "square":
            self.fp, self.tp = SQ_FP, SQ_TP
            self.fpad = (-f) % self.fp  # 1025 -> 1152 (9 patches)
            self.npf = (f + self.fpad) // self.fp
            feat = 2 * self.fp * self.tp
            self.in_proj = nn.Linear(feat, DIM)
            self.out_proj = nn.Linear(DIM, feat)
            ntok = self.npf * (self._tpad_frames() // self.tp)
        elif frontend == "band":
            self.fsplit = f // 2 + f % 2  # 513 low bins, 512 high bins
            lo, hi = self.fsplit, f - self.fsplit
            self.in_lo, self.in_hi = nn.Conv1d(2 * lo, DIM, 1), nn.Conv1d(2 * hi, DIM, 1)
            self.out_lo, self.out_hi = nn.Conv1d(DIM, 2 * lo, 1), nn.Conv1d(DIM, 2 * hi, 1)
            ntok = 2 * self._frames()
        else:
            raise ValueError(frontend)
        self.ntok = ntok
        self.pos = nn.Parameter(torch.randn(1, ntok, DIM) * 0.02)

    # frame count for the fixed clip length L
    def _frames(self) -> int:
        return L // self.stft.hop_length + 1  # center=True

    def _tpad_frames(self) -> int:
        n = self._frames()
        return n + ((-n) % SQ_TP)

    def _cond(self, t, cond):
        t_embed = self.time_embed(t) if t is not None else None
        cond = self.cond_embed(cond)
        return self.cond_combine(t_embed, cond)

    # ---- tokenisers: spec [B,1,F,T] complex <-> tokens [B,N,DIM] ----
    def _tok_column(self, spec):
        return self.in_proj(complex_to_channels(spec)).transpose(1, 2)

    def _untok_column(self, h):
        return self.out_proj(h.transpose(1, 2))  # [B, 2F, T]

    def _tok_square(self, spec):
        b, _c, f, t = spec.shape
        img = torch.cat([spec.real, spec.imag], dim=1)  # [B,2,F,T]
        tpad = self._tpad_frames() - t
        img = torch.nn.functional.pad(img, (0, tpad, 0, self.fpad))  # pad time then freq
        npf, npt = self.npf, img.shape[-1] // self.tp
        img = img.view(b, 2, npf, self.fp, npt, self.tp).permute(0, 2, 4, 1, 3, 5)
        tok = img.reshape(b, npf * npt, 2 * self.fp * self.tp)
        return self.in_proj(tok)

    def _untok_square(self, h, t):
        b = h.shape[0]
        npt = self._tpad_frames() // self.tp
        y = self.out_proj(h).view(b, self.npf, npt, 2, self.fp, self.tp)
        y = y.permute(0, 3, 1, 4, 2, 5).reshape(b, 2, self.npf * self.fp, npt * self.tp)
        y = y[:, :, :FREQ, :t]  # crop pad
        return _no_amp_complex(y[:, 0:1], y[:, 1:2])  # [B,1,F,T] complex

    def _tok_band(self, spec):
        lo = complex_to_channels(spec[:, :, : self.fsplit])
        hi = complex_to_channels(spec[:, :, self.fsplit :])
        t_lo = self.in_lo(lo).transpose(1, 2)  # [B,T,DIM]
        t_hi = self.in_hi(hi).transpose(1, 2)
        return torch.stack([t_lo, t_hi], dim=2).reshape(spec.shape[0], -1, DIM)  # frame-major

    def _untok_band(self, h):
        b = h.shape[0]
        h = h.view(b, -1, 2, DIM)
        s_lo = channels_to_complex(self.out_lo(h[:, :, 0].transpose(1, 2)), 1, self.fsplit)
        s_hi = channels_to_complex(self.out_hi(h[:, :, 1].transpose(1, 2)), 1, FREQ - self.fsplit)
        return torch.cat([s_lo, s_hi], dim=2)  # [B,1,F,T] complex

    def forward(self, x, t=None, cond=None, length=None):
        x = as_waveform(x)
        target = int(length or x.shape[-1])
        cond = self._cond(t, cond)
        spec = waveform_to_stft(x, self.stft)  # [B,1,F,T] complex
        frames = spec.shape[-1]

        if self.frontend in ("column", "bottleneck"):
            h = self._tok_column(spec)
        elif self.frontend == "square":
            h = self._tok_square(spec)
        else:
            h = self._tok_band(spec)
        h = h + self.pos

        for block in self.blocks:
            h = block(h, cond)

        if self.frontend in ("column", "bottleneck"):
            y = self._untok_column(h)
            spec_out = channels_to_complex(y, 1, self.stft.freq_bins)
        elif self.frontend == "square":
            spec_out = self._untok_square(h, frames)
        else:
            spec_out = self._untok_band(h)
        return stft_to_waveform(spec_out, self.stft, length=target)


def make_model(kwargs, dev):
    return ProbeModel(**kwargs).to(dev)


# ---------------------------------------------------------------------------
# data / metrics (mirrors experiments/stft_capacity/probe.py)
# ---------------------------------------------------------------------------
def load_clips(nclip, dev):
    cands = []
    for dp, _, fns in os.walk(ROOT):
        for fn in fns:
            if fn.lower().endswith((".wav", ".flac", ".aif", ".aiff")):
                cands.append(os.path.join(dp, fn))
    random.seed(3)  # same seed/clip set as the capacity probe
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
    tot = 0.0
    for n_fft in (512, 1024, 2048):
        tot += float((_stft_logmag(a, n_fft, n_fft // 4) - _stft_logmag(b, n_fft, n_fft // 4)).abs().mean())
    return tot / 3.0


def run(label, kwargs, nclip, C, CONDS, conditioner, dev, steps, B):
    model = make_model(kwargs, dev)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    ntok = model.ntok
    flow = RectifiedFlow()
    Cn = torch.nn.functional.normalize(CONDS, dim=1)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.01)
    n = C.shape[0]
    t0 = time.time()
    model.train()
    final_loss = float("nan")
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
        final_loss = float(loss.detach())

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
    print(f"  [{label:>9} {nparams:5.1f}M tok={ntok:>3}] "
          f"logSTFT_L1={np.mean(l1s):.3f}  tf_corr(.3/.5/.8)={tf_corr[0.3]:+.2f}/{tf_corr[0.5]:+.2f}/{tf_corr[0.8]:+.2f}  "
          f"own={np.mean(own):.3f}  retr={hits}/{n}  genRMS={np.mean(grms):.3f}(tgt{tgt:.2f})  "
          f"loss={final_loss:.3f}  {dt:.0f}s", flush=True)
    return {"label": label, "params": nparams, "ntok": ntok, "nclip": nclip,
            "logstft_l1": float(np.mean(l1s)), "tf_corr_0.5": tf_corr[0.5],
            "own": float(np.mean(own)), "retr": hits, "genrms": float(np.mean(grms))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--nclips", type=str, default="64")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--fronts", type=str, default="all", help="comma list of labels or 'all'")
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    os.makedirs(CACHE_DIR, exist_ok=True)
    fronts = FRONTENDS if args.fronts == "all" else [f for f in FRONTENDS if f[0] in args.fronts.split(",")]
    conditioner = build_embedding(
        {"type": "matpac", "checkpoint_path": MATPAC_CKPT, "device": str(dev),
         "use_teacher": False, "encode_batch_size": 0, "compile_encoder": False}, device=dev).to(dev).eval()
    results = []
    for nclip in [int(x) for x in args.nclips.split(",")]:
        C, CONDS = get_data(nclip, conditioner, dev)
        off = torch.nn.functional.normalize(CONDS, dim=1)
        offm = float((off @ off.T)[~torch.eye(nclip, dtype=bool, device=dev)].mean())
        print(f"\n=== n={nclip} clips | cond pairwise cos mean={offm:.3f} | steps={args.steps} batch={args.batch} ===", flush=True)
        for label, kwargs in fronts:
            results.append(run(label, kwargs, nclip, C, CONDS, conditioner, dev, args.steps, args.batch))

    print("\n=== summary (sorted by sampled logSTFT_L1, lower=better) ===", flush=True)
    for r in sorted(results, key=lambda r: r["logstft_l1"]):
        print(f"  {r['label']:>9} n={r['nclip']:>3} tok={r['ntok']:>3}  "
              f"logSTFT_L1={r['logstft_l1']:.3f}  tf_corr.5={r['tf_corr_0.5']:+.2f}  "
              f"own={r['own']:.3f}  retr={r['retr']}  genRMS={r['genrms']:.3f}", flush=True)


if __name__ == "__main__":
    main()
