"""Waveform-space vs spectrogram-space FLOW ablation (no src/ edits).

Question: does it matter whether the rectified flow LIVES in waveform space or in
spectrogram space? The two arms share ONE identical trunk + `column` tokeniser (12x
d512/h8 TransformerBlock + MATPAC AdaLN cond, learned abs-pos + RoPE, in_proj/out_proj
Conv1d 2050<->512). The ONLY thing that changes is where noise is injected and where the
loss is measured:

  waveform  noise + interpolant + v-MSE loss in WAVEFORM space; STFT/iSTFT bracket every
            model eval (= production `RectifiedFlow._predict`, = the RESULTS.md `column`).
  spec      noise + interpolant + v-MSE loss in the channelised STFT; the STFT/iSTFT are
            pulled OUTSIDE the flow -> the flow variable IS the spectrogram. One iSTFT at
            the very end of sampling crosses back to waveform purely to score.

Scale match (the one real confound): the waveform arm has data std ~0.28 vs noise std 1.0.
The spec arm scales the channelised STFT by a single GLOBAL scalar so its std also matches
the waveform data std, keeping noise N(0,1) -> identical data/noise SNR ratio. Global (not
per-bin) on purpose: it preserves the natural spectral tilt, so spec-flow sees the same
falling-spectrum-vs-flat-noise profile that waveform-flow does. The scalar is divided back
out before the final iSTFT.

Both arms share data, optimiser, steps, the (idx, t) training sequence (dedicated seeded
generator), the per-clip sampling-noise seeds, and a single inline 25-step Euler sampler;
only the injected-noise dimensionality differs (unavoidable). Metrics mirror
experiments/stft_frontends/probe.py so numbers drop into RESULTS.md.

  python experiments/stft_frontends/flow_space.py --gpu 1 --nclips 64 --steps 6000

The waveform arm reproduces the RESULTS.md `column` control (reseeded: L1 2.847 / own 0.626
/ retr 32 / genRMS 0.248) as a fairness cross-check; the spec arm is the new datapoint.
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

DIM, DEPTH, HEADS, COND_DIM = 512, 12, 8, 384

# (label, flow_space, loss_space): where the noise/interpolant live, where the loss is measured.
#   waveform : noise + v-MSE in waveform (= production)                       [A]
#   wav→spec : noise in waveform, v-MSE on the model's RAW spec output        [B]  isolates loss domain
#   spec     : noise + v-MSE in the channelised STFT                          [D]  full spec flow
# A→B isolates the loss domain (noise held in waveform); B→D isolates the noise domain.
ARMS = [("waveform", "waveform", "waveform"),
        ("wav→spec", "waveform", "spec"),
        ("spec", "spec", "spec")]


def _no_amp_complex(real, imag):
    with torch.amp.autocast(device_type=real.device.type, enabled=False):
        return torch.complex(real.float(), imag.float())


class ProbeModel(nn.Module):
    """Shared trunk usable in either flow space, with a `column` or `patch` front-end.

    The flow variable is ALWAYS the full channelised STFT [B, 2F, T]: the waveform arm
    brackets the core with STFT/iSTFT, the spec arm feeds it directly. The front-end only
    swaps the spectrogram<->token tokeniser INSIDE the core, never the flow variable — so the
    waveform-vs-spec axis is identical across front-ends. Trunk + projections are built under
    a fixed seed so the two arms of a given front-end share an identical init. `patch` ports
    the `square` front-end from probe.py (patch_512x8 = 512 freq × 8 time tiles, 15 tok) with
    an optional JiT input bottleneck (= RESULTS `patch_512x8 + bn16`)."""

    def __init__(self, space: str, frontend: str = "column",
                 patch_f: int = 512, patch_t: int = 8, n_bottleneck: int | None = None):
        super().__init__()
        assert space in ("waveform", "spec")
        assert frontend in ("column", "patch")
        self.space = space
        self.frontend = frontend
        self.sample_rate = SR
        self.stft = STFTConfig.from_dict(STFT)
        f = self.stft.freq_bins  # 1025
        torch.manual_seed(0)  # identical trunk init across arms
        self.time_embed = TimeEmbedding(COND_DIM, time_scale=100.0)
        self.cond_embed = ConditioningEmbedding(COND_DIM, COND_DIM)
        self.cond_combine = ConditioningCombiner(COND_DIM)
        self.blocks = nn.ModuleList(
            [TransformerBlock(DIM, COND_DIM, heads=HEADS, mlp_ratio=8 / 3, rope=True, qk_norm=True)
             for _ in range(DEPTH)]
        )
        if frontend == "column":
            if n_bottleneck:  # JiT input bottleneck 2F→n→DIM, output full-rank (input-only)
                self.in_proj = nn.Sequential(nn.Conv1d(2 * f, n_bottleneck, 1, bias=False),
                                             nn.Conv1d(n_bottleneck, DIM, 1))
            else:
                self.in_proj = nn.Conv1d(2 * f, DIM, 1)
            self.out_proj = nn.Conv1d(DIM, 2 * f, 1)
            self.ntok = self._frames()
        else:  # patch: 2D freq×time tiles + optional JiT input bottleneck
            self.fp, self.tp = patch_f, patch_t
            self.fbins = f
            self.fpad = (-self.fbins) % self.fp
            self.npf = (self.fbins + self.fpad) // self.fp
            feat = 2 * self.fp * self.tp
            if n_bottleneck:
                self.in_proj = nn.Sequential(nn.Linear(feat, n_bottleneck, bias=False),
                                             nn.Linear(n_bottleneck, DIM))
            else:
                self.in_proj = nn.Linear(feat, DIM)
            self.out_proj = nn.Linear(DIM, feat)
            self.ntok = self.npf * (self._tpad_frames() // self.tp)
        self.pos = nn.Parameter(torch.randn(1, self.ntok, DIM) * 0.02)

    def _frames(self) -> int:
        return L // self.stft.hop_length + 1  # center=True

    def _tpad_frames(self) -> int:
        n = self._frames()
        return n + ((-n) % self.tp)

    def _cond(self, t, cond):
        t_embed = self.time_embed(t) if t is not None else None
        cond = self.cond_embed(cond)
        return self.cond_combine(t_embed, cond)

    # ---- patch tokeniser (ports `square` from probe.py): complex spec <-> tokens ----
    def _tok_patch(self, spec):
        b, _c, _f, t = spec.shape
        img = torch.cat([spec.real, spec.imag], dim=1)  # [B,2,F,T]
        img = torch.nn.functional.pad(img, (0, self._tpad_frames() - t, 0, self.fpad))
        npf, npt = self.npf, img.shape[-1] // self.tp
        img = img.view(b, 2, npf, self.fp, npt, self.tp).permute(0, 2, 4, 1, 3, 5)
        return self.in_proj(img.reshape(b, npf * npt, 2 * self.fp * self.tp))

    def _untok_patch(self, h, t):
        b = h.shape[0]
        npt = self._tpad_frames() // self.tp
        y = self.out_proj(h).view(b, self.npf, npt, 2, self.fp, self.tp)
        y = y.permute(0, 3, 1, 4, 2, 5).reshape(b, 2, self.npf * self.fp, npt * self.tp)
        y = y[:, :, : self.fbins, :t]
        return _no_amp_complex(y[:, 0:1], y[:, 1:2])

    def _core(self, spec, cond):  # complex [B,1,F,T] -> complex [B,1,F,T]
        frames = spec.shape[-1]
        if self.frontend == "column":
            h = self.in_proj(complex_to_channels(spec)).transpose(1, 2)
        else:
            h = self._tok_patch(spec)
        h = h + self.pos
        for block in self.blocks:
            h = block(h, cond)
        if self.frontend == "column":
            return channels_to_complex(self.out_proj(h.transpose(1, 2)), 1, self.stft.freq_bins)
        return self._untok_patch(h, frames)

    def forward(self, x, t=None, cond=None, length=None, return_spec=False):
        cond = self._cond(t, cond)
        if self.space == "spec":  # x is the channelised (normalised) spec
            return complex_to_channels(self._core(channels_to_complex(x, 1, self.stft.freq_bins), cond))
        x = as_waveform(x)
        target = int(length or x.shape[-1])
        out = self._core(waveform_to_stft(x, self.stft), cond)  # complex spec
        if return_spec:  # raw (un-iSTFT'd) channelised spec — for the wav-noise / spec-loss arm
            return complex_to_channels(out)
        return stft_to_waveform(out, self.stft, length=target)


# ---------------------------------------------------------------------------
# data / metrics (mirrors experiments/stft_frontends/probe.py)
# ---------------------------------------------------------------------------
def load_clips(nclip, dev):
    cands = []
    for dp, _, fns in os.walk(ROOT):
        for fn in fns:
            if fn.lower().endswith((".wav", ".flac", ".aif", ".aiff")):
                cands.append(os.path.join(dp, fn))
    random.seed(3)  # same seed/clip set as the front-end probe
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


def euler_sample(model, flow, shape, cond, noise, steps, dev):
    """Inline 25-step Euler (mirrors RectifiedFlow.generate, guidance 1.0, no autocast).
    Shape-agnostic: `shape`/`noise` are waveform [1,1,L] or spec [1,2F,T] per the arm."""
    x = noise
    grid = torch.linspace(EPS, 1.0 - EPS, steps + 1, device=dev)
    b = shape[0]
    length = shape[-1] if model.space == "waveform" else None
    for i in range(steps):
        t = grid[i].expand(b)
        dt = grid[i + 1] - grid[i]
        pred = model(x, t=t, cond=cond, length=length)
        x = x + dt * flow.target_to_v(pred, x, t)
    return x


def _a_weight(f):  # linear A-weighting gain at frequencies f (Hz)
    f = f.clamp_min(1.0)
    f2 = f ** 2
    ra = (12194.0 ** 2 * f2 ** 2) / (
        (f2 + 20.6 ** 2) * (f2 + 12194.0 ** 2)
        * torch.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)))
    return 10 ** ((20 * torch.log10(ra) + 2.0) / 20)


def noise_weight(mode, Sc, freq_bins, target_std, dev):
    """Per-bin (per-channel) weight w [1,2F,1] on the DATA spectrogram in spec flow. Noise stays
    N(0,1), so w sets the per-frequency SNR (∝ w·data_std). All modes except 'none' renormalise to
    global std = target_std (the waveform data std), isolating the per-frequency SHAPE:
      none    raw STFT, no renorm (data ≫ noise globally) — tests the global scale.
      const   flat w (= the global-scalar α used so far).
      perbin  w = 1/σ[f]            → flat SNR across frequency (full whitening).
      aweight w = A(f)/σ[f]         → flat SNR in A-weighted perceptual units (boosts mid).
      mel     w = (1/(700+f))/σ[f]  → flat SNR in mel-density units (boosts low/mid)."""
    F = freq_bins
    sigma = torch.cat([Sc[:, :F], Sc[:, F:]], dim=0).std(dim=(0, 2))  # [F] per-bin std (real+imag)
    sigma = sigma.clamp_min(sigma.max() * 1e-3)
    freqs = torch.linspace(0, SR / 2, F, device=dev)
    shape = {"none": torch.ones(F, device=dev),
             "const": torch.ones(F, device=dev),
             "perbin": 1.0 / sigma,
             "aweight": _a_weight(freqs) / sigma,
             "mel": (1.0 / (700.0 + freqs)) / sigma}[mode]
    wc = torch.cat([shape, shape])  # [2F]
    if mode != "none":  # renorm so weighted data has global std = target_std
        wc = wc * (target_std / (Sc * wc[None, :, None]).std())
    return wc[None, :, None]


def run(label, flow_space, loss_space, nclip, C, CONDS, conditioner, dev, steps, B,
        frontend="column", bn=None, scale_mode="const"):
    model = ProbeModel(flow_space, frontend=frontend, n_bottleneck=bn).to(dev)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    flow = RectifiedFlow()
    Cn = torch.nn.functional.normalize(CONDS, dim=1)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.01)
    n = C.shape[0]
    NF = model.stft.freq_bins
    T = model._frames()
    t0 = time.time()

    # spec flow: per-frequency weight on the data spectrogram; noise stays N(0,1) (sets per-freq SNR)
    if flow_space == "spec":
        with torch.no_grad():
            Sc = complex_to_channels(waveform_to_stft(C, model.stft))  # [N,2F,T]
        w_chan = noise_weight(scale_mode, Sc, NF, float(C.std()), dev)

        def to_flow(xw):  # waveform [b,1,L] -> weighted channelised spec [b,2F,T]
            return complex_to_channels(waveform_to_stft(xw, model.stft)) * w_chan
    else:
        w_chan = None

        def to_flow(xw):
            return xw

    def chan_stft(xw):  # waveform -> raw channelised STFT [b,2F,T] (no alpha; loss-scale is Adam-invariant)
        return complex_to_channels(waveform_to_stft(xw, model.stft))

    # identical (idx, t) training sequence across arms via a dedicated generator
    g = torch.Generator(device=dev).manual_seed(1234)
    model.train()
    final_loss = float("nan")
    for _s in range(steps):
        idx = torch.randint(0, n, (B,), device=dev, generator=g)
        t = torch.randn(B, device=dev, generator=g).sigmoid().clamp(EPS, 1 - EPS)
        x1 = to_flow(C[idx])
        x_t, t, x1 = flow.train_tuple(x1, t=t)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            if loss_space == flow_space:  # waveform/waveform or spec/spec
                pred = model(x_t, t=t, cond=CONDS[idx], length=L)
                loss, _ = flow.loss(pred, x1, x_t, t, space="v", loss_type="mse")
            else:  # wav→spec: noise in waveform, loss on the raw spec output vs the spec target
                pred = model(x_t, t=t, cond=CONDS[idx], length=L, return_spec=True)
                loss, _ = flow.loss(pred, chan_stft(x1), chan_stft(x_t), t, space="v", loss_type="mse")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        final_loss = float(loss.detach())

    model.eval()

    def to_wav(flow_x):  # final flow-space tensor -> waveform [1,1,L]
        if flow_space == "spec":
            spec = channels_to_complex(flow_x / w_chan, 1, NF)
            return stft_to_waveform(spec, model.stft, length=L)
        return flow_x

    # FITTING: teacher-forced x1-pred corr (given the TRUE x_t in this arm's space)
    tf_corr = {}
    with torch.no_grad():
        for tval in (0.3, 0.5, 0.8):
            corrs = []
            for i in range(n):
                gi = torch.Generator(device=dev).manual_seed(900 + i)
                x1i = to_flow(C[i:i + 1])
                x0 = torch.randn(x1i.shape, device=dev, generator=gi)
                xt = (1 - tval) * x0 + tval * x1i
                xp = model(xt, t=torch.full((1,), tval, device=dev), cond=CONDS[i:i + 1], length=L)
                a, b = to_wav(xp).flatten(), C[i:i + 1].flatten()
                corrs.append(float((a @ b) / (a.norm() * b.norm() + 1e-9)))
            tf_corr[tval] = float(np.mean(corrs))

    # END-TO-END: sampled output vs true clip
    def embed(a1):
        with torch.no_grad():
            return conditioner(a1, sample_rate=SR, audio_lengths=torch.tensor([a1.shape[-1]], device=dev)).view(-1)

    own, l1s, grms, hits = [], [], [], 0
    with torch.no_grad():
        for i in range(n):
            gi = torch.Generator(device=dev).manual_seed(50 + i)
            shape = (1, 1, L) if flow_space == "waveform" else (1, 2 * NF, T)
            noise = torch.randn(shape, device=dev, generator=gi)
            o = to_wav(euler_sample(model, flow, shape, CONDS[i:i + 1], noise, 25, dev)).clamp(-1, 1)
            l1s.append(logstft_l1(o, C[i:i + 1]))
            eg = torch.nn.functional.normalize(embed(o), dim=0)
            cos = Cn @ eg
            own.append(float(cos[i]))
            if int(cos.argmax()) == i:
                hits += 1
            grms.append(float((o ** 2).mean().sqrt()))
    tgt = float((C ** 2).mean(dim=(-2, -1)).sqrt().mean())
    dt = time.time() - t0
    ascale = f" scale={scale_mode}" if flow_space == "spec" else ""
    print(f"  [{label:>9} {nparams:5.1f}M tok={model.ntok:>3}] "
          f"logSTFT_L1={np.mean(l1s):.3f}  tf_corr(.3/.5/.8)={tf_corr[0.3]:+.2f}/{tf_corr[0.5]:+.2f}/{tf_corr[0.8]:+.2f}  "
          f"own={np.mean(own):.3f}  retr={hits}/{n}  genRMS={np.mean(grms):.3f}(tgt{tgt:.2f})  "
          f"loss={final_loss:.3f}{ascale}  {dt:.0f}s", flush=True)
    return {"arm": label, "logstft_l1": float(np.mean(l1s)), "tf_corr_0.5": tf_corr[0.5],
            "own": float(np.mean(own)), "retr": hits, "genrms": float(np.mean(grms))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--nclips", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--arms", type=str, default="all", help="comma list of arms or 'all'")
    ap.add_argument("--frontend", type=str, default="patch", choices=["column", "patch"])
    ap.add_argument("--bottleneck", type=str, default="16",
                    help="JiT input bottleneck(s), comma list; 0=off. Applies to column or patch.")
    ap.add_argument("--scale", type=str, default="const",
                    help="spec-flow noise scaling(s), comma list: none,const,perbin,aweight,mel.")
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    os.makedirs(CACHE_DIR, exist_ok=True)
    arms = ARMS if args.arms == "all" else [a for a in ARMS if a[0] in args.arms.split(",")]
    fe = args.frontend
    bns = [int(x) or None for x in args.bottleneck.split(",")]
    scales = args.scale.split(",")
    conditioner = build_embedding(
        {"type": "matpac", "checkpoint_path": MATPAC_CKPT, "device": str(dev),
         "use_teacher": False, "encode_batch_size": 0, "compile_encoder": False}, device=dev).to(dev).eval()

    C, CONDS = get_data(args.nclips, conditioner, dev)
    off = torch.nn.functional.normalize(CONDS, dim=1)
    offm = float((off @ off.T)[~torch.eye(args.nclips, dtype=bool, device=dev)].mean())

    for bn in bns:
        for scale in scales:
            base = "patch_512x8" if fe == "patch" else "column"
            fe_label = f"{base} + bn{bn}" if bn else base
            print(f"\n=== n={args.nclips} clips | cond pairwise cos mean={offm:.3f} | steps={args.steps} "
                  f"batch={args.batch} | {fe_label} | spec-noise scale={scale} ===", flush=True)
            results = [run(label, flow_space, loss_space, args.nclips, C, CONDS, conditioner, dev,
                           args.steps, args.batch, fe, bn, scale)
                       for label, flow_space, loss_space in arms]
            print(f"  --- {fe_label} scale={scale}: arm  logSTFT_L1 / tf.5 / own / retr / genRMS ---",
                  flush=True)
            for r in results:
                print(f"      {r['arm']:>9}  {r['logstft_l1']:.3f}  {r['tf_corr_0.5']:+.2f}  "
                      f"{r['own']:.3f}  {r['retr']:>3}  {r['genrms']:.3f}", flush=True)


if __name__ == "__main__":
    main()
