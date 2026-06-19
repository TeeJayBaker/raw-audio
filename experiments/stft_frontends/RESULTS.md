# STFT Front-End & Auxiliary-Loss Overfit Ablations — Consolidated Results

All numbers are from the overfit probes in `experiments/stft_frontends/` and
`experiments/aux_losses/`. Six runs, one shared trunk; only the spectrogram→token
front-end (or the training loss) changes.

## Setup

- **Data / fit:** n=64 one-shot clips (seed=3 selection), MATPAC conditioning, 6000 steps,
  batch 24, AdamW lr 2e-4, bf16 AMP, **v-space MSE**, `sigmoid(randn)` timestep distribution.
- **Trunk (fixed):** d512 / 12 layers / 8 heads transformer (= production `stft_transformer`),
  AdaLN-Zero conditioning, learned abs-pos + RoPE. STFT `n_fft=2048` → **1025 freq bins**,
  hop 512; 0.4 s clips → **38 time frames**.
- **Sampling:** 25-step Euler, guidance 1.0. (Same for every run — there is no 1-NFE variant here.)

### Metrics

| metric | meaning | dir |
|---|---|---|
| **logSTFT-L1** | paired multi-resolution log-mag STFT L1, sampled vs true clip — spectral fidelity | ↓ |
| **tf .3/.5/.8** | teacher-forced x1-pred cosine corr given the *true* `x_t` at t=0.3/0.5/0.8 — **FITTING** capacity | ↑ |
| **own** | MATPAC own-cosine: sampled clip re-embedded, cosine to its target conditioning — semantic fidelity (not trained on) | ↑ |
| **retr** | retrieval hits /64 in MATPAC space | ↑ |
| **genRMS** | sampled RMS energy (target **0.28**) | →tgt |
| **tok** | token count = **RTF proxy** (trunk attention is O(tok²)) | ↓ |

> **Fitting is saturated in every run** (tf ≈ 0.98–0.99 everywhere), so all logSTFT-L1 / own /
> retr differences are **sampling-side**, not capacity.

### Methodology note (reseeding)

**Run 1 (original) did NOT reseed training between configs**, so its RNG drifted and its
numbers are noisier (this is why its bottleneck sweep looked flat/non-monotonic — an
artifact, later corrected). **Runs 2–6 reseed to `seed=1234`**, making the front-end/loss the
only variable. The reseeded `column` control = **2.847 / own 0.626**; the original (Run 1)
`column` = 2.835 / 0.621. Run 1's values are kept below but flagged `†` (no-reseed).

---

## Run 1 — Original front-end probe `probe.py` (no reseed `†`)

The first sweep: column control, the original **bottleneck sweep (64/128/256)**, square
(=128f×8t patch), and band2 (freq band-split).

| config | tok | logSTFT-L1 | tf .3/.5/.8 | own | retr | genRMS |
|---|---|---|---|---|---|---|
| band2 `†` | 76 | 2.266 | .95/.98/.99 | 0.719 | 33 | 0.249 |
| square (128f×8t) `†` | 45 | 2.442 | .97/.98/.99 | 0.637 | 33 | 0.262 |
| bneck64 `†` | 38 | 2.761 | .97/.99/.99 | 0.641 | 40 | 0.258 |
| bneck256 `†` | 38 | 2.830 | .95/.98/.99 | 0.627 | 35 | 0.249 |
| column `†` | 38 | 2.835 | .94/.98/.99 | 0.621 | 33 | 0.242 |
| bneck128 `†` | 38 | 2.835 | .96/.99/.99 | 0.637 | 36 | 0.256 |

*Read at the time as "bottleneck ≈ neutral" — RNG noise; corrected by the reseeded sweep below.*

---

## Run 2 — Auxiliary losses on the column front-end `aux_losses/probe.py`

Each aux added alone on top of v-MSE. Weights **calibrated** mid-baseline so each aux's
contribution ≈ MR-STFT@1.0 (measured magnitudes mr=1.35, wavefm=14.3, complex=328.7 →
w_mr=1.0, w_wavefm=0.095, w_complex=0.004). **logSTFT-L1 is `*`home-field** for the spectral
losses — rank on own/retr/genRMS.

| config | weight | own (Δ) | retr | genRMS | tf .3/.5/.8 | logSTFT-L1* |
|---|---|---|---|---|---|---|
| **+MR-STFT** | 1.000 | **0.725** (+0.099) | **46** | 0.257 | .95/.97/.97 | 2.178 |
| +complex-STFT | 0.004 | 0.716 (+0.090) | 43 | 0.230 | .92/.96/.97 | 2.020 |
| +WaveFM | 0.095 | 0.674 (+0.048) | 34 | 0.246 | .94/.97/.97 | 2.320 |
| baseline (no aux) | — | 0.626 | 32 | 0.248 | .95/.98/.99 | 2.847 |

- **MR-STFT best** (validates production `mr_stft_weight=1.0`; also best energy).
- **complex-STFT** close 2nd on semantics, best spectral, but worst energy (0.230).
- **WaveFM weakest** — *confounded*: its magnitude is phase-dominated, so calibrating on the
  bundle total under-weights its log-mag/mel terms.

---

## Run 3 — Reseeded bottleneck sweep + first patch shapes `followup.py`

| config | tok | logSTFT-L1 (Δ vs col) | tf .3/.5/.8 | own | retr | genRMS |
|---|---|---|---|---|---|---|
| band2 (ref, out-of-budget) | 76 | 2.315 (−0.532) | .95/.98/.99 | 0.710 | 34 | 0.238 |
| patch_ft_256x4 | 50 | 2.377 (−0.470) | .97/.99/.99 | 0.681 | 35 | 0.263 |
| patch_ff_64x16 | 51 | 2.436 (−0.411) | .97/.98/.99 | 0.600 | 24 | 0.218 |
| patch_bal_128x8 | 45 | 2.437 (−0.410) | .97/.98/.99 | 0.648 | 32 | 0.254 |
| patch_strip_128xfullT | 9 | 2.492 (−0.355) | ~.99/.99 | 0.668 | 30 | 0.266 |
| bneck16 | 38 | 2.641 (−0.205) | .99/.99/.99 | 0.683 | 44 | 0.274 |
| bneck32 | 38 | 2.697 (−0.150) | .98/.99/.99 | 0.667 | 41 | 0.264 |
| bneck48 | 38 | 2.749 (−0.098) | .97/.99/.99 | 0.649 | 38 | 0.248 |
| bneck64 | 38 | 2.778 (−0.069) | .97/.99/.99 | 0.648 | 39 | 0.257 |
| bneck128 | 38 | 2.796 (−0.051) | .96/.99/.99 | 0.629 | 36 | 0.244 |
| column (control) | 38 | 2.847 (0.000) | .95/.98/.99 | 0.626 | 32 | 0.248 |

*Clean monotonic bottleneck (tighter=better); `patch_strip` later found length-invalid (its
full-time patch ties the projection to clip length).*

---

## Run 4 — Bottleneck floor + length-robust patches `followup2.py`

| config | tok | logSTFT-L1 | tf .3/.5/.8 | own | retr | genRMS |
|---|---|---|---|---|---|---|
| patch_256x4 | 50 | 2.377 | .97/.99/.99 | 0.681 | 35 | 0.263 |
| patch_512x4 | 30 | 2.442 | .98/.99/.99 | 0.700 | 43 | 0.278 |
| patch_512x2 | 57 | 2.457 | .97/.99/.99 | 0.677 | 37 | 0.245 |
| bneck12 | 38 | 2.608 | .99/.99/.99 | 0.697 | 44 | 0.276 |
| bneck8 | 38 | 2.614 | .99/.99/.99 | 0.698 | 47 | 0.279 |
| bneck4 | 38 | 2.639 | .99/.99/.99 | 0.705 | 45 | 0.285 |
| bneck16 | 38 | 2.641 | .99/.99/.99 | 0.683 | 44 | 0.274 |
| bneck2 | 38 | 2.655 | .99/.99/.99 | 0.700 | 49 | 0.281 |
| column | 38 | 2.847 | .95/.98/.99 | 0.626 | 32 | 0.248 |
| patch_full_t4 (1 band) | 10 | 3.125 | .96/.99/.99 | 0.572 | 29 | 0.260 |
| patch_full_t8 (1 band) | 5 | 3.180 | .96/.99/.99 | 0.531 | 24 | 0.254 |

*Full-frequency (1-band) patches are **worse than column** → frequency locality is essential;
this explains `patch_strip` (its win was its bands, not coarse time).*

---

## Run 5 — Cheap freq-banded patches + first patch×bottleneck `followup3.py`

| config | tok | logSTFT-L1 | tf .3/.5/.8 | own | retr | genRMS |
|---|---|---|---|---|---|---|
| patch_256x8 | 25 | 2.372 | .98/.99/.99 | 0.699 | 41 | 0.267 |
| patch_256x4 | 50 | 2.377 | .97/.99/.99 | 0.681 | 35 | 0.263 |
| patch_512x8 | 15 | 2.413 | .99/.99/.99 | 0.721 | 46 | 0.275 |
| patch_512x4 | 30 | 2.442 | .98/.99/.99 | 0.700 | 43 | 0.278 |
| patch_512x4 + bn8 | 30 | 2.472 | .98/.99/.99 | 0.703 | 45 | 0.261 |
| patch_256x4 + bn8 | 50 | 2.501 | .98/.98/.99 | 0.695 | 42 | 0.266 |
| patch_512x4 + bn4 | 30 | 2.519 | .98/.98/.99 | 0.705 | 44 | 0.283 |
| bneck8 | 38 | 2.614 | .99/.99/.99 | 0.698 | 47 | 0.279 |
| column | 38 | 2.847 | .95/.98/.99 | 0.626 | 32 | 0.248 |

*tp8 beats tp4 with bands; bottleneck on tp4 patches looked neutral/harmful here — refined by Run 6.*

---

## Run 6 — Patch × bottleneck matrix `followup4.py` (GPU 1; cross-GPU anchor matched 2.413/0.721 exactly)

| config | tok | logSTFT-L1 | tf .3/.5/.8 | own | retr | genRMS |
|---|---|---|---|---|---|---|
| **patch_512x8 + bn16** | 15 | **2.321** | .99/.99/.99 | 0.734 | 50 | 0.276 |
| patch_512x8 + bn4 | 15 | 2.348 | .99/.99/.99 | **0.735** | 49 | 0.274 |
| patch_512x8 + bn64 | 15 | 2.351 | .99/.99/.99 | 0.721 | 49 | 0.265 |
| patch_512x8 + bn32 | 15 | 2.364 | .99/.99/.99 | 0.724 | 50 | 0.278 |
| patch_256x8 + bn16 | 25 | 2.365 | .99/.99/.99 | 0.719 | 48 | 0.279 |
| patch_512x8 + bn128 | 15 | 2.381 | .99/.99/1.00 | 0.718 | 48 | 0.268 |
| patch_512x8 + bn8 | 15 | 2.386 | .99/.99/.99 | 0.727 | 50 | 0.281 |
| patch_256x8 + bn128 | 25 | 2.389 | .99/.99/.99 | 0.703 | 43 | 0.280 |
| patch_512x8 (no bn, anchor) | 15 | 2.413 | .99/.99/.99 | 0.721 | 46 | 0.275 |
| patch_256x8 + bn32 | 25 | 2.413 | .99/.99/.99 | 0.697 | 42 | 0.274 |
| patch_256x8 + bn64 | 25 | 2.433 | .99/.99/.99 | 0.693 | 39 | 0.277 |
| patch_512x4 + bn16 | 30 | 2.434 | .99/.99/.99 | 0.714 | 47 | 0.285 |
| patch_512x4 + bn32 | 30 | 2.442 | .98/.99/.99 | 0.701 | 44 | 0.264 |
| patch_512x4 + bn128 | 30 | 2.456 | .98/.99/.99 | 0.699 | 41 | 0.279 |
| patch_512x4 + bn64 | 30 | 2.464 | .98/.99/.99 | 0.683 | 39 | 0.267 |
| patch_256x8 + bn4 | 25 | 2.474 | .98/.99/.99 | 0.716 | 44 | 0.261 |
| patch_256x8 + bn8 | 25 | 2.513 | .98/.99/.99 | 0.704 | 44 | 0.280 |

---

# Consolidated views

## A. Bottleneck on the column front-end (logSTFT-L1 ↓ vs rank n)

**`column` = the original representation: one token per single STFT hop, spanning the full
1025-bin frequency range** (`in_proj` Conv1d 2 × 1025 = 2050 → 512, 38 tokens). The bottleneck
factorises that 2050→512 input projection through a rank-`n` waist (2050→n→512, output full).
Reseeded (seed 1234) curve is the canonical one; original (Run 1, no reseed) shown for comparison.

| n (rank) | logSTFT-L1 | own | retr | genRMS | | original `†` (no reseed) |
|---|---|---|---|---|---|---|
| **none (column 2050→512)** | 2.847 | 0.626 | 32 | 0.248 | | 2.835 / 0.621 |
| 128 | 2.796 | 0.629 | 36 | 0.244 | | 2.835 / 0.637 |
| 64 | 2.778 | 0.648 | 39 | 0.257 | | 2.761 / 0.641 |
| 48 | 2.749 | 0.649 | 38 | 0.248 | | — |
| 32 | 2.697 | 0.667 | 41 | 0.264 | | — |
| 16 | 2.641 | 0.683 | 44 | 0.274 | | — |
| 12 | **2.608** | 0.697 | 44 | 0.276 | | — |
| 8 | 2.614 | 0.698 | 47 | 0.279 | | — |
| 4 | 2.639 | **0.705** | 45 | **0.285** | | — |
| 2 | 2.655 | 0.700 | **49** | 0.281 | | — |
| 256 | — | — | — | — | | 2.830 / 0.627 |

**Soft basin:** spectral L1 bottoms at n≈8–12; semantic metrics keep improving as it tightens
(own peaks n4, retr peaks n2). Even rank-2 doesn't collapse. Sweet spot **n≈4–8**. (Original
no-reseed numbers made it look flat — RNG artifact.)

## B. Patch geometries (no bottleneck, reseeded), sorted by logSTFT-L1

| patch | bands×time | tok | logSTFT-L1 | own | retr | genRMS |
|---|---|---|---|---|---|---|
| **patch_256x8** | 4×5 | 25 | **2.372** | 0.699 | 41 | 0.267 |
| patch_256x4 (=patch_ft) | 4×10 | 50 | 2.377 | 0.681 | 35 | 0.263 |
| patch_512x8 | 2×5 | 15 | 2.413 | **0.721** | 46 | 0.275 |
| patch_ff_64x16 | 16×3 | 51 | 2.436 | 0.600 | 24 | 0.218 |
| patch_bal_128x8 (=square) | 8×5 | 45 | 2.437 | 0.648 | 32 | 0.254 |
| patch_512x4 | 2×10 | 30 | 2.442 | 0.700 | 43 | 0.278 |
| patch_512x2 | 2×19 | 57 | 2.457 | 0.677 | 37 | 0.245 |
| patch_strip (length-invalid) | 8×1 | 9 | 2.492 | 0.668 | 30 | 0.266 |
| patch_full_t4 (1 band) | 1×10 | 10 | 3.125 | 0.572 | 29 | 0.260 |
| patch_full_t8 (1 band) | 1×5 | 5 | 3.180 | 0.531 | 24 | 0.254 |
| *band2 (separate-proj bands, ref)* | *2 tok/frame* | *76* | *2.315* | *0.710* | *34* | *0.238* |

(Band counts are *effective* real bands: patch_512 pads 1025→1536 = 2 full bands + a near-empty
Nyquist remainder token; patch_256 = 4 real bands + remainder.)

## C. Patch × bottleneck matrix (logSTFT-L1 / own)

**Bold = beats that patch's no-bottleneck baseline.**

| front-end (tok) | none | bn4 | bn8 | bn16 | bn32 | bn64 | bn128 |
|---|---|---|---|---|---|---|---|
| column / no-patch (38) | 2.847 / .626 | **2.639 / .705** | **2.614 / .698** | **2.641 / .683** | **2.697 / .667** | **2.778 / .648** | **2.796 / .629** |
| **512×8 (15)** | 2.413 / .721 | **2.348 / .735** | **2.386 / .727** | **2.321 / .734** | **2.364 / .724** | **2.351 / .721** | **2.381 / .718** |
| 256×8 (25) | 2.372 / .699 | 2.474 / .716 | 2.513 / .704 | 2.365 / .719 | 2.413 / .697 | 2.433 / .693 | 2.389 / .703 |
| 512×4 (30) | 2.442 / .700 | 2.519 / .705 | 2.472 / .703 | 2.434 / .714 | 2.442 / .701 | 2.464 / .683 | 2.456 / .699 |
| 256×4 (50) | 2.377 / .681 | — | 2.501 / .695 | — | — | — | — |

*(column's bn columns are sourced from Runs 3–4; every cell beats its `none` baseline. For
column, tighter is better — best ≈ bn8, degrading toward bn128.)*

The bottleneck helps the **broad / redundant-token** front-ends — `column` (each token = a
full-frequency frame, 2050-dim) and `512×8` (each = a big 2-band × 8-frame chunk, 8192-dim) —
where the per-token input is highly correlated, so a low-rank waist strips redundancy and
regularizes. It is neutral-to-harmful for **compact, localized** patches (`256×8`, `512×4`,
`256×4`). Note this is **not** simply raw input dim: `column` (2050-dim) gains a lot while
`256×4` (2048-dim) is hurt — what matters is per-token *redundancy*, not size. Among the
patches, the bottleneck stacks only on `512×8`.

## D. Top configs overall (sorted by logSTFT-L1)

| rank | config | tok | logSTFT-L1 | own | retr | genRMS |
|---|---|---|---|---|---|---|
| 1 | **patch_512x8 + bn16** | **15** | **2.321** | 0.734 | 50 | 0.276 |
| 2 | patch_512x8 + bn4 | 15 | 2.348 | **0.735** | 49 | 0.274 |
| 3 | patch_512x8 + bn64 | 15 | 2.351 | 0.721 | 49 | 0.265 |
| 4 | patch_256x8 + bn16 | 25 | 2.365 | 0.719 | 48 | 0.279 |
| 5 | patch_256x8 (no bn) | 25 | 2.372 | 0.699 | 41 | 0.267 |
| — | *band2 (76 tok ref)* | *76* | *2.315* | *0.710* | *34* | *0.238* |
| — | column (control) | 38 | 2.847 | 0.626 | 32 | 0.248 |

---

# Key findings

1. **Everything is sampling-side** — fitting (tf_corr) is saturated in all 6 runs; front-end /
   loss changes move how the velocity field *integrates*, not what it can fit.
2. **Frequency locality is essential.** ~2–5 real freq bands win; 1 band (full-freq) is *worse
   than column*; 16 bands (`patch_ff`) also bad. This explains `patch_strip` (its bands, not
   coarse time, were the win — and its full-time patch is length-invalid, so unusable).
3. **Coarser time helps with bands** (tp8 ≥ tp4 ≥ tp2) and is cheaper.
4. **Bottleneck (JiT) is a real lever, monotone-ish, soft basin (n≈4–8).** Original "neutral"
   read was RNG noise. On the *column* front-end it's the main quality lever (even rank-2 works).
5. **Bottleneck × patch stacks only when the patch projection is high-dim** (512×8, 8192-dim),
   not for moderate-dim patches.
6. **Aux losses help (sampling-side):** MR-STFT best (keep at 1.0), complex-STFT close 2nd
   (watch energy), WaveFM inconclusive (phase-confounded calibration). Orthogonal to the front-end.

## Recommended front-end + next step

**`patch_512x8` (2 real bands × time-patch 8, 15 tokens) + input bottleneck n≈8–16 on its
projection** → logSTFT-L1 ≈ 2.32, own ≈ 0.73, retr 50 — the best quality **and** the cheapest
(15 tok vs column 38; matches `band2`'s spectral score at 1/5 the tokens and beats its
own-cosine). Then **stack with MR-STFT loss** (orthogonal axis) on a real run. Cleanup worth
doing: drop the wasted near-empty Nyquist-remainder band token (`patch_512` is really 2 bands).

> Terminology: the **"original single-timestep / full-frequency" front-end is `column`** — one
> token per single STFT hop, covering the full 1025-bin frequency range (no time- or freq-patching).
> Its complete bottleneck sweep (rank n = 2…128 reseeded, plus the original no-reseed 64/128/256)
> is in consolidated **view A** and every `column` row above. (All runs share v-space training and
> 25-step Euler sampling — there is no separate 1-NFE variant in this series.)
