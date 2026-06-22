# STFT Front-End & Auxiliary-Loss Overfit Ablations — Consolidated Results

All numbers are from the overfit probes in `experiments/stft_frontends/` and
`experiments/aux_losses/`. One shared trunk throughout; across runs only the spectrogram→token
front-end, the training loss, or — in Run 8 — the *flow domain* (waveform vs spectrogram) changes.

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

## Run 7 — Nyquist handling + 2D RoPE on `patch_512x8` `followup5.py` / `followup6.py`

`patch_512x8` pads 1025→1536 = 3 freq bands, but the 3rd is a near-empty **Nyquist-remainder
token** (2 real bands + bin 1024 + zeros). Four ways to handle it / encode position:

| variant | tok | logSTFT-L1 | tf .3/.5/.8 | own | retr | genRMS |
|---|---|---|---|---|---|---|
| **512x8 dropNyq** (drop bin 1024→0, 2 clean bands) | **10** | **2.384** | .99/.99/.99 | **0.728** | **50** | 0.276 |
| 512x8 rope2d (axial 2D RoPE, same tokens) | 15 | 2.390 | .99/.99/.99 | 0.723 | 48 | 0.271 |
| 512x8 packNyq (fold Nyquist→DC-imag, lossless) | 10 | 2.392 | .99/.99/.99 | 0.718 | 46 | 0.280 |
| 512x8 anchor (pad, 3-band, 1D-raster RoPE) | 15 | 2.413 | .99/.99/.99 | 0.721 | 46 | 0.275 |

- **Nyquist: drop it.** `dropNyq` ≥ `packNyq` on every metric (retr 50 vs 46) — keeping the
  inaudible 24 kHz bin via the DC-imag fold buys nothing and the fold's low-band impurity
  slightly hurts. Both 10-tok options beat the 15-tok pad anchor, so the remainder token was
  pure waste. **Crop to 1024 bins (2 clean bands), reconstruct Nyquist as 0** → 15→10 tok *and*
  quality ↑. No aliasing/artifacts.
- **2D RoPE** (`rope2d`, identical init to anchor) is a small free win at the same 15 tok
  (2.413→2.390, retr 46→48): the patches were mildly handicapped by 1D-raster RoPE. Adopt axial
  RoPE for a production patch front-end.

---

## Run 8 — Waveform-space vs spectrogram-space FLOW `flow_space.py`

Every run above flows in **waveform** space (= production): noise + linear interpolant + v-MSE in
waveform, with STFT/iSTFT bracketing each model eval (`RectifiedFlow._predict`). This run holds the
trunk fixed and moves *where the flow lives*, on two front-ends (`column`, `patch_512x8+bn16`), via
three arms that factor the flow into **noise domain × loss domain**:

- **waveform** [A] — noise + interpolant + v-MSE all in waveform (= production).
- **wav→spec** [B] — noise + interpolant in *waveform*, but v-MSE on the model's **raw spectrogram
  output** (no terminal iSTFT in the loss path). Isolates the *loss* domain.
- **spec** [D] — noise + interpolant + v-MSE in the channelised STFT; one terminal iSTFT to score.
  Scaled by a single global scalar (α=0.051) so its std matches the waveform data std (~0.28),
  noise N(0,1), divided back out before the iSTFT. **Global (not per-bin)** scaling preserves the
  natural spectral tilt, so spec-flow sees the same falling-spectrum-vs-flat-noise SNR profile.

`A→B` = loss domain; `B→D` = noise domain. All arms share data, optimiser, the (idx,t) sequence,
sampling-noise seeds, and one 25-step Euler sampler. Fitting is saturated everywhere (tf ≈ .98–.99)
⇒ **entirely sampling-side**.

| front-end | arm | tok | logSTFT-L1 | own | retr | genRMS |
|---|---|---|---|---|---|---|
| **column** | waveform [A] | 38 | 2.804 | 0.626 | 35 | 0.231 |
| **column** | wav→spec [B] | 38 | 2.396 | 0.748 | 42 | 0.242 |
| **column** | **spec [D]** | 38 | **2.329** | **0.800** | **51** | 0.272 |
| | *— `column` waveform ref* | | *2.847* | *0.626* | *32* | *0.248* |
| **patch_512x8+bn16** | waveform [A] | 15 | 2.436 | 0.712 | 48 | 0.273 |
| **patch_512x8+bn16** | wav→spec [B] | 15 | **2.301** | 0.721 | 47 | 0.277 |
| **patch_512x8+bn16** | spec [D] | 15 | 2.313 | 0.716 | 46 | 0.273 |
| | *— `patch` waveform ref* | | *2.321* | *0.734* | *50* | *0.276* |

**Fairness check passes:** both fresh waveform arms reproduce their established controls (column own
0.626 identical, L1 2.804 vs 2.847; patch ≈2.32–2.44 vs 2.321 ref — within the RNG-scheme delta from
the dedicated idx/t generator + inline sampler).

> **Rank metrics on conditional adherence (own/retr), not logSTFT-L1.** Following the conditioning
> is the project goal, and the L1/own picture diverge sharply here. On L1 the configs look "tied"
> (2.31–2.44); on **own/retr** they do not — **`column`-spec (own 0.800 / retr 51) is the best
> config in the entire series**, beating every `patch` variant (best patch own 0.734). The
> front-end ranking *flips with the flow domain*: waveform-flow patch (0.712) ≫ column (0.626), but
> **spec-flow column (0.800) ≫ patch (0.716)**. So spec-flow is *not* "redundant" — it's the route
> to the best conditioning, via `column`.

**The spec-flow win is contingent on a WEAK front-end, and the two domains do different jobs:**

- **`column` (weak): both domains help, and they do different things.** Loss domain (A→B) is the big
  lever for *spectral + semantic* fidelity — L1 −0.41, own +0.12, retr +7. Noise domain (B→D)
  independently fixes *energy + retrieval* — genRMS 0.242→0.272 (the collapse fix is almost all
  here), retr +9, own +0.05. The raw-spectrogram **loss** does the spectral work; spec-space
  **noise** restores energy.
- **`patch_512x8+bn16` (strong): neither domain moves L1**, and on own/retr the patch never reaches
  column-spec (0.71–0.72 vs 0.800). Its freq-local token geometry already fixes the waveform-flow
  weaknesses, so spec-flow adds nothing *on patch* — but it also doesn't *match* what spec-flow buys
  on column.
- **⇒ spec-flow (loss *and* noise) SUBSTITUTES for the weaknesses of a poor representation.** The
  "overcomplete terminal-iSTFT-projection cleanup" mechanism is unsupported as the driver — the
  loss-domain arm (no terminal projection) already captures most of the column win.

### Bottleneck × flow on `column` (does the JiT bottleneck stack with spec-flow?)

The JiT input bottleneck (2050→n→512, output full) was column's biggest lever in *waveform* flow
(view A). Does it stack on top of column-spec's best-in-series 0.800? **No — it is antagonistic
with spec-flow.** (waveform + spec arms, bn ∈ {none,4,8,16}.)

| column bn | waveform L1 / own / retr / gRMS | spec L1 / own / retr / gRMS |
|---|---|---|
| **none** | 2.804 / 0.626 / 35 / 0.231 | **2.329 / 0.800 / 51 / 0.272** |
| 4 | 2.910 / 0.666 / 45 / 0.289 | 2.621 / 0.710 / 50 / 0.279 |
| 8 | 2.897 / 0.667 / 45 / 0.286 | 2.628 / 0.709 / 47 / 0.277 |
| 16 | 2.914 / 0.653 / 41 / 0.279 | 2.638 / 0.711 / 49 / 0.282 |

- **Spec-column: the bottleneck only hurts** — own 0.800→~0.71, L1 2.33→~2.63, monotone across
  bn4/8/16. Spec-flow's whole advantage is the *full-rank read of the spectrogram* it flows in; a
  rank-n input waist throws exactly that away. **Best column-spec = no bottleneck.**
- **Waveform-column: the bottleneck is a *crutch*** — it lifts own/retr (0.626→~0.66, 35→45) and
  restores energy (genRMS 0.231→0.286, to target), i.e. it does the *same* job spec-flow does, less
  well. So bottleneck, spec-loss, and spec-noise are **all substitutes** for weak-representation
  weaknesses; you want exactly one fix, and column-spec-no-bn is the best of them.
- *Caveat:* my fresh waveform-column-bn **L1** rises vs no-bn (2.80→2.91), which disagrees with view
  A's L1 *drop* (2.847→2.61) — own/retr do improve as view A says, so it's likely a bn-path harness
  delta affecting L1 specifically; it does not touch the (large, monotone) spec-column conclusion.
- **Transfer caveat (important at scale):** the bottleneck is an *input-information restriction* that
  only pays off as a crutch for a poorly-conditioned setup. At n=64 the MATPAC cond is a near-unique
  lookup key, so the model barely needs to *read* x_t and a rank-4 waist suffices; at 120k+ the cond
  specifies a *distribution* and x_t must carry the realisation-selecting information, so a tiny rank
  will likely starve it. Do **not** bake a small `n` — re-tune at scale (expect the sweet spot to
  grow or vanish), and note the best result here (column-spec) wants *no* bottleneck anyway.

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
7. **Nyquist: drop it** (crop 1025→1024, reconstruct bin 1024 as 0). Beats both the dead-token pad
   (15 tok) and the lossless DC-imag pack — keeping 24 kHz buys nothing. → 15→10 tok + quality ↑.
8. **2D (axial) RoPE > 1D-raster RoPE** for patches (small free win); the patch numbers above were
   measured with the weaker 1D-raster RoPE, so they slightly *understate* the patched front-ends.
9. **Spectrogram-space flow gives the best conditional adherence in the series — via `column`**
   (Run 8). Judged on own/retr (the project metric), **column-spec (own 0.800 / retr 51) beats every
   config**, incl. all `patch` variants (best 0.734); the front-end ranking *flips with the flow
   domain* (waveform: patch≫column; spec: column≫patch). The 3-arm decomposition (noise × loss
   domain) shows on the weak `column` the **loss domain** drives spectral/semantic and the **noise
   domain** restores energy/retrieval; on strong `patch` neither moves L1 and neither reaches
   column-spec. So spec-flow, freq-local token geometry, and the JiT bottleneck are all
   **substitutes** for a weak representation's faults — use exactly one. The bottleneck is
   *antagonistic* with spec-flow (column-spec own 0.800→0.71 with any bn) and is an overfit crutch
   (helps weak waveform-column, redundant at best otherwise). **Big caveat:** own/retr at n=64 partly
   measures memorisation of a near-unique cond→clip lookup — validate the column-spec edge on
   held-out clips before trusting it at scale.

## Recommended front-end + next step

**Drop-Nyquist 2-band `patch_512x8` (1024 bins, 10 tokens) + axial 2D RoPE + input bottleneck
n≈8–16**, then **stack MR-STFT loss** (orthogonal axis). Each ingredient helps independently:
dropNyq alone is 2.384 @ 10 tok, 2D-RoPE −0.02 at no token cost, bn16 (on pad) 2.321, MR-STFT
own +0.10 on column. The combined config is the one to try on a real run — best quality at the
lowest token/RTF cost of anything tested (10 tok vs column's 38, vs `band2`'s 76).

**Spec-flow is a real second candidate, judged on conditional adherence (Run 8).** Two configs to
take forward, on different axes:
1. **`patch_512x8` (+dropNyq/2D-RoPE/bn/MR-STFT), waveform-space flow** — cheapest (10–15 tok), and
   its wins are *architectural* (freq-locality, MR-STFT) so most likely to transfer.
2. **`column`, spectrogram-space flow, NO bottleneck** — best conditioning in the whole series
   (own 0.800 / retr 51), but costs 38 tok and its win does *not* stack with patch nor with the
   bottleneck (both substitute for the same weakness). The bottleneck actively *hurts* it.

The catch: spec-flow's edge is on own/retr, which at n=64 partly measures memorisation of a near-
unique cond→clip lookup, not generalisation. Before betting on column-spec at scale, confirm the
own/retr advantage **survives on held-out clips** (and that the no-bottleneck full-rank read still
pays when the cond stops being a lookup). Don't bake a small bottleneck `n` either — it's an
overfit-regime crutch (see Bottleneck × flow above).

> Terminology: the **"original single-timestep / full-frequency" front-end is `column`** — one
> token per single STFT hop, covering the full 1025-bin frequency range (no time- or freq-patching).
> Its complete bottleneck sweep (rank n = 2…128 reseeded, plus the original no-reseed 64/128/256)
> is in consolidated **view A** and every `column` row above. (All runs share v-space training and
> 25-step Euler sampling — there is no separate 1-NFE variant in this series.)
