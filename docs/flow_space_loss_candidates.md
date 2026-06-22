# Flow-space & spectral-loss implementation candidates

Handoff note for an implementation session. These came out of a research thread on the **"phasey at hop 512" artifact**, **waveform-vs-complex-STFT flow space**, and **aux-loss behaviour**. Full reasoning lives in the auto-loaded memory: `spec_flow_phasey_verdict`, `flow_space_principled_verdict`, `aux_loss_flow_conflict`, `weekend_runs_metric_gaps`, `wandb_metrics_access`.

**Cross-cutting context (read first):**
- The backbone is always spectrogram→spectrogram; STFT n_fft=2048, hop=512 → **~4× overcomplete** (valid spectrograms occupy ~25% of the real+imag space). This overcompleteness is the root of the phasey artifact and of the spec-flow degeneracy.
- **All current metrics (MR-STFT log-mag, CLAP MMD/`mind`, MATPAC `embedding_cosine`) are phase-blind**, and genRMS/retrieval aren't logged at all → the phasey artifact is currently invisible. Adding a phase metric is prerequisite to evaluating any of the losses below.
- **n=64 overfit probes do NOT predict scale** (they mispredicted both spec-flow and aux-loss vs the real runs). Validate any of these on real-scale wandb runs, not just probes.

---

## 1. Consistency residual

**What:** how far the model's predicted complex spectrogram `S` is from a *valid* (real-signal) STFT — the direct fingerprint of the phasey / hop-rate (93.75 Hz) AM artifact.
`resid = ‖S − STFT(iSTFT(S))‖ / ‖S‖` (relative complex L1/L2). Exactly 0 for any consistent STFT; grows with cross-frame phase incoherence. **Reference-free** (no target needed).

**Two uses — same quantity, different role:**
- **(a) Eval metric** on *sampled* clips → makes the phasey artifact visible. This is the cheapest, most diagnostic phase metric; add it to `src/validation.py`.
- **(b) Training regularizer** = the "STFT overlap/consistency loss" (see §3).

**Implementation:** compute on the model's **raw channelised spec output (pre-iSTFT)**, NOT the output waveform (a real waveform's STFT is trivially consistent → residual ≈ 0, useless). Use `backbone.io`: `channels_to_complex → stft_to_waveform (iSTFT) → waveform_to_stft (re-STFT)` then relative complex L1. **Working reference impl:** `experiments/aux_losses/aux_tmin_sweep.py::consistency_residual()`.

**Gotcha:** for the column waveform-flow model this residual is high (~0.9) across the board — column predicts very inconsistent spectrograms. It's a relative/comparative metric.

---

## 2. Complex-STFT L1 (already in code)

**What:** paired complex (real+imag) L1 between the model's predicted spectrogram and `STFT(target)` at the model resolution. Penalises **phase** (unlike magnitude/log-mag losses, which are phase-blind). **Already implemented:** `losses.audio.complex_stft_loss` (with Flow2GAN energy-inverse weighting `1/√(|S|²+ε)`, clamped). Wired as an aux via `trainer._aux_losses`, gated by `aux_t_min`; weekend runs used weight 0.005 and it's part of the aux bundle that **helps at scale**.

**Distinction from §1 (important — don't conflate):**
- complex_stft_loss is **referenced** (needs the target `x1`; teacher-forced training loss) and conflates *wrong phase* with *inconsistent phase*.
- The consistency residual (§1) is **reference-free** and isolates *only* inconsistency — usable on free-running samples where there is no target.
They are complementary, not substitutes.

---

## 3. STFT consistency / overlap loss — and the tight-frame question

**What:** the consistency residual (§1) used as a **training term**: penalise `‖S − STFT(iSTFT(S))‖` on the model's raw predicted spectrogram. Pushes the model to emit *valid* STFTs even at the off-manifold `x_t` it drifts through during sampling. This is RFWave's "overlap loss" and is the **direct fix for the phasey artifact**, and the ingredient that makes complex-STFT-space flow viable. Reference-free, so it constrains the output everywhere (not just at the training target, which is what complex_stft_loss does).

**Why waveform-flow needs it less:** waveform-flow's per-eval `iSTFT→STFT` implicitly re-projects onto the consistent subspace each step; spec-flow has only a terminal iSTFT, so it needs this loss explicitly.

**Tight-frame question — does `AᵀA = cI` make the overlap loss redundant? → NO.**
`AᵀA = cI` (analysis is a Parseval isometry on the *signal* side) only fixes the loss-space conditioning and makes synthesis a scaled adjoint. But a tight frame is **still overcomplete**: on the *coefficient* side `AAᵀ = c·P` where `P` is the orthogonal projection onto `range(A)` — still `≠ I`. So a consistent subspace still exists and the model can still emit off-subspace (inconsistent) spectrograms; the overlap loss penalises exactly that component, which the tight-frame property does **not** remove.
It is redundant **only** for a **critically-sampled / orthonormal** transform (square `A`, `AAᵀ = I`, e.g. **MDCT**): no overcompleteness → every coefficient vector is a valid signal → consistency residual ≡ 0 → overlap loss vacuous. But MDCT is a different basis (real-valued, DCT-IV), would break the tuned freq-local-patch / 2D-RoPE wins, and empirically underperforms STFT for generation. **So: keep overcomplete STFT + overlap loss, OR go orthonormal MDCT (no overlap loss) — a tight-frame STFT is not a middle ground that removes the loss.**

---

## 4. Magnitude compression `c̃ = β·|c|^α·e^{i∠c}` (α≈0.5, β≈0.15)

**What:** SGMSE-style amplitude compression of the complex STFT **before** flow noise is added: raise magnitude to `α<1` (≈0.5), scale by `β` (≈0.15), **keep phase**. Compresses the heavy-tailed magnitude / steep spectral tilt **without flattening it** (partial whitening), so high-freq bins aren't SNR-starved under flat `N(0,1)` noise. Invert (`|·|^(1/α)`, ÷β) before the iSTFT.

**Where:** a polar map inside spec-flow's `to_flow` / `from_flow` (`src/flow/fm.py`) — apply after STFT / before the std-match scaling + noise; invert after de-scaling / before iSTFT.

**Why this and not full whitening:** this is the *correct* version of the "the spec data isn't whitened before noise" instinct. **Full per-bin whitening over-flattens** the tilt → amplifies HF hiss and double-counts MR-STFT (which already up-weights HF). Compression (α≈0.5) is the mild middle. This is the main predicted lever for *why the global-α spec-flow attempt underperformed*.

**Note:** the codebase already has `noise_weight(mode=perbin/aweight/mel/...)` in `experiments/stft_frontends/flow_space.py` as the *full per-bin whitening* alternative — compression is the milder, recommended option; they're on the same axis (don't stack naively). Not yet in `src/`.

**Scope — this is a flow-VARIABLE transform, NOT an internal model feature.** It only does its job when applied to the thing that is noised + scored, so it belongs in spec-flow (or a whitened-waveform flow). Under **plain waveform-flow** (noise + loss in the waveform) it does **not** apply: there the STFT is only an internal backbone feature, and a fixed invertible compress→backbone→decompress leaves the flow target and the noise geometry unchanged (the noise is already in the waveform). The per-frequency SNR starvation *does* still exist in waveform-flow (colored data + white noise → HF buried first), but the fix is either (a) **whiten/compress the waveform around the flow** (a pre/post filter — but that *changes the flow space*, so it's no longer "plain waveform"), or (b) **per-frequency loss weighting** (MR-STFT already does some). Internal STFT compression can't fix it.

---

## 5. EDM preconditioning (`c_in`, `c_out`, `c_skip`)

**What:** Karras et al. preconditioning: scale the network **input** to unit variance at every noise level — `c_in = 1/√(σ² + σ_data²)` — and parametrise the **output** via `c_out`/`c_skip` so the regression target keeps consistent magnitude. Decouples "scale" from "task"; fixes the per-noise-level (t-axis) input conditioning.

**Where:** at the model boundary (input scaling as a function of `t`; output recombination). Applies whether the flow is waveform or spec.

**Relationship to §4 (orthogonal, stackable):** compression fixes the per-**frequency** (f-axis) SNR tilt; EDM `c_in` fixes the per-**noise-level** (t-axis) input scale. Note the current parametrisation is v-space / x-prediction with a `1/(1−t)²` weight (`RectifiedFlow`); EDM is an alternative boundary parametrisation of the same thing.

**Caveat — LOW value under Mean Flow / 1-NFE (deprioritise):** EDM precond's payoff is a *training-time* conditioning fix across a *wide* noise-level range. Here that range is tiny: FM input variance `≈ (1−t)²σ_noise² + t²σ_data²` spans only ~13× over `t∈[0,1]` with `rms_lift=false`, and just ~2× once data std is scale-matched to noise (vs diffusion's orders-of-magnitude σ). The dominant scale fix is therefore a **single global data↔noise std match** (`wav_scale`/`rms_lift` — already a knob), which makes per-`t` `c_in` largely redundant; the transformer's RMSNorm absorbs the rest. MF *training* does still span all `t` (the MeanFlow identity needs interior `t` + a `du/dt` JVP), so it's not literally "no noise levels" — but MF *1-NFE inference* queries only the noise input (a single `t`), where `c_in` is a constant rescale = no-op. And EDM's `c_out`/`c_skip` are derived for the *denoising* target — they don't transfer to MF's average-velocity target (whose scale the adaptive `mf_loss(p,c)` already partly normalises), and making them `t`-dependent adds terms to the MF JVP. **Net: skip EDM precond for MF; ensure data≈noise std globally, and spend effort on the f-axis lever (§4), not the t-axis.**

---

## Suggested priority / relationships

1. **Add the consistency residual + complex-STFT L1 as eval metrics first** (§1 eval, §2) — without a phase metric, none of the loss changes are measurable (everything else is phase-blind).
2. **STFT overlap/consistency loss (§3)** is the direct training fix for "phasey" and is largely flow-space-agnostic.
3. **If pursuing complex-STFT-space flow:** it needs **both** §4 (compression) and §3 (overlap loss) to be competitive (the unwhitened, no-overlap global-α attempt is the worst case). Otherwise keep waveform-flow + §5 (`c_in`) + a per-frequency loss weighting.
4. **§4 and §5 are orthogonal** (f-axis vs t-axis) and stack.
5. §1-train and §3 are the **same quantity**; implement once, use as metric and (optionally) loss.

**Validate on real-scale wandb runs, not n=64 probes** (pull via `wandb.Api()`, project `tom-baker/raw-audio` — see `wandb_metrics_access`). Keep the existing aux bundle (MR-STFT@1.0 + complex@0.005) — at scale it's a clear win.
