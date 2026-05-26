# Evaluation

**Purpose:** track what we measure generative quality with, and increasingly what we *train against*, for raw-sample-space audio. Image FID informs audio FAD; recent work attacks FID along three independent axes:

1. **Embedding saturation** — Inception/VGGish are old classifier features; SOTA generators now score below the val FID. Fix: multi-backbone FDr^k, FD-as-loss with modern φ's.
2. **Gaussian closed-form rank deficiency / hackability** — sliced-Wasserstein (MIND) is a proper distance and ~100× cheaper than FID.
3. **Single-metric fragility** — no single fidelity-or-diversity metric passes all sanity checks ([Räisä 2025](../../papers/2505.22450.md)). Fix: report a bundle with independent failure modes, use the open-sourced sanity suite per embedding before trusting any single number.

The three fixes are orthogonal and stack. Recommended audio bundle: **FAD-multi-backbone + Density&Coverage + Vendi (+ CCDS if conditional) + per-embedding sanity-pass rate**.

## What FID is and what's wrong with it

A representation φ maps a sample to a fixed-d vector; FID models real and generated as multivariate Gaussians in φ-space:

  FID = ‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2 (Σ_r Σ_g)^{1/2})

Three structural problems:

| Problem | Why it bites | Mitigation |
|---|---|---|
| **Statistical cost** — empirical Σ is rank-deficient when n ≤ d; canonical FID needs ~50k samples for d=2048 Inception (~10k for d=128 VGGish). | Per-epoch eval on a CPU-target model is unaffordable at 50k; FAD usually only at endpoints. | **MIND** — sliced-W estimator with dimension-free sample complexity; ~5k samples gives stable rankings. |
| **Not a proper distance.** FID = 0 only requires matching first two moments; degenerate discrete distributions on 2d atoms hit FID = 0. | Moment-matching attacks lower FID without perceptual improvement (FID retains 11% under attack, μFID 2.6%). | **MIND** is a proper distance (sliced-W is); retains 31% under the same attack. MMD and Sinkhorn divergence also proper but heavier. |
| **Embedding saturation.** Inception-v3 / VGGish are old classifier features; SOTA generators score below val FID; different φ's rank quality by 1–2 OOM. | We over-optimise to whatever φ encodes; same risk on FAD. | **FDr^k** — average normalised FD across K modern backbones; val = 1.0 by construction. **FD-as-loss** lets us train against any φ. |

## Metric families

### Fidelity / distribution-match

| Instance | Backbone φ | Form | Proper dist | Sample size | Domain |
|---|---|---|---|---|---|
| FID | Inception-v3 (d=2048) | Gaussian 2-W | No | ~50k | image |
| FAD | VGGish (d=128) on log-mel | Gaussian 2-W | No | ~10k | audio |
| FD-DINOv2 / FD-MAE / FD-SigLIP / FD-CLIP | modern SSL / VL encoders | Gaussian 2-W | No | ~50k | image |
| **FDr^k** | K-backbone normalised ratio averaged | Gaussian 2-W per φ | No | ~50k | image — val = 1.0 by construction |
| **MIND** | any | Sliced 1D W via sorting | Yes | ~5k | φ-agnostic — ~100× faster, 3× more attack-resistant |
| MMD (Gaussian / Laplace) | any | n² kernel matrix | Yes | mid | both — bandwidth σ hyperparameter |
| Sinkhorn divergence | any | entropic OT | Yes (limit) | mid | both — ε hyperparameter |
| **Signature-kernel scoring rule** ([2510.19110](../../papers/2510.19110.md)) | any path-valued | path-signature kernel via Goursat PDE, `O(l²)` mem | Yes (strictly proper) | low (path-paired) | sequence / audio-native; matches cross-time / cross-frequency structure no `μ + Σ` encodes |

### Diversity / mode-coverage

| Instance | Reference? | Mechanism | Audio kernel | Catches |
|---|---|---|---|---|
| **Density & Coverage** ([2002.09797](../../papers/2002.09797.md)) | Yes (real set) | k-NN manifold; Coverage = fraction of *real* points with ≥1 fake in their k-NN ball | Random-init CNN features (§4.2: R64 beats T4096 on SC09 spectrograms) | Outlier-robust coverage. High-Density / low-Coverage = adversarial sharpness-but-narrow |
| **Vendi Score** ([2210.02410](../../papers/2210.02410.md)) | No — reference-free | `exp(-Σ λᵢ log λᵢ)` over eigenvalues of K/n; von Neumann entropy of similarity matrix | CLAP / MERT / AudioMAE / raw-Laplace | Tail diversity (small λᵢ contribute log-weighted). Runnable during training |
| **CCDS** ([2505.08175](../../papers/2505.08175.md), ARC) | No (within-batch) | Same-prompt within-batch CLAP cosine distance | CLAP | Conditional collapse (CFG-style mode pinning) |

**Diagnostic signature of adversarial diversity collapse: FAD↓ + Vendi↓.** Fidelity improves while effective sample count drops — instrument on any 1-NFE audio adversarial pipeline.

**Closed-form expectations under P=Q** (D&C): `E[density] = 1`, `E[coverage] → 1 − 1/2^k`. Pick k so `E[coverage] ≥ 0.95` — k=5 with N=M=10k is the principled default. P&R has no such closed-form.

### Meta-evaluation: every metric is flawed

[Räisä 2025 (2505.22450)](../../papers/2505.22450.md) sits over every metric above. Six desiderata + 17 named sanity-check constructions run programmatically — every popular fidelity/diversity metric fails a large fraction (FID, KID, I-Prec/Rec, Density/Coverage, IAP/IBR, P-Prec/Rec, symPrec/Rec).

Three operational consequences:

1. **No single audio metric is safe.** Adopt a bundle with *independent* failure modes; when the bundle agrees, trust the verdict; when it disagrees, that's the diagnostic.
2. **No absolute thresholds** — D4 (clear bounds) fails for nearly every metric. Every claim relative, never absolute.
3. **Run Räisä's sanity-check suite per audio embedding** before fixing a headline metric. The library is domain-agnostic (Gaussians, GMMs, hypercubes, hyperspheres, sphere-vs-torus, mode-drop sequences, Pareto-outlier injection, scaled-one-dim) — applies directly in audio embedding space.

## Three orthogonal axes of fix

1. **Replace Gaussian closed form with sliced-Wasserstein** (MIND). Keep the embedding; swap `tr(Σ_X + Σ_Y - 2(Σ_Y Σ_X)^{1/2})` for averaged 1D-sorted distance over M random projections. Properness, lower sample cost, lower compute, attack resistance. Doesn't touch embedding saturation.
2. **Replace / augment the embedding** (FDr^k, FD-as-loss, multi-backbone stacks). Keep Gaussian form; swap Inception/VGGish for modern SSL/VL backbones, average over several. Less saturated, more perceptually aligned. Doesn't touch rank/hackability.
3. **Drop the embedding entirely; use a path-native kernel** (signature kernel). Skip "encode then compare in φ-space" for short clips. Strictly proper scoring rule over the raw waveform path; captures structural moments no `μ + Σ` summary encodes. `O(l²)` Goursat memory caps practical length at ~1–2 s @ 22 kHz on a 4090.

These compose. **MIND on top of a multi-backbone audio embedding stack** (sliced-W on each of {VGGish, CLAP, AudioMAE}, then averaged) is the natural metric for clips beyond ~2 s. **For short clips (0.1–1 s) signature-kernel direct-on-waveform** is the cleanest fallback — no encoder choice, strictly proper, native to path structure.

## FD as a loss, not only an evaluator

Until 2026 FD was eval-only because reliable estimation requires ~50k samples that swamp any training batch. **[FD-loss (2604.28190)](../../papers/2604.28190.md)** breaks this: decouple the population used to *estimate* moments from the batch carrying *gradients* via either (a) a MoCo-style queue of recent generated features or (b) EMA on first/second moments. Backprop only through the current ~1k batch.

Consequence: a one-step generator post-trained with FD-loss reaches **0.72 FID on ImageNet 256² @ 1 NFE** — beating multi-step baselines. Multi-step generators can be repurposed to 1-NFE without teacher distillation, GAN training, or per-sample targets.

MIND's sliced-W form is also differentiable so the same "loss not just metric" path is open — though FD-loss has stronger empirical evidence for use as a loss surface.

**MMD-as-training-loss in raw sample space.** [IMM (2503.07565)](../../papers/2503.07565.md) uses MMD between distributions of raw samples themselves (not features) as the training objective. Same statistical machine as FD-loss (kernel MMD with CPD kernels matches all moments; FD-with-Gaussian matches only first two) at a different layer: IMM's MMD is in sample space between the model's prediction at two starting times, with no φ involved, and `M = 2–4` particles per group *is* the population. Stays valid because IMM only needs MMD to be zero-in-expectation at the optimum (`M ≥ 2`), not a population estimate. FD-loss needs 50k because it estimates `μ_r, Σ_r` in `d ≈ 2048`-dim φ-space; IMM operates pairwise in raw `D`-dim sample space.

## FID saturation in practice

Modern generators (REPA-E, RAE-XL, SiT-XL) score below the real validation set's FID (val FID = 1.68 on ImageNet 256). Humans still distinguish samples from real images. FD-loss paper diagnosis:

- Inception-v3 encodes "ImageNet-classifier-relevant" information, over-optimised by years of generators.
- Different feature spaces rank generators differently and disagree by 1–2 OOM in raw FD.
- Optimising Inception FD alone can *worsen* DINOv2/MAE/SigLIP FD — closer to training distribution under one φ but visibly worse perceptually.

**FAD inherits all of this.** VGGish is older and smaller than Inception (d=128 vs d=2048), classifier-trained on AudioSet. CRASH reports a test-vs-noisy-test FAD floor ≈ 0.72 — direct audio-side evidence that FAD differences below ~0.5 are inside the metric's noise band.

## Audio operational takeaways

1. **Don't report only FAD.** Adopt an FDr^k-style multi-rep metric: at minimum `{VGGish, CLAP, AudioMAE}` ratio'd against held-out validation. Validation should score 1.0 by construction.
2. **Use MIND as the per-epoch eval signal** on top of one or more of those audio embeddings. 5k samples is the difference between an evaluator we run every epoch and one we never run.
3. **Use FD-loss for post-training one-step audio generators.** The decoupling trick is φ-agnostic; nothing assumes images. Frozen audio encoder + precomputed `(μ_r, Σ_r)` over our corpus is all that's needed.
4. **Pick φ that complements waveform fidelity.** Image FD-SIM mixes Inception (classifier) + ConvNeXt (modern CNN) + MAE (reconstruction). Audio analogue should include a reconstruction-trained encoder (EnCodec, AudioMAE) alongside classifier-trained ones (VGGish, CLAP) so phase/waveform artefacts have a chance of being penalised.
5. **Stress-test for moment-matching.** Once an audio embedding is chosen, repeat MIND's Prop 4.1 adversarial construction (atoms at `μ ± α uᵢ` along Σ's eigendirections) and confirm whatever FAD / FD-stack we use can be hacked while MIND notices.
6. **Short-clip FD is open.** Most audio φ assume ≥ 1 s clips. For 0.1 s targets, we may need a custom small φ trained on short windows or a temporal-pooling strategy that doesn't degenerate.

## Open questions

- What is the "FD-SIM for audio"? Concrete encoder mix to standardise on. Default proposal: VGGish + CLAP + AudioMAE/EnCodec encoder. Apply MIND on each, average ratios FDr^k-style.
- Does MIND inherit an audio embedding's blind spots the way FAD does? Empirical test: two audio sets with matched mean/covariance in φ-space but different samples.
- Can a tiny in-house φ (small AudioMAE trained on our corpus) substitute for pretrained encoders?
- Does FD-loss (or MIND-as-loss) converge on 1-D waveform generators? No public evidence.
- **Sample-space loss for raw audio.** IMM's energy kernel matches first moment only; Laplace matches all moments at single-time; signature matches path-structural moments. The right sample-space loss is open. Cheapest experiment: signature-kernel MMD on raw 1 s clips vs energy-distance baseline (AudioDEAR) — same loss shape, different kernel, no φ.
- Reference-corpus size N for short audio clips: image FD-loss saturates at N ≈ 50k; MIND at n ≈ 5k. Audio likely needs higher per-clip variance compensation.
- Right scaling factor α for MIND on audio embeddings. Paper uses α=3d for FID-scale matching; no FAD-scale to match — drop the scaling?
