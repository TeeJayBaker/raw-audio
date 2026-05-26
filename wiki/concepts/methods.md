# Methods

**Purpose:** from-scratch training paradigms for raw-audio generative modelling — diffusion, flow matching, MeanFlow family, consistency, IMM, Drifting / WGF, EqM, scoring-rule, from-scratch GAN. What does each commit to, what does it cost, and which ones survive the 1-NFE bar that this project lives under?

> Methods that require a pretrained teacher (adversarial distillation, APT, FD-loss, score / trajectory distillation, representation distillation) live in [post-training](post-training.md). The split is by whether a warm-start checkpoint is needed.

## Why 1-NFE dominates

CPU is the inference target, per-step cost is fixed by the architecture, total cost is `NFE × cost-per-step`. NFE ≥ 50 (classical diffusion) is 50–500× slower than real-time on CPU. Everything else on this page is downstream of getting to 1 NFE.

Two methods break that framing on different axes. **EqM** drops time-conditional dynamics entirely — inference is per-sample-adaptive optimizer steps on a static energy field. **TiM** keeps the PF-ODE but supervises the network on a continuous-time finite-interval transition identity for all `(t, r)` simultaneously — one checkpoint serves 1-NFE through 128-NFE with monotonic quality.

## Families

### Score-based diffusion (and the audio baseline)

Continuous-time forward SDE (VP / sub-VP), ε-parameterised denoiser, `λ(t) ≈ σ²(t)` weighting, SDE / DDIM / ODE sampling. **Audio reference: CRASH** (raw 44.1 kHz drums, 1D σ-FiLM U-Net, cosine σ): FAD ≈ 1.0 at 400 NFE, ≈ 1.5 at DDIM-50, GPU-only. Treated as the multi-step ceiling 1-NFE recipes want to match, not a path forward in itself.

### Flow matching / rectified flow

Same Gaussian corruption, but learn the velocity `v = x_1 − x_0`; linear interpolant. On STFT-frame audio (RFWave), 10-NFE RF cleanly beats 50-NFE DDPM at matched backbone — the FM-vs-score gap matters most at low NFE.

**Audio-specific FM design rules** ([LinDiff](../../papers/2306.05708.md) first, Flow2GAN / WaveFM / WavFlow independently rederive on different backbones):

- **Predict the clean target, not the velocity** — silent regions and quiet bands force `v ≈ −x_0`, an expensive noise-cancellation problem; clean-target prediction removes it and unlocks auxiliary mel / phase / STFT losses (which cannot be meaningfully stacked on a velocity prediction). LinDiff (`x_0` under linear interpolant, 2023), Flow2GAN, WaveFM, [WavFlow](../../papers/2605.18749.md) all converge on this; LinDiff is the earliest, WavFlow is the largest-scale (1 B-param multimodal at 16 / 44.1 kHz) and the most directly inherits the JiT/pMF manifold argument with audio-domain framing. **WavFlow's specific combination** — `x`-prediction at the network output, `v`-loss on the recovered velocity `v_θ = (x̂_1 − x_t)/(1 − t)` — wins on distribution coverage (IS) over both pure-`x` and pure-`v` ablations; the pattern is identical to pMF's `x` → `u` → `V_θ` conversion.
- **Spectral-energy-inverse loss weighting** (Flow2GAN): `1/√(S(x_1) + ε)` clamped to `[0.01, 100]` across (time, freq) — upweights quiet regions in both axes; strictly more comprehensive than PriorGrad / RFWave per-frame scaling.
- **Mel-conditioned diagonal-Gaussian prior** (WaveFM): per-sample `Σ` interpolated from mel energy, clamped at 10⁻³. Co-load-bearing with x-prediction for 1-step quality.
- **Phase handling**: WaveFM uses an explicit `atan2`-wrapped phase-angle loss on multi-res STFT; Flow2GAN runs on the complex STFT so phase is implicit in the iSTFT readout. Both work — choice is open.
- **Amplitude lifting** ([WavFlow](../../papers/2605.18749.md)): raw audio RMS is typically < 0.2 with sharp zero-centred distributions — most of the dynamic range is unused, so the loss has near-zero magnitude on quiet samples and the signal is "submerged by noise" against the `𝒩(0, 1)` flow prior. One-line fix: `x_lift = s_a · clamp(r_*/rms(x) · x, −1, 1)` with `r_* = 0.33`, `s_a = 3.0`. ~15 FD_PaSST swing in WavFlow's ablation (isolating RMS-norm at `s_a = 1`). Drop-in for any raw-waveform FM training loop; inverse-rescale + LUFS-normalise at inference. Strictly *required* preprocessing for raw-waveform FM at this scale — the original-amplitude regime did not train.
- **Don't carry over image-domain noise-level shift** ([WavFlow](../../papers/2605.18749.md) §E). The `t_s = t/(t + s(1−t))` reweighting toward higher noise that helps pixel-space ViT diffusion (Hoogeboom et al. 2023; Li & He 2025) **degrades raw-audio FM monotonically** at `s ∈ {3.0, 5.0}` across all metrics. Mechanism: image pixels have wide dynamic range where high-noise training catches low-frequency global structure; audio waveforms have inherently low signal density even after the amplitude lift, so further noise burying makes the signal unrecoverable. Default `s = 1.0` on raw audio.

#### FM + post-training to 1-step: audio FM+CM vs FM+GAN

Two finishing moves that turn an FM teacher into 1-NFE audio. Detail lives in [post-training](post-training.md); the headline is the CPU axis:

| Stage 2 | Stability | Backbone | 1-NFE CPU xRT (Xeon, 24 kHz) |
|---|---|---|---|
| **CM** ([WaveFM](../../papers/2503.16689.md)) — L₂ distance to EMA teacher, truncated-Gaussian `t` near 0 | high (deterministic target) | 1-D U-Net at sample rate | 0.64 (sub-RT) |
| **GAN** ([Flow2GAN](../../papers/2512.23278.md), [PeriodWave-Turbo](../../papers/2408.08019.md)) — MPD+MRD finetune | moderate (GAN tuning) | STFT-domain ConvNeXt | 4.85 (Flow2GAN) — first >RT CPU number |

The architectural axis (sample-rate waveform vs hop-rate STFT) dominates CPU cost by ~10× regardless of NFE or recipe. The untested cell is **FM + CM on STFT-domain backbone** — WaveFM's distillation on Flow2GAN's architecture.

### MeanFlow family (fastforward FM)

Reformulate FM so one forward pass produces `x_0 ← x_1 − u_θ(x_1, t)` (average-velocity parameterisation) or `x_0 = f_θ(x_1, 1, 0)` (solution-function parameterisation). The family is the largest single cluster in the wiki and is best read along two axes: **what parameterisation** and **what to do about the JVP**.

**Ancestor: [Shortcut Models](../../papers/2410.12557.md).** One DiT conditioned on `(t, d)` over a dyadic step-size grid, trained end-to-end with a discrete binary self-consistency identity `s(x_t, t, 2d) = (s(x_t, t, d) + s(x'_{t+d}, t+d, d))/2` plus an FM anchor at `d=0`. Batch ratio k=1/4 FM + 3/4 self-consistency, EMA bootstrap (0.999), WD 0.1. No JVP, no teacher, single end-to-end run, ~16% overhead. Variable-NFE on one checkpoint. Three later unifications all place Shortcut at a specific point in their design space (AlphaFlow's α=½, SplitMeanFlow's `s=(r+t)/2` discrete corner, TiM's dyadic-`(t,r)` slice). The "75% bootstrap / 25% empirical" heuristic in MeanFlow originates here as Shortcut's `k=1/4`.

**Why the original MeanFlow is unstable.** The variance-reduction paper (2605.09235) shows the conditional velocity `v_cond` plays two distinct roles inside the loss: regression target (correct role) and Monte-Carlo control variate inside the JVP tangent (where vanilla MeanFlow assigns coefficient `β=0`, suboptimal). Per-step gradient variance is proportional to the spectral norm of the Jacobi factor `J = (t−r)∂_{x_t}u_θ − I` and grows unboundedly with `‖J‖²`; the stop-gradient hides this from the optimizer at convergence (non-decreasing loss + unbounded variance pathology). The closed-form optimal tangent-mixing coefficient is `β* = κ/(κ+1) · σ²d/(σ²d+‖b‖²)` — product of noise-cancellation and James-Stein shrinkage. Vanilla MeanFlow's `β=0` is optimal only at the trivial-Jacobian corner.

**The seven independent fixes** sit at different points in the training graph:

| Mechanism | Paper | What's changed | JVP |
|---|---|---|---|
| Regression target | [iMF](../../papers/2512.02012.md) | v-loss with network-independent target; CFG-as-input | kept, stop-grad |
| Parameterisation + local FD | [SoFlow](../../papers/2512.15657.md) | Taylor FD over `f_θ` solution-consistency | eliminated |
| Algebraic identity | [SplitMeanFlow](../../papers/2507.16884.md) | Interval-splitting identity on student forwards | eliminated |
| Teacher trajectory | [IntMeanFlow](../../papers/2510.07979.md) | Teacher's discrete Euler trajectory as target | eliminated |
| Objective decomposition + curriculum | [AlphaFlow](../../papers/2510.20771.md) | Sigmoid α-anneal across the FM↔MeanFlow family | kept |
| Statistical efficiency | [Variance-Reduction MF](../../papers/2605.09235.md) | EMA-tangent proxy + FM anchor | kept, stop-grad |
| Identity construction | [TVM](../../papers/2511.19797.md) | Differentiate the integral w.r.t. terminal `s` only | kept, but `J=0` by construction |
| Pixel + x-prediction | [pMF](../../papers/2601.22158.md) | JiT-style `x`-pred + perceptual loss; iMF v-loss otherwise | kept, stop-grad |

**Rule of thumb across the family** (and the deeper rule under variance-reduction): the regression target must not depend, or must only stop-grad-depend, on the network — and `v_cond` should remain the target; only its appearance as the JVP tangent needs replacement (Proposition 3 of 2605.09235 proves the asymmetry — substituting in the tangent carries an `(t−r)`-multiplied bias that vanishes at the `r=t` boundary; substituting in the target produces bias persistent at every `(r, t)`).

**Practical leaderboard read-off** (matched ImageNet-256, ~600M params, 240+ epochs; see the leaderboard table below):

- **iMF** wins on raw quality (FID 1.72) when JVP/memory-attention is not a deployment blocker.
- **SoFlow** wins on training memory/throughput when the backbone needs FlashAttention.
- **SplitMeanFlow** is the only family member deployed at production scale on speech audio (Doubao TTS); conceptually cleanest JVP-free identity.
- **AlphaFlow** is SOTA on vanilla DiT (FID 2.58) with no architecture tricks.
- **TVM** is the only family member with an explicit 2-Wasserstein guarantee, and the cleanest matched-protocol improvement over vanilla MeanFlow.
- **EMA-tangent** is the cheapest stability port (~22% overhead) — but FID at the β=1 corner is worse than vanilla because of the FID-MSE landscape mismatch; the value is gradient stability, not headline FID.
- **pMF** is the only viable 1-NFE pixel-space MeanFlow.

**Audio applications, three points so far** — all conditional, all over frozen intermediate representations, none CPU-benchmarked:

- **[SplitMeanFlow](../../papers/2507.16884.md)** on Seed-TTS acoustic-diffusion latent, JVP-free, deployed at 1–2 NFE in Doubao TTS.
- **[MeanFlowSE](../../papers/2509.23299.md)** on a WaveVAE latent conditioned on frozen WavLM features for speech enhancement. 40.7M DiT — smallest MeanFlow at any modality. Decisive SSL-vs-VAE-latent conditioning ablation establishes representation-space as a quality knob, not just compute. See [representation-space](representation-space.md).
- **[IntMeanFlow](../../papers/2510.07979.md)** on F5-TTS / CosyVoice2 mel-spectrograms via teacher-Euler distillation. **Task-dependent NFE**: 1-NFE viable for token2mel (dense conditioning), 3-NFE needed for text2mel (sparse). General lesson: 1-NFE is most feasible when conditioning is dense and time-aligned. Companion **O3S** ternary-search step-placement is orthogonal and composes with any continuous-`(t, r)` few-step generator.

Raw-waveform unconditional MeanFlow with a CPU benchmark is still empty across the family.

### Transition Models (TiM)

Strict generalisation of MeanFlow / CM at the regression-target axis. Parameterise `f_θ(x_t, t, r)` as a finite-interval transition operator between any two timesteps, supervise with an exact continuous-time identity (State Transition Identity, Eq. 8 of TiM):

```
d/dt ( B_{t,r} · ( α̂_t x + σ̂_t ε − f_{θ,t,r} ) ) = 0
```

The product rule decomposes this into PF-ODE supervision plus time-slope matching. The latter is what fixed-endpoint methods (CM, MeanFlow, Shortcut) throw away by design and is the structural reason consistency-family methods saturate past their training endpoint.

`r → t` reduces to FM / diffusion; `r = 0`, `d = ‖·‖²` reduces to continuous-time CM up to a weighting; MeanFlow / iMF / SoFlow are transport-specific special cases at fixed `(t, r)` sampling; CTM / PCM / Shortcut are discrete-`(t, r)` predecessors.

**The portable trick — DDE.** The training target needs `df_{θ−,t,r}/dt`. MeanFlow / iMF / SoFlow / sCM use JVP, blocking FlashAttention and FSDP. TiM uses a central finite difference

```
df/dt ≈ ( f(x_{t+ε}, t+ε, r) − f(x_{t−ε}, t−ε, r) ) / (2ε)
```

— two extra forward evals, ~2× faster than JVP at matched FID, FSDP/FlashAttn compatible. Drop-in replacement for any JVP-based fastforward training loop.

**The headline property** is monotonic NFE-quality on one checkpoint (GenEval 0.67 at 1-NFE → 0.83 at 128-NFE on 865M TiM). Every distilled few-step competitor (FLUX.1-Schnell, SDXL-Turbo, SD3.5-Turbo) degrades past their training endpoint.

**TiM vs TVM** are the closest structural neighbours: same parameterisation, different identity. TiM supervises the all-`(t, r)` STI via DDE-over-`t`; TVM supervises the single terminal-velocity condition via JVP-over-`s` (one-dimensional, `J=0` in propagated gradient). Different supervision philosophies — TiM gives variable-NFE monotonicity, TVM gives a distribution-level `W_2` bound. Matched-protocol head-to-head is the missing experiment; neither paper publishes it.

### Drifting / WGF-then-compress

Don't integrate an ODE at inference, integrate one in distribution space at training, then compress the trajectory into a static 1-NFE generator. Unifying template (2603.09936): pick energy `F[q]` with `F=0 ⇔ q=p`, write its Wasserstein gradient flow `∂_t q_t = ∇·(q_t V_t)` with `V_t = −∇ δF/δq`, explicit-Euler-discretise, regress the generator to match each step with stop-grad on the target (which is the JKO frozen-field discretisation).

**`f`-divergence recipe book (2603.10592).** Under KDE smoothing with a characteristic + uniform-gradient-bound kernel, every `f`-divergence's WGF velocity factors as a divergence-specific density-ratio weight times the common direction `∇ log p_kde − ∇ log q_kde`:

| Divergence | Weight `w(x)` | Failure mode |
|---|---|---|
| forward KL (Drifting) | `1` | mode blur |
| reverse KL | `p_kde/q_kde` | mode collapse |
| χ² | `q_kde/p_kde` | over-coverage |
| MMD² | direct `∇(p_kde − q_kde)` | both blur and coverage |

Convex combinations of valid divergences yield valid divergences; mixed flows (e.g. rev-KL + χ²) give a structural mode-collapse-vs-blur dial complementary to W-Flow's coupling-level fix.

**Family points:**

- **[Drifting](../../papers/2602.04770.md)** — hand-designed anti-symmetric mean-shift drift in φ-space. Forward-KL row under Gaussian kernel.
- **[Sinkhorn-Drifting](../../papers/2603.12366.md)** — two-sided Sinkhorn scaling; rederives the drift as the WGF of the Sinkhorn divergence, closes identifiability rigorously.
- **[W-Flow](../../papers/2605.11755.md)** — full Sinkhorn-WGF instantiation with explicit-Euler discretisation, two-batch self-transport estimator (replaces Drifting's diagonal-mask hack), velocity-level CFG. 1-NFE SOTA on ImageNet-256 (FID 1.29 XL/2). Convergence theorem in `sup_t W₂`. Non-adversarial mode-coverage preservation on imbalanced data is the structurally interesting property for multimodal audio corpora.
- **Mixed-divergence ([2603.10592](../../papers/2603.10592.md))** — rev-KL + χ² mix on the same field; validated only on 2D Swiss roll.
- **Spectral ([2603.09936](../../papers/2603.09936.md))** — Drifting under Gaussian = `σ²∇log(p_σ/q_σ)`; Landau damping shows Laplacian beats Gaussian on high-frequency modes (relevant for audio). Exponential bandwidth annealing gives `O(log K_max)` convergence.

Empirical `F`-choice on ImageNet-256 (W-Flow Tab 2a, B/2 100-epoch): Drifting heuristic 8.46, MMD-WGF 10.40, KL-WGF 10.17, Sinkhorn-WGF 7.29. MMD/KL underperform the heuristic; on natural images the energy has to be `S_ε` or close.

**Kernel-choice constraint** (K1–K4 of 2603.10592). Gaussian satisfies K1–K4. Laplace violates K4 — empirically works but numerically jitters at convergence. Three resolutions: mollified Laplace; Gaussian + exponential bandwidth annealing; or vMF / spherical-log kernel on a normalised audio-φ space (K4 vacuous on compact manifolds, cleanest target if a JEPA-style audio encoder is available).

**The family is φ-dependent at scale.** Pixel-space ImageNet does not converge without a strong learned encoder; audio transfer hinges on AudioMAE / EnCodec / waveform-MAE.

### Equilibrium Matching (implicit-energy gradient field)

The only off-tree paradigm. Throws out time-conditional dynamics — learns a single time-invariant gradient field `f_θ(x)` that is the gradient of an implicit energy landscape with minima on the data manifold. Loss: `‖f_θ(γx + (1-γ)ε) − (ε−x)·c(γ)‖²` with `γ ~ U(0,1)` not seen by the network; `c(1) = 0`. Best `c`: truncated decay `c_trunc(γ; a=0.8)`, gradient multiplier `λ=4`.

Sampling is gradient descent (vanilla GD, NAG with `μ=0.35`, or any optimizer). **Per-sample adaptive compute** by stopping when `‖f_θ(x)‖₂ < g_min` — up to 60% compute reduction vs fixed 250-step at comparable FID. Step-size robustness: FID is roughly flat over wide `η`.

FID 1.90 on ImageNet-256 with SiT-XL/2 (same backbone, `t` input set to 0). Not 1-NFE — needs ~100–250 GD steps; loses to Drifting / iMF / pMF at strict 1-step. Loses to FM on CIFAR-10 (smaller scale).

Optional **EqM-E** variant parameterises a scalar `g(x)` and unlocks EBM byproducts: OOD detection via `g(x)`, sample composition by summing energies, partial-noise denoising. Costs some FID.

Theoretical justification leans on high-d Gaussian concentration; weakens at lower dimensions. Audio `d ≈ 22k` (1s @ 22kHz mono) sits between CIFAR `d ≈ 3k` and ImageNet `d ≈ 196k` — empirical question whether the concentration holds enough.

The interpretation that matters: **sampling-as-optimization decouples inference budget from training**. NFE is a runtime choice. Coarse-then-fine `η` becomes principled (decaying-LR GD), per-sample early stopping is mathematically natural — the entire optimization literature opens as inference-design space.

### Consistency-from-scratch and IMM

Train a one- or few-step generator end-to-end without a teacher. Two structurally different approaches:

- **Consistency Models + SSL anchor (EPG).** Train a CT objective on PF-ODE trajectories; the [EPG](../../papers/2510.12586.md) recipe makes this work in pixel space without a teacher via SSL encoder pre-training + an auxiliary contrastive loss between `f_θ(x_{t_n}, t_n)` and `x_0` through a frozen copy of the pre-trained encoder. First successful CT trained pixel-space-from-scratch (FID 8.82 at 1-NFE).
- **[Inductive Moment Matching](../../papers/2503.07565.md)** — parametrise a one-step map `f^θ_{s,t}` between any two stochastic-interpolant times; loss is MMD between (i) current model jumping `t→s` and (ii) EMA-copy jumping `r→s` with `s≤r≤t`. `M≥2` particles + Laplace / pseudo-Huber CPD kernel matches all moments. Lemma 1: CM is the `M=1`, first-moment (energy-kernel) corner of IMM, with both repulsion and higher moments dropped — and CM instability follows. **Practical implication: start any consistency-on-waveform attempt with M=2 and a Laplace kernel.**

IMM unifies the divergence-axis side of the consistency family; TiM unifies the regression-target side via all-`(t, r)` supervision; AlphaFlow unifies the per-sample-objective axis via its α-family; Variance-Reduction MF unifies the statistical-efficiency axis within JVP-containing MeanFlow objectives. The four are complementary — a concrete method sits in all four frames at different positions.

### SSL pre-training

Decompose the generator into encoder + decoder, pre-train the encoder with SSL, attach a decoder, fine-tune end-to-end. EPG's recipe: two NT-Xent terms (contrastive on augmentation pairs + representation consistency on adjacent PF-ODE points). Annealed contrastive temperature τ from 0.1 to 0.2 replaces brittle EMA scheduling. Pre-training cost ~57 h on 8×H200 for RCM-B/16, cheaper than training the StableDiffusion VAE.

Ablation: removing the contrastive loss makes downstream generation *worse* than from-scratch — the encoder collapses to a uniform representation across noise levels.

**The SSL stage is avoidable.** pMF achieves stronger 1-NFE pixel-space FID without it. Keep SSL as the tool for consistency-from-scratch specifically.

### From-scratch GAN

Train a generator with only adversarial + reconstruction losses, no teacher, no diffusion/FM prior. Three anchors:

- **[HiFi-GAN](../../papers/2010.05646.md)** — foundational audio reference (MPD+MSD+feature-matching+mel-L1 inherited by every modern audio adversarial paper). V3 (1.46M): 13.4× RT on 2020 CPU at 22 kHz — load-bearing existence proof that a 1-NFE conv stack can do faster-than-RT speech audio on CPU. Conditional vocoder. No R1, no SAN.
- **[R3GAN](../../papers/2501.05441.md)** — modern minimal GAN. RpGAN (relativistic pairing, loss landscape has no bad mode-dropping local minima) + R1 + R2 zero-centered gradient penalties on a ConvNeXt-style ResNet. Drops every StyleGAN2 trick. 1000/1000 mode coverage on Stacked-MNIST. **Highest-EV audio experiment: add R2 to MPD/MRD discriminators** (currently only R1 is used sporadically).
- **[ComVo](../../papers/2603.11589.md)** — STFT-domain ConvNeXt vocoder with native complex-valued layers throughout generator + cMRD discriminator. Vanilla `MPD + cMRD + feature-matching + mel-L1` hinge GAN, no FM pretrain, no consistency, 1-NFE by construction. Two portable contributions: (i) **phase quantization** `z_q = |z|·e^{i·(2π/N_q)·round(N_q θ/2π)}`, architecture-agnostic; (ii) block-matrix CVNN implementation for training-time saving. Matched-memory ablation `G_C D_R` beats `G_R D_R 2×` — win is from complex-domain modelling, not doubled per-parameter memory. No CPU benchmark.

The Flow2GAN / PeriodWave-Turbo *GAN-finetune stage* is post-training, covered in [post-training](post-training.md); their FM-pretrain stage is the from-scratch FM section above.

### Scoring-rule one-step (energy distance, signature kernel)

Train a one-step head directly with a strictly proper scoring rule — no score field, no ODE, no teacher. Universal shape: `L = (1/m(m−1)) Σ_{r≠s} k(x_r, x_s) − (2/m) Σ_r k(x_r, y)` over a PSD kernel `k`. Strict propriety = `k` characteristic.

Three rungs in order of structural expressivity:

| Kernel | Matches | Audio relevance |
|---|---|---|
| Energy `−‖·‖` ([AudioDEAR](../../papers/2605.00329.md)) | Sample-marginal 1st moment | Treats waveform as a length-`d` vector |
| Laplace / pseudo-Huber CPD ([IMM](../../papers/2503.07565.md) at `M≥2`) | All sample-marginal moments | Laplacian preferred (Landau damping) |
| Signature `K^Sig` ([2510.19110](../../papers/2510.19110.md)) | All path-structural moments | Native — waveforms are paths; stereo Lévy area is a free phase feature |

Goursat memory is `O(l²)`, capping natural audio length at ~1 s @ 22 kHz on a 4090 — exactly the project's 0.1–1 s sweet spot. Numerical-stability frontier on waveform `l ≈ 10^4` is past the paper's tested range (heuristic `1/√(I·J)` pre-scaling validated up to `l=2048`); RBF static kernel σ=1 is hard-locked (linear / polynomial explode). `sigkernel` Python package available.

**The signature kernel composes** as a fourth IMM kernel option and as the `F` in Drifting's `V = −∇ δF/δq` template (sig-kernel-WGF — autodiff through `sigkernel` gives the drift). Drop-in upgrade for AudioDEAR's loss: replace `−‖·‖` with `K^Sig` + basepoint/time augmentations.

AudioDEAR's auxiliary representation-distillation (`L_one_step + λ · MSE(h_S, h_T)`, λ≈1000, final-layer hidden state, λ≈1000) composes with any one-step loss when a teacher is around; requires backbone shape parity.

> "Score" is overloaded. Score matching learns `∇ log p_t(x)` and iterates an ODE/SDE for many NFE; scoring-rule training optimises a sample-based statistical divergence directly and never needs a density. Different families.

### Masked-token generation / discrete AR with bit modelling

AR over a sequence of discrete tokens in a learned latent (VQ / FSQ / LFQ / BSQ); lineage MaskGIT → MAR → MaskBit → Infinity → [BAR](../../papers/2602.09024.md). **Filed for completeness, not a project-recommended path** — codec floor + serial AR inference are disqualifiers against CPU + 1-NFE.

BAR's **MBM head** (iterative bitwise unmasking over `N=2–5` steps on top of the AR transformer's per-token context vector) is the one mechanism worth knowing — replaces both softmax-over-`2^k` (OOMs past `k≈18`) and Infinity's per-bit independent sigmoid (quality collapses at large `k`). Drop-in for any AR audio LM with vocab `≥2¹⁸`.

**Critical clarification:** BAR's "bits" are bits of a learned discrete codebook index (FSQ token), NOT bits of raw pixel intensity or sample amplitude. The natural raw-audio analogue is Analog Bits (Chen et al. 2022) — structurally different (continuous diffusion on binary data) and not reachable from BAR's recipe.

## Cross-cutting axes

### From-scratch vs post-training × needs-φ

|  | **From scratch (this page)** | **Post-training ([post-training.md](post-training.md))** |
|---|---|---|
| **No φ required** | EPG, iMF / SoFlow / pMF / AlphaFlow / TVM (fastforward FM), IMM, AudioDEAR, HiFi-GAN, R3GAN, ComVo, EqM, TiM, SplitMeanFlow (loss is from-scratch-compatible but only validated under two-stage distillation) | DMD2, ADD, LADD, Seaweed-APT, ARC, Flow2GAN GAN-stage, PeriodWave-Turbo GAN-stage, AudioDEAR distillation, SplitMeanFlow as deployed, IntMeanFlow |
| **Needs strong frozen φ** | Drifting, W-Flow | FD-loss, SiD-DiT |

### The JVP question

The MeanFlow family produces seven independent answers to "what about `torch.func.jvp`?" (table above under the family description). The variance-reduction frame (2605.09235) unifies five of them — iMF, AlphaFlow, Modular MF, Re-MeanFlow, TVM, Decoupled MF — as practical realisations of the same `β → 1` (or `J → 0`) optimum. **SplitMeanFlow, SoFlow, TiM-DDE, and IntMeanFlow** sit on a separate axis (JVP-elimination via identity change, finite difference, or teacher trajectory); they bypass the variance amplification rather than counteract it.

The two axes compose. Examples of immediately constructable but untested recipes: EMA-tangent + DDE (variance-reduced + FlashAttention-friendly); AlphaFlow curriculum + SplitMeanFlow at α=0 (JVP-free terminal phase + curriculum); AlphaFlow + DDE; SplitMeanFlow + EMA-tangent on the boundary anchor only; TVM + DDE-over-`s`-only (no custom JVP kernel needed).

## From-scratch 1-NFE leaderboard

ImageNet-256, ~600–680 M params, 240+ epochs:

| Method | 1-NFE FID | 2-NFE | Space | Needs φ? | Notes |
|---|---|---|---|---|---|
| W-Flow-XL/2 | **1.29** | — | latent | yes | Overall SOTA |
| W-Flow-L/2 | 1.35 | — | latent | yes | |
| W-Flow-B/2 | 1.52 | — | latent | yes | 133 M generator |
| Drifting (latent, L/2) | 1.54 | — | latent | yes | |
| Drifting (pixel) | 1.61 | — | pixel | yes | |
| iMF-XL/2 | 1.72 | 1.54 | latent | no | |
| pMF-H/16 | 2.22 | — | **pixel** | no | First viable 1-NFE pixel MF |
| pMF-L/16 | 2.52 | — | pixel | no | |
| α-Flow-XL/2+ | 2.58 | 2.15 | latent | no | SOTA on vanilla DiT, no JVP elimination |
| α-Flow-XL/2 | 2.95 | 2.34 | latent | no | |
| SoFlow-XL/2 | 2.96 | 2.66 | latent | no | JVP-free |
| pMF-B/16 | 3.12 | — | pixel | no | 118 M / 33 GFLOPs |
| TVM-XL/2 | 3.29 | 2.80 | latent | no | Explicit `W_2` bound; 4-NFE FID 1.99 |
| MeanFlow-XL/2 | 3.43 | 2.93 | latent | no | Original fastforward baseline |
| IMM | 8.05 | — | latent | no | Strong at 8-NFE (1.99), poor at 1 |
| EPG-L/16 (CT) | 8.82 | — | pixel | self-trained | First pixel-CT-from-scratch |
| Shortcut-XL | 10.6 | — | latent | no | Lineage anchor; 128-NFE FID 3.8 on same checkpoint |

FD-loss post-training on a pMF-H base reaches **0.72 FID @ 1 NFE** — beats every from-scratch number above. W-Flow + FD-loss is untested.

**Off-axis points** (not strict 1-NFE):

- **EqM-XL/2** reaches FID 1.90 with ~100–250 NAG-GD steps; the paradigm trades 1-step for adaptive per-sample compute.
- **TiM** serves 1-NFE through 128-NFE with monotonic quality on one checkpoint. The wiki's missing experiment is TiM-XL/2 at matched protocol against W-Flow / iMF / pMF.

## Translation to raw audio

Most directly portable, in order:

0. **[Shortcut Models](../../papers/2410.12557.md) as the simplest MeanFlow-family baseline.** Before AlphaFlow / SplitMeanFlow / iMF / TiM / pMF, the question is whether the original discrete-dyadic recipe converges on raw waveform. Dual FiLM `φ_t(t) + φ_d(d)` on a CRASH-style 1-D U-Net, `M ∈ {16, 32}` (much smaller than image-side 128 — natural audio NFE budgets are 1–16), `k = 1/4` empirical + 3/4 self-consistency, EMA bootstrap 0.999, WD 0.1. **Audio reparameterisation requirement:** Shortcut predicts a velocity-like `s`, so the `v ≈ −x_0` failure mode applies — port must predict `x_1` and convert. If this simplest recipe converges, the strict generalisations become straightforward upgrades.

1. **pMF** is the single most direct image-to-audio port in the wiki. Three pieces, all dimension-agnostic: `x`-prediction → convert via `x → u → V_θ`; perceptual loss directly on `x_θ` (MR-STFT + a learned waveform distance like AudioMAE-φ or MERT; gate at `t ≤ 0.8`); patch-size-as-cost-knob with fixed 256-token budget (1 s @ 22.05 kHz → 86-sample patches; 10 s → 860-sample patches; per-token FLOPs flat).

2. **iMF** is dimension-agnostic. For a 1-D conv U-Net, fall back to FiLM in place of in-context conditioning. No SSL stage, no audio φ.

3. **AlphaFlow** is the purest training-recipe port — no architecture / parameterisation / identity change. Wrap any existing MeanFlow waveform loop with the α-Flow loss (Eq. 8) + sigmoid α-schedule (Algorithm 2: `γ = 25, η = 5×10⁻³`, `k_s/K ≈ 0.4, k_e/K ≈ 0.65`) + adaptive loss `ω = α/(‖Δ‖²+c)` + v̄_{s,t} = v_t (FM velocity, **not** Shortcut's u_θ⁻) + `r=t` ratio at 25%. On a 1-D conv backbone the JVP isn't a hardware blocker, so AlphaFlow's main competitor on raw audio is vanilla MeanFlow trained naively — and on the image evidence, AlphaFlow strictly dominates that.

   **+1. Variance-Reduction MF (Algorithm 1)** is the cheapest stability port and sits underneath any of the above as a drop-in. Wrap with Polyak EMA (`μ = 0.9999`), compute the JVP tangent through `u_θ̄` rather than `v_cond` (keep `v_cond` as the regression target — Proposition 3 asymmetry), add a small FM anchor at interior offset `‖u_θ(x_t, t−δ, t) − v_cond‖²` with `δ ∈ [0, 10⁻³]` and `λ ≈ 0.1`. ~22% overhead. At waveform `d ≈ 22k` the Jacobi-factor amplification grows more severely than at ImageNet latent `d ≈ 4k`. Anisotropy of `Σ_{v'}` (highly frequency-band-dependent on waveform) suggests a direction-dependent β per STFT band — the natural audio extension.

4. **SplitMeanFlow** — direct audio prior-art port. Two pieces: the Interval Splitting Consistency loss (Eq. 11) on any 1-D audio FM teacher; the two-stage distillation pipeline (FM warm-start, boundary anchor at `r=t` for fraction `p ≥ 0.5`, ISC on the remainder). Realistic given any pretrained waveform FM teacher (CRASH, RFWave, Flow2GAN's FM-pretrain). Seed-TTS operates on acoustic features, not raw 22 kHz; sample-space behaviour and from-scratch convergence are both open.

5. **SoFlow** is the JVP-free alternative when the parameterisation needs to be `f_θ`. Relevance on 1-D conv (no attention, JVP works) is limited; the edge dominates only on Transformer-on-waveform.

6. **TiM** is the variable-NFE-on-one-checkpoint option. DDE is the most portable trick (~2× faster than JVP, FSDP/FlashAttn compatible). Decoupled `(t, Δt)` maps to dual FiLM on a 1-D conv U-Net (separate `φ_t(t)` and `φ_Δt(Δt)` summed FiLM-style). Interval-aware attention applies only with a Transformer bottleneck. Audio CPU-UX payoff: ship one checkpoint, default to 1-NFE, expose a quality slider with monotonic improvement.

6a. **TVM** — terminal-time twin of TiM and the cheapest port of the JVP-keeping fix to a conv backbone. Three pieces: the TVM loss (Eq. 12); the RMSNorm-everywhere Lipschitz fix (mandatory on DiT, decorative or load-bearing on 1-D conv is an open empirical question); Adam `β₂ = 0.95` + predict-`x_1` + scaled CFG `(s−t)·w·F_θ` for conditional. No JVP-hardware blocker on 1-D conv ⇒ custom FlashAttention-JVP kernel is irrelevant. `J=0` by construction means the dimension-amplified variance pathology (which grows from `d≈4k` to `d≈22k`) should *increase* TVM's edge over vanilla MeanFlow at audio scale.

7. **IMM** is the cleanest few-step (not one-step) recipe and the most "no-extras" — no teacher, no GAN, no LPIPS, no VAE, no JVP, no SSL, no audio φ. 1-NFE is weak; 2–8 NFE competitive. Right starting point for any consistency-on-waveform attempt.

8. **EqM** — only candidate where inference budget is decided per sample at runtime. Two FM-loop changes (drop `t` conditioning, multiply target by `c_trunc(γ; a=0.8) × λ=4`); same SiT-XL/2 architecture; identical train compute. Coarse-then-fine `η` becomes principled (large `η` → envelope/onset; small `η` → timbre refinement); per-sample early stopping is natural. Not 1-NFE — open whether 4–8 step EqM is FAD-competitive on raw audio.

9. **Scoring-rule one-step (energy → Laplace → signature).** Energy distance is the cheapest first pass. Laplace / pseudo-Huber CPD at `M ≥ 2` is the IMM kernel-axis upgrade. Signature kernel is the natively-suited choice on paths (waveforms ARE paths) at `O(l²)` memory cost — fits ~1 s @ 22 kHz on a 4090, the project's sweet spot.

10. **R3GAN-style from-scratch GAN** on a 1-D conv backbone with MPD+MRD, RpGAN + R1 + R2. "Drop the bag of tricks" splits the audio bag into stabilisers (feature-matching, mel-L1 — likely droppable under R1+R2) and physics encoders (MPD, MRD, snake, anti-aliased filters — keep).

10a. **ComVo-style from-scratch GAN on a complex-valued STFT-domain ConvNeXt** — STFT-frame-rate companion to (10). Three audio-portable ideas: genuinely complex-valued layers throughout (beats matched-memory RVNN); phase quantization (architecture-agnostic, one-line layer); block-matrix CVNN implementation for training-time saving.

11. **W-Flow / Drifting** requires an audio φ first. Within the family, W-Flow is empirical SOTA — Sinkhorn-Knopp on the batch (L=10), two-batch self-transport, velocity-level CFG. The two-batch self-transport is the mode-collapse fix multimodal audio corpora need. Mixed rev-KL + χ² drift is the cheapest non-adversarial precision/coverage dial. Kernel constraint: Laplace violates K4 — use mollified Laplace, Gaussian + bandwidth annealing, or vMF on a normalised audio-φ space.

12. **EPG** is portable but requires defining audio augmentations (random gain, time-shift, EQ; pitch-shift only on instrument-class data). Audio-MoCo is its own ablation question.

13. **Post-training pathways** (SiD-DiT, FD-loss, ADD/LADD/DMD2/APT, Flow2GAN-style finetune, AudioDEAR distillation, SplitMeanFlow as deployed) require a pretrained teacher and live on [post-training](post-training.md). Typical stacked pipelines: from-scratch one-step → FD-loss polish; from-scratch FM → APT; from-scratch FM → SplitMeanFlow distillation.

### Audio-specific lessons (don't lose these on a port)

- **Predict `x_1`, not `v`** in FM (Flow2GAN, WaveFM). Loss: `‖g_θ − x_1‖²` with `1/(1−t)²` dropped to upweight small `t`.
- **Spectral-energy-inverse loss weighting** (Flow2GAN): `1/√(S(x_1) + ε)` clamped to `[0.01, 100]` across (time, freq).
- **Use Laplacian kernel for any kernel-based loss on audio** — Gaussian has exponential slowdown on high-frequency modes (Landau damping).
- **Mel-conditioned diagonal-Gaussian prior** (WaveFM) is co-load-bearing with `x_1`-prediction for 1-step quality on conditional models.
- **AudioDEAR's representation-distillation** (final-layer hidden-state MSE, λ ≈ 1000) composes with any one-step loss when a teacher is around; constraint is backbone-shape parity.

### Composition / hybrid plays

- From-scratch one-step (EPG / iMF / SoFlow / pMF / IMM / energy-distance / Drifting-with-φ) + FD-loss polish at 1 NFE. iMF + FD-loss is canonical; pMF + FD-loss is the natural pixel-space analogue.
- FM-pretrain + GAN finetune is the only audio recipe with a >RT CPU number. EqM-pretrain + GAN-finetune is the untested variant.
- FM-pretrain + SplitMeanFlow distillation is the actually-deployed audio recipe (Doubao TTS).
- SiD-DiT is the right teacher-based path when there *is* a multi-step FM audio teacher — doesn't require teacher conversion.
- EqM + EqM-E composition for audio layering: sum gradients of two class-conditional EqM-E models at sampling time (kick + snare → coherent layered mix).

## Open questions

A small, deduplicated set of questions that actually drive experiments. "Does X work on 1-D audio" is the implicit umbrella for all of them — listed below are the ones where the answer would change a project decision.

- **Does the pixel-MF (MeanFlow) recipe work on raw waveform?** WavFlow has now confirmed the pixel-*FM* recipe (x-pred + v-loss + patched waveform transformer) does work on raw audio at scale — so the manifold-assumption argument carries to 1-D. The remaining MF-specific question: pMF on a CRASH-style or LinDiff-style backbone at 1 s / 22 kHz / 256 tokens with `x`-pred + perceptual loss (MR-STFT + AudioMAE) + iMF v-loss + JVP. If yes, this is the project's default 1-NFE audio recipe; the next likely failure mode is the perceptual-loss choice rather than the prediction-target choice.
- **Variable-length via patch scaling.** Holding 256 tokens across 1 s / 5 s / 10 s @ 22.05 kHz (86 / 430 / 860-sample patches). Does the same architecture train cleanly across all three at flat per-token FLOPs? If yes — single waveform generator at fixed inference cost across clip length, a hard requirement for usable interactive audio gen.
- **Shortcut Models on raw waveform** — does the original discrete-dyadic recipe converge at all on audio with `M ∈ {16, 32}` and `x_1`-prediction reparameterisation? If yes, AlphaFlow / SplitMeanFlow / iMF / TiM all become straightforward upgrades; if no, the failure mode itself is informative.
- **JVP-free MeanFlow head-to-head on 1-D backbone.** SplitMeanFlow vs SoFlow vs iMF+DDE at matched compute on CRASH-backbone. Image papers don't run this; the SplitMeanFlow paper doesn't cite iMF/SoFlow. Determines the JVP-free audio default.
- **AlphaFlow curriculum on 1-D conv U-Net** — does the FM→MeanFlow sigmoid α-anneal give the same ~15% relative 1-NFE FID improvement on a CRASH-style backbone that it gives on vanilla DiT?
- **Does the Jacobi-factor amplification hold on 1-D conv?** The image measurement (71× peak loss-variance gap at `t = 0.9`) used DiT-B/4. 1-D conv has different Jacobian structure (translation-equivariant, no cross-token attention) — whether the gap is similar or smaller is open. Drives whether variance-reduction recipes are necessary on audio at all.
- **FAD-MSE mismatch on waveform.** Image headline: FID-optimal `β = 0` despite gradient-MSE-optimal `β ≈ 0.94`. Does FAD behave the same way (over-penalising bias) or is it more bias-tolerant? If the latter, the EMA-tangent corner becomes the deployment choice on audio.
- **Anisotropic `Σ_{v'}` direction-dependent β on waveform.** Theorem 4 assumes scalar isotropy. Waveform variance is highly frequency-band-dependent — estimate `Σ_{v'}` per STFT band, train with per-band `β`. Cleanest audio-specific extension of the variance-reduction framework.
- **TVM on a 1-D conv waveform backbone** — drop the loss + gap-sampling + `x_1`-reparameterisation + EMA `γ=0.99` onto CRASH; compare 1/2/4-NFE FAD against vanilla MeanFlow and AlphaFlow at matched compute.
- **TiM at small-data audio scale.** TiM's monotonic-NFE property is shown on 33 M images / 865 M params. Does it survive at ~30 k drum clips / ~5 M params, or does it require data scale we don't have?
- **EqM few-step viability on audio.** Headline is ~100–250 GD steps. Smallest experiment: CRASH-backbone + EqM objective on 1 s drums; sweep step counts (4, 8, 16, 32, 250) at fixed `η`. Where does FAD plateau? Lower bound: at what NFE does EqM lose to FM at matched model size?
- **Sig-kernel vs energy-distance head-to-head** at matched 1-step audio head (`m=2`, same backbone, same data). Answers whether matching path-structural moments actually beats marginal-sample distance on raw audio — the single most informative scoring-rule audio experiment.
- **Sig-kernel numerical stability on `l ≈ 10^4`** — paper's heuristic pre-scaling validated only up to `l=2048`. Where does Inf/NaN appear on waveforms?
- **W-Flow / mixed-divergence Drifting on multimodal audio corpora.** 95% pop / 5% jazz — does the jazz mode survive in samples? Direct test of whether the FFHQ-imbalanced mode-coverage result generalises to audio.
- **Audio-φ for Drifting / W-Flow.** Mel-only is the obvious failure mode (discards phase). EnCodec / AudioMAE / waveform-MAE worth comparing; whether SSL pre-training is the prerequisite is the headline open question for the WGF family on audio.
