# Inference

**Purpose:** the project's headline metric. CPU latency for 0.1–10 s samples, and everything that determines it — NFE, schedule, sampler, model cost. Every method we read or implement gets a row here.

## What CPU latency actually depends on

`total inference cost = NFE × cost-per-step + decoder-cost`. Three knobs are coupled:

1. **NFE budget.** Dominant lever. Per-step cost is fixed by the architecture; reducing NFE is what [methods](methods.md) is about.
2. **Operating sequence length.** STFT-frame ops are an order of magnitude cheaper per step than sample-rate ops — see [architecture](architecture.md). ELIT adds a runtime variant: a single DiT-like checkpoint trained with tail-dropping over a latent interface serves dozens of per-step FLOP points (`J̄` chosen per call), orthogonal to NFE.
3. **Schedule × sampler choice.** Couple tightly with NFE — the schedule that wins at 400 NFE is rarely the best at DDIM-50.

For score-based diffusion specifically, the schedule × sampler × NFE coupling is non-trivial enough to be the main reason naive few-step diffusion fails.

**Two methods break the fixed-NFE framing.** [EqM](../../papers/2510.02300.md) replaces ODE sampling with gradient descent on a learned static energy landscape; step count is runtime-chosen per sample, `η` is robust. [TiM](../../papers/2509.04394.md) keeps the ODE primitive but trains the network on an exact finite-interval transition identity for all `(t, r)` simultaneously — one checkpoint serves 1-NFE through 128-NFE with monotonic quality improvement. Different runtime knobs: EqM's is optimizer steps on a static field; TiM's is ODE-style finite-interval transitions. They could in principle compose.

**A third cost shape — `L × (1 + N)` two-loop AR + bit-head inference** ([BAR](../../papers/2602.09024.md)). Masked-token generation in a learned discrete latent. `L` = AR sequence length (256 / 64 / 16 for BAR-B / B/2 / B/4); `N` = MBM iteration count per token (2–5). No 1-NFE option exists within the family — the AR loop is fundamentally serial. Filed here for completeness; sits outside the project's CPU + 1-NFE target.

## Why CPU is the metric

- Removes batching artefacts: single-sample latency is what end users feel in interactive use.
- Removes vendor-specific GPU FLOPs accounting.
- Sets a hard ceiling on model size and NFE that's much tighter than GPU comparisons suggest. A model that's 2× faster than real time on a 3090 with batch 180 might be 20–50× slower than real time single-sample on a laptop CPU.

We now have one published audio CPU benchmark: Flow2GAN reports xRT on Intel Xeon Platinum 8457C at batch 16, 24 kHz mono, 1 s clips, for several waveform-generation baselines. Conditional (mel / EnCodec → waveform), so it answers the vocoder question, not the unconditional-generation question — but it gives the first calibrated map of what existing audio architectures cost on CPU.

## Audio CPU numbers

Intel Xeon Platinum 8457C, batch 16, 1 s clips, 24 kHz mono. xRT = generated audio seconds / wall-clock seconds (higher = better; >1 = above real-time).

| Method | NFE | Params (M) | CPU xRT | GPU xRT (H100) | Note |
|---|---|---|---|---|---|
| BigVGAN | 1 | 112.4 | 0.214 | 69.8 | Single-step GAN but heavy |
| Vocos | 1 | 13.5 | **387.6** | 6440.8 | Small ConvNeXt over STFT; fastest, PESQ 3.62 |
| [RFWave](../../papers/2403.05010.md) | 10 | 18.1 | 0.37 | 158.8 | Multi-band RF on complex STFT |
| PeriodWave-Turbo | 4 | 70.2 | 0.12 | 43.7 | FM + GAN finetune, single-resolution |
| [WaveFM](../../papers/2503.16689.md) | 1 | 19.5 | 0.64 | 226.3 | FM + CM, HiFi-GAN-MRF 1-D U-Net at sample rate |
| [Flow2GAN](../../papers/2512.23278.md) | 1 | 78.9 | **4.85** | 851.7 | STFT-branch ConvNeXt, FM-pretrain + GAN-finetune. PESQ 4.19 |
| [Flow2GAN](../../papers/2512.23278.md) | 2 | 78.9 | 2.46 | 449.3 | PESQ 4.44 |
| [Flow2GAN](../../papers/2512.23278.md) | 4 | 78.9 | 1.35 | 228.5 | PESQ 4.48, MOS 4.58 |
| [HiFi-GAN V1](../../papers/2010.05646.md) | 1 | 13.9 | 1.43 (i7 laptop) | 167.9 | Different hardware (2020 Intel i7), MOS 4.36 @ 22 kHz |
| [HiFi-GAN V2](../../papers/2010.05646.md) | 1 | 0.92 | 9.74 (i7 laptop) | 764.8 | MOS 4.23 — smallest viable 1-NFE audio GAN |
| [HiFi-GAN V3](../../papers/2010.05646.md) | 1 | 1.46 | **13.44 (i7 laptop)** | 1186.8 | MOS 4.05 @ 22 kHz. Existence proof that 1-NFE conv stack does >10× RT on CPU at near-MOS-ceiling quality |
| [ARC / Stable Audio Open Small](../../papers/2505.08175.md) | 8 | 497 total | ~1.82 (Arm mobile, Int8) | 156 (H100) | 12 s @ 44.1 kHz stereo in 6.6 s on Vivo X200 Pro. First TTA with mobile-class deployment. No desktop CPU number |

**Read-off:**

- Most published audio vocoders are sub-real-time on CPU even at batch 16 (BigVGAN, RFWave, PeriodWave-Turbo, WaveFM all < 1× RT) — all run convs at sample rate.
- **STFT-domain ConvNeXt is the right shape for CPU, regardless of NFE or post-training recipe.** Vocos (387× RT) and Flow2GAN (4.85× RT at 1-step, much higher quality) both run conv stacks at hop-rate.
- **Post-training recipe (CM vs GAN) does NOT change the CPU verdict.** WaveFM (FM + CM, 1-NFE, 19.5 M, waveform): 0.64× RT. PeriodWave-Turbo (FM + GAN, 4-NFE, 70 M, waveform): 0.12× RT. Flow2GAN (FM + GAN, 1-NFE, 78.9 M, STFT): 4.85× RT. All three use the same `x₁`-prediction reparameterisation but the backbone axis dominates by ~10×.
- Per-step CPU cost scales ~linearly with NFE in Flow2GAN. For budget calculations: ≈ 0.2 s audio / CPU-second / step at 78.9 M, 24 kHz, batch 16.
- Batch 16 inflates throughput — for interactive single-shot UX we care about batch=1, not reported, would be lower.
- The notable missing CPU number is [ComVo](../../papers/2603.11589.md) — fully complex-valued STFT-domain ConvNeXt vocoder, GPU xRT 819 vs Vocos 4658 at matched params (~5.7× slower on GPU). Memory ~2× per parameter. Complex-valued kernels in CPU BLAS are less optimised than float32 so the CPU gap likely widens. Until measured, ComVo is an architectural lesson, not a deployment recipe.
- **Cross-paper xRT confirmation for the STFT-domain ConvNeXt family**: [WaveNeXt](../../papers/wavenext.md) independently reports RTF 0.10 (= xRT 10) at 24 kHz and RTF 0.16 (= xRT 6.25) at 48 kHz on a single AMD EPYC 7542 core at batch 1, for both Vocos and WaveNeXt (head swap is essentially free). Different hardware and batching from the Flow2GAN benchmark above, but consistent ordering — Vocos-class trunks are firmly >>real-time on CPU, with the readout choice (iSTFT vs learned-linear-upsampler) not visibly affecting CPU cost.

### Older audio, GPU-only

| Method | NFE | Hardware | Reported | Generated | Effective speed |
|---|---|---|---|---|---|
| [CRASH](../../papers/2106.07431.md), SDE 400 | 400 | RTX-3090, batch 180 | 12 h | 27 000 × 0.48 s ≈ 3.5 h | ~0.3× RT (GPU, batched) |
| [CRASH](../../papers/2106.07431.md), DDIM 50 | 50 | RTX-3090, batch 180 | 1.5 h | 27 000 × 0.48 s ≈ 3.5 h | ~2.3× RT (GPU, batched) |
| [LinDiff](../../papers/2306.05708.md), 1 step | 1 | RTX-3090, batch n/r | RTF 0.004 | 22.05 kHz mono | 250× RT (GPU) |
| [LinDiff](../../papers/2306.05708.md), 3 steps | 3 | RTX-3090, batch n/r | RTF 0.013 | 22.05 kHz mono | 77× RT (GPU); MOS 4.12 |
| [LinDiff](../../papers/2306.05708.md), 100 steps | 100 | RTX-3090, batch n/r | RTF 0.520 | 22.05 kHz mono | 1.9× RT (GPU); MOS 4.18 |

LinDiff is the only published patched-ViT waveform model with a reported RTF. The 3-step quality plateau (MOS 4.12 within 0.06 of the 100-step ceiling 4.18) ports the "few-step rectified-flow Euler captures most quality" lesson from image-domain RF to raw audio. **No CPU number** — the project's `waveform_transformer.yaml` at depth 12 / dim 512 / patch 512 / 48 kHz would be the first published CPU benchmark for this architecture family.

## Image calibration for porting

GFLOPs per generated 256² image, including any auxiliary nets (VAE) and CFG:

| Method | NFE | Params | GFLOPs / img | Notes |
|---|---|---|---|---|
| DiT-XL/2 (LDM) | 250 | 84 + 675 M | 312 + 119 (VAE) = 431 | EPG Tab. 1 |
| SiT-XL/2 (LDM) | 250 | 84 + 675 M | 312 + 119 = 431 | |
| [EPG](../../papers/2510.12586.md)-XL/16 | 75 | 583 M | 128 | Pixel-space ViT; cheaper than DiT despite no VAE shortcut |
| [EPG](../../papers/2510.12586.md)-G/16 | 75 | 1391 M | 321 | FID 1.58 on ImageNet-256 |
| [EPG](../../papers/2510.12586.md)-L/16 (consistency) | 1 | 540 M | ~4 (321/75) | FID 8.82 one-step |
| [Drifting](../../papers/2602.04770.md) B/16 (pixel) | 1 | 134 M | 87 | FID 1.76 one-step |
| [Drifting](../../papers/2602.04770.md) L/16 (pixel) | 1 | 464 M | ~300 (est) | FID 1.61 one-step pixel |
| [Drifting](../../papers/2602.04770.md) L/2 (latent) | 1 | 463 + 49 M VAE | n/r | FID 1.54 |
| **[pMF](../../papers/2601.22158.md)-B/16 (pixel)** | 1 | 118 M | **33** | FID 3.12 — cheapest 1-NFE result anywhere |
| [pMF](../../papers/2601.22158.md)-L/16 (pixel) | 1 | 410 M | 117 | FID 2.52 |
| [pMF](../../papers/2601.22158.md)-H/16 (pixel) | 1 | 956 M | 271 | FID 2.22 — closes gap to latent iMF (1.72) to <0.5 |
| [pMF](../../papers/2601.22158.md)-H/32 (pixel, 512²) | 1 | 962 M | 272 | FID 2.48 — same GFLOPs as 256² thanks to fixed 256-token budget |
| [ELIT-DiT-XL](../../papers/2603.12245.md) (512²) | 40 | 698 M | 386–831 TF/iter (knob) | Single checkpoint, 60 inference budgets. 25% tokens: FID 12.5 vs DiT 18.8 at 47% compute |
| [BAR](../../papers/2602.09024.md)-B/2 (token-shuffle, 256²) | 64 × (1+4) | 415 M + tokenizer | n/r | gFID 1.35; H200 150 img/s — matches MeanFlow throughput at lower FID |
| [BAR](../../papers/2602.09024.md)-L (discrete, 256²) | 256 × (1+4) | 1.1 B + tokenizer | n/r | gFID 0.99 with CFG — SOTA across discrete + continuous; H200 10.65 img/s |

### The 1-NFE pixel cost/quality Pareto

| Method | GFLOPs | Params | 1-NFE FID | Headline |
|---|---|---|---|---|
| pMF-B/16 | **33** | 118 M | 3.12 | Cheapest 1-NFE anywhere |
| Drifting-B/16 | 87 | 134 M | 1.76 | Best FID/GFLOP, needs φ |
| EPG-L/16 (CT) | ~4 | 540 M | 8.82 | Cheapest GFLOPs but FID gap |
| pMF-L/16 | 117 | 410 M | 2.52 | Mid-range pMF |
| pMF-H/16 | 271 | 956 M | 2.22 | Best 1-NFE pixel FID without φ |
| Drifting-L/16 | ~300 | 464 M | 1.61 | Best 1-NFE pixel FID overall (needs φ) |
| StyleGAN-XL | 1574 | 166 M | 2.30 | Old GAN baseline; 5–50× more compute |

Pareto front for 1-NFE pixel-space: pMF-B → Drifting-B → pMF-H → Drifting-L. pMF wins when no learned φ is available; Drifting wins absolute FID at the cost of a separate feature stack.

## Sampler × schedule × NFE coupling

For score-based diffusion / FM, the three knobs don't optimise independently.

**Headline lesson from CRASH:** the σ schedule that wins at 400 NFE (sub-VP cosine) is the *worst* cosine schedule at DDIM-50; the schedule that wins at DDIM-50 (VP cosine) isn't the best at 400 NFE. Picking a schedule without specifying NFE budget is a category error for this project.

### Findings worth keeping

- **Cosine σ schedule beats linear-β VP** across all samplers in CRASH. Linear-β has σ growing very fast near t=0; equally-spaced t-steps spend almost no resolution on the perceptually critical low-noise regime. Cosine `σ(t) = ½[1 − cos((1−s)πt)]` with s=0.006 gives smoother, slower-rising σ near 0.
- **DDIM is a discretisation of the ODE** (CRASH §3): integrating `d(x/m) = d(σ/m) ε(x, σ)` between `t_i` and `t_{i+1}` recovers exactly the DDIM update. Same trained network → choose SDE / ODE / DDIM at inference freely. No DDIM-specific training.
- **RFF embedding of σ** lets the network skip learning σ(t); slightly stabilises training across schedule variants.
- **`t ∈ [η, 1]` with σ(η)=10⁻⁴.** Don't train on the truly-noise-free tail; imperceptible, loss is wasted.
- **Equal-straightness Euler step allocation** (RFWave). For RF/FM, define cumulative straightness `S(t) = ∫₀ᵗ E‖(X_1 − X_0) − v(X_τ, τ|C)‖² dτ`. For target NFE n, pick the n+1 time points so `ΔS` is constant — clusters steps near endpoints, spreads them in the easy middle. Computed once per trained model from a single batch of 96, then frozen. Model-agnostic; applies to any RF/FM model post-hoc.

### CRASH schedule × NFE FAD

| Schedule | SDE 400 | ODE 400 | DDIM 50 |
|---|---|---|---|
| VP exp (Song 2021a) | 4.11 | 3.96 | 5.11 |
| VP cos | 1.29 | 1.10 | **1.56** |
| sub-VP cos | 1.34 | **0.98** | 3.36 |
| sub-VP 1-1 cos | 1.41 | 1.23 | 2.93 |

Best at 400 NFE (sub-VP cos) is third-best at DDIM 50. Pick the schedule for your NFE budget.

### Implications

- Any few-step / one-step method we benchmark must report results on a schedule chosen for the relevant NFE budget, not asymptotic-quality schedule.
- VP cosine is the default for ≤50-NFE sampling on raw waveform.
- DDIM-style ODE samplers should be the default few-step sampler before moving to consistency / distillation methods.

## Sampling-as-optimization (EqM)

EqM is the only method in the wiki where inference budget is *not* fixed at training time. The trained model is a gradient field `f_θ(x) = ∇E(x)` for an implicit energy `E`; sampling is gradient descent on `E` from `x_0 ~ N(0, I)`. NFE is a runtime choice per sample; sampler is a free choice from the optimization literature.

### Properties this unlocks for CPU UX

1. **Per-sample adaptive compute.** Stop when `‖f_θ(x_k)‖₂ < g_min`. Simple/percussive samples converge faster than complex/sustained ones — the model decides. Up to ~60% reduction off fixed 250-step compute. On audio, the right primitive for an interactive UI: preview after `‖f‖₂ < g_threshold`, refine on demand.
2. **Coarse-then-fine `η` schedule.** Decaying-LR GD is principled, not heuristic. On audio: large `η` early → envelope/onset; small `η` late → timbre. Clean perceptual interpretation, untested.
3. **Sampler choice is open.** NAG-GD beats vanilla GD (FID 1.93 → 1.90); Adam / RMSProp / LBFGS / line-search all apply. The optimization literature is decades-deep and most of it is portable.
4. **Step-size robustness.** FID is roughly flat over a wide `η` range; FM has a single sharp optimum (off by 2× → significant quality loss). Less tuning sensitivity, easier defaults across hardware.
5. **ODE samplers as a special case.** Appendix D: standard Euler-ODE is a specific `η`-schedule on the EqM gradient field. Existing schedule-and-sampler tooling still works as a fallback; GD/NAG-GD strictly subsume it.

### Caveats

- **Not 1-NFE.** Headline FID 1.90 needs ~100–250 NAG-GD steps; adaptive cuts ~60% off. At strict 1-step EqM has no published number.
- **No CPU wall-clock data.** Paper reports FID and step-count distributions only; the 40%-of-fixed-compute number is FLOPs-equivalent, not CPU-time-measured.
- **Backbone cost still dominates.** Per-step is a full SiT-XL/2 forward pass (~119 GFLOPs). 100 steps × 119 GFLOPs = 12 TFLOPs per image. For audio at smaller backbone sizes (1-D conv U-Net, 1-D ViT-B) per-step CPU cost is much smaller, but the step-count multiplier remains — adaptive compute doesn't make EqM cheaper than 1-NFE; it makes it cheaper than fixed-many-step.

## Variable-NFE with monotonic quality (TiM)

The only method in the wiki where one trained checkpoint serves the full 1-NFE → many-NFE range with monotonic quality improvement on a standard ODE-style sampler. Runtime knob is the number of finite-interval transitions taken along the PF-ODE; per-step cost is a standard DiT forward pass. Different axis from EqM (optimizer steps on a static field).

### Why this matters for CPU UX

Every other 1-NFE method in the wiki saturates or *degrades* past its training endpoint. FLUX.1-Schnell: GenEval 0.68 → 0.67 → 0.62 → 0.58 at 1/2/32/128 NFE. SDXL-Turbo, SD3.5-Turbo: same pattern. iMF / pMF / SoFlow / W-Flow are trained for strict 1-NFE; running them at higher NFE either does nothing useful or hits the few-step ceiling. **TiM 865M T2I**: GenEval 0.67 → 0.76 → 0.80 → 0.83 at 1/8/32/128 NFE — strictly monotonic, beating SD3.5-Large (8B) and FLUX.1-Dev (12B) at every step count.

For an interactive audio CPU UI: default to 1-NFE preview, expose an opt-in quality slider that runs 8/32/128 NFE — quality improves every time, on the same checkpoint. No alternative method supports this.

Per-step cost is still a standard DiT forward — TiM doesn't change FLOPs/step, it makes the NFE-quality curve well-behaved.

### Caveats

- **No CPU number, no GFLOPs/sample reported** — standard image-paper omission.
- **128-NFE on CPU is still expensive.** TiM resolves quality monotonicity, not the latency problem — for our target (>1× RT on CPU) we still need the 1-NFE point to be deployable. The value is "graceful refinement when affordable", not "guaranteed real-time at every NFE".
- **No head-to-head 1-NFE FID against W-Flow / iMF / pMF at matched ImageNet-256 protocol.** TiM's only ImageNet number is TiM-B/4 / 80 ep — not comparable to iMF-XL/2 / 240 ep / 1.72 FID.

### Stacking with ELIT

ELIT's runtime per-step-FLOPs knob (`J̄`) is on a different axis from TiM's NFE knob. Stacking gives a two-axis runtime budget knob on one checkpoint: per-step compute *and* number of finite-interval transitions independently per call. Compositionally clean; untested.

## What to capture for every future paper

1. Params (or FLOPs / MACs per forward pass).
2. NFE at the reported quality level.
3. Wall time as reported (any hardware, any batch).
4. Single-sample CPU forward-pass cost if we can measure or estimate it.

## Open questions

- What does a single forward pass through the CRASH U-Net cost on a recent laptop CPU at 21 000 samples? Until we have this number, we don't know how far we are from the goal.
- At which (architecture, NFE) point does the diffusion family cross from "slower than RT on CPU" to "faster than RT on CPU"?
- How does a 134 M DiT-B/16 Drifting one-step (87 GFLOPs) translate to CPU latency at 1 s audio @ ~86–256 tokens? Most promising cost/quality at 1 NFE *with* a frozen φ.
- How does a 118 M pMF-B/16 (33 GFLOPs) translate to CPU latency? Cheapest 1-NFE image data point and the most no-frills audio-portable recipe.
- Does pMF's patch-size-scaling-at-fixed-GFLOPs hold for variable audio clip length? Image: 256 tokens across 256² / 512² / 1024² → GFLOPs flat. Audio: 256 tokens across 1 s / 5 s / 10 s @ 22.05 kHz → patches of 86 / 430 / 860 samples.
- Reproduce Flow2GAN's CPU xRT on our hardware (laptop CPU, not Xeon Platinum) at batch=1 as well as batch=16.
- Does STFT-branch ConvNeXt retain its CPU advantage in *unconditional* generation, or does the absence of mel-conditioning destabilise the spectral-coefficient regression?
- **ComVo CPU xRT at matched setup** (Xeon Platinum 8457C, batch 16, 1 s @ 24 kHz). Does CVNN-everywhere stay competitive on CPU, or does the complex-kernel-vs-float32-BLAS gap blow the GPU 5.7× slowdown out to 10–20× on CPU?
- How does cosine σ schedule interact with consistency-model training, where the trajectory is collapsed across t?
- Higher-order ODE samplers (DPM-Solver, Heun) aren't tested in CRASH; would likely move few-step quality up at fixed NFE.
- Does ELIT's runtime-budget knob transfer to 1-D audio, and does it compose with 1-NFE fastforward FM?
- EqM at few-step on audio: where does FAD plateau? Sweep 4 / 8 / 16 / 32 / 100 NAG-GD steps on 1 s drums; report FAD and CPU wall-clock per sample.
- EqM adaptive-compute on audio: per-sample step-count distribution? Do simple percussive clips converge faster than sustained tones?
- TiM monotonic-NFE survival on 1-D audio. Headline is 865M T2I; does the property hold for a ~5 M 1-D conv U-Net on 30 k drum clips?
- TiM + ELIT two-axis runtime knob on audio. Composes cleanly; untested.
- DDE training-time speedup on 1-D backbones. ~2× faster than JVP at matched FID on DiT — does the same ratio hold on 1-D conv U-Net (where JVP works) and 1-D Transformer-on-waveform (where JVP blocks FlashAttn)?
- TiM + FD-loss post-train. TiM gives variable-NFE base, FD-loss polishes any single NFE point. If 1-NFE TiM + FD-loss matches W-Flow's 1.29 ImageNet FID *and* retains monotonic-NFE, that's the project's preferred recipe for audio.
