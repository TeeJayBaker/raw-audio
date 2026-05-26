# Post-training

**Purpose:** given an already-trained generator, what do you do with it? Reduce NFE, raise quality, escape the teacher ceiling, repurpose for a new objective. The cross-cutting page for methods that **require a pretrained model as input**. Distinct from [methods](methods.md), which catalogues training paradigms that build generators from scratch.

## When to post-train vs train from scratch

| Situation | Track |
|---|---|
| Working multi-step audio FM / diffusion checkpoint, want CPU-fast inference | **Post-training, this page.** APT or score distillation is faster and cheaper than re-training a 1-NFE model from scratch. |
| No pretrained model, target 1-NFE pixel/sample-space generation | **From-scratch, [methods](methods.md)** — pMF, iMF, IMM, Drifting, R3GAN. |
| Working pretrained model, want to fix a specific failure mode (sharpness, diversity, prompt adherence) | **Post-training**, specifically the relevant fix below. |
| Want 1-NFE *and* sample quality better than the teacher | **APT** (paradigm B below) — Seaweed-APT beats its own 25-NFE teacher on human eval at 1-NFE. The teacher is a soft ceiling, not a hard one. |

Every post-training paper in the wiki uses dramatically less compute than the pretrain it came from: PeriodWave-Turbo 1k fine-tune steps; Seaweed-APT ~300 G-updates; ARC 100k iters at batch 256 on 8×H100; DMD2 SDXL 5 days. Cost is dominated by hardware and batch size, not iteration count.

## The three adversarial post-training paradigms

| Paradigm | What gets trained | Distillation? | Refs |
|---|---|---|---|
| **(A) Adversarial distillation** | Student initialised from teacher; trained on both score-regression *and* adversarial vs real data. | Yes, throughout. | [ADD/SD-Turbo](../../papers/2311.17042.md), LADD, [DMD2](../../papers/2405.14867.md), SiDA |
| **(B) Adversarial post-training (APT)** | Distillation only as a warm-start; then drop the distillation loss entirely and continue with pure adversarial-vs-real-data. | Only as init. | [Seaweed-APT](../../papers/2501.08316.md), [ARC](../../papers/2505.08175.md), AAPT |
| **(C) FM-pretrain + GAN finetune** | Stage 1: FM pretrain from scratch (see [methods](methods.md)). Stage 2: unroll `N=1/2/4` steps and GAN-finetune end-to-end. | None — the stage-1 pretrain *is* the teacher knowledge. | [Flow2GAN](../../papers/2512.23278.md), [PeriodWave-Turbo](../../papers/2408.08019.md) |

The 2025 paradigm shift was recognising that (B) APT exists as a separate finishing move from (A) distillation. Seaweed-APT proved that after diffusion pretraining the right finishing move is pure adversarial against real data; ARC proved the same for audio. **Score distillation is not an inevitable component.** For raw-audio CPU inference the recommended track is (B) APT because: no teacher in memory after warm-start (lighter VRAM than DMD2's three-network setup); CFG-free at inference via ARC's contrastive trick (halves per-step cost); ARC reports diversity *increases* vs teacher (paradigm A's known weakness).

### (A) Adversarial distillation — ADD lineage

[ADD](../../papers/2311.17042.md): `L = L_adv^G + 2.5 · L_distill`. Adversarial = hinge-GAN with a frozen DINOv2 backbone + trainable lightweight heads at multiple feature scales (R1=10⁻⁵, text+image-CLS conditioned). Distillation = score regression on a re-noised student output with NFSD weighting on `c(t)`. Structural lesson: freeze a strong perceptual backbone + train small heads → solves "GAN-from-scratch unstable". Audio analogue is freezing CLAP/MERT/AudioMAE/EnCodec, but audio backbones are weaker than DINOv2 so the head ensemble must compensate. ADD acknowledges but does not address mode collapse — DMD2 is the principled fix.

[DMD2](../../papers/2405.14867.md) — the cleanest score-distillation + real-data GAN recipe and the right baseline for distilling a future raw-audio FM teacher. Three changes vs original DMD:

1. Drop the LPIPS regression dataset (was 700 A100-days at SDXL scale).
2. **TTUR** — 5 fake-critic updates per generator update; stabilises pure distribution matching.
3. **GAN loss against real data** plugs the hole: reverse-KL is mode-seeking → without GAN, score distillation collapses modes. The discriminator attaches to the fake critic's bottleneck so most of D's params are already the fake critic.

Ablations: DMD+TTUR alone (FID 2.61) and GAN alone (2.56) are strictly worse than the combo (1.51) — complementary, not redundant. Backward simulation for multi-step closes the train-test mismatch. 4-step SDXL beats its own 100-NFE teacher. Three-network training (teacher + fake critic + discriminator) is the VRAM cost — tight on a 24 GB 4090 for audio.

### (B) Adversarial post-training — APT lineage

[Seaweed-APT](../../papers/2501.08316.md) — paradigm-defining image+video paper. Drops all distillation/score/trajectory objectives after pretraining; finishes with pure non-saturating GAN against real data. Three load-bearing ingredients:

- **Approximated R1** `‖D(x) − D(x+σε)‖²` — first-order, FSDP/FlashAttention-compatible (true R1 requires double-backward, not supported inside FSDP). λ=100, σ=0.01 image / 0.1 video. Without it the discriminator collapses to 0 within a few hundred steps.
- **Transformer discriminator initialised from the diffusion model**, with cross-attention logit heads tapped at multiple depths (16, 26, 36) — depth ensemble in D.
- **Timestep ensemble** input: `D(x,c) := E_{t ~ shift(U(0,T), s)} D̂(x_t, c)`. s=1 image / s=12 video. D's backbone was trained to consume noised inputs across all t, so the ensemble is over the parameter of the network, not a corruption of the data.

First 1-NFE 720p24 2 s video; beats own 25-NFE teacher on visual fidelity — distillation ceiling is real and APT escapes it.

[ARC](../../papers/2505.08175.md) — audio-domain APT applied to Stable Audio Open (RF latent stereo, 44.1 kHz, ~12 s). Distillation-free, CFG-free, 8-NFE generator. Loss:

```
L_R + λ · L_C,  λ = 1
L_R(φ,ψ) = E[ f(Δ_gen − Δ_real) ]              # relativistic, paired by prompt
L_C(ψ)   = E[ f(Δ_real(x,P[c]) − Δ_real(x,c))] # contrastive on shuffled prompts
f(x) = −log(1 + e^{−x})
```

Two non-obvious choices: disentangled gen/disc noise levels (`p_gen` and `p_disc` are different distributions sampled independently per pair); pairing by prompt (Bradley-Terry preference interpretation). Discriminator initialised from 75% of the RF DiT blocks + a small conv head. Ping-pong sampler (denoise-renoise) at 8 steps. **8-step ARC beats 100-step SAO on FD and CCDS with Div MOS 4.4 vs 4.0** — the relativistic+contrastive structure escapes the ADD/LADD mode-collapse trap. No R1 reported — relies on relativistic structure for stability.

Together these define the APT design vocabulary: warm-start → drop distillation → adversarial against real data, with R1-approx as the standard stability fix when D is a transformer. ARC proved the recipe works on conditional latent audio; the open question is whether it ports to raw-waveform / unconditional generation.

### (C) FM-pretrain + audio-specific finetune — FM+GAN / FM+CM dichotomy

Stage 1: train an FM model from scratch with audio fixes — `x₁`-prediction, spectral-energy-inverse weighting *or* mel-conditioned prior, multi-resolution STFT loss; see [methods](methods.md). Stage 2 splits into two finishing moves:

**(C1) FM + GAN finetune** — unroll for `N=1/2/4` steps and adversarially fine-tune against MPD+MRD for ~1k–110k iters. Separate model per N.

- [Flow2GAN](../../papers/2512.23278.md): STFT-domain ConvNeXt; 1-NFE at 78.9 M, CPU xRT 4.85 on Xeon — the only published >real-time CPU number for unconditional-style audio. FM-only at 16 steps loses to FM-pretrain + 1-NFE GAN finetune — the GAN stage is load-bearing at low NFE.
- [PeriodWave-Turbo](../../papers/2408.08019.md): waveform-domain multi-period U-Net at 70 M, CPU xRT 0.12 — same recipe but ~40× slower than Flow2GAN.
- Both train with multi-scale mel-perceptual reconstruction. Raw-STFT L1 alone produces electric noise (phase-error gradient misleads). DMD-style score distillation does *not* help when mel conditioning is strong — the conditional distribution is already pinned. Open whether DMD matters under weaker / unconditional conditioning.

**(C2) FM + consistency-distillation finetune** — same FM pretrain, but the 1-NFE finishing move is consistency distillation against an EMA teacher, not adversarial.

- [WaveFM](../../papers/2503.16689.md): waveform-domain HiFi-GAN-MRF 1-D U-Net (19.5 M); 1 M FM pretrain steps + 25k CM-distillation steps (~2 h on a 4090) with truncated-Gaussian `t ~ Ñ(0, 0.33²)` on `[0, 0.99]`, EMA μ=0.999, Δt=0.01, L₂ + STFT-with-phase + mel-L1. Targets set to clean waveforms at `t=1` (deviates from standard CM's `c_skip x + c_out F_θ` parameterisation, claimed incompatible with their `x₁`-reparameterisation). CPU xRT 0.64 — sub-RT. Per IMM Lemma 1, this is the M=1 energy-kernel corner of inductive MMD — natural upgrades are M=2 + Laplace / signature-kernel.

**The architectural axis (STFT-domain hop-rate vs waveform-domain sample-rate) dominates everything else in the CPU budget, and the finishing move (GAN vs CM) does NOT change this.** Flow2GAN (STFT) beats both PeriodWave-Turbo (waveform + GAN) and WaveFM (waveform + CM) by ~10× on CPU despite the latter being the smallest of the three.

FM+CM vs FM+GAN at audio (summary):

| Axis | FM + CM (WaveFM) | FM + GAN (Flow2GAN, PeriodWave-Turbo) |
|---|---|---|
| Stage 2 supervision | EMA teacher's prediction at next `t+Δt` | Adversarial vs real audio + mel-L1 |
| Networks in VRAM | Student + EMA teacher (~2× student) | Generator + MPD + MRD (3 networks) |
| Stage 2 stability | High (deterministic target) | Moderate (GAN tuning) |
| Stage 2 iters | 25k total | 11–110k per N |
| Separate model per NFE? | No | Yes (`N=1/2/4` separate) |
| Mode-collapse failure | Brittle to distribution shift (1-NFE OOD gap wider than 6-NFE) | "Plastic-looking", needs diversity diagnostics |
| Quality at 1-NFE (LibriTTS PESQ) | 3.51 (waveform, 19.5 M) | 4.19 (STFT, 78.9 M) |
| CPU xRT @ 1-NFE | 0.64 | 4.85 |

The untested cell is FM + CM on an STFT-domain backbone — WaveFM's distillation on Flow2GAN's architecture. The empty cell on the *waveform* domain is the unconditional version of either recipe (both Flow2GAN and WaveFM are mel-conditional vocoders).

## Non-adversarial post-training

### Distillation: trajectory vs score

When there's a pretrained multi-step teacher (the realistic audio scenario), the dominant divide:

- **Trajectory distillation** (consistency / sCM, LCM, SANA-Sprint) — student reproduces the teacher's ODE path. Sensitive to schedule, preconditioning, parameterisation. SANA-Sprint had to convert rectified-flow checkpoints into TrigFlow before consistency distillation worked.
- **Score distillation** (DMD, SiD, [SiD-DiT](../../papers/2509.25127.md), Diff-Instruct, VSD) — student matches only the teacher's score field (Fisher-style divergence) at sampled `(x_t, t)`. Free to take shortcuts; consistently better few-step quality on diffusion benchmarks.

SiD-DiT is the cleanest score-distillation recipe for FM teachers: under Gaussian corruption, FM has an implicit score `S = −(x_t + (1−t)v^{FM})/t` derivable from velocity, so Fisher divergence is well-defined for any FM teacher (rectified-flow, TrigFlow, v-prediction). Same SiD code across SANA / SD3 / SD3.5 / FLUX with only timestep rescaling — no teacher finetuning, no architectural change.

Useful mental model: every loss (`x₀ / ε / v / FM-velocity`) has the same optimal solution `E[x_0 | x_t]`. They differ only by the weight-normalised timestep distribution `π(t) = w_t · p(t) / ∫ w_t · p(t) dt`. Distillation methods derived for one parameterisation port to others by re-tuning `p(t)` and `w_t` so `π(t)` matches.

Trajectory-vs-score composes with the adversarial axis: ADD = trajectory + adversarial; DMD2 = score + adversarial; LADD = trajectory + adversarial (discriminator inside backbone). Full design space: `{trajectory, score} × {pure, +real-data-GAN, +R1-approx, +contrastive}`.

### FD-loss as post-training

[Representation Fréchet Loss](../../papers/2604.28190.md) breaks the "FD as eval-only" wall by decoupling the population used to *estimate* moments from the batch that carries *gradients* — MoCo-style queue of recent generated features or EMA on first/second moments. Backprop only through the current ~1k batch.

One-step generator post-trained with FD-loss reaches **0.72 FID @ 1 NFE on ImageNet-256**. Repurposes multi-step generators into 1-NFE generators without teacher distillation, GAN training, or per-sample targets. φ-agnostic — drops onto any audio embedding (VGGish / CLAP / AudioMAE).

Stacked pipeline: train one-step from scratch (iMF, pMF, IMM); polish with FD-loss to push distributional match further at 1 NFE. The FD-loss paper post-trains an iMF base — iMF + FD-loss is canonical; pMF + FD-loss is the pixel-space analogue.

### Representation-level distillation (AudioDEAR)

Audio-domain auxiliary: `L = L_one_step + λ · MSE(h_S, h_T)` with `h` = final-layer backbone hidden state, `λ ≈ 1000`. Student free to learn its own output map; only its internal representation is pulled toward the teacher's. Composes with any one-step loss; constraint is matching backbone shape.

[AudioDEAR](../../papers/2605.00329.md) shows this auxiliary closes most of the gap to a 100-step teacher while keeping the one-step loss in the lead role.

## Stability ingredients that compose across paradigms

| Technique | Role | Modern usage | Audio analog |
|---|---|---|---|
| **R1** (Mescheder 2018) | `‖∇_x D(x_real)‖²`; locally convergent | StyleGAN, ADD, DMD2, R3GAN, ARC | Drop-in on MPD/MRD; sporadic in audio |
| **R2** (R3GAN insight) | R1 on fakes too; closes fake-side gradient loop | R3GAN; not widely adopted yet | High-EV one-line addition, untested in audio |
| **R1-approximation** (Seaweed-APT) | `‖D(x)−D(x+σε)‖²`; first-order, FSDP-friendly | Mandatory for scaled transformer discs | Critical if transformer disc; less so for small conv discs |
| **Spectral norm** (Miyato 2018) | 1-Lipschitz D via power iteration | BigGAN, StyleGAN-T, HiFi-GAN MSD | Already common; compose with R1+R2 |
| **Frozen pretrained backbone** (ADD) | Discriminator = frozen feature net + small heads | ADD, LADD, ARC, Projected-GAN, GigaGAN | Audio backbones weaker than DINOv2 — compensate with more head capacity |
| **TTUR** (Heusel 2017) | Different LR for G and D, or `n_D` updates per `n_G` | DMD2 (5:1 D:G), most StyleGAN-era | Universal; no reason not to use |
| **Discriminator-init-from-generator-backbone** | Reuse the trained backbone as D's feature extractor | ADD, LADD, ARC (75% of RF DiT) | Direct port: use the FM model's stack as D's backbone |

## Mode/diversity collapse — three orthogonal fixes

Known failure mode of adversarial post-training: sharpness at the cost of diversity. Three composable fixes:

1. **RpGAN pairing + R2** ([R3GAN](../../papers/2501.05441.md)) — Sun et al.'s landscape result + Mescheder's local-convergence proof extended to RpGAN. Stacked-MNIST 1000/1000 modes. Strictly from-scratch result but the loss form ports to post-training.
2. **Contrastive discriminator term** ([ARC](../../papers/2505.08175.md)) — D must score correctly-paired (audio, condition) above shuffled-paired. Replaces CFG without baking it in. ARC reports diversity *increases* over teacher.
3. **Real-data GAN on top of score distillation** ([DMD2](../../papers/2405.14867.md)) — reverse-KL is mode-seeking; real-data discriminator injects mode-covering pressure. Complementary, not redundant.

Orthogonal — target different mechanisms. A future audio recipe could stack all three: RpGAN+R1+R2 (fix loss landscape) + contrastive D (replace CFG) + real-data GAN as part of any score-distillation stage. See [evaluation](evaluation.md) for the diagnostic stack (Vendi + Coverage + CCDS).

## Cost and structural footprint

| Method | Networks at train time | NFE achieved | Iters needed | Notes |
|---|---|---|---|---|
| ADD | G + frozen DINO + teacher + heads | 1–4 | medium (~100k SDXL) | Distillation + GAN both active; mode collapse |
| LADD | G + teacher + heads-inside-backbone | 1–4 | medium | D moves inside the backbone |
| DMD2 | G + frozen teacher + fake critic + small D | 1–4 | medium | Three networks in VRAM — heaviest |
| Seaweed-APT | G + transformer D | 1 | small (~300 G-updates) | Drop teacher after warm-start |
| ARC | G + D (~75% G's backbone + conv head) | 8 | medium (100k iters) | Pure adversarial; CFG-free at inference |
| PeriodWave-Turbo | G + MPD + MS-SB-CQTD | 2–4 | tiny (~1k steps) | FM-pretrain dominates; finetune sharpens only |
| Flow2GAN | G + MPD + MRD | 1/2/4 | small (~11–110k) | Separate model per N; STFT backbone |
| WaveFM | G + EMA teacher (no D) | 1 (also runs at 6) | tiny (25k steps, ~2h on 4090) | One model serves multiple N |
| FD-loss | G + frozen φ + EMA stats | 1 | small | No D, no teacher, no per-sample target |
| SiD-DiT | G + frozen FM teacher + fake critic | 4 | medium | Score distillation only, no GAN |
| AudioDEAR | G + frozen multi-step teacher (backbone-matched) | 1 | medium | Auxiliary composes with one-step loss |

APT (B) is the lightest among adversarial options; FD-loss is the lightest overall (no discriminator). DMD2 is the heaviest.

## Translation to raw audio

Most directly portable, in order:

1. **APT on an FM audio teacher.** ARC already proved this works for latent T2A. The recipe ports directly — warm-start FM with deterministic distillation, drop the distillation loss, continue with relativistic + contrastive adversarial loss against real data. Discriminator initialised from the FM backbone. **Recommended primary track.**
2. **FD-loss post-training** of any one-step or multi-step generator. φ-only and architecture-agnostic. Canonical stacked pipeline for the project: from-scratch iMF/pMF/IMM/EqM → FD-loss polish at 1 NFE.
3. **DMD2-style score-distillation + real-data GAN** if we want a teacher-supervised path. Heavier on VRAM but the cleanest principled solution to mode collapse for score-distillation students. Compare against APT on the same audio teacher.
4. **ARC's contrastive discriminator** as a CFG-killer for any conditional audio model — possibly without the full APT machinery. One-line addition; testable cheaply.
5. **SiD-DiT** for pure score distillation when there's a multi-step FM audio teacher. Parameterisation-agnostic; works on rectified-flow / TrigFlow / v-prediction without conversion.
6. **AudioDEAR's representation distillation** as a backbone-matched auxiliary to any from-scratch one-step recipe.
7. **PeriodWave-Turbo / Flow2GAN FM-pretrain + GAN finetune** is the audio-specific recipe with the best CPU number today; structurally a hybrid between this page and [methods](methods.md).
8. **WaveFM FM-pretrain + CM finetune** is the CM analogue of (7) — same FM pretrain skeleton, deterministic-teacher CM finishing move. Cheaper / more stable stage 2 (no D balancing, no mode-collapse instrumentation), single student model serving multiple NFE counts. Drop-in upgrade per IMM Lemma 1: M=2 + Laplace / signature kernel.

### Audio-specific lessons (don't lose these on a port)

- **R1-approx is mandatory** if the discriminator is a transformer (Seaweed-APT collapses without it). For small conv-only audio discs (MPD/MRD/MS-SB-CQTD) it's recommended, not load-bearing.
- **R2 on audio discriminators is the highest-EV single experiment** suggested by R3GAN's theory — no audio paper has tested it.
- **Discriminator-init-from-FM-backbone** (ADD/LADD/ARC). Audio doesn't have a DINOv2-quality external backbone; using the FM model's own learned features as D's backbone sidesteps that gap.
- **Multi-scale mel-perceptual reconstruction** stays; raw STFT L1 is a dead end (PeriodWave-Turbo: collapses to electric noise — phase-error gradient misleading).
- **CFG-free via contrastive D** is a real win — audio FM models that bake in CFG pay double per-NFE; ARC's contrastive loss removes the doubling without losing prompt adherence.

## Open questions

- Does Seaweed-APT-style pure adversarial post-training work for **unconditional raw-audio FM teachers** at small (50–500 M) scale with conv discriminators? ARC proved it for latent T2A; nothing tested for raw waveform or unconditional.
- Does ARC's contrastive discriminator term remove CFG cleanly from existing audio FM models without the full APT machinery?
- Does DMD2-style score-distillation + real-data GAN beat APT on a weakly-conditional or unconditional audio teacher? PeriodWave-Turbo's null DMD result was strong-conditioning-specific.
- **Does adding R2 to MPD/MRD discriminators** improve stability and diversity on audio? One-line change; R3GAN's theory predicts yes.
- R1-approx vs true R1 on small conv audio discriminators — does the FD form give equivalent stability when both are tractable?
- Does iMF + FD-loss or pMF + FD-loss on a raw-waveform student work, and does the image-side 0.72-FID-from-3.12 jump carry over?
- Stability cost of dropping mel conditioning for unconditional generation in paradigm-C recipes (Flow2GAN-style and WaveFM-style)? Both papers depend on mel conditioning to pin the marginal.
- **FM + CM vs FM + GAN at matched architecture.** WaveFM and Flow2GAN aren't comparable head-to-head (different backbones). Clean A/B: pick STFT-domain ConvNeXt, pretrain reparameterised FM once, finetune two copies (CM vs GAN) at matched compute.
- **FM + CM on an STFT-domain backbone.** WaveFM's distillation + Flow2GAN's architecture. Most natural unconditional waveform candidate; untested.
