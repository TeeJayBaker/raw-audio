# Representation space (pixel/sample-space vs latent)

**Purpose:** answer "why generate in *raw* sample space when latent diffusion is the dominant paradigm?" — the framing that defines this whole project. Track what the image-domain literature has established about the trade and what ports to raw audio.

## The trade

| Axis | Pixel space (raw audio) | Latent space (VAE / codec then diffuse) |
|---|---|---|
| **Quality ceiling** | Bounded only by training compute. | Bounded by the VAE/codec. Reconstructions outside the encoder's distribution are imperfect; that floor is permanent. |
| **Training pipeline** | One model. | Two: VAE/codec, then generator. Compounded failure modes. |
| **Per-step FLOPs** | High — model operates on raw samples. | Low — model operates on compressed latents (8× or 64×). |
| **Total inference cost** | High per step × NFE. | Low per step × NFE + decoder cost. |
| **Adaptability to new data** | Direct. | Limited by VAE distribution; retraining the VAE is expensive. |
| **Phase / waveform-specific concerns (audio)** | Native. | Codec may discard phase or introduce artefacts at low bitrate. |

The pixel-space penalty is a compute penalty, not a fundamental one — and that penalty is shrinking.

## Current image-side evidence

Pixel-space diffusion was treated as a research curiosity for years (ADM, RIN, VDM++, SimpleDiffusion all lagged latent diffusion in FID *and* compute). EPG broke this for multi-step pixel diffusion in 2025; **JiT** showed a plain ViT with `x`-prediction alone — no SSL, no perceptual, no enc/dec asymmetry — was sufficient; **pMF** closed the 1-NFE gap in 2026 by stacking JiT's `x`-prediction with iMF's MeanFlow.

## Audio-side evidence

The image-side argument now has its first audio counterpart: **[WavFlow](../../papers/2605.18749.md) (2026)** ports JiT's `x`-prediction + patched-transformer recipe to **raw 16 / 44.1 kHz waveform** and matches latent SOTA (MMAudio, HunyuanVideo-Foley) on VGGSound V2A and AudioCaps T2A at 624 M / 1.03 B params. No codec, no VAE, no learned audio tokenizer in the inference graph — entry/exit is parameter-free `T → (C, D)` patchify / unpatchify.

What WavFlow demonstrates *audio-side* that pMF / JiT demonstrated image-side:

- **Manifold assumption carries to 1-D audio.** `x`-prediction with `v`-loss strictly beats pure `x`- or `v`-prediction in WavFlow's ablation, on the same coverage-vs-fidelity axis pMF identifies in pixel space.
- **Patched-ViT is the working backbone family at scale.** 200-sample patches at 16 kHz (12.5 ms per token) — below the human auditory resolution threshold — produces the equivalent of pMF's 256-token regime, transferred to 1-D.
- **No SSL pretrain, no VAE, no perceptual loss required to converge.** Same minimal stack as JiT (no SSL) + pMF (no GAN, no LPIPS). Audio-domain additions are amplitude lifting (preprocessing, parameter-free) and frozen CLIP/Synchformer for conditioning (no impact on the unconditional argument).

What WavFlow does *not* demonstrate:

- **1-NFE viability on raw audio.** WavFlow is 50 NFE; no MeanFlow / consistency / few-step variant is reported. The pMF → audio 1-NFE port remains an open recipe.
- **CPU deployment.** 1 B params × 50 NFE × 8 s @ 16/44.1 kHz is well above the project's CPU + 1-NFE target. WavFlow validates the *recipe*, not the deployment shape.
- **Sub-1-second clips.** WavFlow operates exclusively at 8 s, where 5 M clips of supervision is needed. The project's 0.1–1 s target with much smaller corpora is a different operating point.

The "this is the bet this project is making" framing is now: the recipe is real (WavFlow), the smaller-scale 1-NFE-on-CPU instantiation remains the open work.

### Structural progenitor — JiT

JiT demonstrates that pixel-space transformer generation requires almost nothing beyond predicting `x` instead of `ε`/`v`. Mechanistic argument: under the manifold assumption, clean data lies on a low-d submanifold of `ℝ^D` while noise and velocity fill the full ambient space; a limited-capacity network can output a manifold-valued quantity (encoding `d` numbers/token) but cannot output a full-dim Gaussian (would need `D` numbers/token, exceeding capacity).

JiT-B/16 ImageNet-256 (Tab. 2a): `x`-pred works across all three losses; `ε`/`v`-pred fail catastrophically across all three losses (FID ~370–395). Catastrophic regime is specific to **per-patch dim ≳ network hidden dim** — at JiT-B/4 with 48-d patches (low-dim regime), all nine prediction × loss combinations work.

Headline (multi-step Heun-50, no SSL/perceptual/GAN/representation-alignment): JiT-G/16 FID 1.82 on ImageNet-256 at 2 B params / 383 GFLOPs. Token count constant at 256 across 64× pixel-count growth (256² → 1024²); compute essentially flat (25 → 30 GFLOPs for JiT-B). Origin of the patch-size-scaling-at-fixed-FLOPs property.

Implications across the wiki:

- **pMF inherits JiT's `x`-prediction as a dependency.** pMF Tab. 2: `u`-prediction collapses to 164 FID at patch-dim 768; `x`-prediction stays at 9.56. Without JiT's analysis, pMF's pixel-space MeanFlow does not exist.
- **EPG's asymmetric enc/dec + SSL is one route, not the only one.** JiT's plain symmetric ViT with no SSL reaches FID 1.82 at 256², within ~0.2 FID of EPG-G/16's 1.58 at similar compute.
- **Multi-step pixel-space at low GFLOPs is now solved** — JiT-B/16 at 25 GFLOPs / 131 M / FID 3.66 is well below DiT-XL/2's ~168 GFLOPs end-to-end including VAE.
- **1-NFE is not solved by JiT** — collapses without pMF's MeanFlow machinery. JiT is the multi-step baseline; pMF is the 1-NFE upgrade.

### Multi-step regime — EPG

EPG-XL/16 reaches DiT/SiT-level FID at ~30% of DiT-XL's GFLOPs (accounting for the VAE forward), with no VAE training stage. EPG-G/16 (1.39 B / 321 GFLOPs / FID 1.58) sets multi-step pixel-space SOTA.

### Discrete-tokenizer regime — BAR

The pixel-vs-latent axis above implicitly assumes "latent" means *continuous* learned latent (SD-VAE, MAR-VAE). A third operating space — **discrete learned tokenizer** (VQ-VAE, FSQ, BSQ, LFQ) — was historically treated as strictly inferior. [BAR](../../papers/2602.09024.md) demonstrates this is purely a compression-ratio artifact.

Bit Budget metric for direct comparison:

```
B_discrete    = (H/f)·(W/f)·⌈log₂ C⌉              # spatial · bits-per-token
B_continuous  = (H/f)·(W/f)·D·16                  # spatial · channels · 16 bits mixed-precision
```

At matched bit budget, BAR-FSQ at `C = 2³²` reaches rFID 0.33, beating SD-VAE (0.62) and MAR-VAE (0.53). BAR-L 1.1B sets new ImageNet-256 SOTA across both paradigms (gFID 0.99 with CFG). The historical discrete-vs-continuous gap was a tokenizer-bit-budget gap, not a generator-paradigm gap.

**Cost-shape caveat — discrete with scaled bits is NOT cheap inference.** The win is in quality and parameter efficiency. Generator is standard AR transformer + Masked Bit Modeling head: `L = 256` outer AR steps × `N = 2–5` inner MBM steps. No 1-NFE BAR variant. See [methods → Masked-token generation](methods.md#masked-token-generation--discrete-ar-with-bit-modelling).

### One-step regime — pMF closes the gap, W-Flow reopens it

| Method | Space | NFE | Params | GFLOPs | FID |
|---|---|---|---|---|---|
| W-Flow-XL/2 | latent | 1 | 679 M + 49 M φ + VAE | — | **1.29** |
| W-Flow-L/2 | latent | 1 | 463 M + 49 M φ + VAE | — | 1.35 |
| Drifting-L/2 | latent | 1 | 463 M + 49 M φ + VAE | — | 1.54 |
| iMF-XL/2 (φ-free) | latent | 1 | 610 M + 415 M | 146 + 106 (VAE) | 1.72 |
| **pMF-H/16** | **pixel** | 1 | 956 M | **271** | **2.22** |
| pMF-L/16 | pixel | 1 | 410 M | 117 | 2.52 |
| **pMF-B/16** | **pixel** | 1 | **118 M** | **33** | **3.12** |
| StyleGAN-XL (prior pixel SOTA, GAN) | pixel | 1 | 166 M | 1574 | 2.30 |
| EPG-L/16 (prior pixel SOTA, diffusion) | pixel | 1 | 540 M | 113 | 8.82 |

Read-off:

1. **The pixel-vs-latent 1-NFE gap is now ~1 FID** (pMF-H 2.22 vs W-Flow-XL 1.29). Wider than the pMF-vs-iMF gap but small compared to the 1-step-vs-multi-step bridge.
2. **Both latent SOTAs at 1-NFE (W-Flow, Drifting) depend on a learned feature encoder φ.** φ-free latent (iMF) sits at 1.72; with-φ (W-Flow) at 1.29. φ-dependence is the sharpest axis: Drifting/W-Flow paid a φ-pretrain stage to escape the iMF/pMF ceiling, but neither has a working pixel result. On audio, the φ-pretrain converts into needing AudioMAE / EnCodec / waveform-MAE.
3. **pMF-B/16 is the cheapest 1-NFE result in absolute terms** — 33 GFLOPs, 118 M, 3.12 FID, no VAE, no φ. Closest existing data point to "could plausibly run faster than RT on CPU".
4. **The 2026 jumps are structural, not incremental.** pMF stacks iMF's v-loss/JVP machinery + JiT-style x-prediction + pixel-space perceptual loss (LPIPS + ConvNeXt-V2 directly on `x_θ`, only possible because the network outputs pixels). The latent-side jump (1.72 → 1.29) is also structural: kernel mean-shift (MMD-WGF) → Sinkhorn-WGF + two-batch self-transport via the `V = −∇ δF/δq` template. Neither family has yet ported its trick to the other side.

### High-resolution pixel-space at fixed compute (pMF)

| Resolution | Patch | Patch dim | Params | GFLOPs | 1-NFE FID |
|---|---|---|---|---|---|
| 256² | 16² | 768 | 956 M | 271 | 2.22 |
| 512² | 32² | 3072 | 962 M | 272 | 2.48 |
| 1024² | 64² | 12288 | ~970 M | ~273 | 4.58 |

512² costs the same GFLOPs as 256² — extreme overlap with the cost shape we want for variable-length audio (hold token count constant, grow patch size with clip length).

## Why this matters for raw audio

Audio has three "tokenised latent" paradigms in active use, which map onto the image-side discrete-vs-continuous axis differently:

1. **Continuous learned latent** — Stable Audio / AudioLDM VAEs. The direct audio analogue of SD-VAE / MAR-VAE. Same trade as image-side latent diffusion: codec floor, two stages, phase artefacts.
2. **Discrete learned latent (small codebook)** — EnCodec, DAC, SoundStream at typical bitrate (RVQ 4–8 codebooks × 1024 entries). Audio-side equivalent of pre-BAR discrete tokenizers, sitting at modest bit budgets (~32 bits/frame).
3. **Discrete learned latent (scaled codebook)** — no published audio analogue exists. BAR's central thesis (codebook scaling closes the discrete-continuous gap) has never been tested for audio tokenizers. Open question BAR opens for audio.

For path 2 (the dominant audio path today via MusicGen / VALL-E / AudioGen / Stable Audio Open's RVQ stack), BAR's contribution is the MBM head as a vocabulary-scaling primitive — if anyone scaled an audio tokenizer's codebook to `2²⁰+`, MBM is the only currently-published way to predict it without softmax OOM. Currently-uninteresting because no audio tokenizer is bit-budget-starved enough to need it.

Audio latent-space inheritances:

- **Codec floor.** EnCodec at low bitrate is audibly compressed. The generator can never exceed reconstruction quality.
- **Two training stages.** Most audio codecs are trained on speech-heavy mixed corpora; music reconstruction is mediocre.
- **Phase coding artefacts.** Codecs throw away or quantise phase; instrument transients and reverbs suffer.
- **Compute savings are real.** A codec at 25 Hz token rate vs 16 kHz waveform = 640× compression. The compute argument for latent audio is *much* stronger than for latent image (8× compression).

EPG's lesson: the right architecture (asymmetric ViT, encoder/decoder split, patch scaling) + the right training recipe (SSL pre-training, then end-to-end) closes most of the FLOPs gap at multi-step. pMF's lesson is sharper: at 1-NFE the gap is effectively closed without SSL pretraining at all — three structural pieces (x-prediction, v-loss decoupling, pixel-space perceptual loss) suffice. **WavFlow has now ported the multi-step half of this argument to audio at large scale**, with two audio-specific additions (amplitude lifting; no noise-level shift). The 1-NFE half — pMF's MeanFlow upgrade applied to a raw-waveform patched ViT — is still open and is the project's central recipe gap.

## Four operating spaces for audio generative modelling

Raw waveform is the project's target; the other three are floors-with-different-shapes that this project rejects but the wiki tracks for differentiation.

| Axis | **Raw waveform** | Mel-spectrogram | Waveform VAE latent | SSL-feature space |
|---|---|---|---|---|
| Transform type | None (identity) | Deterministic, lossy (linear mel filterbank on `\|STFT\|`, log-compression) | Learned, lossy (encoder trained jointly w/ decoder reconstruction loss) | Learned, semantic (SSL pretrain w/ contrastive / masked-prediction loss, denoise-invariant by design) |
| Pretrain stages before generator | 0 | 0 (deterministic) | 1 (VAE) | 1 (SSL pretrain) — or 2 stacked over a VAE generator |
| Compression ratio | 1× (16 kHz samples) | ~200× (80 mel bins × 80 Hz frames vs 16 kHz) | 25–80 Hz tokens × 64–256 d (~80–600×) | WavLM-Large: 50 Hz × 1024 d (~16× of raw values, but 1024-d carries phonetic abstraction not signal energy) |
| What's preserved | Phase, transients, full bandwidth | Magnitude envelope only (phase discarded); vocoder must reconstruct phase from scratch | Whatever the VAE encoder kept — phase compromised, narrowband artefacts | Phonetic content, prosody, long-range dependencies; signal-level detail removed by design |
| What's lost | Nothing | Phase (must be hallucinated); time-frequency resolution from STFT window choice | Codec/VAE floor on detail recovery | All signal-level reconstruction |
| Inversion to waveform | n/a | Griffin-Lim / learned vocoder | VAE decoder | Separate decoder required — composes the SSL floor with the decoder floor |
| Audio MeanFlow data point | None — empty cell | IntMeanFlow (CosyVoice2 / F5-TTS) | SplitMeanFlow (Doubao TTS, Seed-TTS acoustic latent) | MeanFlowSE (WavLM-Large conditioning + WaveVAE generative latent) |
| Other audio examples | CRASH (diffusion, no MeanFlow) | RFWave on complex STFT — different from mel | Stable Audio, AudioLDM, AudioDEAR backbone | Some non-MeanFlow audio LMs (SELM-style) |

Every published audio MeanFlow operates on a frozen intermediate representation; raw-waveform MeanFlow is the empty cell. The three published cases inherit different floors:

- **IntMeanFlow** on mel — phase entirely discarded, must be reconstructed by a downstream vocoder. Two stages: MeanFlow on mel + frozen vocoder. Useless for the project's goal (it's a TTS recipe; the vocoder *is* the codec floor).
- **SplitMeanFlow** on Seed-TTS acoustic-diffusion latent — VAE-style learned latent with whatever phase/transient artefacts that codec inherited. Two stages: MeanFlow on latent + frozen decoder.
- **MeanFlowSE** on SSL + WaveVAE — three compounded floors (WavLM speech-only representation, WaveVAE frozen reconstruction, noise level). RTF 0.013 on 4090 is dominated by frozen WavLM-Large (316 M) doing the phonetic abstraction outside the trainable budget. The "compact 40.7 M MeanFlow" framing is misleading once the full pipeline is accounted for. No CPU number.

Compositional structure shared by all three: trainable MeanFlow head + frozen feature extractor / decoder. The trainable component is small (4–60 M) because the heavy lifting (perceptual abstraction, phase reconstruction, signal-level detail) is offloaded to frozen pretrained modules outside the budget. **Raw-waveform audio MeanFlow doesn't get to offload anything.**

**Why this matters for the project's lens:** every intermediate representation is a quality knob, not just a compute knob. MeanFlowSE's SSL-vs-VAE-latent conditioning ablation gives 2× WER degradation (8.5 → 18.4) at fixed model size and inference cost — the representation choice changes the achievable quality ceiling, not just the FLOPs cost. For unconditional raw-waveform generation, where there is no conditioning signal, this translates to "what perceptual loss / feature distance does the training objective use?" — which lands in pMF's perceptual-loss-on-`x_θ` recipe (MR-STFT + AudioMAE-φ + MERT combinations). The intermediate representations are the *conditioning* answer; pMF-style perceptual loss is the *raw-waveform* answer to the same underlying "use high-level features somewhere in the pipeline" intuition. The crucial difference: in the raw-waveform answer the high-level features are used as a *loss target*, not piped through the inference graph — no floor is inherited, no frozen module dominates runtime.

## What JiT, EPG, and pMF port to audio

From **JiT** (the simplest, lowest-floor recipe):

1. **`x`-prediction + `v`-loss.** The single structural lever. Verbatim port. First audio experiment: replicate JiT's nine-cell prediction × loss matrix on 1-D audio at typical configs (P·C ≤ 1000). Predict: `x`-pred succeeds across all losses; `ε`/`v`-pred catastrophic failure depends on whether per-patch dim ≳ hidden_dim — at audio's natural patch sizes the failure may *not* trigger.
2. **Plain 1-D ViT with general-purpose Transformer ingredients** — SwiGLU, RMSNorm, RoPE-1D, QK-norm, in-context class tokens × 32. All dimension-agnostic.
3. **Token-count-constant patch-size scaling.** Constant FLOPs across clip length — single most important structural property for variable-length audio.
4. **No SSL / no perceptual / no GAN / no representation alignment.** Audio JiT can be *more* minimal than image JiT — no LPIPS or DINO are available for audio at competitive quality anyway. Strip everything; add MR-STFT loss only if vanilla MSE underperforms.

From **EPG**:

1. **Patched ViT enc/dec.** Patch length is the cost knob. Pick it so token count stays in 256–1024.
2. **Asymmetric scaling.** Spend capacity on the decoder, keep encoder modest.
3. **SSL pre-training** (optional now that pMF avoids it). Pre-train encoder with contrastive + temporal-consistency.
4. **One-step consistency directly in waveform space** with frozen-encoder anchor loss. Direct port if SSL stage retained.

From **pMF**:

1. **Decouple prediction space from loss space.** Network outputs `x_θ` (raw waveform on the audio manifold); loss lives in v-space via `x → u → V_θ` conversion + iMF's v-loss. Dimension-agnostic.
2. **Patch-size scaling at fixed 256-token budget.** Hold token count constant, grow patch size with clip length × sample rate.
3. **Pixel-space perceptual loss directly on `x_θ`.** Audio: MR-STFT + optionally a learned waveform-domain perceptual distance (AudioMAE-φ, MERT). Gate at low `t` (≤ 0.8) so the prediction is clean enough.
4. **CFG as conditioning, not doubled NFE.** Inference stays 1 forward pass even with guidance.

## What does *not* port cleanly

- **Image augmentations** (EPG-style MoCo / colour jitter). Audio analogue: random gain, time-shift, additive noise, EQ; pitch-shift only on instrument-class data.
- **VGG-LPIPS / ConvNeXt-V2** (pMF's perceptual losses). These are image-feature backbones; audio analogue is MR-STFT + an audio embedding distance (AudioMAE / MERT).
- **Reconstruction loss form.** Both EPG (L2-ish) and pMF (L2 + LPIPS) use pixel-domain losses; for audio we need phase-aware terms. MR-STFT is the obvious default.

## Open questions

- Can the EPG/pMF GFLOPs-competitiveness be replicated in 1-D? Audio's signed/phase-sensitive nature may force a more expensive backbone, or may not.
- For short audio (< 1 s), is the data manifold simple enough to skip SSL pre-training entirely (pMF's path) and train one-step end-to-end with pMF-style x-prediction + MR-STFT? WavFlow confirms SSL is skippable for 8 s at 5 M samples; whether this holds at the project's ~30 k drum-clip scale where audio-side SSL pretrain might be needed to compensate for data scarcity is still open.
- What's the minimum CPU-affordable pixel-space audio model that matches a frozen-codec latent baseline at the same training budget? pMF-B/16 (118 M / 33 GFLOPs / FID 3.12) is the closest 1-NFE point; does the 1-D analogue run faster than RT on CPU at 1 s @ 22 kHz?
- Does pMF's "u-prediction collapses at high patch dim, x-prediction is robust" finding (patch-dim 768 collapse) hold on 1-D audio? Audio patch dims (e.g. 86×1 = 86) sit well below pMF's 768 threshold — the collapse may not trigger and u-pred could remain viable.
- Does BAR-style codebook scaling close the discrete-vs-continuous reconstruction gap for *audio* tokenizers? Untested.
