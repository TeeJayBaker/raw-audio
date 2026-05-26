---
title: "WaveNeXt: ConvNeXt-Based Fast Neural Vocoder Without iSTFT Layer"
arxiv_id: null
year: 2023
authors: [Takuma Okamoto, Haruki Yamashita, Yamato Ohtani, Tomoki Toda, Hisashi Kawai]
domain: audio
tags: [vocoder, gan, convnext, mel-conditioned, learned-upsampling, cpu-benchmark, vocos, hifi-captain, jets, e2e-tts, asru-2023]
status: ingested
last_reviewed: 2026-05-22
---

## TL;DR (project-relevant)

WaveNeXt is **Vocos with the iSTFT readout swapped for a learned 256× linear pixel-shuffler**. The ConvNeXt trunk is byte-for-byte identical to Vocos; only the head changes. Same training recipe (MPD + MRD + feature matching + mel-L1 hinge GAN), same input (mel-spectrogram), same operating sequence length (frame-rate, hop=256 @ 24 kHz). On Hi-Fi-CAPTAIN (Japanese, single-speaker) the paper measures **RTF 0.10 at 24 kHz = ~10× faster than real-time** on a single AMD EPYC 7542 core, **indistinguishable from Vocos's RTF 0.10** at the same setup — and higher MOS than Vocos on every condition tested (analysis-synthesis female / male, JETS-based E2E TTS female / male, full-band E2E TTS at 48 kHz). The argument: time-domain GAN loss does not constrain the predicted (magnitude, phase) pair under the overlap-add redundancy, so estimating them as an intermediate target is wasted variance; direct waveform-sample prediction matches the loss. **For this project the relevance is the readout choice itself, not the conditional-vocoder result** — the same learned-upsampler head can swap into any frame-rate trunk that would otherwise iSTFT to waveform.

## Why this matters here

Three slots:

1. **The first paper to break iSTFT readout from the STFT-domain ConvNeXt family.** Vocos / RFWave / Flow2GAN / ComVo all iSTFT the final ConvNeXt output back to waveform. WaveNeXt drops the iSTFT for `Linear(d → n_fft) → Linear(n_fft → hop_length, bias=False) → Reshape`. Trunk, training loss, and CPU cost are unchanged; subjective quality goes up. The iSTFT in this family is an architectural prior, not a structural requirement.
2. **The readout swap is orthogonal to every other axis tracked in the wiki.** Trunk variants (single-branch / subband-batched / multi-resolution-branches / complex-valued) all currently terminate in iSTFT; WaveNeXt's head can in principle replace the readout in any of them. None of the combinations is tested.
3. **The paper's phase-redundancy diagnostic (Fig. 2) is the actual contribution**, not the architecture change. Vocos's estimated mag/phase don't match ground truth, but when you re-analyze the synthesized waveform, mag/phase look like ground truth. Many (mag, phase) configurations produce the same waveform under overlap-add → the time-domain GAN loss is degenerate over them → the network is forced to pick one without supervision. This generalises beyond vocoders: any model with `representation → invertible transform → loss` may face the same loss-degeneracy problem, and the WaveNeXt solution (skip the transform, predict in the loss-space directly) applies whenever the transform is invertible-but-redundant.

The repo already has `convnext-to-wavenext.md` (head-swap implementation note) and `configs/backbone/convnext_wavenext.yaml` — this paper is the source for both.

## The head swap

Vocos head (Fig. 1d):

```
... ConvNeXt → Linear(d → 2·(n_fft/2 + 1)) → split(m, p)
            → M = exp(m), φ = atan2(sin p, cos p)
            → complex_spec = M · e^{iφ}
            → iSTFT → waveform
```

WaveNeXt head (Fig. 1e, Fig. 3):

```
... ConvNeXt → Linear(d → n_fft)
            → Linear(n_fft → hop_length, bias=False)
            → Reshape (B, hop_length, T) → (B, 1, hop_length · T)
            → waveform
```

Key properties of the WaveNeXt head:

- **No activation or norm** between the two linears.
- **Bias-disabled** on the second linear (inherited from MS-HiFi-GAN's data-driven subband-filter convention).
- **Pure concatenation**, not overlap-add — each frame's `hop_length` predicted samples are placed back-to-back; no windowing.
- **Parameter cost** ≈ `d·n_fft + n_fft·hop_length` (≈ 786 k for d=512, n_fft=1024, hop=256) vs Vocos head's ≈ `d·(n_fft+2)` (≈ 525 k). Negligible compared to the ConvNeXt trunk (~13.5 M).
- **256× upsample in one linear layer** at 24 kHz (hop=256), 512× at 48 kHz (hop=512). The 4× FC-HiFi-GAN linear upsampler is a much easier problem; this is the same trick at much larger ratio.

Equivalent to a 1-D pixel-shuffle / sub-pixel convolution with a 1×1 conv producing `hop_length` channels, then reshape — the second linear is the 1×1 conv along the feature axis. The paper explicitly notes this equivalence.

## Why iSTFT-free helps (Section 4.1, Fig. 2)

The diagnostic:

- Compute STFT(ground-truth waveform) → ground-truth (mag, phase) panels.
- Compute Vocos-estimated (mag, phase) → visible degradation in magnitude, visible drift in phase vs ground truth.
- Synthesize Vocos waveform via iSTFT of estimated (mag, phase), then re-STFT → re-analyzed (mag, phase) panels look like ground truth.

Reading: the iSTFT + overlap-add map is **many-to-one** in the (mag, phase) → waveform direction at the granularity the loss can see. The time-domain GAN losses (MPD + MRD + feature matching + mel-L1) compare *waveforms*, not (mag, phase) pairs. So multiple (mag, phase) outputs satisfy the loss equally; the network has no signal to pick the "right" one. Vocos converges to *a* solution, not *the* ground-truth one.

WaveNeXt sidesteps this by predicting in the same space the loss compares — waveform samples directly. Training-loss curves (Fig. 5) confirm: Vocos loss drops faster early, WaveNeXt loss converges more slowly but reaches a lower asymptote, on both analysis-synthesis and E2E TTS settings. WaveNeXt asks more from optimization (1-million-iteration training schedule needed to fully exploit it) but the ceiling is higher.

The wider claim — predicting in the loss space is better than predicting in an invertible intermediate when the intermediate is loss-redundant — generalises beyond vocoding. Diffusion / FM in raw audio that goes `noise → predicted-clean-STFT → iSTFT → waveform → loss` would inherit the same problem; the WaveNeXt-style fix would be to predict the waveform directly from the trunk output.

## Training, data, eval

- **Loss**: identical to Vocos (Eqs. 1–2 in the paper). MPD + MRD adversarial hinge + feature matching + mel-L1; `w_MRD = 0.1`, `w_mel = 45`. No phase loss, no spectral loss beyond mel-L1.
- **JETS extension**: replace the HiFi-GAN generator inside JETS with Vocos or WaveNeXt; FastSpeech 2 acoustic model + monotonic-alignment-search alignment module trained jointly. Adds `w_var · ℓ_var + w_align · ℓ_align` to LG.
- **Data**: Hi-Fi-CAPTAIN Japanese single-speaker corpora, separate female and male (~20 hours each, 18 372 / 250 / 40 train/val/test utterances). 24 kHz analysis-synthesis + 24 kHz / 48 kHz JETS-based E2E TTS. Mel: 80-d, band-limited to 7600 Hz, FFT/hop = 1024/256 (24 kHz) or 2048/512 (48 kHz).
- **n=8 ConvNeXt blocks** (same as Vocos default), 1 M training iterations on A100-40GB.
- **CPU measurement**: single core of AMD EPYC 7542 (2.9 GHz, Zen2), PyTorch 1.13.1, batch = 1 single-sample (TTS-style measurement, distinct from Flow2GAN's batch=16 Xeon Platinum benchmark — not directly comparable).

## Results

Numbers are the project-relevant slice; Tab. 1, 2 and Fig. 6, 7 of the paper for full table.

### 24 kHz analysis-synthesis (Tab. 1, female / male)

| Model | RTF↓ | MOS female | MOS male |
|---|---|---|---|
| HiFi-GAN V1 | 0.92 | 4.62 | 4.12 |
| HiFi-GAN V2 | 0.10 | 4.15 | 3.97 |
| MS-iSTFT-HiFi-GAN | 0.19 | 4.47 | 4.30 |
| MS-FC-HiFi-GAN | 0.18 | 4.62 | 4.24 |
| Vocos | 0.10 | 4.04 | 3.97 |
| **WaveNeXt** | **0.10** | **4.38** | **4.37** |
| Ground truth | – | 4.62 | 4.47 |

WaveNeXt-vs-Vocos delta: +0.34 MOS female, +0.40 MOS male, at the same single-core CPU RTF (= ~10× RT). MCD is *worse* than Vocos (2.86 vs 2.74 female, 4.32 vs 3.05 male) — this is expected: WaveNeXt is no longer estimating (mag, phase) at all, so a mel-cepstral-distortion metric (which is an STFT-derived comparison) penalises it for taking a different path to the same perceptual output. CER and log-f0 RMSE are equivalent.

### 24 kHz JETS E2E TTS (Tab. 1, female / male)

| Model | RTF↓ | MOS female | MOS male |
|---|---|---|---|
| HiFi-GAN V1 | 0.97 | 4.31 | 4.09 |
| MS-iSTFT-HiFi-GAN | 0.24 | 4.19 | 3.57 |
| MS-FC-HiFi-GAN | 0.23 | 4.20 | 3.27 |
| Vocos | 0.15 | 3.52 | 3.46 |
| **WaveNeXt** | **0.15** | **4.36** | **3.57** |

E2E TTS is where Vocos hurts most (significantly behind HiFi-GAN-based models); WaveNeXt closes that gap entirely, matching the best HiFi-GAN variant on female and tying MS-iSTFT-HiFi-GAN on male, at half their RTF.

### 48 kHz full-band JETS E2E TTS (Tab. 2, female)

| Model | RTF↓ | MOS |
|---|---|---|
| HiFi-GAN V1 | 1.08 | 4.14 |
| HiFi-GAN V2 | 0.17 | – |
| MS-FC-HiFi-GAN | 0.30 | 4.21 |
| Vocos | 0.16 | 2.39 |
| **WaveNeXt** | **0.16** | **3.82** |

Vocos collapses at 48 kHz (MOS 2.39 — large objective failure too: MCD 5.87, log-f0 RMSE 0.33 vs everything else ~0.24); WaveNeXt recovers to MOS 3.82 at the same RTF. HiFi-GAN V1 wins MOS at 48 kHz but at ~7× higher CPU cost (RTF 1.08 = slower than real-time on a single core); WaveNeXt is the fastest model that produces credible 48 kHz output. The asymmetry between Vocos's 48 kHz collapse and WaveNeXt's survival is the strongest evidence in the paper that **the (mag, phase)-prediction strategy scales worse than direct waveform prediction as sample rate / sequence length grows** — at 48 kHz the readout is doing 512× upsampling, and the loss-degeneracy hurts more.

### CPU cost translation

Convert RTF (paper convention: wall-clock / audio_duration, lower = faster) → xRT (audio_duration / wall-clock, higher = faster, used elsewhere in this wiki):

- **WaveNeXt 24 kHz: xRT 10** (single-core AMD EPYC 7542, batch 1)
- **WaveNeXt 48 kHz: xRT 6.25** (same hardware)
- **Vocos numbers are identical to WaveNeXt** in this setup — the linear-layer head is essentially free vs the ConvNeXt trunk

Not directly comparable to the inference.md primary table (Xeon Platinum 8457C, batch 16): hardware and batching differ. But the relative ordering is consistent — both benchmarks place Vocos and WaveNeXt at >>real-time on CPU at single-step inference.

## Project-relevant takeaways

- **Same trunk, same CPU, higher quality — the head matters.** The simplest possible "what if we drop the iSTFT?" ablation in the STFT-domain ConvNeXt family. Result: better MOS at every condition tested, no CPU penalty, no architectural regression. The iSTFT in Vocos / RFWave / Flow2GAN / ComVo is now a defensible choice rather than a default; the WaveNeXt readout is a defensible alternative.
- **The readout swap composes with every other STFT-domain ConvNeXt variant.** RFWave + WaveNeXt readout, Flow2GAN + WaveNeXt readout, ComVo + WaveNeXt readout — all are well-defined architectures and none has been tested. For Flow2GAN specifically the readout swap would mean each of the three resolution branches emits a waveform directly (no iSTFT per branch) and the branch outputs are summed; the per-branch hop lengths (`{256, 128, 64}`) become the per-branch pixel-shuffle ratios.
- **For unconditional sample-space audio gen, the readout choice carries the loss-redundancy lesson directly.** Any frame-rate trunk (ConvNeXt at hop-rate, ViT on STFT patches, 1-D U-Net with a frame-rate bottleneck) that emits a waveform via an invertible transform faces the same problem WaveNeXt diagnoses for Vocos. Predicting in the loss space (waveform samples, or whatever space the loss compares) is the safer default.
- **48 kHz full-band scaling is the second-strongest result in the paper.** Vocos collapses (MOS 2.39 from MOS 4.04 at 24 kHz); WaveNeXt survives (MOS 3.82 from MOS 4.38 at 24 kHz). At fixed sample rate the head choice is a moderate quality lever; at higher sample rate it becomes the difference between collapse and credibility. **Bears watching for any STFT-domain method this project ports beyond 24 kHz.**
- **The paper's recipe is a strict subset of Vocos's** — no new loss, no new optimizer, no new normalisation, no new activation. Reproduction cost is minimal; the repo's existing `convnext_wavenext.yaml` config encodes the entire change. This is the "easiest win" in the audio architecture space tracked here.
- **The phase-redundancy argument (§4.1) is a transferable diagnostic.** Whenever a model predicts an intermediate representation `R` that maps to a loss space `L` via an invertible transform `T`, check whether `T` is many-to-one at the granularity the loss can see. If yes, `R` is loss-redundant; predict `L` directly. Diffusion / FM models that go `noise → predicted-clean-STFT → iSTFT → time-domain-loss` should expect this.

## Limitations / open questions

- **HiFi-GAN V1 still beats WaveNeXt on subjective MOS** at 48 kHz (4.14 vs 3.82) and on male E2E TTS at 24 kHz (4.09 vs 3.57). Open whether closing this gap requires extended ConvNeXt v2 / LightVoc-style tricks (paper's flagged future work, refs [65, 66]) or a deeper trunk.
- **MCD regression vs Vocos.** WaveNeXt's MCD is worse than Vocos's despite higher MOS. The paper doesn't explore this; presumably WaveNeXt's STFT-derived spectral envelope drifts more from ground truth than Vocos's (which is *trained* to fit a magnitude target), while the time-domain waveform sounds better. **MCD is not a useful metric for WaveNeXt-class models** — if porting the recipe to a different dataset, use UTMOS / MOS / PESQ instead.
- **Convergence is slower.** Vocos loss drops faster early; WaveNeXt needs roughly the full 1 M iterations to fully exploit the architecture. On a budget-constrained training run, Vocos finishes ahead. **Should be checked**: does WaveNeXt's loss curve still overtake Vocos's at fewer iterations on smaller datasets, or is full convergence load-bearing?
- **Batch = 1 single-core CPU benchmark only.** No multi-core, no batched, no laptop-CPU (M-series Mac, mobile) numbers. Real interactive deployment on consumer hardware is untested.
- **Only conditional vocoding** (mel → waveform). The argument about loss-redundancy of the (mag, phase) intermediate transfers to unconditional generation in principle, but no paper has tested it.
- **No comparison to BigVGAN at this sample rate.** Hi-Fi-CAPTAIN is unusual choice for benchmarking; cross-paper comparison vs the Vocos paper's VCTK benchmark requires re-running.
- **No exploration of head depth.** Two linear layers is the minimum; whether a slightly deeper head (e.g. Linear → GELU → Linear → Reshape) helps or hurts is untested. The paper picks the minimum that works.

## Links

- Concept pages: [architecture](../wiki/concepts/architecture.md) (frame-rate ConvNeXt family, readout axis), [inference](../wiki/concepts/inference.md) (CPU benchmark, cross-paper xRT).
- Implementation note in this repo: [`convnext-to-wavenext.md`](../convnext-to-wavenext.md) — exact head diff and parameter constraints.
- Config: [`configs/backbone/convnext_wavenext.yaml`](../configs/backbone/convnext_wavenext.yaml).
- Direct comparators in the wiki: [Flow2GAN (2512.23278)](2512.23278.md) (multi-resolution iSTFT readout, the obvious composition target), [RFWave (2403.05010)](2403.05010.md) (subband-batched iSTFT readout), [ComVo (2603.11589)](2603.11589.md) (CVNN trunk + iSTFT readout — composition would give CVNN trunk + learned readout), [HiFi-GAN (2010.05646)](2010.05646.md) (the gradual-upsampler baseline WaveNeXt-with-Vocos-trunk beats at half the CPU cost).
- Code / samples: https://is.gd/duv0DF (ESPnet2-TTS-based, official).
- Predecessor (cited): FC-HiFi-GAN / MS-FC-HiFi-GAN (Yamashita et al. 2023, IEICE Tech. Rep., Japanese) — same author group; introduces the trainable linear upsampler on a HiFi-GAN trunk before applying it to Vocos.
