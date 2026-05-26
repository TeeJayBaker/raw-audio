# Backbone CPU RTF Results

Benchmarked on 2026-05-21 with PyTorch 2.12.0, macOS-26.1-arm64, batch size 1, 48 kHz mono, 1.0 second target audio, 3 warmup iterations, and 10 timed iterations. The target for short one-shot drum generation is 1 second in <=100 ms, equivalent to xRT >=10 or RTF <=0.10.

**Update 2026-05-22:** `waveform_transformer` rescaled from the original smoke-sized config (0.6 M params, `patch_size=32, dim=96, depth=4`) to a serious transformer (`patch_size=512, dim=512, depth=12, heads=8`, WaveNeXt-style head with `n_fft=2048`, 59.4 M params). Its rows in both tables below reflect the new config; all other rows are unchanged from 2026-05-21.

**Update 2026-05-23:** Two architectural changes following the literature survey (notably LinDiff arxiv 2306.05708 as the only published precedent for transformer-on-raw-waveform generation, and the absence of any published transformer-on-raw-STFT generator with 2D ViT patching).

- `waveform_transformer`: WaveNeXt head removed (no published precedent for WaveNeXt-on-transformer-trunk). Two head options now available: `conv` (three `Conv1d(1, 1, kernel=13)` layers with SiLU between, 42 params) and `convnext` (Conv1d expand to `hidden=16` → `ConvNeXtBlock1d(K=7)` → Conv1d project back, 2353 params). Default is `conv` — `convnext` adds the proper depthwise+pointwise inductive bias at ~9 ms single-thread cost (38.6 ms median vs 29.8 ms for `conv`), kept as an escalation path if boundary artefacts emerge in training. New param count with `conv` head: **57.59 M**.
- `stft_transformer`: 2D STFT patching dropped. Replaced with a freq-as-channel design (Voicebox/F5-TTS pattern applied to STFT): `Conv1d(2 × 1025, 512, kernel=1, stride=1)` consumes one complex STFT frame as a 2050-channel token, 12-layer / dim-512 transformer trunk at frame rate (94 tokens for 1 s), `ConvTranspose1d(512, 2 × 1025, 1, 1)` projects back to complex STFT. Matches `waveform_transformer`'s `dim/depth/heads/cond_dim` for fair A/B. Conditioning switched from `context_tokens` to `adaln_zero` to match. New param count: **59.16 M**. Same `io: stft` benchmark caveat as before — output is complex STFT, iSTFT is not in the timing.

Rows for both backbones in the tables below reflect the 2026-05-23 configs; all other rows are unchanged from 2026-05-21.

Raw CSVs:

- `reports/backbone_rtf_raw_1thread.csv`
- `reports/backbone_rtf_raw_default_threads.csv`

Benchmark command:

```bash
uv run python scripts/benchmark_backbone_rtf.py configs/backbone/*.yaml --seconds 1 --warmup 3 --iters 10 --threads 1 --csv reports/backbone_rtf_raw_1thread.csv
uv run python scripts/benchmark_backbone_rtf.py configs/backbone/*.yaml --seconds 1 --warmup 3 --iters 10 --csv reports/backbone_rtf_raw_default_threads.csv
```

Important caveat: configs with `io: stft` were measured in their declared STFT domain, emitting complex STFT tensors shaped `[B, C, F, frames]`. Those timings do not include final iSTFT/audio reconstruction unless the config itself performs it. Direct waveform feasibility is best judged from the `io: waveform` rows.

## Single-Thread CPU

| Backbone | IO | Params | Mean ms / 1s | xRT | RTF | Meets <=100 ms? |
|---|---:|---:|---:|---:|---:|---|
| vocos | stft | 27.74M | 28.7 | 34.80 | 0.0287 | yes, STFT-domain |
| convnext_wavenext | waveform | 28.79M | 28.9 | 34.57 | 0.0289 | yes |
| stft_transformer | stft | 59.16M | 28.9 | 34.60 | 0.0289 | yes, STFT-domain |
| waveform_transformer | waveform | 57.59M | 29.8 | 33.55 | 0.0298 | yes |
| hifigan_v3 | waveform | 3.04M | 80.1 | 12.48 | 0.0801 | yes |
| flow2gan | waveform | 81.97M | 84.0 | 11.90 | 0.0840 | yes |
| comvo | stft | 48.79M | 93.2 | 10.73 | 0.0932 | yes, STFT-domain |
| rfwave | stft | 20.03M | 96.7 | 10.34 | 0.0967 | yes, STFT-domain |
| hifigan_v1 | waveform | 16.04M | 469.2 | 2.13 | 0.4692 | no |
| crash | waveform | 32.20M | 646.1 | 1.55 | 0.6461 | no |
| wavefm | waveform | 20.37M | 774.0 | 1.29 | 0.7740 | no |

## Default PyTorch Threads

Default PyTorch used 8 CPU threads. For this batch-size-1 workload, default threading was slower for most Conv1d/ConvNeXt configs, likely due to thread overhead at short sequence lengths. It helped the transformer configs (`waveform_transformer`, `stft_transformer`), plus `hifigan_v3`, `crash`, `hifigan_v1`, and `wavefm`.

| Backbone | IO | Mean ms / 1s | xRT | RTF | Meets <=100 ms? |
|---|---:|---:|---:|---:|---|
| stft_transformer | stft | 38.6 | 25.89 | 0.0386 | yes, STFT-domain |
| waveform_transformer | waveform | 40.5 | 24.67 | 0.0405 | yes |
| hifigan_v3 | waveform | 55.6 | 17.99 | 0.0556 | yes |
| convnext_wavenext | waveform | 218.0 | 4.59 | 0.2180 | no |
| vocos | stft | 230.6 | 4.34 | 0.2306 | no |
| hifigan_v1 | waveform | 249.4 | 4.01 | 0.2494 | no |
| crash | waveform | 331.2 | 3.02 | 0.3312 | no |
| wavefm | waveform | 405.1 | 2.47 | 0.4051 | no |
| rfwave | stft | 487.1 | 2.05 | 0.4871 | no |
| flow2gan | waveform | 770.5 | 1.30 | 0.7705 | no |
| comvo | stft | 868.5 | 1.15 | 0.8685 | no |

## Feasibility Takeaways

For direct 1-second waveform generation under 100 ms on CPU, the viable single-thread configs are `convnext_wavenext`, `waveform_transformer`, `hifigan_v3`, and `flow2gan`; in the STFT-domain column, `stft_transformer` (new freq-as-channel design), `vocos`, `comvo`, and `rfwave` all meet target. Under default 8-thread scheduling only `stft_transformer`, `hifigan_v3`, and `waveform_transformer` remain under 100 ms.

The strongest latency candidates after the 2026-05-23 refactor:

- `stft_transformer` (new, freq-as-channel): 28.9 ms single-thread, 38.6 ms default-thread at 59.16 M params. Fastest STFT-domain backbone in both threading modes and the fastest backbone overall under default threads. Note the STFT-output caveat: iSTFT (~5–15 ms at `n_fft=2048, hop=512`) is not in this timing.
- `convnext_wavenext`: 28.9 ms single-thread (28.79 M params), end-to-end waveform. Still the fastest *end-to-end-waveform* single-thread backbone, but falls apart under default threading (218 ms) from depthwise Conv1d thread overhead.
- `waveform_transformer` (new, 3-conv smoother): 29.8 ms single-thread, 40.5 ms default-thread at 57.59 M params. Three single-channel `Conv1d(K=13)` with SiLU after the linear unpatchify — 42 head params, nonlinear, effectively free at sample rate. Tied with `convnext_wavenext` single-thread and ahead of it under default threading.
- `hifigan_v3`: stable under 100 ms in both CPU modes, 80.1 ms single-thread and 55.6 ms default-thread.
- `flow2gan`: under target single-thread at 84.0 ms, but too slow with default threading.

The three frame-rate backbones (`convnext_wavenext`, `waveform_transformer`, `stft_transformer`) are now clean A/B/C comparisons at matched `dim=512`-class compute:

| | Input domain | Trunk | Output strategy |
|---|---|---|---|
| `convnext_wavenext` | STFT (internal) | ConvNeXt-1D, depth 8 | WaveNeXt head (learned linear synthesis) → waveform |
| `waveform_transformer` | raw waveform | Transformer, depth 12 | Linear unpatchify + 3-conv sample-rate smoother → waveform |
| `stft_transformer` | STFT | Transformer, depth 12 | ConvTranspose1d (per-frame linear) → complex STFT (iSTFT external) |

Trunk type (Transformer vs ConvNeXt) and input domain (raw waveform vs STFT) are the two main axes; the patchify-trunk-unpatchify-smooth recipe for `waveform_transformer` follows LinDiff (arxiv 2306.05708) in spirit at minimal capacity, the freq-as-channel STFT transformer follows the Voicebox/F5-TTS pattern applied to STFT, and the ConvNeXt + WaveNeXt-head substrate follows Vocos/WaveNeXt. The heavier waveform UNet-style configs, `crash`, `hifigan_v1`, and `wavefm`, are not feasible for the 100 ms CPU target without substantial architecture or inference optimization.

