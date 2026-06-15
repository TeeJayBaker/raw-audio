# FM Training Notes

Working notes for the one-shot FM runs. Supersedes `fm_runs_failure_investigation.md`,
`wavenext_conditioning_diagnosis.md`, `sampling_clipping_diagnosis.md`,
`gpu_fm_validation_recipe.md` (all deleted; recoverable via git history).

## Live issues

- Guidance rescale and CFG interval are still un-implemented. `src/flow/fm.py::_model_v`
  does plain `x_null + s*(x_cond - x_null)` then returns the converted `v` (lines 99-112).
  Peak-norm in `sample()` masks the ~1.8x amplitude inflation at `s=3.5`; rescale fixes
  it at source. Config-gate under `sampling.guidance_rescale` (e.g. `phi=0.7`) and
  `sampling.guidance_interval` (e.g. `[0.0, 0.8]`).
- iSTFT head default in `src/backbone/convnext.py:20` is still `magphase`, which NaNs at
  init under FM (`exp()` blows up). The `vocos` experiment config sets `realimag`
  explicitly. Either flip the default or guard against `magphase` for FM.
- The MATPAC `ConditioningEmbedding` (`src/backbone/conditioning.py:40-48`) is a 2-layer
  MLP. Ablations showed `sep_norm` works even without a learned projector; the projector
  is kept for dim-matching only. No action required, just don't expect the projector to
  carry conditioning fidelity.

## Settled fixes (working tree, uncommitted)

- Conditioning combiner now RMS-normalises both the time and cond paths before the sum
  (`src/backbone/conditioning.py:51-76`, `ConditioningCombiner`). Fixes `||t_emb||`
  runaway (2.7 -> ~200) that drove the ConvNeXt 200k run to silence.
- ConvNeXt block carries a zero-init AdaLN gate (`src/backbone/blocks.py:135,148`,
  `groups=3`). Mirrors the transformer's per-sub-block gate; adds conditioning headroom.
- `RFTrainer.sample()` per-item peak-normalises to 1.0 instead of hard-clamping
  (`src/trainer.py:388-389`). Removes the CFG-oversaturation clipping artefact from
  logged samples and val-embedding metrics.
- Recommended recipe = ConvNeXt trunk + iSTFT `realimag` head (`vocos` config) +
  `sep_norm` + gate, x-pred/v-loss. Overfit: retrieval 8/8, genRMS 0.306 at 10k steps.
- `ConditioningEmbedding` maps `cond=None` to the zero vector through the cond MLP
  (`src/backbone/conditioning.py`), so the CFG uncond branch now matches the
  cond-dropout null the model trained on. Previously `_model_v`'s `cond=None` bypassed
  the MLP and the combiner fell back to bare `t_embed` — an untrained null (output rms
  delta 0.55 vs 0.93 signal). Recon check, stft-b64 100k, s=3.5 / 25 steps, 10 clips:
  mean MATPAC cosine 0.712 -> 0.775, 9/10 improved
  (`experiments/cfg_sampling/listen/`).
- Refuted hypotheses (do not revisit): `rms_lift=false` is not the cause of the
  original collapse; the x-pred `1/(1-t)^2` v-loss weighting is the published recipe,
  not a bug; the ODE sampler (Heun/step count) is not the issue; STFT framing is fine.

## Open questions / follow-ups

- Re-run the cond-vs-no-cond probe and a 50k vocos run end-to-end with the merged
  fixes to confirm rail% ~ 0 and intelligible audio across `s in {1.0, 1.5, 2.5, 3.5}`.
- CFG interval sweep on stft-200k @ 50k (`experiments/cfg_sampling/sweep.py`, n=64,
  out/sweep_fm-oneshots-mars-stft-200k_step_00050000_zeros.json): guidance hurts near
  *data*, not near noise — `[0, 0.8)` t-interval (on from noise, off for the final
  ~20%) at s=2.5–3.5 beats both full-interval CFG (KAD 6.45–6.56 vs 7.9–8.8 at 16
  steps) and no guidance (6.91); cos/retr 0.81/0.80 vs 0.786/0.73. Optimum is flat
  over hi ∈ [0.6, 0.9] and s ∈ [2.5, 3.5]; lo-exclusion does nothing on KAD but
  `[0.2, 0.8)` trims genRMS inflation (0.185 vs 0.211). Image-domain folklore
  (drop high-noise guidance) does NOT transfer. Guidance-rescale (`phi`) still
  untested; `sampling.guidance_interval` still un-implemented in src.
- Transformer at `patch_size ~ 128` is the documented fallback; current
  `fm_oneshots_mars_waveform.yaml` uses a larger patch (verify before any retrain).
- WaveNeXt head is slow on amplitude, not broken: if keeping it, enable
  `loss.mr_stft_weight` and budget for substantially longer training.
- Commit the working-tree conditioning/peak-norm changes so the fixes have a hash to
  reference.

## Validation recipe

After the CPU smoke path passes:

```bash
PYTHONPATH=src uv run python scripts/validate_fm_batch.py \
  --config configs/experiment/fm_baseline.yaml \
  conditioner.checkpoint_path=/path/to/matpac.ckpt

PYTHONPATH=src uv run python train.py \
  --config-name experiment/fm_baseline \
  conditioner.checkpoint_path=/path/to/matpac.ckpt \
  train.max_steps=100 train.sample_every=10 train.ckpt_every=50 sampling.steps=10
```

Expect finite `val_x_mse` / `val_v_mse`, checkpoints under
`runs/fm-baseline/checkpoints/`, finite WAVs under `runs/fm-baseline/samples/`.
Unconditional sanity check: `train.py --config-name experiment/fm_wavenext_smoke`.
