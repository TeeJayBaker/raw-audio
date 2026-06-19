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
- MeanFlow uses the measured fixed guidance interval `[0.0, 0.8]`, the exact mirrored
  iMF timestep mean `logit_mean=0.4`, and pMF-style MR-STFT gating at repo `t>=0.2`
  (pMF `t<=0.8` on its opposite axis). The guided MF target remains stochastic, but
  the DDE/JVP tangent uses the model's conditioned boundary velocity to retain iMF's
  variance reduction.
- MeanFlow `flow.mf.dudt=jvp` now works: the fused SDPA kernels lack forward-AD, so
  `Attention` (`src/backbone/blocks.py`) carries a class-level `set_triton_jvp` flag that
  `MeanFlow.u_and_dudt` flips around `torch.func.jvp`. On CUDA it routes to the ported
  Triton flash-attention-JVP kernel (`src/backbone/jvp_flash_attention.py`, from
  `../flow-one-shot`); on CPU (and any triton-less env) it falls back to the MATH SDPA
  backend, which supports forward-AD. Normal training keeps the optimised fused SDPA + its
  fast backward. Requires `setuptools` for triton import. `u_and_dudt` also threads
  `return_spec` through both DDE and JVP so the complex-STFT aux loss applies in `_mf_step`.
  DDE is still the config default; jvp is a validated, faster alternative (kernel correctness
  vs the MATH reference in `tests/test_jvp_flash_attention.py`, GPU-only; dde≈jvp A/B on CPU
  in `tests/test_mean_flow.py`). The Triton backward is naive O(N²), fine at our ≤512 token
  count.
- MeanFlow DDE `du/dt` must run its two probe forwards in fp32. They form a central
  difference `/(2*dde_eps)`; under the trainer's bf16 autocast the subtraction cancels
  catastrophically — measured ~85% relative error vs fp32 on the step-25k stft model (152%
  on a toy Linear). The `.float()` in `u_and_dudt` was too late (model already returned
  bf16). This poisoned only the MF branch: `train/mf_mse` climbed ~0.05 -> 0.3–1.2 with
  spikes while `train/rf_mse` (h=0, no du/dt) stayed ~0.01–0.08, and 1-NFE samples drifted
  to noise over ~40k steps (archived run `runs/mf-oneshots-mars-stft-bf16dde-degraded`).
  Fix (`src/flow/mf.py::u_and_dudt`): disable autocast and cast inputs/conditioning to fp32
  for the dde probes; primal `u` stays bf16. Real-ckpt du/dt error 0.85 -> 0.00; regression
  guard `test_dde_dudt_runs_fp32_under_bf16_autocast`. The `jvp` path was unaffected (exact
  forward-AD). Cost: 2 extra fp32 forwards on MF micro-batches.
- Refuted hypotheses (do not revisit): `rms_lift=false` is not the cause of the
  original collapse; the x-pred `1/(1-t)^2` v-loss weighting is the published recipe,
  not a bug; the ODE sampler (Heun/step count) is not the issue; STFT framing is fine.

## 2026-06-17 — MeanFlow 1-NFE degradation: root cause (du/dt blow-up; missing iMF v-head)

Both MF runs (`mf-oneshots-mars-stft`, `-noaux`) degrade monotonically from the RF-200k
init: every eval metric slides from the first val (step ~1000) onward, `train/mf_mse`
swings 0.02→1.0, while `train/rf_mse` stays 0.01–0.08. Diagnosed via a du/dt probe
(`/tmp/mf_probe/probe.py`) on the 25k (degraded) vs 200k (RF-init) checkpoints, plus a
diff against the pMF reference (`/tmp/pMF`, github Lyy-iiis/pMF — `pmf.py`).

Ruled out, with evidence:
- **complex_stft aux loss** (the original suspect): logged terms are pre-weight; ×0.005 it is
  <2% of the loss and *decreases* (14→2) while eval worsens. Not the cause.
- **All aux losses**: the `-noaux` ablation (`mr_stft=0, complex_stft=0`) still degrades.
- **bf16-dde bug**: the fp32 fix is confirmed active (`mf.py:72`, run process started after the
  fix landed) yet the failure is identical → that was a separate, already-fixed bug.
- **MF target math**: dde≈jvp to ~5% on the real model; `target_to_v` sign/scale correct;
  `gap_embed`/`omega_embed` are `_zero_init_time_embed` (clean warm-start). All correct.

Localization (the key discriminator): failure is **h>0 / du-dt-branch only**. `rf_mse` (h=0,
instantaneous velocity) is stable end-to-end; the h>0 field (what 1-NFE sampling uses) diverges.

Mechanism (probe, t=0.8 row): at large h the correction term `‖h·du/dt‖` explodes —
RF-init 0.49 → degraded **1.85**, exceeding `‖u‖≈‖v_tgt‖≈1.0`, so `V = u − h·du/dt` overshoots
the target ~2× and the residual is essentially the entire uncorrected du/dt term. `‖du/dt‖`
is ~2–4× the RF-init. This is the known MeanFlow JVP-variance / `‖J‖²` blow-up
([2605.09235](../wiki/papers/2605.09235.md)), which that paper notes is *worse* at waveform's
high dimension. At h≈0 the residual is small and identical to RF-init (matches the stable rf_mse).

Amplifier, not driver: the adaptive weight `1/(‖Δ‖²+c)` with `c=1e-3` ≪ the residual scale
(0.03–3.4) degenerates to pure inverse and down-weights the diverging large-h rows ~70×
(w 0.8 vs 50). Setting `adaptive_p=0` (plain MSE) only *slows* the degradation. Provenance:
the adaptive weight is a **from-scratch** MeanFlow trick (original MF / iMF / pMF all train MF
from scratch and carry it over); **MeanAudio** — the only RF→MF fine-tune precedent — uses
plain L2, no adaptive weight.

Root cause vs pMF: **the repo is missing pMF/iMF's auxiliary v-head.** In pMF (`pmf.py`) the net
outputs `(u, v)`; the v-head is (1) directly supervised (`loss_v`, `pmf.py:435`) and (2) used as
the JVP z-tangent — `jvp(u_fn, (z,t,r), (v_c, 1, 0))` with `v_c` from the v-head
(`pmf.py:419`, comment "Different from original MeanFlow, we use predicted v in the jvp"). That is
the iMF / β→1 variance reduction that keeps du/dt bounded. Our repo has **no v-head and no
`loss_v`**; it takes the JVP tangent from the u-network's own boundary eval (`guided_velocity_target`
→ `target_to_v(model(...))`), i.e. the high-variance regime where du/dt diverges.
Secondary diff: pMF clips the velocity denominator `(z−x)/clip(t, 0.05, 1)` (`pmf.py:397`);
our `target_to_v` divides by `(1−t)` floored only at 1e-5 → near-unbounded near the data end.

Next step: add an auxiliary v-head — supervise it (`loss_v`) and use its prediction as the JVP
tangent — per pMF/iMF; the cheap variant is the EMA-tangent + FM-anchor of
[2605.09235](../wiki/papers/2605.09235.md) Alg. 1. Reference: `/tmp/pMF/pmf.py`.

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
