# FM Training Scaffold Checklist

This checklist is for follow-up agents to audit and complete the first FM baseline.

## Implemented Scaffold To Review

- [x] `configs/experiment/fm_baseline.yaml`: verify Hydra defaults, train knobs, trainer target, conditioner fields, and backbone conditioning dim.
- [x] `configs/backbone/wavenext.yaml`: verify the default FM baseline backbone belongs in the backbone config group and can be swapped from the experiment config.
- [x] `configs/data/audio_directory.yaml`: verify directory dataset defaults and sample-rate/clip-length conventions.
- [x] `configs/flow/fm.yaml`: verify `prediction_target: x | v`, `100 * t` convention, and sampling eps.
- [x] `configs/eval/preliminary.yaml`: verify disabled placeholder metrics are explicit and not reported as real numbers.
- [x] `src/data/audio_dataset.py`: audit extension discovery, resampling, crop/pad behavior, channel handling, normalization, and collate output.
- [x] `src/emb/matpac.py`: verify it uses the checkpoint-adjacent `config.yaml` and local MATPAC implementation.
- [x] `src/flow/fm.py`: verify linear interpolant math, x/v target contracts, time broadcasting, and `x -> v` conversion near `t=1`.
- [x] `src/losses/audio.py`: verify primary loss scaling and auxiliary x-loss behavior for both x-target and v-target training.
- [x] `src/ema.py`: verify state dict, apply/restore, and update behavior with trainable-only params.
- [x] `src/sampling/fm_sampler.py`: verify Euler update and NFE handling for x-prediction and v-prediction.
- [x] `src/eval/audio_metrics.py`: verify placeholder signatures and return-shape documentation before real metric implementation.
- [x] `src/fm_trainer.py`: audit device placement, AMP, validation, sample writing, checkpoint schema, and EMA sampling/validation.
- [x] `train.py`: verify generic Hydra entrypoint works from repo root with `PYTHONPATH=src` and can swap experiments via `--config-name`.
- [x] `scripts/validate_fm_batch.py`: verify one-batch validation path works with null conditioner; real MATPAC path is documented for GPU validation.
- [x] `scripts/sample_fm.py`: verify checkpoint loading, EMA option needs, null conditioning fallback, and output normalization policy.
- [x] `tests/test_fm_scaffold.py`: expand tests as real losses/metrics become available.

## Required Follow-Up Implementations

- [x] Replace the approximate `spectral_energy_inverse_weight()` with the exact Flow2GAN loss: smoothed power spectrogram, linear filterbank, inverse sqrt weighting, and clamp `[0.01, 100]`.
- [x] Decide whether spectral-energy-inverse weighting should weight waveform loss, STFT loss, or a dedicated Flow2GAN spectral term, then test gradients.
- [x] Harden MR-STFT loss with configurable resolutions, window caching, stereo policy, and optional log-magnitude term.
- [x] Add validation losses for both x-MSE and v-MSE regardless of training target.
- [x] Implement embedding cosine monitoring using frozen MATPAC embeddings for real vs generated audio.
- [x] Implement FAD or FD-style audio distance over a selected embedding backend.
- [x] Implement MIND/Monge Audio Distance as sliced 1-D Wasserstein over audio embeddings.
- [x] Implement Vendi and Density/Coverage after selecting default embeddings/kernels.
- [x] Add metric sanity-check fixtures based on the evaluation wiki before using metrics as headline numbers.
- [ ] Add pMF scaffolding only after FM baseline runs: deferred until a real MATPAC/GPU FM baseline run exists, to keep this patch FM-only.
- [ ] Add GAN fine-tuning as a separate stage: deferred until FM baseline quality/listening checks justify a second training stage.
- [x] Add CPU xRT reporting for batch 1 and batch 16 sampling, matching the repo reporting convention.
- [x] Add a real GPU validation recipe documenting required MATPAC checkpoint path and expected 10-100 step outputs.

## Acceptance Before Serious Training

- [x] `uv run pytest` passes.
- [x] `uv run --env PYTHONPATH=src python scripts/validate_fm_batch.py --use-null-conditioner` passes.
- [ ] On the GPU machine, `conditioner.checkpoint_path=/path/to/matpac.ckpt` loads and returns the expected embedding dim.
- [x] A 10-100 step FM run produces finite losses, checkpoints, and sample WAVs.
- [x] A follow-up agent has reviewed every implemented scaffold item above and either checked it off or filed concrete fixes.
