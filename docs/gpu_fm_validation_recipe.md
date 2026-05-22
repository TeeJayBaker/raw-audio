# GPU FM Validation Recipe

Use this after the CPU smoke path passes.

```bash
PYTHONPATH=src uv run python scripts/validate_fm_batch.py \
  --config configs/experiment/fm_baseline.yaml \
  conditioner.checkpoint_path=/path/to/matpac.ckpt

PYTHONPATH=src uv run python train.py \
  --config-name experiment/fm_baseline \
  conditioner.checkpoint_path=/path/to/matpac.ckpt \
  train.max_steps=100 \
  train.sample_every=10 \
  train.ckpt_every=50 \
  sampling.steps=10
```

Expected outputs:

- MATPAC loads from `/path/to/matpac.ckpt` and uses the checkpoint-adjacent `config.yaml`.
- The conditioner is projected to `backbone.conditioning.cond_dim` by default.
- The 10-100 step run reports finite training loss and validation `val_x_mse` / `val_v_mse`.
- `runs/fm-baseline/checkpoints/` contains step checkpoints.
- `runs/fm-baseline/samples/` contains clipped finite WAV files.

For an unconditional machine check without MATPAC:

```bash
PYTHONPATH=src uv run python train.py --config-name experiment/fm_wavenext_smoke
```
