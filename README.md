# raw-audio

Experimental code for lightning-fast one-shot and few-step raw-audio generation with data-space flow matching models.

The project focuses on generating waveform samples directly, without a VAE, codec, mel decoder, or other latent audio representation in the generation path. The current implementation is a PyTorch/Hydra training scaffold for flow matching on short audio clips, with WaveNeXt-style and related backbone experiments.

## Repository Layout

```text
configs/      Hydra configs for data, flow objectives, backbones, eval, and experiments
docs/         Engineering notes and validation recipes
matpac/       Local MATPAC embedding/checkpoint integration code
scripts/      Validation, sampling, benchmarking, and model-stat utilities
src/          Training, model, loss, data, flow, embedding, and eval modules
tests/        Unit tests for backbones, losses, metrics, and FM scaffolding
train.py      Generic Hydra entrypoint for experiments
```

Local-only directories are intentionally ignored by git:

```text
data/         Audio datasets and smoke-test wave files
runs/         Training outputs, samples, checkpoints, and configs
outputs/      Hydra runtime output directories
wiki/         Local research notes, papers, PDFs, and reports
.claude/      Local assistant workflow files
```

## Setup

This repo uses `uv` for dependency management.

```bash
uv sync
```

Check the PyTorch install:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If a GPU machine resolves to CPU-only PyTorch, install the CUDA wheel explicitly:

```bash
uv pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Common Commands

Run the test suite:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```

Validate a single flow-matching batch with a null conditioner:

```bash
PYTHONPATH=src uv run python scripts/validate_fm_batch.py --use-null-conditioner
```

Run the CPU smoke experiment:

```bash
PYTHONPATH=src uv run python train.py --config-name experiment/fm_wavenext_smoke
```

Run the baseline experiment:

```bash
PYTHONPATH=src uv run python train.py --config-name experiment/fm_baseline
```

Sample from a checkpoint:

```bash
PYTHONPATH=src uv run python scripts/sample_fm.py runs/fm-wavenext-smoke/checkpoints/step_00000002.pt --out sample.wav
```

Benchmark backbone throughput:

```bash
PYTHONPATH=src uv run python scripts/benchmark_backbone.py
```

## Configuration

Experiments are composed from `configs/`:

- `configs/experiment/` selects complete runnable experiments.
- `configs/train/` defines shared training and logging defaults.
- `configs/backbone/` defines backbone families and sizes.
- `configs/data/` defines audio dataset loading.
- `configs/flow/` defines the flow-matching objective.
- `configs/eval/` defines preliminary audio metrics.

The default entrypoint is:

```bash
PYTHONPATH=src uv run python train.py --config-name experiment/fm_baseline
```

Hydra overrides can be appended as usual, for example:

```bash
PYTHONPATH=src uv run python train.py --config-name experiment/fm_wavenext_smoke train.max_steps=10 sampling.steps=1
```

## Weights & Biases

Training initializes wandb by default using shared defaults from `configs/train/default.yaml`. Configure the run with Hydra overrides:

```bash
PYTHONPATH=src uv run python train.py \
  --config-name experiment/fm_baseline \
  train.wandb.entity=my-team \
  train.wandb.name=fm-baseline-001
```

Disable wandb for a local run:

```bash
PYTHONPATH=src uv run python train.py --config-name experiment/fm_wavenext_smoke train.wandb.enabled=false
```

Use offline wandb logging:

```bash
PYTHONPATH=src uv run python train.py --config-name experiment/fm_baseline train.wandb.mode=offline
```

Wandb receives scalar train/validation metrics plus audio monitor pairs from a fixed validation set. Reference audio is logged once; generated audio is logged on every validation run with fixed per-example initial noise for reproducible comparisons.

## Notes

The code is not packaged as an installable Python package. `pyproject.toml` configures `uv`, pytest, and ruff; modules are imported from `src/` with `PYTHONPATH=src`.

Generated assets, checkpoints, datasets, PDFs, reports, and local research notes are deliberately excluded from git so the repository stays focused on reusable experiment code.
