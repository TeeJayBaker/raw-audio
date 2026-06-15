#!/bin/bash
# MR-STFT aux-loss A/B: identical sweeps (seed 0 -> same reference clips + noise)
# on fm-oneshots-mars-stft-b64-200k (mr_stft_weight=0.0) vs fm-oneshots-mars-stft-200k
# (mr_stft_weight=1.0) at matched checkpoint steps. CPU-only: both GPUs are occupied
# (the with-arm is still training on one of them). Thread-capped + niced so the live
# run's dataloader workers keep priority.
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

run_sweep() {
  nice -n 10 uv run python experiments/cfg_sampling/sweep.py \
    --device cpu --ckpt "$1" \
    --steps 1,8,32 --w 1.5,3.5 --intervals "" --save-audio 8
}

run_sweep runs/fm-oneshots-mars-stft-b64-200k/checkpoints/step_00050000.pt
run_sweep runs/fm-oneshots-mars-stft-200k/checkpoints/step_00050000.pt
echo "=== 50k-vs-50k pair done ==="

run_sweep runs/fm-oneshots-mars-stft-b64-200k/checkpoints/step_00100000.pt

CKPT=runs/fm-oneshots-mars-stft-200k/checkpoints/step_00100000.pt
for _ in $(seq 1 120); do [ -f "$CKPT" ] && break; sleep 300; done
if [ -f "$CKPT" ]; then
  sleep 180  # let the 870MB write finish
  run_sweep "$CKPT"
else
  echo "timed out waiting for $CKPT"
fi
echo "=== AB DONE ==="
