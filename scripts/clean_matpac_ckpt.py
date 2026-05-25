"""Strip Lightning hparams / callbacks from a MATPAC .ckpt and resave as a
portable {"state_dict": ...} file with no pickled cross-project classes.

The original training project (flow-one-shot / diff_one_shot) pickles class
references into the Lightning checkpoint, so `torch.load(weights_only=False)`
from raw-audio fails with `ModuleNotFoundError: src.conditioning`. This script
must be run from a venv that has the original pickle classes available (e.g.
the flow-one-shot venv). The output is a plain dict of tensors that any venv
can load.

Usage (from flow-one-shot venv):
    cd /home/tom/projects/flow-one-shot
    PYTHONPATH=. uv run python /home/tom/projects/raw-audio/scripts/clean_matpac_ckpt.py \
        /media/NAS/neutone/diff_one_shot/checkpoints/whole-violet-235/last-v1.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def clean(src: Path, dst: Path | None = None) -> Path:
    src = src.expanduser().resolve()
    if dst is None:
        dst = src.with_name(src.stem + "_clean" + src.suffix)
    dst = dst.expanduser().resolve()

    print(f"Loading: {src}")
    ckpt = torch.load(str(src), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit(f"Expected dict at top level, got {type(ckpt).__name__}")
    if "state_dict" not in ckpt:
        raise SystemExit(f"No 'state_dict' key in {src} (keys: {list(ckpt.keys())[:10]})")

    state_dict = {k: v.detach().cpu() for k, v in ckpt["state_dict"].items()}
    print(f"  {len(state_dict)} tensors, first 3 keys: {list(state_dict)[:3]}")

    torch.save({"state_dict": state_dict}, str(dst))
    print(f"Wrote: {dst}")

    # Sanity-check round-trip with weights_only=True (no class lookups).
    reloaded = torch.load(str(dst), map_location="cpu", weights_only=True)
    assert set(reloaded["state_dict"]) == set(state_dict)
    print("Round-trip verified (weights_only=True load works).")
    return dst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path, help="Path to original Lightning .ckpt")
    parser.add_argument("--dst", type=Path, default=None, help="Output path (default: <src>_clean.<ext>)")
    args = parser.parse_args()
    clean(args.src, args.dst)


if __name__ == "__main__":
    main()
