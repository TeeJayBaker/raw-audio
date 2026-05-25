#!/usr/bin/env python3
"""Measure crest factor / WavFlow-lift clipping on a one-shot dataset.

Standalone: needs only numpy + soundfile (the loader the project already uses).
Copy anywhere and run against the NAS dataset root:

    python crest_factor_check.py /path/to/dataset
    python crest_factor_check.py /path/to/dataset --limit 3000 --r-star 0.33 0.2 0.1

It replicates the trainer's per-clip RMS (root-mean-square over all channels and
time, src/trainer.py:329) and, for each candidate r_* (RMS-normalize target), reports
how much of each waveform the WavFlow lift's clamp(-1,1) would hard-clip. The clamp
fires when |x| > rms / r_*, i.e. when the crest factor exceeds 1/r_* (= 20*log10(1/r_*) dB).
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SUPPORTED_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}
SILENCE_RMS = 1e-6


def find_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS
    )


def pct(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="Dataset root (searched recursively).")
    ap.add_argument("--limit", type=int, default=0, help="Randomly sample at most N files (0 = all).")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed for --limit.")
    ap.add_argument("--r-star", type=float, nargs="+", default=[0.33, 0.2, 0.1],
                    help="RMS-normalize targets to evaluate (WavFlow default 0.33).")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"error: root does not exist: {args.root}", file=sys.stderr)
        return 2

    files = find_files(args.root)
    if not files:
        print(f"error: no audio files under {args.root}", file=sys.stderr)
        return 2
    if args.limit and len(files) > args.limit:
        random.Random(args.seed).shuffle(files)
        files = sorted(files[: args.limit])

    crest_db: list[float] = []
    durations: list[float] = []
    # per-r_* accumulators: per-file fraction of samples that get clipped
    clip_frac: dict[float, list[float]] = {r: [] for r in args.r_star}

    n_read = n_silent = n_err = 0
    for i, path in enumerate(files):
        if i and i % 500 == 0:
            print(f"  ...{i}/{len(files)}", file=sys.stderr)
        try:
            audio, sr = sf.read(path, always_2d=True, dtype="float32")
        except Exception as exc:  # noqa: BLE001 - report-and-skip unreadable files
            n_err += 1
            if n_err <= 10:
                print(f"  skip (read error): {path.name}: {exc}", file=sys.stderr)
            continue

        x = audio.reshape(-1)
        x = x[np.isfinite(x)]
        if x.size == 0:
            n_err += 1
            continue

        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        if rms < SILENCE_RMS:
            n_silent += 1
            continue

        peak = float(np.max(np.abs(x)))
        n_read += 1
        durations.append(audio.shape[0] / float(sr))
        crest_db.append(20.0 * math.log10(peak / rms))

        absx = np.abs(x)
        for r in args.r_star:
            clip_frac[r].append(float(np.mean(absx > rms / r)))

    if n_read == 0:
        print("error: no readable non-silent files", file=sys.stderr)
        return 2

    crest = np.asarray(crest_db)
    dur = np.asarray(durations)

    print("=" * 64)
    print(f"root              : {args.root}")
    print(f"files analysed    : {n_read}  (silent skipped {n_silent}, errors {n_err})")
    print(f"duration (s)      : median {np.median(dur):.2f}  mean {dur.mean():.2f}  max {dur.max():.2f}")
    print()
    print("CREST FACTOR  peak/RMS over the whole clip, in dB  (RMS as the trainer computes it)")
    print(f"  min {crest.min():5.1f} | p10 {pct(crest,10):5.1f} | median {np.median(crest):5.1f} "
          f"| mean {crest.mean():5.1f} | p90 {pct(crest,90):5.1f} | p99 {pct(crest,99):5.1f} | max {crest.max():5.1f}")
    print()

    # histogram of crest dB
    edges = [0, 3, 6, 9, 12, 15, 18, 21, 24, 30, 1e9]
    labels = ["0-3", "3-6", "6-9", "9-12", "12-15", "15-18", "18-21", "21-24", "24-30", "30+"]
    counts, _ = np.histogram(crest, bins=edges)
    width = max(counts.max(), 1)
    print("  crest dB   count")
    for lab, c in zip(labels, counts):
        bar = "#" * int(round(40 * c / width))
        print(f"  {lab:>7} | {c:6d} {bar}")
    print()

    print("LIFT CLIPPING  what the clamp(-1,1) destroys at each r_*  (lifted RMS stays ~1 if s_a = 1/r_*)")
    print(f"  {'r_*':>5} {'ceiling':>8} {'files w/ clip':>14} {'mean clipped':>13} {'median clipped':>15}")
    for r in args.r_star:
        ceil_db = 20.0 * math.log10(1.0 / r)
        cf = np.asarray(clip_frac[r])
        files_clipped = float(np.mean(cf > 0.0))           # any sample clipped
        print(f"  {r:>5.2f} {ceil_db:>6.1f}dB {files_clipped*100:>13.1f}% "
              f"{cf.mean()*100:>12.3f}% {np.median(cf)*100:>14.3f}%")
    print()
    print("Reading: 'files w/ clip' = share of files with at least one clipped sample (peak above the")
    print("ceiling). 'mean/median clipped' = per-file share of samples hard-clipped. Note one-shots loaded")
    print("with zero/start-pad in training have lower RMS than measured here, so real clipping is somewhat worse.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
