from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from backbone.audio_ops import STFTConfig  # noqa: E402
from backbone.factory import build_backbone, load_backbone_config  # noqa: E402
from scripts.model_stats import count_params, format_param_count  # noqa: E402


def _conditioning(cfg, batch_size: int) -> torch.Tensor | None:
    conditioning = cfg.get("conditioning", {})
    if conditioning.get("mode", "none") == "none":
        return None
    return torch.randn(batch_size, int(conditioning.get("cond_dim", 16)))


def _total_upsample(cfg) -> int:
    product = 1
    for factor in cfg.get("decoder", {}).get("up_factors", []):
        product *= int(factor)
    return product


def _input_for_one_second(cfg, batch_size: int, output_samples: int) -> tuple[torch.Tensor, int]:
    io = cfg.get("io", {})
    channels = int(io.get("channels", 1))
    target = output_samples

    if io.get("type") == "stft":
        stft = STFTConfig.from_dict(io)
        # torch.stft(center=True) emits floor(samples / hop) + 1 frames.
        frames = output_samples // stft.hop_length + 1
        x = torch.randn(batch_size, channels, stft.freq_bins, frames, dtype=torch.complex64)
        return x, target

    if cfg.get("_target_", "").endswith("Upsampler") and io.get("input_projection") == "stft":
        stft = STFTConfig.from_dict(io.get("stft", io))
        required_frames = (output_samples + _total_upsample(cfg) - 1) // _total_upsample(cfg)
        input_samples = max(stft.hop_length, (required_frames - 1) * stft.hop_length)
        return torch.randn(batch_size, int(io.get("input_channels", 1)), input_samples), target

    return torch.randn(batch_size, channels, output_samples), target


def benchmark(config: Path, batch_size: int, seconds: float, warmup: int, iters: int, threads: int | None) -> dict[str, str | float | int]:
    if threads is not None:
        torch.set_num_threads(threads)
    cfg = load_backbone_config(config)
    sample_rate = int(cfg.get("sample_rate", 48000))
    output_samples = int(round(seconds * sample_rate))
    model = build_backbone(cfg).eval()
    x, target = _input_for_one_second(cfg, batch_size, output_samples)
    cond = _conditioning(cfg, batch_size)

    with torch.inference_mode():
        for _ in range(warmup):
            y = model(x, cond=cond, length=target)
        timings = []
        for _ in range(iters):
            start = time.perf_counter()
            y = model(x, cond=cond, length=target)
            timings.append(time.perf_counter() - start)

    mean_s = statistics.mean(timings)
    median_s = statistics.median(timings)
    xrt = seconds / mean_s
    rtf = mean_s / seconds
    return {
        "config": str(config),
        "name": str(cfg.get("name", config.stem)),
        "target": str(cfg.get("_target_", "")),
        "io": str(cfg.get("io", {}).get("type", "waveform")),
        "params": count_params(model),
        "params_h": format_param_count(count_params(model)),
        "input_shape": tuple(x.shape),
        "output_shape": tuple(y.shape),
        "mean_ms": mean_s * 1000,
        "median_ms": median_s * 1000,
        "xrt": xrt,
        "rtf": rtf,
        "threads": torch.get_num_threads(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark all backbone configs for CPU real-time factor.")
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--threads", type=int, default=None, help="Torch CPU threads. Omit for PyTorch default.")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    rows = [benchmark(path, args.batch_size, args.seconds, args.warmup, args.iters, args.threads) for path in args.configs]
    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    for row in rows:
        print(
            f"{row['name']}: mean={row['mean_ms']:.3f} ms median={row['median_ms']:.3f} ms "
            f"xRT={row['xrt']:.2f} RTF={row['rtf']:.4f} params={row['params_h']} "
            f"input={row['input_shape']} output={row['output_shape']}"
        )


if __name__ == "__main__":
    main()
