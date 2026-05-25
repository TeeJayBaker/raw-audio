from __future__ import annotations

import argparse
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

from backbone.factory import build_backbone, load_backbone_config  # noqa: E402
from scripts.model_stats import count_params, format_param_count  # noqa: E402


def _dummy_input(cfg, batch_size: int, length: int):
    return torch.randn(batch_size, int(cfg.get("channels", 1)), length)


def _dummy_cond(cfg, batch_size: int):
    return torch.randn(batch_size, int(cfg.get("conditioning", {}).get("cond_dim", 16)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a Hydra-configured backbone.")
    parser.add_argument("config", nargs="?", default="configs/backbone/flow2gan.yaml")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    cfg = load_backbone_config(args.config)
    model = build_backbone(cfg).eval()
    x = _dummy_input(cfg, args.batch_size, args.length)
    cond = _dummy_cond(cfg, args.batch_size)
    length = args.length

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x, cond=cond, length=length)
        timings = []
        for _ in range(args.iters):
            start = time.perf_counter()
            y = model(x, cond=cond, length=length)
            timings.append(time.perf_counter() - start)

    mean_s = statistics.mean(timings)
    median_s = statistics.median(timings)
    samples_per_sec = args.batch_size * y.shape[-1] / mean_s
    sample_rate = int(cfg.get("sample_rate", 48000))
    xrt = samples_per_sec / sample_rate
    print(f"config: {args.config}")
    print(f"params: {format_param_count(count_params(model))} ({count_params(model)} total)")
    print(f"mean: {mean_s * 1000:.3f} ms")
    print(f"median: {median_s * 1000:.3f} ms")
    print(f"samples/sec: {samples_per_sec:.1f}")
    print(f"xRT: {xrt:.2f}")
    print(f"output: {tuple(y.shape)}")


if __name__ == "__main__":
    main()
