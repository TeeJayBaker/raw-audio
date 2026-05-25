from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from backbone.factory import build_backbone  # noqa: E402
from flow.fm import RectifiedFlow  # noqa: E402


def _load_config(path: Path):
    configs_root = (ROOT / "configs").resolve()
    resolved = path.resolve()
    if path.suffix == ".yaml" and configs_root in resolved.parents:
        config_name = resolved.relative_to(configs_root).with_suffix("").as_posix()
        with hydra.initialize_config_dir(version_base=None, config_dir=str(configs_root)):
            return hydra.compose(config_name=config_name)
    return OmegaConf.load(path)


def benchmark(
    cfg,
    batch_size: int,
    steps: int,
    warmup: int,
    iters: int,
    threads: int | None,
    seconds: float,
) -> dict[str, float | int]:
    if threads is not None:
        torch.set_num_threads(threads)
    model = build_backbone(cfg.backbone).eval()
    shape = (
        batch_size,
        int(cfg.data.get("channels", 1)),
        int(round(seconds * int(cfg.data.sample_rate))),
    )
    cond_dim = int(cfg.backbone.conditioning.cond_dim)
    cond = torch.zeros(batch_size, cond_dim)
    flow = RectifiedFlow()
    method = str(cfg.sampling.get("method", "euler"))

    with torch.inference_mode():
        for _ in range(warmup):
            flow.sample(model, shape=shape, cond=cond, steps=steps, method=method)
        timings = []
        for _ in range(iters):
            start = time.perf_counter()
            flow.sample(model, shape=shape, cond=cond, steps=steps, method=method)
            timings.append(time.perf_counter() - start)

    mean_s = statistics.mean(timings)
    median_s = statistics.median(timings)
    generated_s = float(seconds) * batch_size
    return {
        "batch_size": batch_size,
        "steps": steps,
        "threads": torch.get_num_threads(),
        "mean_ms": mean_s * 1000.0,
        "median_ms": median_s * 1000.0,
        "xrt": generated_s / mean_s,
        "rtf": mean_s / generated_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FM sampling CPU xRT.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment/fm_baseline.yaml")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="sample length in seconds (defaults to data.max_seconds)",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    steps = int(args.steps or cfg.sampling.steps)
    seconds = float(args.seconds if args.seconds is not None else cfg.data.max_seconds)
    for batch_size in args.batch_sizes:
        row = benchmark(
            cfg, batch_size, steps, args.warmup, args.iters, args.threads, seconds
        )
        print(
            f"batch={row['batch_size']} steps={row['steps']} threads={row['threads']} "
            f"mean={row['mean_ms']:.3f} ms median={row['median_ms']:.3f} ms "
            f"xRT={row['xrt']:.2f} RTF={row['rtf']:.4f}"
        )


if __name__ == "__main__":
    main()
