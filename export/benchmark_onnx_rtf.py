from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ONNX backbone real-time factor.")
    parser.add_argument("model", type=Path)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("onnxruntime is required: uv add onnxruntime") from exc

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(
        args.model,
        sess_options=options,
        providers=[args.provider],
    )

    shapes = {value.name: value.shape for value in session.get_inputs()}
    cond_dim = shapes["conditioning"][-1]
    if not isinstance(cond_dim, int):
        raise SystemExit("Conditioning width must be static in the ONNX model.")

    if "spec" in shapes:
        frames = round(args.seconds * args.sample_rate) // args.hop_length + 1
        inputs = {
            "spec": np.random.randn(args.batch_size, shapes["spec"][1], frames).astype(
                np.float32
            ),
            "timestep": np.random.rand(args.batch_size).astype(np.float32),
            "conditioning": np.random.randn(args.batch_size, cond_dim).astype(np.float32),
        }
        seconds = args.seconds
    else:
        samples = shapes["x"][-1]
        if not isinstance(samples, int):
            raise SystemExit("Waveform state length must be static in the ONNX model.")
        inputs = {
            "x": np.random.randn(args.batch_size, shapes["x"][1], samples).astype(np.float32),
            "timestep": np.random.rand(args.batch_size).astype(np.float32),
            "conditioning": np.random.randn(args.batch_size, cond_dim).astype(np.float32),
        }
        seconds = samples / args.sample_rate
    for _ in range(args.warmup):
        session.run(None, inputs)

    timings = []
    for _ in range(args.iters):
        start = time.perf_counter()
        session.run(None, inputs)
        timings.append(time.perf_counter() - start)

    mean = statistics.mean(timings)
    median = statistics.median(timings)
    print(
        f"mean={mean * 1000:.3f} ms median={median * 1000:.3f} ms "
        f"xRT={seconds / mean:.2f} RTF={mean / seconds:.4f} "
        f"threads={args.threads} provider={session.get_providers()[0]}"
    )


if __name__ == "__main__":
    main()
