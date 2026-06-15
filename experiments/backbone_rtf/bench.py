"""RTF benchmark of STFT-frame transformer backbone variants — transformer core only.

The STFT/iSTFT are deliberately excluded: the ONNX conv-STFT is an export artifact, not
representative of a real C++ FFT, and its cost is ~constant across variants — including it
would drown out the differences we care about. So we benchmark the spec-domain graph
(spec in -> in_proj -> blocks -> out_proj -> spec out), the same path as
export/export_backbone.py --external-stft. One forward == 1 NFE; multiply by the sampler's
step count for end-to-end latency.

Variants (each an isolated change vs the original stft_transformer config):
  original   n_fft 2048 / hop 512, dim 512, depth 12 — freq folded into channels (2050->512)
  width768   block.dim 768 (report evolution step 1, "width d768")
  hop256     stft.hop_length 256 (2x frames/tokens)
  banded-K   freq split into K tokens/frame, full attention over time*freq (no 2050->512 squeeze)

The banded variants are scratch modules built from the same src building blocks; they are
benchmark stand-ins, not a training implementation.
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT, ROOT / "export"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import export_backbone as ex  # noqa: E402  (OnnxStftBackbone, _transformer_blocks)
from backbone.blocks import TransformerBlock  # noqa: E402
from backbone.conditioning import (  # noqa: E402
    ConditioningCombiner,
    ConditioningEmbedding,
    TimeEmbedding,
)
from backbone.factory import build_backbone, load_backbone_config  # noqa: E402
from backbone.io import STFTConfig  # noqa: E402

SECONDS = 1.0
OUT_DIR = ROOT / "experiments" / "backbone_rtf" / "onnx"


def _frames(cfg) -> int:
    sr = int(cfg.get("sample_rate", 48_000))
    hop = int(cfg.stft.hop_length)
    return round(SECONDS * sr) // hop + 1


def _spec_dim(cfg) -> int:
    n_fft = int(cfg.stft.n_fft)
    channels = int(cfg.get("channels", 1))
    return 2 * channels * (n_fft // 2 + 1)


class BandedStftCore(nn.Module):
    """Spec-domain transformer core with frequency-banded tokens: each frame's 2*F channels
    are split into `n_bands` tokens, so attention mixes time AND frequency (cross-frequency
    sharing) instead of squeezing 2*F -> dim per frame. Reuses the export module's block
    runner so it times on the same footing as the OnnxStftBackbone variants."""

    def __init__(self, cfg, n_bands: int):
        super().__init__()
        self.n_bands = n_bands
        self.channels = int(cfg.channels)
        self.stft = STFTConfig.from_dict(OmegaConf.to_container(cfg.stft, resolve=True))
        dim = int(cfg.block.dim)
        depth = int(cfg.block.depth)
        heads = int(cfg.block.get("heads", 8))
        cond_dim = int(cfg.conditioning.cond_dim)
        embed_dim = int(cfg.conditioning.get("embed_dim", cond_dim))
        time_scale = float(cfg.conditioning.get("time_scale", 1.0))

        self.time_embed = TimeEmbedding(cond_dim, time_scale=time_scale)
        self.cond_embed = ConditioningEmbedding(embed_dim, cond_dim)
        self.cond_combine = ConditioningCombiner(cond_dim)

        self.full_ch = 2 * self.channels * self.stft.freq_bins
        self.band_in = math.ceil(self.full_ch / n_bands)
        self.in_proj = nn.Linear(self.band_in, dim)
        self.out_proj = nn.Linear(dim, self.band_in)
        self.blocks = nn.ModuleList(
            TransformerBlock(dim, cond_dim, heads=heads, rope=True, qk_norm=True)
            for _ in range(depth)
        )

    def forward(self, spec, timestep, conditioning):
        cond = self.cond_combine(self.time_embed(timestep), self.cond_embed(conditioning))
        b, ch, frames = spec.shape
        spec = F.pad(spec, (0, 0, 0, self.n_bands * self.band_in - ch))
        tokens = (
            spec.reshape(b, self.n_bands, self.band_in, frames)
            .permute(0, 1, 3, 2)
            .reshape(b, self.n_bands * frames, self.band_in)
        )
        h = self.in_proj(tokens)
        h = ex._transformer_blocks(h, cond, self.blocks)
        y = self.out_proj(h)
        return (
            y.reshape(b, self.n_bands, frames, self.band_in)
            .permute(0, 1, 3, 2)
            .reshape(b, self.n_bands * self.band_in, frames)[:, :ch, :]
        )


def export_onnx(model: nn.Module, cfg, path: Path) -> None:
    """Export the spec-domain core at fixed batch 1 and fixed frame count (the RTF regime).
    Random weights — shapes are what matter for timing."""
    cond_dim = int(cfg.conditioning.cond_dim)
    inputs = (
        torch.randn(1, _spec_dim(cfg), _frames(cfg)),
        torch.rand(1),
        torch.randn(1, cond_dim),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            inputs,
            str(path),
            input_names=["spec", "timestep", "conditioning"],
            output_names=["output"],
            opset_version=17,
            dynamo=False,
        )


def benchmark(path: Path, threads: int, iters: int = 30, warmup: int = 8) -> float:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    shapes = {v.name: v.shape for v in session.get_inputs()}
    inputs = {
        "spec": np.random.randn(1, shapes["spec"][1], shapes["spec"][2]).astype(np.float32),
        "timestep": np.random.rand(1).astype(np.float32),
        "conditioning": np.random.randn(1, shapes["conditioning"][-1]).astype(np.float32),
    }
    for _ in range(warmup):
        session.run(None, inputs)
    timings = []
    for _ in range(iters):
        start = time.perf_counter()
        session.run(None, inputs)
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def build_variants():
    base = load_backbone_config(str(ROOT / "configs" / "backbone" / "stft_transformer.yaml"))

    def cfg_with(**over):
        c = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
        for dotted, val in over.items():
            OmegaConf.update(c, dotted, val)
        return c

    yield "original", base, ex.OnnxStftBackbone(build_backbone(base).eval())
    c = cfg_with(**{"block.dim": 768})
    yield "width768", c, ex.OnnxStftBackbone(build_backbone(c).eval())
    c = cfg_with(**{"stft.hop_length": 256})
    yield "hop256", c, ex.OnnxStftBackbone(build_backbone(c).eval())
    for k in (2, 4, 8):
        yield f"banded-K{k}", base, BandedStftCore(base, n_bands=k).eval()


def main() -> None:
    rows = []
    for name, cfg, model in build_variants():
        params = sum(p.numel() for p in model.parameters())
        path = OUT_DIR / f"{name}.onnx"
        export_onnx(model, cfg, path)
        t1 = benchmark(path, threads=1)
        t4 = benchmark(path, threads=4)
        rows.append((name, params, _frames(cfg), t1, t4))
        print(f"benched {name:12s} tokens-deep params={params/1e6:5.1f}M 1t={t1*1e3:6.2f}ms 4t={t4*1e3:6.2f}ms")

    base_t1 = rows[0][3]
    print("\n" + "=" * 80)
    print(f"{'variant':12s} {'params':>8s} {'frames':>7s} {'1-thread':>11s} {'vs orig':>8s} {'4-thread':>11s}")
    print("-" * 80)
    for name, params, frames, t1, t4 in rows:
        print(
            f"{name:12s} {params / 1e6:6.1f}M {frames:7d} {t1 * 1e3:9.2f}ms {t1 / base_t1:7.2f}x {t4 * 1e3:9.2f}ms"
        )
    print("=" * 80)
    print("Transformer core only (STFT/iSTFT excluded). Per-forward = 1 NFE; CPUExecutionProvider, batch 1.")


if __name__ == "__main__":
    main()
