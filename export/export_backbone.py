from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backbone.convnext import ConvNeXt, IStftHead, WaveNeXtHead  # noqa: E402
from backbone.factory import build_backbone, load_backbone_config  # noqa: E402
from backbone.transformer import Transformer  # noqa: E402


def _window(n_fft: int, win_length: int, device: torch.device) -> torch.Tensor:
    window = torch.hann_window(win_length, device=device)
    if win_length == n_fft:
        return window
    left = (n_fft - win_length) // 2
    return F.pad(window, (left, n_fft - win_length - left))


def _crop_or_pad(x: torch.Tensor, length: int) -> torch.Tensor:
    diff = x.shape[-1] - length
    if diff >= 0:
        start = diff // 2
        return x[..., start : start + length]
    left = -diff // 2
    return F.pad(x, (left, -diff - left))


def _stft(x: torch.Tensor, stft) -> torch.Tensor:
    n_fft, hop = stft.n_fft, stft.hop_length
    window = _window(n_fft, stft.win_length, x.device)
    phase = 2 * torch.pi * torch.arange(stft.freq_bins, device=x.device)[:, None]
    phase = phase * torch.arange(n_fft, device=x.device)[None, :] / n_fft
    kernels = torch.cat((torch.cos(phase), -torch.sin(phase))) * window
    batch, channels = x.shape[:2]
    padded = F.pad(x.flatten(0, 1).unsqueeze(1), (n_fft // 2,) * 2, mode="reflect")
    spec = F.conv1d(padded, kernels.unsqueeze(1), stride=hop)
    real, imag = spec.split(stft.freq_bins, dim=1)
    return torch.cat(
        (
            real.reshape(batch, channels * stft.freq_bins, -1),
            imag.reshape(batch, channels * stft.freq_bins, -1),
        ),
        dim=1,
    )


def _istft(x: torch.Tensor, channels: int, stft, length: int) -> torch.Tensor:
    n_fft, hop, bins = stft.n_fft, stft.hop_length, stft.freq_bins
    window = _window(n_fft, stft.win_length, x.device)
    real, imag = x.split(channels * bins, dim=1)
    real = real.reshape(-1, bins, x.shape[-1])
    imag = imag.reshape(-1, bins, x.shape[-1])

    phase = 2 * torch.pi * torch.arange(bins, device=x.device)[:, None]
    phase = phase * torch.arange(n_fft, device=x.device)[None, :] / n_fft
    weights = torch.ones(bins, device=x.device)
    if n_fft > 2:
        weights[1:-1] = 2
    frames = (
        (real * weights[None, :, None]).transpose(1, 2) @ torch.cos(phase)
        - (imag * weights[None, :, None]).transpose(1, 2) @ torch.sin(phase)
    ).transpose(1, 2)
    frames = frames * (window / n_fft)[None, :, None]

    basis = torch.eye(n_fft, device=x.device).unsqueeze(1)
    waveform = F.conv_transpose1d(frames, basis, stride=hop)
    envelope = F.conv_transpose1d(
        torch.ones_like(frames[:, :1]),
        window.square().reshape(1, 1, n_fft),
        stride=hop,
    )
    waveform = waveform / envelope.clamp_min(1e-11)
    start = n_fft // 2
    return waveform[..., start : start + length].reshape(-1, channels, length)


class OnnxBackbone(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.backbone, Transformer):
            return self._transformer(x, timestep, conditioning)
        if isinstance(self.backbone, ConvNeXt):
            return self._convnext(x, timestep, conditioning)
        return self.backbone(x, t=timestep, cond=conditioning, length=x.shape[-1])

    def _conditioning(self, timestep: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        model = self.backbone
        return model.cond_combine(
            model.time_embed(timestep),
            model.cond_embed(conditioning),
        )

    def _transformer(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        model = self.backbone
        cond = self._conditioning(timestep, conditioning)
        h_in = _stft(x, model.stft) if model.stft is not None else x
        h = model.in_proj(h_in).transpose(1, 2)
        h = _transformer_blocks(h, cond, model.blocks)
        y = model.out_proj(h.transpose(1, 2))
        if model.stft is not None:
            y = _crop_or_pad(y, h_in.shape[-1])
            return _istft(y, model.out_channels, model.stft, x.shape[-1])
        return _crop_or_pad(model.head(y), x.shape[-1]).float()

    def _convnext(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        model = self.backbone
        cond = self._conditioning(timestep, conditioning)
        outputs = []
        for branch in model.branches:
            h = branch.in_proj(_stft(x, branch.stft))
            for block in branch.trunk:
                h = block(h, cond)
            if isinstance(branch.head, WaveNeXtHead):
                y = branch.head.proj_hop(branch.head.proj_fft(h.transpose(1, 2)))
                outputs.append(_crop_or_pad(y.reshape(y.shape[0], 1, -1), x.shape[-1]))
            elif isinstance(branch.head, IStftHead):
                y = branch.head.proj(h)
                if branch.head.parameterisation == "magphase":
                    mag, phase = y.chunk(2, dim=1)
                    magnitude = torch.exp(torch.clamp(mag, max=1e2))
                    y = torch.cat(
                        (magnitude * torch.cos(phase), magnitude * torch.sin(phase)),
                        dim=1,
                    )
                outputs.append(_istft(y, model.out_channels, branch.stft, x.shape[-1]))
            else:
                raise TypeError(f"Unsupported head: {type(branch.head).__name__}")
        return torch.stack(outputs).sum(dim=0).float()


def _transformer_blocks(
    x: torch.Tensor,
    cond: torch.Tensor,
    blocks: nn.ModuleList,
) -> torch.Tensor:
    head_dim = blocks[0].attn.head_dim
    pos = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)
    freq = torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim
    angles = pos[:, None] / (10000**freq)[None, :]
    rope = (angles.cos()[None, None], angles.sin()[None, None])

    for block in blocks:
        scale1, shift1, gate1, scale2, shift2, gate2 = block.ada(cond)
        h = block.norm1(x) * (1 + scale1[:, None]) + shift1[:, None]
        x = x + gate1[:, None] * _attention(h, block.attn, rope)
        h = block.norm2(x) * (1 + scale2[:, None]) + shift2[:, None]
        x = x + gate2[:, None] * block.mlp(h)
    return x


def _attention(
    x: torch.Tensor,
    attention: nn.Module,
    rope: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    batch, length, dim = x.shape
    q, k, v = attention.qkv(x).reshape(
        batch, length, 3, attention.heads, attention.head_dim
    ).unbind(dim=2)
    q, k, v = (value.transpose(1, 2) for value in (q, k, v))
    if attention.rope:
        cos, sin = rope
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
    if attention.qk_norm:
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
    y = F.scaled_dot_product_attention(q, k, v)
    return attention.out(y.transpose(1, 2).reshape(batch, length, dim))


def _apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class OnnxStftBackbone(nn.Module):
    def __init__(self, backbone: Transformer):
        super().__init__()
        if backbone.stft is None:
            raise ValueError("External STFT export requires an STFT transformer.")
        self.backbone = backbone

    def forward(
        self,
        spec: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        model = self.backbone
        cond = model.cond_combine(
            model.time_embed(timestep),
            model.cond_embed(conditioning),
        )
        h = model.in_proj(spec).transpose(1, 2)
        h = _transformer_blocks(h, cond, model.blocks)
        return model.out_proj(h.transpose(1, 2))


def load_checkpoint(model: nn.Module, path: Path, use_ema: bool) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        state = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    model.load_state_dict(state)

    if use_ema and checkpoint.get("ema") is not None:
        state = model.state_dict()
        state.update(checkpoint["ema"]["shadow"])
        model.load_state_dict(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a configured backbone to ONNX.")
    parser.add_argument("config", help="Backbone config name or YAML path.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument(
        "--external-stft",
        action="store_true",
        help="Export an STFT-domain graph with fixed batch 1 and dynamic frame length.",
    )
    args = parser.parse_args()

    cfg = load_backbone_config(args.config)
    model = build_backbone(cfg).eval()
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, use_ema=not args.no_ema)

    samples = round(args.seconds * int(cfg.get("sample_rate", 48_000)))
    channels = int(cfg.get("channels", 1))
    cond_dim = int(cfg.conditioning.cond_dim)
    if args.external_stft:
        if not isinstance(model, Transformer) or model.stft is None:
            raise SystemExit("--external-stft currently requires an STFT transformer config.")
        if args.batch_size != 1:
            raise SystemExit("--external-stft uses fixed batch size 1.")
        frames = samples // model.stft.hop_length + 1
        inputs = (
            torch.randn(1, 2 * channels * model.stft.freq_bins, frames),
            torch.rand(1),
            torch.randn(1, cond_dim),
        )
        export_model = OnnxStftBackbone(model)
        input_names = ["spec", "timestep", "conditioning"]
        dynamic_axes = {"spec": {2: "frames"}, "output": {2: "frames"}}
    else:
        batch = args.batch_size
        inputs = (
            torch.randn(batch, channels, samples),
            torch.rand(batch),
            torch.randn(batch, cond_dim),
        )
        export_model = OnnxBackbone(model)
        input_names = ["x", "timestep", "conditioning"]
        dynamic_axes = {
            "x": {0: "batch"},
            "timestep": {0: "batch"},
            "conditioning": {0: "batch"},
            "output": {0: "batch"},
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            export_model,
            inputs,
            args.output,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            dynamo=False,
        )
    domain = "STFT-domain" if args.external_stft else "waveform"
    print(
        f"Exported {args.output} ({domain}, {samples} samples at "
        f"{cfg.get('sample_rate', 48_000)} Hz)"
    )


if __name__ == "__main__":
    main()
