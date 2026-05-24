from __future__ import annotations

import torch
from torch import nn

from backbone.audio_ops import (
    STFTConfig,
    as_waveform,
    channels_to_complex,
    complex_to_channels,
    stft_to_waveform,
    waveform_to_stft,
)
from backbone.blocks import center_crop_or_pad
from backbone.complex_blocks import ComplexPointwiseConv1d
from backbone.conditioning import TimeEmbedding, conditioning_mode, prepare_conditioning
from backbone.convnext_blocks import ConvNeXtBlock1d


def _mode(branches: dict | str | None) -> str:
    cfg = {"mode": branches} if isinstance(branches, str) else (branches or {})
    mode = cfg.get("mode", cfg.get("type", "single"))
    return "multi_resolution" if mode == "multi" else mode


def _subband_slices(freq_bins: int, count: int, overlap_bins: int = 0) -> list[tuple[int, int, int, int]]:
    if count < 1:
        raise ValueError("subband count must be >= 1")
    bounds = torch.linspace(0, freq_bins, count + 1).round().to(torch.int64).tolist()
    slices = []
    for i in range(count):
        core_start = int(bounds[i])
        core_end = int(bounds[i + 1])
        start = max(0, core_start - overlap_bins)
        end = min(freq_bins, core_end + overlap_bins)
        slices.append((start, end, core_start, core_end))
    return slices


def _prepare_global_cond(cond: torch.Tensor, dim: int) -> torch.Tensor:
    if cond.ndim == 1:
        cond = cond[:, None]
    elif cond.ndim == 3 and cond.shape[-1] == 1:
        cond = cond.squeeze(-1)
    if cond.ndim != 2:
        raise ValueError(f"Expected global conditioning, got {tuple(cond.shape)}")
    if cond.shape[1] == 1 and dim != 1:
        cond = cond.expand(-1, dim)
    if cond.shape[1] != dim:
        raise ValueError(f"Expected conditioning dim {dim}, got {cond.shape[1]}")
    return cond


class IStftHead(nn.Module):
    def __init__(self, dim: int, out_channels: int, stft: STFTConfig, parameterisation: str = "magphase"):
        super().__init__()
        self.out_channels = out_channels
        self.stft = stft
        self.parameterisation = parameterisation
        if parameterisation not in {"realimag", "magphase"}:
            raise ValueError(f"Unknown iSTFT parameterisation: {parameterisation}")
        self.proj = nn.Conv1d(dim, 2 * out_channels * stft.freq_bins, 1)

    def spec(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x)
        if self.parameterisation == "magphase":
            b, _cf, t = y.shape
            mag, phase = y.split(self.out_channels * self.stft.freq_bins, dim=1)
            mag = mag.reshape(b, self.out_channels, self.stft.freq_bins, t)
            phase = phase.reshape(b, self.out_channels, self.stft.freq_bins, t)
            mag = torch.exp(torch.clamp(mag, max=1e2))
            spec = torch.polar(mag, phase)
        else:
            spec = channels_to_complex(y, self.out_channels, self.stft.freq_bins)
        return spec

    def forward(self, x: torch.Tensor, length: int) -> torch.Tensor:
        spec = self.spec(x)
        return stft_to_waveform(spec, self.stft, length=length)


class ComplexIStftHead(nn.Module):
    def __init__(self, dim: int, out_channels: int, stft: STFTConfig):
        super().__init__()
        if dim % 2:
            raise ValueError(f"Complex iSTFT head requires even real/imag channels, got {dim}")
        self.out_channels = out_channels
        self.stft = stft
        self.proj = ComplexPointwiseConv1d(dim // 2, out_channels * stft.freq_bins)

    def spec(self, x: torch.Tensor) -> torch.Tensor:
        return channels_to_complex(self.proj(x), self.out_channels, self.stft.freq_bins)

    def forward(self, x: torch.Tensor, length: int) -> torch.Tensor:
        return stft_to_waveform(self.spec(x), self.stft, length=length)


class WaveNeXtHead(nn.Module):
    def __init__(self, dim: int, stft: STFTConfig, out_channels: int = 1):
        super().__init__()
        if out_channels != 1:
            raise ValueError("WaveNeXt head currently emits mono waveform output")
        self.n_fft = stft.n_fft
        self.hop_length = stft.hop_length
        self.proj_fft = nn.Linear(dim, stft.n_fft)
        self.proj_hop = nn.Linear(stft.n_fft, stft.hop_length, bias=False)

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.proj_hop(self.proj_fft(x))
        return x.reshape(x.shape[0], 1, -1)

    def forward(self, x: torch.Tensor, length: int) -> torch.Tensor:
        return center_crop_or_pad(self.raw(x), length)


def _layer_scale(value, depth: int) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value == "1/depth":
            return 1.0 / max(1, depth)
        return float(value)
    return float(value)


class _Trunk(nn.Module):
    def __init__(self, channels: int, block: dict, backend: str, conditioning: dict | None):
        super().__init__()
        depth = int(block.get("depth", 4))
        scale = _layer_scale(block.get("layer_scale", 1e-6), depth)
        self.blocks = nn.ModuleList(
            [
                ConvNeXtBlock1d(
                    channels,
                    kernel_size=block.get("kernel_size", 7),
                    expansion=block.get("expansion", 4),
                    backend=backend,
                    activation_name=block.get("activation", "gelu"),
                    norm_name=block.get("norm", "layernorm"),
                    conditioning=conditioning,
                    layer_scale=scale,
                    grn=bool(block.get("grn", False)),
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cond)
        return x


class _STFTBranch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stft_cfg: dict | None,
        head_cfg: dict | None,
        block: dict,
        backend: str,
        conditioning: dict | None,
        freq_bins: int | None = None,
    ):
        super().__init__()
        self.stft = STFTConfig.from_dict(stft_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.freq_bins = freq_bins or self.stft.freq_bins
        width = int(block.get("channels", 32))
        self.backend = backend
        if backend == "complex":
            if width % 2:
                raise ValueError("Complex STFT branches require an even block.channels value")
            self.in_proj = ComplexPointwiseConv1d(in_channels * self.freq_bins, width // 2)
        else:
            in_kernel = int(block.get("input_kernel_size", 7))
            self.in_proj = nn.Conv1d(2 * in_channels * self.freq_bins, width, in_kernel, padding=in_kernel // 2)
        self.trunk = _Trunk(width, block, backend, conditioning)
        head_cfg = head_cfg or {}
        head_type = head_cfg.get("type", "istft")
        if head_type == "istft":
            self.head = IStftHead(width, out_channels, self.stft, head_cfg.get("parameterisation", "realimag"))
        elif head_type == "complex_istft":
            if backend != "complex":
                raise ValueError("complex_istft head requires backend.type: complex")
            self.head = ComplexIStftHead(width, out_channels, self.stft)
        elif head_type == "wavenext":
            self.head = WaveNeXtHead(width, self.stft, out_channels)
        else:
            raise ValueError(f"Unknown ConvNeXt head type: {head_type}")

    def features(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        return self.trunk(self.in_proj(x), cond)

    def spec(self, spec: torch.Tensor, cond: torch.Tensor | None, quantize=None) -> torch.Tensor:
        if spec.shape[2] != self.freq_bins:
            raise ValueError(f"Expected {self.freq_bins} freq bins, got {spec.shape[2]}")
        h = self.in_proj(complex_to_channels(spec))
        if quantize is not None:
            h = quantize(h)
        h = self.trunk(h, cond)
        if not isinstance(self.head, IStftHead | ComplexIStftHead):
            raise ValueError("Only iSTFT heads can emit complex STFT output")
        return self.head.spec(h)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None, length: int, quantize=None) -> torch.Tensor:
        spec = waveform_to_stft(x, self.stft)
        h = self.in_proj(complex_to_channels(spec))
        if quantize is not None:
            h = quantize(h)
        h = self.trunk(h, cond)
        return self.head(h, length)


class ConvNeXt(nn.Module):
    def __init__(
        self,
        io: dict | None = None,
        branches: dict | str | None = None,
        backend: dict | str | None = None,
        block: dict | None = None,
        conditioning: dict | None = None,
        stft: dict | None = None,
        head: dict | None = None,
        sample_rate: int = 48000,
        name: str | None = None,
    ):
        super().__init__()
        self.io = io or {"type": "waveform", "channels": 1}
        self.sample_rate = sample_rate
        self.name = name
        self.branch_cfg = {"mode": branches} if isinstance(branches, str) else (branches or {"mode": "single"})
        backend_cfg = {"type": backend} if isinstance(backend, str) else (backend or {"type": "real"})
        self.backend = backend_cfg.get("type", "real")
        if self.backend == "complex" and backend_cfg.get("phase_quantization"):
            self.phase_bins = int(backend_cfg["phase_quantization"])
        else:
            self.phase_bins = None
        block = block or {}
        self.mode = _mode(self.branch_cfg)
        self.in_channels = int(self.io.get("channels", 1))
        self.out_channels = int(self.io.get("out_channels", self.in_channels))
        self.output_stft = self.io.get("type") == "stft"
        self.cond_mode = conditioning_mode(conditioning)
        self.cond_dim = int((conditioning or {}).get("cond_dim", int(block.get("channels", 32))))
        self.time_embed = (
            TimeEmbedding(self.cond_dim, (conditioning or {}).get("time_hidden_dim"), time_scale=(conditioning or {}).get("time_scale", 1.0))
            if self.cond_mode != "none"
            else None
        )
        self.uses_stft = stft is not None or head is not None or self.io.get("type") == "stft"

        if self.uses_stft:
            self._build_stft_paths(stft or self.io, head or {"type": "istft"}, block, conditioning)
        else:
            self._build_waveform_paths(block, conditioning)

    def _build_waveform_paths(self, block: dict, conditioning: dict | None) -> None:
        width = int(block.get("channels", 32))
        count = int(self.branch_cfg.get("count", 1 if self.mode == "single" else 2))
        self.in_proj = nn.Conv1d(self.in_channels, width, 3, padding=1)
        self.branches = nn.ModuleList([_Trunk(width, block, self.backend, conditioning) for _ in range(count)])
        self.mix = nn.Conv1d(width * count, width, 1)
        self.out_proj = nn.Conv1d(width, self.out_channels, 3, padding=1)

    def _build_stft_paths(self, stft_cfg: dict, head_cfg: dict, block: dict, conditioning: dict | None) -> None:
        if self.mode == "multi_resolution":
            resolutions = self.branch_cfg.get("resolutions")
            if not resolutions:
                raise ValueError("branches.resolutions is required when branches.mode is multi_resolution")
            branch_blocks = self.branch_cfg.get("blocks")
            if branch_blocks is not None and len(branch_blocks) != len(resolutions):
                raise ValueError("branches.blocks must match multi-resolution branch count")
            self.resolution_branches = nn.ModuleList(
                [
                    _STFTBranch(
                        self.in_channels,
                        self.out_channels,
                        res,
                        head_cfg,
                        dict(block) | dict(branch_blocks[i] if branch_blocks is not None else {}),
                        self.backend,
                        conditioning,
                    )
                    for i, res in enumerate(resolutions)
                ]
            )
            return

        self.stft = STFTConfig.from_dict(stft_cfg)
        width = int(block.get("channels", 32))
        self.head_type = head_cfg.get("type", "istft")
        if self.mode == "subband":
            count = int(self.branch_cfg.get("count", 2))
            if self.head_type == "wavenext":
                raise ValueError("subband ConvNeXt does not support wavenext head")
            self.subband_count = count
            self.subband_slices = _subband_slices(self.stft.freq_bins, count, int(self.branch_cfg.get("overlap_bins", 0)))
            self.subband_conditioning = bool(self.branch_cfg.get("subband_conditioning", self.branch_cfg.get("condition_subband", False)))
            self.subband_embed = nn.Embedding(count, int((conditioning or {}).get("cond_dim", width))) if self.subband_conditioning else None
            self.subband_width = max(end - start for start, end, _core_start, _core_end in self.subband_slices)
            self.subband_in = nn.Conv1d(2 * self.in_channels * self.subband_width, width, 3, padding=1)
            self.subband_trunk = _Trunk(width, block, self.backend, conditioning)
            if self.head_type == "istft":
                self.subband_out = nn.Conv1d(width, 2 * self.out_channels * self.subband_width, 1)
                self.head = None
            else:
                raise ValueError(f"Unknown ConvNeXt head type: {self.head_type}")
            return

        self.branch = _STFTBranch(
            self.in_channels,
            self.out_channels,
            stft_cfg,
            head_cfg,
            block,
            self.backend,
            conditioning,
        )

    def _quantize_phase(self, spec: torch.Tensor) -> torch.Tensor:
        if self.phase_bins is None:
            return spec
        mag = spec.abs()
        phase = torch.angle(spec)
        step = 2 * torch.pi / self.phase_bins
        rounded_phase = torch.round(phase / step) * step
        phase = phase + (rounded_phase - phase).detach()
        return torch.polar(mag, phase)

    def _quantize_latent_phase(self, x: torch.Tensor) -> torch.Tensor:
        if self.phase_bins is None:
            return x
        if x.shape[1] % 2:
            raise ValueError("phase quantization requires real/imag channel pairs")
        real, imag = x.chunk(2, dim=1)
        spec = torch.complex(real, imag)
        quantized = self._quantize_phase(spec)
        return torch.cat([quantized.real, quantized.imag], dim=1)

    def _forward_waveform(self, x: torch.Tensor, cond: torch.Tensor | None, target: int) -> torch.Tensor:
        h = self.in_proj(x)
        outs = []
        for idx, branch in enumerate(self.branches):
            hb = h
            if self.mode == "subband" and len(self.branches) > 1:
                hb = torch.roll(hb, shifts=idx, dims=-1)
            hb = branch(hb, cond)
            if self.mode == "subband" and len(self.branches) > 1:
                hb = torch.roll(hb, shifts=-idx, dims=-1)
            outs.append(hb)
        return center_crop_or_pad(self.out_proj(self.mix(torch.cat(outs, dim=1))), target)

    def _forward_subband_spec(self, spec: torch.Tensor, cond: torch.Tensor | None, return_aux: bool = False):
        spec_out = spec.new_zeros(spec.shape[0], self.out_channels, self.stft.freq_bins, spec.shape[-1])
        weight = spec.real.new_zeros(spec.shape[0], 1, self.stft.freq_bins, spec.shape[-1])
        chunks = []
        widths = []
        for start, end, _core_start, _core_end in self.subband_slices:
            chunk = spec[:, :, start:end]
            widths.append(end - start)
            if end - start < self.subband_width:
                pad = self.subband_width - (end - start)
                chunk = torch.nn.functional.pad(chunk, (0, 0, 0, pad))
            chunks.append(complex_to_channels(chunk))
        h = self.subband_in(torch.cat(chunks, dim=0))
        subband_cond = cond
        if self.subband_embed is not None:
            ids = torch.arange(self.subband_count, device=h.device).repeat_interleave(spec.shape[0])
            subband_cond = self.subband_embed(ids).to(dtype=h.dtype)
            if cond is not None:
                cond = _prepare_global_cond(cond, subband_cond.shape[1])
                subband_cond = subband_cond + cond.repeat(self.subband_count, 1)
        h = self.subband_trunk(h, subband_cond)
        preds = channels_to_complex(self.subband_out(h), self.out_channels, self.subband_width).chunk(self.subband_count, dim=0)
        aux = []
        for pred, width, (start, end, core_start, core_end) in zip(preds, widths, self.subband_slices, strict=True):
            pred = pred[:, :, :width]
            spec_out[:, :, start:end] = spec_out[:, :, start:end] + pred
            weight[:, :, start:end] = weight[:, :, start:end] + 1
            aux.append({"prediction": pred, "start": start, "end": end, "core_start": core_start, "core_end": core_end})
        spec_out = spec_out / weight.clamp_min(1).to(dtype=spec_out.real.dtype)
        return (spec_out, {"subbands": aux}) if return_aux else spec_out

    def _forward_subband(self, x: torch.Tensor, cond: torch.Tensor | None, target: int, return_aux: bool = False):
        spec = waveform_to_stft(x, self.stft)
        result = self._forward_subband_spec(spec, cond, return_aux=return_aux)
        if return_aux:
            spec_out, aux = result
            return stft_to_waveform(spec_out, self.stft, length=target), aux
        return stft_to_waveform(result, self.stft, length=target)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        length: int | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor:
        if t is not None and self.time_embed is not None:
            t = self.time_embed(t)
        cond = prepare_conditioning(t, cond, self.cond_mode, self.cond_dim)
        if self.output_stft and not self.uses_stft:
            raise ValueError("io.type: stft requires STFT paths")
        if self.output_stft:
            if not torch.is_complex(x) or x.ndim != 4:
                raise ValueError(f"Expected complex STFT [B, C, F, T], got {tuple(x.shape)}")
            if self.mode == "subband":
                return self._forward_subband_spec(x, cond, return_aux=return_aux)
            if self.mode != "single":
                raise ValueError("STFT-output ConvNeXt currently supports single or subband branches")
            return self.branch.spec(x, cond, quantize=self._quantize_latent_phase if self.backend == "complex" else None)
        x = as_waveform(x)
        target = int(length or x.shape[-1])
        if not self.uses_stft:
            return self._forward_waveform(x, cond, target)
        if self.mode == "multi_resolution":
            ys = [branch(x, cond, target) for branch in self.resolution_branches]
            return torch.stack(ys, dim=0).sum(dim=0)
        if self.mode == "subband":
            return self._forward_subband(x, cond, target, return_aux=return_aux)
        if self.mode != "single":
            raise ValueError(f"Unknown ConvNeXt branch mode: {self.mode}")
        return self.branch(x, cond, target, quantize=self._quantize_latent_phase if self.backend == "complex" else None)
