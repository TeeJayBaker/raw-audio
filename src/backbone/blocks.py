from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from backbone.conditioning import make_conditioning


def _is_sequence(value) -> bool:
    return not isinstance(value, str) and hasattr(value, "__iter__")


def activation(name: str = "silu", channels: int | None = None) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    if name == "prelu":
        if channels is None:
            raise ValueError("prelu activation requires channels")
        return nn.PReLU(channels)
    if name == "silu":
        return nn.SiLU()
    if name == "snake_beta":
        if channels is None:
            raise ValueError("snake_beta activation requires channels")
        return SnakeBeta(channels)
    raise ValueError(f"Unknown activation: {name}")


class SnakeBeta(nn.Module):
    def __init__(self, channels: int, alpha: float = 1.0):
        super().__init__()
        value = torch.full((1, channels, 1), alpha).log()
        self.log_alpha = nn.Parameter(value.clone())
        self.log_beta = nn.Parameter(value.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.log_alpha.exp()
        beta = self.log_beta.exp()
        return x + (1.0 / (beta + 1e-8)) * torch.sin(alpha * x).pow(2)


def center_crop_or_pad(x: torch.Tensor, length: int) -> torch.Tensor:
    diff = x.shape[-1] - length
    if diff == 0:
        return x
    if diff > 0:
        start = diff // 2
        return x[..., start : start + length]
    left = (-diff) // 2
    right = -diff - left
    return F.pad(x, (left, right))


class ResidualBlock1d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        activation_name: str = "silu",
        conditioning: dict | None = None,
    ):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.GroupNorm(1, channels),
            activation(activation_name, channels),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.GroupNorm(1, channels),
            activation(activation_name, channels),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
        )
        self.cond = make_conditioning(conditioning, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        return x + self.net(self.cond(x, cond))


class CRASHDBlock1d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        activation_name: str = "silu",
        conditioning: dict | None = None,
    ):
        super().__init__()
        if len(dilations) != 4:
            raise ValueError("crash_dblock requires exactly four dilations")
        layers: list[nn.Module] = []
        for dilation in dilations:
            pad = int(dilation) * (kernel_size - 1) // 2
            layers.extend(
                [
                    activation(activation_name, channels),
                    nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=int(dilation)),
                ]
            )
        self.net = nn.Sequential(*layers)
        self.cond = make_conditioning(conditioning, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        return x + self.net(self.cond(x, cond))


class _MRFBranch(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: list[int] | tuple[int, ...],
        sublayer_depth: int,
        convs_per_sublayer: int,
        activation_name: str,
    ):
        super().__init__()
        if sublayer_depth < 1:
            raise ValueError("MRF sublayer_depth must be >= 1")
        if convs_per_sublayer < 1:
            raise ValueError("MRF convs_per_sublayer must be >= 1")
        if not dilations:
            raise ValueError("MRF dilations cannot be empty")
        self.sublayers = nn.ModuleList()
        for i in range(sublayer_depth):
            dilation = int(dilations[min(i, len(dilations) - 1)])
            layers: list[nn.Module] = []
            for conv_idx in range(convs_per_sublayer):
                conv_dilation = dilation if conv_idx == 0 else 1
                pad = conv_dilation * (kernel_size - 1) // 2
                layers.extend(
                    [
                        activation(activation_name, channels),
                        nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=conv_dilation),
                    ]
                )
            self.sublayers.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for sublayer in self.sublayers:
            x = x + sublayer(x)
        return x


class MRFBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int | list[int] | tuple[int, ...] = 3,
        dilations: list[int] | tuple[int, ...] | list[list[int]] | tuple[tuple[int, ...], ...] = (1, 3, 5),
        sublayer_depth: int = 3,
        convs_per_sublayer: int = 2,
        activation_name: str = "snake_beta",
        conditioning: dict | None = None,
    ):
        super().__init__()
        self.cond = make_conditioning(conditioning, channels)
        kernel_sizes = list(kernel_size) if _is_sequence(kernel_size) else [int(kernel_size)]
        dilations = list(dilations)
        if dilations and _is_sequence(dilations[0]):
            branch_dilations = [list(ds) for ds in dilations]
        else:
            branch_dilations = [list(dilations) for _ in kernel_sizes]
        if len(branch_dilations) != len(kernel_sizes):
            raise ValueError("MRF dilations must either be shared or match kernel_size branches")
        self.branches = nn.ModuleList(
            [
                _MRFBranch(
                    channels,
                    int(k),
                    branch_dilations[i],
                    int(sublayer_depth),
                    int(convs_per_sublayer),
                    activation_name,
                )
                for i, k in enumerate(kernel_sizes)
            ]
        )
        self.blocks = self.branches

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        h = self.cond(x, cond)
        return sum(branch(h) for branch in self.branches) / len(self.branches)


class Downsample1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int = 2, kernel_size: int | None = None, post_conv1x1: bool = False):
        super().__init__()
        self.factor = factor
        kernel_size = kernel_size or 2 * factor
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=factor, padding=kernel_size // 2)
        self.post = nn.Conv1d(out_channels, out_channels, 1) if post_conv1x1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.post(self.conv(x))


class Upsample1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int = 2, mode: str = "interp", weight_norm: bool = False):
        super().__init__()
        self.factor = factor
        self.mode = mode
        if mode == "transpose":
            kernel = factor * 2
            self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel, stride=factor, padding=factor // 2)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "transpose":
            return self.conv(x)
        x = F.interpolate(x, scale_factor=self.factor, mode="linear", align_corners=False)
        return self.conv(x)
