from __future__ import annotations

import torch
from torch import nn

from backbone.blocks import activation
from backbone.complex_blocks import (
    ComplexConv1d,
    ComplexLayerNorm,
    ComplexPointwiseConv1d,
    SplitGELU,
)
from backbone.conditioning import make_conditioning


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class GRN1d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class BiasNorm1d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - self.bias
        rms = centered.pow(2).mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        return centered * rms * self.log_scale.exp()


class ConvNeXtBlock1d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        expansion: int = 4,
        backend: str = "real",
        activation_name: str = "gelu",
        norm_name: str = "layernorm",
        conditioning: dict | None = None,
        layer_scale: float | None = 1e-6,
        grn: bool = False,
    ):
        super().__init__()
        if backend == "complex":
            if channels % 2:
                raise ValueError(f"Complex ConvNeXt blocks require even channels, got {channels}")
            depthwise = ComplexConv1d(channels // 2, kernel_size)
        elif backend == "real":
            depthwise = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        self.depthwise = depthwise
        self.backend = backend
        if backend == "complex":
            self.norm = ComplexLayerNorm(channels // 2)
        elif norm_name == "biasnorm":
            self.norm = BiasNorm1d(channels)
        elif norm_name in {"layernorm", "ln"}:
            self.norm = ChannelLayerNorm(channels)
        else:
            raise ValueError(f"Unknown ConvNeXt norm: {norm_name}")
        hidden = channels * expansion
        if backend == "complex":
            if hidden % 2:
                raise ValueError(f"Complex ConvNeXt hidden width must be even, got {hidden}")
            layers: list[nn.Module] = [ComplexPointwiseConv1d(channels // 2, hidden // 2), SplitGELU()]
        else:
            layers = [nn.Conv1d(channels, hidden, 1), activation(activation_name, hidden)]
        if grn:
            if backend == "complex":
                raise ValueError("GRN is not implemented for complex ConvNeXt blocks")
            layers.append(GRN1d(hidden))
        layers.append(ComplexPointwiseConv1d(hidden // 2, channels // 2) if backend == "complex" else nn.Conv1d(hidden, channels, 1))
        self.pointwise = nn.Sequential(*layers)
        self.cond = make_conditioning(conditioning, channels)
        self.layer_scale = nn.Parameter(torch.ones(1, channels, 1) * layer_scale) if layer_scale is not None else None

    def forward(self, x, cond=None):
        residual = x
        x = self.depthwise(x)
        x = self.cond(self.norm(x), cond)
        x = self.pointwise(x)
        if self.layer_scale is not None:
            x = x * self.layer_scale
        return residual + x
