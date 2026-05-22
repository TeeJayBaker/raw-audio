from __future__ import annotations

import torch
from torch import nn


class ComplexConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, groups: int | None = None):
        super().__init__()
        padding = kernel_size // 2
        groups = channels if groups is None else groups
        self.real = nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=groups)
        self.imag = nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % 2:
            raise ValueError(f"ComplexConv1d requires even real/imag channels, got {x.shape[1]}")
        real, imag = x.chunk(2, dim=1)
        return torch.cat([self.real(real) - self.imag(imag), self.real(imag) + self.imag(real)], dim=1)


class ComplexPointwiseConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.real = nn.Conv1d(in_channels, out_channels, 1)
        self.imag = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % 2:
            raise ValueError(f"ComplexPointwiseConv1d requires even real/imag channels, got {x.shape[1]}")
        real, imag = x.chunk(2, dim=1)
        return torch.cat([self.real(real) - self.imag(imag), self.real(imag) + self.imag(real)], dim=1)


class ComplexLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.weight_rr = nn.Parameter(torch.ones(1, channels, 1))
        self.weight_ii = nn.Parameter(torch.ones(1, channels, 1))
        self.weight_ri = nn.Parameter(torch.zeros(1, channels, 1))
        self.bias_real = nn.Parameter(torch.zeros(1, channels, 1))
        self.bias_imag = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 2 * self.channels:
            raise ValueError(f"Expected {2 * self.channels} complex channels, got {x.shape[1]}")
        real, imag = x.chunk(2, dim=1)
        real = real - real.mean(dim=(1, 2), keepdim=True)
        imag = imag - imag.mean(dim=(1, 2), keepdim=True)
        var_rr = real.pow(2).mean(dim=(1, 2), keepdim=True) + self.eps
        var_ii = imag.pow(2).mean(dim=(1, 2), keepdim=True) + self.eps
        cov_ri = (real * imag).mean(dim=(1, 2), keepdim=True)
        det = (var_rr * var_ii - cov_ri.pow(2)).clamp_min(self.eps)
        scale = det.rsqrt()
        wr = scale * var_ii.sqrt()
        wi = scale * var_rr.sqrt()
        wc = -scale * cov_ri / (var_rr.sqrt() + var_ii.sqrt()).clamp_min(self.eps)
        nr = wr * real + wc * imag
        ni = wc * real + wi * imag
        yr = self.weight_rr * nr - self.weight_ri * ni + self.bias_real
        yi = self.weight_ri * nr + self.weight_ii * ni + self.bias_imag
        return torch.cat([yr, yi], dim=1)


class SplitGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = x.chunk(2, dim=1)
        return torch.cat([torch.nn.functional.gelu(real), torch.nn.functional.gelu(imag)], dim=1)
