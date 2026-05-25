from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from backbone.blocks import ConvNeXtBlock1d, TransformerBlock
from backbone.conditioning import AdaLN, TimeEmbedding, prepare_conditioning
from backbone.convnext import ConvNeXt, IStftHead, WaveNeXtHead
from backbone.factory import build_backbone, load_backbone_config
from backbone.io import (
    STFTConfig,
    channels_to_complex,
    complex_to_channels,
    stft_to_waveform,
    waveform_to_stft,
)
from backbone.transformer import Transformer
from scripts.model_stats import count_params, format_param_count

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["wavenext", "vocos", "flow2gan", "stft_transformer", "waveform_transformer"]
LENGTH = 4096


@pytest.mark.parametrize("name", CONFIGS)
def test_forward_contract(name):
    cfg = load_backbone_config(name)
    model = build_backbone(cfg).eval()
    x = torch.randn(2, int(cfg.channels), LENGTH)
    cond = torch.randn(2, int(cfg.conditioning.cond_dim))
    with torch.inference_mode():
        y = model(x, t=torch.rand(2), cond=cond, length=LENGTH)
    assert y.shape == (2, int(cfg.get("out_channels", cfg.channels)), LENGTH)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()


def test_conditioning_alone_or_timestep_alone():
    cfg = load_backbone_config("wavenext")
    model = build_backbone(cfg).eval()
    x = torch.randn(1, 1, LENGTH)
    with torch.inference_mode():
        assert model(x, cond=torch.randn(1, int(cfg.conditioning.cond_dim)), length=LENGTH).shape == (1, 1, LENGTH)
        assert model(x, t=torch.rand(1), length=LENGTH).shape == (1, 1, LENGTH)


def test_multi_resolution_branches_sum():
    cfg = load_backbone_config("flow2gan")
    model = build_backbone(cfg)
    assert len(model.branches) == len(cfg.branches.resolutions)


def test_factory_accepts_path_and_dict_config():
    path_model = build_backbone("configs/backbone/wavenext.yaml")
    cfg_model = build_backbone(OmegaConf.load(ROOT / "configs" / "backbone" / "wavenext.yaml"))
    assert type(path_model) is type(cfg_model)


def test_trunks_resolve_to_expected_classes():
    assert isinstance(build_backbone("wavenext"), ConvNeXt)
    assert isinstance(build_backbone("waveform_transformer"), Transformer)


def test_prepare_conditioning_combines_timestep_and_conditioning():
    t = torch.randn(2, 4)
    cond = torch.randn(2, 4)
    assert torch.allclose(prepare_conditioning(t, cond, 4), t + cond)
    assert torch.allclose(prepare_conditioning(t, None, 4), t)
    assert torch.allclose(prepare_conditioning(None, cond, 4), cond)
    with pytest.raises(ValueError, match="timestep"):
        prepare_conditioning(None, None, 4)


def test_prepare_conditioning_requires_correct_shape():
    assert prepare_conditioning(None, torch.randn(2, 4), 4).shape == (2, 4)
    assert prepare_conditioning(None, torch.randn(4), 4).shape == (1, 4)  # bare vector gets a batch dim
    for bad in (torch.randn(2, 3), torch.randn(2, 4, 1), torch.randn(2, 4, 3)):
        with pytest.raises(ValueError, match=r"\[B, 4\]"):
            prepare_conditioning(None, bad, 4)


def test_adaln_zero_init_is_identity_modulation():
    ada = AdaLN(4, 8, groups=2)
    scale, shift = ada(torch.randn(2, 4))
    assert torch.allclose(scale, torch.zeros_like(scale))
    assert torch.allclose(shift, torch.zeros_like(shift))


def test_transformer_block_forward():
    block = TransformerBlock(8, cond_dim=4, heads=2)
    assert block(torch.randn(2, 5, 8), torch.randn(2, 4)).shape == (2, 5, 8)


def test_convnext_block_forward():
    block = ConvNeXtBlock1d(8, cond_dim=4, kernel_size=3)
    assert block(torch.randn(2, 8, 16), torch.randn(2, 4)).shape == (2, 8, 16)


def test_stft_round_trip():
    cfg = STFTConfig(n_fft=64, hop_length=16, win_length=64)
    x = torch.randn(2, 1, 1024)
    recon = stft_to_waveform(waveform_to_stft(x, cfg), cfg, length=1024)
    assert recon.shape == x.shape
    assert torch.allclose(recon[..., 64:-64], x[..., 64:-64], atol=1e-4)


def test_complex_channel_layout_round_trip():
    spec = torch.randn(2, 2, 5, 7, dtype=torch.complex64)
    channelized = complex_to_channels(spec)
    assert channelized.shape == (2, 2 * 2 * 5, 7)
    assert torch.allclose(channelized[:, :10].reshape(2, 2, 5, 7), spec.real)
    assert torch.allclose(channels_to_complex(channelized, channels=2, freq_bins=5), spec)


def test_time_embedding_time_scale_changes_scalar_embeddings():
    base = TimeEmbedding(8, features=8, time_scale=1.0)
    scaled = TimeEmbedding(8, features=8, time_scale=100.0)
    scaled.load_state_dict(base.state_dict())
    t = torch.tensor([0.25, 0.5])
    assert not torch.allclose(base(t), scaled(t))


def test_wavenext_head_structure_and_raw_length():
    cfg = load_backbone_config("wavenext")
    head = build_backbone(cfg).branches[0].head
    assert isinstance(head, WaveNeXtHead)
    assert head.proj_fft.out_features == int(cfg.stft.n_fft)
    assert head.proj_hop.in_features == int(cfg.stft.n_fft)
    assert head.proj_hop.out_features == int(cfg.stft.hop_length)
    assert head.proj_hop.bias is None


@pytest.mark.parametrize("parameterisation", ["magphase", "realimag"])
def test_istft_head_parameterisations_preserve_shape(parameterisation):
    stft = STFTConfig(n_fft=14, hop_length=4, win_length=14)
    head = IStftHead(8, 1, stft, parameterisation=parameterisation)
    assert head(torch.randn(2, 8, 16), length=48).shape == (2, 1, 48)


def test_istft_magphase_bias_interpretation():
    stft = STFTConfig(n_fft=14, hop_length=4, win_length=14)
    head = IStftHead(8, 1, stft, parameterisation="magphase")
    torch.nn.init.zeros_(head.proj.weight)
    with torch.no_grad():
        head.proj.bias[:8].fill_(torch.log(torch.tensor(2.0)))
        head.proj.bias[8:].fill_(torch.pi / 2)
    spec = head.spec(torch.zeros(1, 8, 3))
    assert torch.allclose(spec.real, torch.zeros_like(spec.real), atol=1e-5)
    assert torch.allclose(spec.imag, torch.full_like(spec.imag, 2.0), atol=1e-5)


def test_model_stats_helpers():
    model = torch.nn.Linear(3, 2)
    assert count_params(model) == 8
    assert format_param_count(999) == "999"
    assert format_param_count(1_500) == "1.50K"
    assert format_param_count(2_000_000) == "2.00M"


def test_benchmark_backbone():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "benchmark_backbone.py"),
            "configs/backbone/wavenext.yaml",
            "--length",
            str(LENGTH),
            "--warmup",
            "1",
            "--iters",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "params:" in result.stdout
    assert "xRT:" in result.stdout
