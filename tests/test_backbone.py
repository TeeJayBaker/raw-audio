from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from backbone.blocks import AdaLN, TransformerBlock
from backbone.conditioning import ConditioningCombiner, TimeEmbedding
from backbone.factory import build_backbone, load_backbone_config
from backbone.io import (
    STFTConfig,
    channels_to_complex,
    complex_to_channels,
    istft_channels,
    stft_channels,
    stft_to_waveform,
    waveform_to_stft,
)
from backbone.transformer import Transformer
from flow.fm import RectifiedFlow
from scripts.model_stats import count_params, format_param_count

ROOT = Path(__file__).resolve().parents[1]
TINY_STFT = {"n_fft": 64, "hop_length": 16, "win_length": 64}


def _tiny_transformer(patching=None, bottleneck=None, gap_embed=False, guidance_embed=False, interval_embed=False):
    return Transformer(
        channels=1,
        stft=TINY_STFT,
        patching=patching,
        bottleneck=bottleneck,
        block={"dim": 32, "depth": 1, "heads": 2},
        conditioning={
            "cond_dim": 16,
            "embed_dim": 16,
            "gap_embed": gap_embed,
            "guidance_embed": guidance_embed,
            "interval_embed": interval_embed,
        },
        sample_rate=8000,
    )


def _spec(model, batch=2, frames=11):
    return torch.randn(batch, 2 * model.channels * model.stft.freq_bins, frames)


# --- backbone forward contract: channelised spec in -> channelised spec out ---

@pytest.mark.parametrize(
    "patching,bottleneck",
    [
        ({"scheme": "column"}, None),
        ({"scheme": "square", "patch_f": 16, "patch_t": 4}, None),  # square -> npf>1 -> axial RoPE
        ({"scheme": "square", "patch_f": 16, "patch_t": 4}, 8),  # + input bottleneck
    ],
)
def test_forward_contract(patching, bottleneck):
    torch.manual_seed(0)
    model = Transformer(
        channels=1,
        stft=TINY_STFT,
        patching=patching,
        bottleneck=bottleneck,
        block={"dim": 32, "depth": 2, "heads": 2},
        conditioning={"cond_dim": 16, "embed_dim": 16},
        sample_rate=8000,
    ).eval()
    x = _spec(model, frames=19)
    with torch.inference_mode():
        y = model(x, t=torch.rand(2), cond=torch.randn(2, 16))
    assert y.shape == x.shape
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()


def test_stft_transformer_config_builds_and_runs():
    cfg = load_backbone_config("stft_transformer")
    model = build_backbone(cfg).eval()
    assert isinstance(model, Transformer)
    x = torch.randn(1, 2 * 1 * model.stft.freq_bins, 5)
    with torch.inference_mode():
        y = model(x, t=torch.rand(1), cond=torch.randn(1, int(cfg.conditioning.cond_dim)))
    assert y.shape == x.shape


def test_conditioning_alone_or_timestep_alone():
    model = _tiny_transformer().eval()
    x = _spec(model, batch=1)
    with torch.inference_mode():
        assert model(x, cond=torch.randn(1, 16)).shape == x.shape
        assert model(x, t=torch.rand(1)).shape == x.shape


def test_bottleneck_factorises_in_proj_only():
    model = _tiny_transformer(patching={"scheme": "square", "patch_f": 16, "patch_t": 4}, bottleneck=8)
    assert isinstance(model.in_proj, nn.Sequential)
    waist, lift = model.in_proj
    assert waist.bias is None and waist.out_channels == 8  # low-rank, biasless waist
    assert lift.out_channels == model.dim
    assert isinstance(model.out_proj, nn.Conv1d)  # output stays full-rank
    assert model.out_proj.out_channels == model.patcher.feat(model.out_channels)


def test_factory_accepts_path_and_dict_config():
    path_model = build_backbone("configs/backbone/stft_transformer.yaml")
    cfg_model = build_backbone(OmegaConf.load(ROOT / "configs" / "backbone" / "stft_transformer.yaml"))
    assert type(path_model) is type(cfg_model) is Transformer


# --- flow._predict: the single waveform<->spectrogram crossing -------------

def test_predict_round_trips_through_spectrogram():
    torch.manual_seed(0)
    flow = RectifiedFlow()
    model = _tiny_transformer().eval()  # aux_depth=0 -> no v-head
    x = torch.randn(2, 1, 256)
    t = torch.tensor([0.3, 0.7])
    cond = torch.randn(2, 16)
    with torch.inference_mode():
        pred, aux, wav, spec = flow._predict(model, x, t=t, cond=cond)
    assert wav.shape == (2, 1, 256)
    assert wav.dtype == torch.float32
    assert aux is None  # no v-head
    assert pred is wav  # waveform space: the flow-space prediction IS the waveform
    # spec is the backbone's channelised (pre-iSTFT) prediction; iSTFT reproduces the waveform.
    assert spec.shape == (2, 2 * model.out_channels * model.stft.freq_bins, spec.shape[-1])
    recon = istft_channels(spec, model.stft, out_channels=model.out_channels, length=x.shape[-1])
    assert torch.allclose(recon, wav, atol=1e-5)


# --- conditioning / blocks (unchanged behaviour) ---------------------------

def test_conditioning_combiner_normalises_each_path_then_sums():
    combiner = ConditioningCombiner(4)
    t = torch.randn(2, 4)
    cond = torch.randn(2, 4)
    assert torch.allclose(combiner(t, cond), combiner.time_norm(t) + combiner.cond_norm(cond))
    assert torch.allclose(combiner(t, None), combiner.time_norm(t))
    assert torch.allclose(combiner(None, cond), combiner.cond_norm(cond))
    with pytest.raises(ValueError, match="timestep"):
        combiner(None, None)


def test_conditioning_combiner_requires_correct_shape():
    combiner = ConditioningCombiner(4)
    assert combiner(None, torch.randn(2, 4)).shape == (2, 4)
    assert combiner(None, torch.randn(4)).shape == (1, 4)  # bare vector gets a batch dim
    for bad in (torch.randn(2, 3), torch.randn(2, 4, 1), torch.randn(2, 4, 3)):
        with pytest.raises(ValueError, match=r"\[B, 4\]"):
            combiner(None, bad)


def test_adaln_zero_init_is_identity_modulation():
    ada = AdaLN(4, 8, groups=2)
    scale, shift = ada(torch.randn(2, 4))
    assert torch.allclose(scale, torch.zeros_like(scale))
    assert torch.allclose(shift, torch.zeros_like(shift))


def test_transformer_block_forward():
    block = TransformerBlock(8, cond_dim=4, heads=2)
    assert block(torch.randn(2, 5, 8), torch.randn(2, 4)).shape == (2, 5, 8)


# --- STFT / channel-layout helpers -----------------------------------------

def test_stft_round_trip():
    cfg = STFTConfig(n_fft=64, hop_length=16, win_length=64)
    x = torch.randn(2, 1, 1024)
    recon = stft_to_waveform(waveform_to_stft(x, cfg), cfg, length=1024)
    assert recon.shape == x.shape
    assert torch.allclose(recon[..., 64:-64], x[..., 64:-64], atol=1e-4)


def test_stft_channels_round_trip():
    cfg = STFTConfig(n_fft=64, hop_length=16, win_length=64)
    x = torch.randn(2, 1, 1024)
    channels = stft_channels(x, cfg)
    assert channels.shape == (2, 2 * 1 * cfg.freq_bins, channels.shape[-1])
    assert channels.dtype == torch.float32
    recon = istft_channels(channels, cfg, out_channels=1, length=1024)
    assert recon.shape == (2, 1, 1024)
    assert recon.dtype == torch.float32
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
            "configs/backbone/stft_transformer.yaml",
            "--length",
            "4096",
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


# --- MeanFlow aux conditioning embeds (zero-init RF warm start) -------------

def test_aux_embeds_zero_init_are_exact_rf_warm_start():
    torch.manual_seed(0)
    rf = _tiny_transformer()
    mf = _tiny_transformer(gap_embed=True, guidance_embed=True, interval_embed=True)
    missing, unexpected = mf.load_state_dict(rf.state_dict(), strict=False)
    assert not unexpected
    assert all(
        any(tag in key for tag in ("gap_embed", "omega_embed", "lo_embed", "hi_embed"))
        for key in missing
    )

    x = _spec(rf)
    t = torch.tensor([0.3, 0.7])
    cond = torch.randn(2, 16)
    kw = {
        "h": torch.tensor([0.5, 0.2]),
        "omega": torch.tensor([3.0, 5.0]),
        "t_lo": torch.tensor([0.1, 0.2]),
        "t_hi": torch.tensor([0.8, 0.9]),
    }
    assert torch.allclose(mf(x, t=t, cond=cond, **kw), rf(x, t=t, cond=cond), atol=1e-6)
    assert torch.allclose(mf(x, t=t, cond=cond), rf(x, t=t, cond=cond), atol=1e-6)


def test_aux_inputs_without_embeds_raise():
    rf = _tiny_transformer()
    x = _spec(rf, batch=1)
    t = torch.tensor([0.5])
    cond = torch.randn(1, 16)
    with pytest.raises(ValueError, match="gap_embed"):
        rf(x, t=t, h=torch.tensor([0.5]), cond=cond)
    with pytest.raises(ValueError, match="guidance_embed"):
        rf(x, t=t, omega=torch.tensor([3.0]), cond=cond)
    with pytest.raises(ValueError, match="interval_embed"):
        rf(x, t=t, t_hi=torch.tensor([0.8]), cond=cond)


# --- MeanFlow twin x-pred heads (aux_depth) --------------------------------

def _twin(depth, aux_depth):
    return Transformer(
        channels=1,
        stft=TINY_STFT,
        block={"dim": 32, "depth": depth, "heads": 2, "aux_depth": aux_depth},
        conditioning={"cond_dim": 16, "embed_dim": 16, "gap_embed": True, "guidance_embed": True},
        sample_rate=8000,
    )


def test_aux_depth_zero_is_single_head_rf():
    model = _tiny_transformer()  # aux_depth defaults to 0
    assert model.aux_depth == 0 and model.v_blocks is None and model.v_out_proj is None
    assert not any("v_blocks" in k or "v_out_proj" in k for k in model.state_dict())
    x = _spec(model)
    out = model(x, t=torch.rand(2), cond=torch.randn(2, 16), return_aux=True)
    assert torch.is_tensor(out) and out.shape == x.shape  # no v-head -> single tensor even if asked


def test_aux_depth_returns_twin_heads():
    torch.manual_seed(0)
    model = _twin(depth=3, aux_depth=2)
    assert model.trunk_depth == 1 and len(model.v_blocks) == 2
    x = _spec(model)
    kw = {"t": torch.rand(2), "h": torch.rand(2), "cond": torch.randn(2, 16)}
    assert torch.is_tensor(model(x, **kw))  # return_aux default False -> u-head only
    u_spec, v_spec = model(x, return_aux=True, **kw)
    assert u_spec.shape == x.shape and v_spec.shape == x.shape
    assert not torch.allclose(u_spec, v_spec)


def test_v_head_warm_start_from_rf_checkpoint():
    # An RF checkpoint (aux_depth=0) loads into an aux_depth>0 model with only the v-head missing:
    # `blocks`/`out_proj` keep identical names, so the shared trunk + u-head load as-is.
    rf, mf = _twin(depth=2, aux_depth=0), _twin(depth=2, aux_depth=1)
    missing, unexpected = mf.load_state_dict(rf.state_dict(), strict=False)
    assert not unexpected
    assert missing and all(("v_blocks" in k or "v_out_proj" in k) for k in missing)
