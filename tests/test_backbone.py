from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from backbone.audio_ops import STFTConfig, channels_to_complex, complex_to_channels
from backbone.blocks import CRASHDBlock1d, MRFBlock, SnakeBeta
from backbone.complex_blocks import ComplexLayerNorm, ComplexPointwiseConv1d, SplitGELU
from backbone.conditioning import TimeEmbedding, combine_time_conditioning
from backbone.convnext_blocks import BiasNorm1d
from backbone.convnexts import ComplexIStftHead, IStftHead
from backbone.factory import build_backbone, load_backbone_config
from backbone.transformer_blocks import TransformerBlock
from backbone.unets import UNet, Upsampler
from scripts.model_stats import count_params, format_param_count

ROOT = Path(__file__).resolve().parents[1]
MAIN_CONFIGS = [
    "hifigan_v1",
    "hifigan_v3",
    "crash",
    "wavefm",
    "vocos",
    "rfwave",
    "flow2gan",
    "comvo",
    "wavenext",
    "stft_transformer",
    "waveform_transformer",
]


def _cond(cfg, batch_size: int):
    conditioning = cfg.get("conditioning", {})
    if conditioning.get("mode", "none") == "none":
        return None
    return torch.randn(batch_size, int(conditioning.get("cond_dim", 4)))


def _input(cfg, batch_size: int = 2, length: int = 64):
    io = cfg.get("io", {})
    channels = int(io.get("channels", 1))
    if io.get("type") == "stft":
        return torch.randn(batch_size, channels, int(io.get("freq_bins", 8)), 16, dtype=torch.complex64)
    if io.get("input_projection") == "stft":
        channels = int(io.get("input_channels", 1))
    length = _waveform_length(cfg, length)
    return torch.randn(batch_size, channels, length)


def _waveform_length(cfg, length: int) -> int:
    io = cfg.get("io", {})
    length = max(length, int(io.get("length", length)))
    if "stft" in io:
        length = max(length, int(io["stft"].get("n_fft", 0)) + 1)
    if "stft" in cfg:
        length = max(length, int(cfg["stft"].get("n_fft", 0)) + 1)
    if cfg.get("patching", {}).get("type") == "1d":
        length = max(length, int(cfg["patching"].get("patch_size", 0)))
    for resolution in cfg.get("branches", {}).get("resolutions", []):
        length = max(length, int(resolution.get("n_fft", 0)) + 1)
    return length


@pytest.mark.parametrize("name", MAIN_CONFIGS)
def test_load_and_instantiate_main_configs(name):
    model = build_backbone(name)
    assert isinstance(model, torch.nn.Module)


@pytest.mark.parametrize("name", MAIN_CONFIGS)
def test_main_configs_forward(name):
    cfg = load_backbone_config(ROOT / "configs" / "backbone" / f"{name}.yaml")
    model = build_backbone(cfg).eval()
    x = _input(cfg)
    cond = _cond(cfg, x.shape[0])
    target_length = int(cfg.get("io", {}).get("length", x.shape[-1])) if cfg["_target_"].endswith("Upsampler") else x.shape[-1]
    with torch.inference_mode():
        y = model(x, cond=cond, length=target_length)
    expected_channels = int(cfg.get("io", {}).get("out_channels", cfg.get("io", {}).get("channels", 1)))
    if cfg.get("io", {}).get("type") == "stft":
        assert torch.is_complex(y)
        assert y.shape == (x.shape[0], expected_channels, int(cfg["io"]["freq_bins"]), x.shape[-1])
    else:
        assert y.shape == (x.shape[0], expected_channels, target_length)


def test_factory_accepts_path_and_dict_config():
    path_model = build_backbone("configs/backbone/flow2gan.yaml")
    cfg = OmegaConf.load(ROOT / "configs" / "backbone" / "flow2gan.yaml")
    cfg_model = build_backbone(cfg)
    assert type(path_model) is type(cfg_model)


def test_complex_stft_channel_layout_round_trip_and_freq_validation():
    spec = torch.randn(2, 2, 5, 7, dtype=torch.complex64)
    channelized = complex_to_channels(spec)
    assert torch.allclose(channelized[:, :10].reshape(2, 2, 5, 7), spec.real)
    assert torch.allclose(channelized[:, 10:].reshape(2, 2, 5, 7), spec.imag)
    assert torch.allclose(channels_to_complex(channelized, channels=2, freq_bins=5), spec)
    with pytest.raises(ValueError, match="freq_bins"):
        STFTConfig.from_dict({"n_fft": 14, "freq_bins": 9})


def test_unet_symmetric_and_asymmetric_lengths():
    symmetric = UNet(
        io={"channels": 1, "out_channels": 1},
        encoder={"channels": [4, 8], "down_factors": [2]},
        decoder={"up_factors": [2], "skip": "concat"},
        block={"depth": 1},
    )
    asymmetric = UNet(
        io={"channels": 1, "out_channels": 1},
        encoder={"channels": [4, 8, 12], "down_factors": [2, 3]},
        decoder={"up_factors": [3, 2], "skip": "none"},
        block={"depth": 1},
    )
    x = torch.randn(1, 1, 65)
    assert symmetric(x, length=65).shape[-1] == 65
    assert asymmetric(x, length=65).shape[-1] == 65


def test_upsampler_length_contract():
    model = Upsampler(
        io={"channels": 3, "out_channels": 1, "length": 80},
        decoder={"channels": [8, 4], "up_factors": [2]},
        block={"depth": 1},
    )
    y = model(torch.randn(2, 3, 24))
    assert y.shape == (2, 1, 80)


@pytest.mark.parametrize(
    ("branches", "backend", "channels"),
    [
        ({"type": "single"}, {"type": "real"}, 1),
        ({"type": "subband", "count": 2}, {"type": "real"}, 1),
        ({"type": "multi", "count": 2}, {"type": "complex"}, 2),
    ],
)
def test_convnext_branch_and_backend_modes(branches, backend, channels):
    model = build_backbone(
        {
            "_target_": "backbone.convnexts.ConvNeXt",
            "io": {"channels": channels, "out_channels": channels},
            "branches": branches,
            "backend": backend,
            "block": {"channels": 8, "depth": 1},
            "conditioning": {"mode": "add", "cond_dim": 4},
        }
    )
    y = model(torch.randn(2, channels, 48), cond=torch.randn(2, 4), length=48)
    assert y.shape == (2, channels, 48)


@pytest.mark.parametrize("mode", ["none", "add", "film", "adaln", "context_tokens"])
def test_conditioning_modes(mode):
    model = build_backbone(
        {
            "_target_": "backbone.convnexts.ConvNeXt",
            "io": {"channels": 1, "out_channels": 1},
            "branches": {"type": "single"},
            "backend": {"type": "real"},
            "block": {"channels": 8, "depth": 1},
            "conditioning": {"mode": mode, "cond_dim": 4},
        }
    )
    cond = None if mode == "none" else torch.randn(2, 4)
    assert model(torch.randn(2, 1, 32), cond=cond).shape == (2, 1, 32)


def test_conditioning_global_only_contract():
    model = build_backbone(
        {
            "_target_": "backbone.convnexts.ConvNeXt",
            "io": {"channels": 1, "out_channels": 1},
            "branches": {"mode": "single"},
            "backend": {"type": "real"},
            "block": {"channels": 8, "depth": 1},
            "conditioning": {"mode": "add", "cond_dim": 4},
        }
    )
    x = torch.randn(2, 1, 32)
    assert model(x, cond=torch.randn(2), length=32).shape == (2, 1, 32)
    assert model(x, cond=torch.randn(2, 4), length=32).shape == (2, 1, 32)
    assert model(x, cond=torch.randn(2, 4, 1), length=32).shape == (2, 1, 32)
    with pytest.raises(ValueError, match="global"):
        model(x, cond=torch.randn(2, 4, 3), length=32)

    pooled = build_backbone(
        {
            "_target_": "backbone.convnexts.ConvNeXt",
            "io": {"channels": 1, "out_channels": 1},
            "branches": {"mode": "single"},
            "backend": {"type": "real"},
            "block": {"channels": 8, "depth": 1},
            "conditioning": {"mode": "add", "cond_dim": 4, "pool": "mean"},
        }
    )
    assert pooled(x, cond=torch.randn(2, 4, 3), length=32).shape == (2, 1, 32)


def test_time_conditioning_combines_with_explicit_conditioning():
    t = torch.randn(2, 4)
    cond = torch.randn(2, 4)
    assert torch.allclose(combine_time_conditioning(t, cond, "add"), t + cond)
    with pytest.raises(ValueError, match="no configured conditioning"):
        combine_time_conditioning(t, cond, "none")


@pytest.mark.parametrize(
    "branches",
    [
        {"mode": "single"},
        {"mode": "subband", "count": 2},
        {"mode": "multi_resolution", "resolutions": [{"n_fft": 14, "hop_length": 4, "win_length": 14}, {"n_fft": 30, "hop_length": 8, "win_length": 30}]},
    ],
)
def test_convnext_stft_branch_modes(branches):
    model = build_backbone(
        {
            "_target_": "backbone.convnexts.ConvNeXt",
            "io": {"channels": 1, "out_channels": 1},
            "branches": branches,
            "stft": {"n_fft": 14, "hop_length": 4, "win_length": 14},
            "head": {"type": "istft"},
            "backend": {"type": "real"},
            "block": {"channels": 8, "depth": 1},
            "conditioning": {"mode": "none"},
        }
    )
    assert model(torch.randn(2, 1, 48), length=48).shape == (2, 1, 48)


def test_convnext_multi_resolution_requires_explicit_resolutions():
    with pytest.raises(Exception, match="branches.resolutions"):
        build_backbone(
            {
                "_target_": "backbone.convnexts.ConvNeXt",
                "io": {"channels": 1, "out_channels": 1},
                "branches": {"mode": "multi_resolution", "count": 2},
                "stft": {"n_fft": 14, "hop_length": 4, "win_length": 14},
                "head": {"type": "istft"},
                "backend": {"type": "real"},
                "block": {"channels": 8, "depth": 1},
                "conditioning": {"mode": "none"},
            }
        )


def test_wavenext_head_structure_and_raw_length():
    cfg = load_backbone_config("configs/backbone/wavenext.yaml")
    model = build_backbone(cfg)
    width = model.branch.head.proj_fft.in_features
    n_fft = int(cfg.stft.n_fft)
    hop_length = int(cfg.stft.hop_length)
    assert width == int(cfg.block.channels)
    assert model.branch.head.proj_fft.out_features == n_fft
    assert model.branch.head.proj_hop.in_features == n_fft
    assert model.branch.head.proj_hop.out_features == hop_length
    assert model.branch.head.proj_hop.bias is None
    h = torch.randn(2, width, 5)
    assert model.branch.head.raw(h).shape == (2, int(cfg.io.out_channels), 5 * hop_length)
    length = n_fft + 1
    assert model(torch.randn(2, int(cfg.io.channels), length), length=length).shape == (2, int(cfg.io.out_channels), length)


def test_transformer_waveform_and_stft_paths():
    wave_cfg = load_backbone_config("configs/backbone/waveform_transformer.yaml")
    stft_cfg = load_backbone_config("configs/backbone/stft_transformer.yaml")
    wave = build_backbone(wave_cfg)
    stft = build_backbone(stft_cfg)
    wave_length = int(wave_cfg.patching.patch_size)
    assert wave(
        torch.randn(1, int(wave_cfg.io.channels), wave_length),
        cond=torch.randn(1, int(wave_cfg.conditioning.cond_dim)),
        length=wave_length,
    ).shape == (1, int(wave_cfg.io.out_channels), wave_length)
    spec = torch.randn(1, int(stft_cfg.io.channels), int(stft_cfg.io.freq_bins), 16, dtype=torch.complex64)
    y = stft(spec, cond=torch.randn(1, int(stft_cfg.conditioning.cond_dim)), length=40)
    assert torch.is_complex(y)
    assert y.shape == spec.shape
    with pytest.raises(ValueError, match="Real STFT"):
        stft(
            torch.randn(1, int(stft_cfg.io.channels), int(stft_cfg.io.freq_bins), 16),
            cond=torch.randn(1, int(stft_cfg.conditioning.cond_dim)),
            length=40,
        )


def test_transformer_rectangular_stft_patching_preserves_shape():
    model = build_backbone(
        {
            "_target_": "backbone.transformers.Transformer",
            "io": {"type": "stft", "channels": 2, "out_channels": 2, "n_fft": 18, "hop_length": 4, "win_length": 18, "freq_bins": 10},
            "patching": {"type": "2d", "patch_shape": [4, 3]},
            "block": {"dim": 12, "depth": 1, "heads": 3},
            "conditioning": {"mode": "context_tokens", "cond_dim": 4},
        }
    )
    assert model.in_proj.kernel_size == (4, 3)
    assert model.out_proj.stride == (4, 3)
    spec = torch.randn(1, 2, 10, 17, dtype=torch.complex64)
    y = model(spec, cond=torch.randn(1, 4), length=64)
    assert torch.is_complex(y)
    assert y.shape == spec.shape


def test_mrf_topology_and_snakebeta_log_parameters():
    snake = SnakeBeta(4)
    assert hasattr(snake, "log_alpha")
    assert hasattr(snake, "log_beta")
    assert snake(torch.randn(2, 4, 8)).shape == (2, 4, 8)

    block = MRFBlock(4, kernel_size=[3, 5, 7], dilations=[[1, 2], [2, 4], [3, 6]], sublayer_depth=2, convs_per_sublayer=2)
    assert len(block.branches) == 3
    assert all(len(branch.sublayers) == 2 for branch in block.branches)
    convs = [m for m in block.modules() if isinstance(m, torch.nn.Conv1d)]
    assert len(convs) == 12
    assert block(torch.randn(2, 4, 16)).shape == (2, 4, 16)


def test_context_token_conditioning_is_token_only():
    cfg = load_backbone_config("configs/backbone/stft_transformer.yaml")
    model = build_backbone(cfg)
    assert model.cond.mode == "none"
    tokens = []

    def capture(_module, inputs):
        tokens.append(inputs[0].shape[1])

    handle = model.blocks[0].register_forward_pre_hook(capture)
    try:
        model(
            torch.randn(1, int(cfg.io.channels), int(cfg.io.freq_bins), 16, dtype=torch.complex64),
            cond=torch.randn(1, int(cfg.conditioning.cond_dim)),
            length=40,
        )
    finally:
        handle.remove()
    assert tokens and tokens[0] > 1


def test_transformer_adaln_zero_is_per_block_and_time_is_not_silent():
    cfg = load_backbone_config("configs/backbone/waveform_transformer.yaml")
    model = build_backbone(cfg)
    assert all(isinstance(block, TransformerBlock) and block.ada is not None for block in model.blocks)
    length = int(cfg.patching.patch_size)
    assert model(torch.randn(1, int(cfg.io.channels), length), t=torch.tensor([0.5]), length=length).shape == (
        1,
        int(cfg.io.out_channels),
        length,
    )
    unconditioned = build_backbone(
        {
            "_target_": "backbone.transformers.Transformer",
            "io": {"type": "waveform", "channels": 1},
            "patching": {"type": "1d", "patch_size": 4},
            "block": {"dim": 8, "depth": 1, "heads": 2},
            "conditioning": {"mode": "none"},
        }
    )
    with pytest.raises(ValueError, match="no configured conditioning"):
        unconditioned(torch.randn(1, 1, 16), t=torch.tensor([0.5]))


def test_config_specific_macro_knobs_are_active():
    flow_cfg = load_backbone_config("configs/backbone/flow2gan.yaml")
    flow = build_backbone(flow_cfg)
    assert isinstance(flow.resolution_branches[0].trunk.blocks[0].norm, BiasNorm1d)
    assert any(isinstance(m, torch.nn.PReLU) for m in flow.resolution_branches[0].trunk.blocks[0].pointwise)
    assert [branch.stft.hop_length for branch in flow.resolution_branches] == [
        int(resolution.hop_length) for resolution in flow_cfg.branches.resolutions
    ]

    vocos_cfg = load_backbone_config("configs/backbone/vocos.yaml")
    vocos = build_backbone(vocos_cfg)
    assert vocos.branch.in_proj.kernel_size == (int(vocos_cfg.block.get("input_kernel_size", vocos_cfg.block.kernel_size)),)
    assert vocos.time_embed is not None

    wavenext = build_backbone("configs/backbone/wavenext.yaml")
    assert wavenext.time_embed is not None

    hifigan_cfg = load_backbone_config("configs/backbone/hifigan_v1.yaml")
    hifigan = build_backbone(hifigan_cfg)
    boundary_kernel = int(hifigan_cfg.block.boundary_kernel_size)
    assert hifigan.in_proj.kernel_size == (boundary_kernel,)
    assert hifigan.out_proj.kernel_size == (boundary_kernel,)
    assert hasattr(hifigan.in_proj, "weight_g")

    assert hifigan.frame_proj is not None
    assert hifigan.frame_proj.in_channels == 2 * int(hifigan_cfg.io.input_channels) * hifigan.stft.freq_bins
    assert hifigan.in_proj.in_channels == int(hifigan_cfg.io.channels)
    assert hifigan.time_embed is not None

    wavefm_cfg = load_backbone_config("configs/backbone/wavefm.yaml")
    wavefm = build_backbone(wavefm_cfg)
    assert wavefm.time_embed.time_scale == float(wavefm_cfg.conditioning.time_scale)
    assert wavefm.in_proj.in_channels == int(wavefm_cfg.io.channels)
    assert wavefm.in_proj.kernel_size == (int(wavefm_cfg.block.boundary_kernel_size),)
    assert isinstance(wavefm.mid, torch.nn.Identity)
    assert isinstance(wavefm.final_activation, SnakeBeta)
    cond = torch.randn(1, int(wavefm_cfg.conditioning.cond_dim))
    assert wavefm(torch.randn(1, int(wavefm_cfg.io.channels), 64), cond=cond, length=64).amin() >= -1
    assert wavefm(torch.randn(1, int(wavefm_cfg.io.channels), 64), cond=cond, length=64).amax() <= 1

    assert len(wavefm.dec_blocks[0]) == int(wavefm_cfg.block.decoder_depth) + 1
    assert isinstance(wavefm.dec_blocks[0][1], MRFBlock)
    assert len(wavefm.dec_blocks[0][1].branches) == len(wavefm_cfg.block.decoder_kernel_sizes)
    assert wavefm.enc_blocks[0][0].net[2].kernel_size == (int(wavefm_cfg.block.kernel_size),)
    assert wavefm.enc_blocks[0][0].net[2].in_channels == int(wavefm_cfg.encoder.channels[1])
    assert hasattr(wavefm.in_proj, "weight_g")
    assert count_params(wavefm) < int(wavefm_cfg.get("max_params", 25_000_000))

    crash_cfg = load_backbone_config("configs/backbone/crash.yaml")
    crash = build_backbone(crash_cfg)
    assert isinstance(crash.enc_blocks[0][0], CRASHDBlock1d)
    assert crash.downs[0].conv.kernel_size == (int(crash_cfg.encoder.down_kernel_size),)
    assert len(crash.enc_blocks) == len(crash_cfg.encoder.channels)
    assert len(crash.downs) == len(crash_cfg.encoder.down_factors)
    total_downsample = 1
    for down in crash.downs:
        total_downsample *= down.factor
    expected_downsample = 1
    for factor in crash_cfg.encoder.down_factors:
        expected_downsample *= int(factor)
    assert total_downsample == expected_downsample
    expected_enc_channels = [
        int(crash_cfg.encoder.channels[min(i + 1, len(crash_cfg.encoder.channels) - 1)])
        for i in range(len(crash_cfg.encoder.down_factors))
    ]
    assert [blocks[0].net[1].in_channels for blocks in crash.enc_blocks] == expected_enc_channels
    assert isinstance(crash.mid, torch.nn.Identity)


def test_subband_wavenext_rejected_and_rfwave_overlap_conditioning():
    with pytest.raises(Exception, match="subband.*wavenext"):
        build_backbone(
            {
                "_target_": "backbone.convnexts.ConvNeXt",
                "io": {"channels": 1, "out_channels": 1},
                "branches": {"mode": "subband", "count": 2},
                "stft": {"n_fft": 14, "hop_length": 4, "win_length": 14},
                "head": {"type": "wavenext"},
                "block": {"channels": 8, "depth": 1},
                "conditioning": {"mode": "none"},
            }
        )

    model = build_backbone("configs/backbone/rfwave.yaml")
    cfg = load_backbone_config("configs/backbone/rfwave.yaml")
    assert model.subband_embed is not None
    assert len(model.subband_slices) == int(cfg.branches.count)
    assert not isinstance(model.subband_trunk, torch.nn.ModuleList)
    assert any((end - start) > (core_end - core_start) for start, end, core_start, core_end in model.subband_slices)
    spec = torch.randn(2, int(cfg.io.channels), int(cfg.io.freq_bins), 12, dtype=torch.complex64)
    y, aux = model(spec, return_aux=True)
    assert y.shape == spec.shape
    assert len(aux["subbands"]) == int(cfg.branches.count)


def test_istft_magphase_and_layer_scale_depth():
    head = IStftHead(8, 1, STFTConfig(n_fft=14, hop_length=4, win_length=14), parameterisation="magphase")
    assert IStftHead(8, 1, STFTConfig(n_fft=14, hop_length=4, win_length=14)).parameterisation == "magphase"
    assert head(torch.randn(2, 8, 16), length=48).shape == (2, 1, 48)
    torch.nn.init.zeros_(head.proj.weight)
    with torch.no_grad():
        head.proj.bias[:8].fill_(torch.log(torch.tensor(2.0)))
        head.proj.bias[8:].fill_(torch.pi / 2)
    spec = head.spec(torch.zeros(1, 8, 3))
    assert torch.allclose(spec.real, torch.zeros_like(spec.real), atol=1e-5)
    assert torch.allclose(spec.imag, torch.full_like(spec.imag, 2.0), atol=1e-5)

    cfg = load_backbone_config("configs/backbone/vocos.yaml")
    model = build_backbone(cfg)
    scale = model.branch.trunk.blocks[0].layer_scale
    assert scale is not None
    assert torch.allclose(scale, torch.full_like(scale, 1 / int(cfg.block.depth)))


def test_time_embedding_time_scale_changes_scalar_embeddings():
    base = TimeEmbedding(8, features=8, time_scale=1.0)
    scaled = TimeEmbedding(8, features=8, time_scale=100.0)
    scaled.load_state_dict(base.state_dict())
    t = torch.tensor([0.25, 0.5])
    assert not torch.allclose(base(t), scaled(t))


def test_complex_convnext_backend_and_phase_quantization_are_wired():
    cfg = load_backbone_config("configs/backbone/comvo.yaml")
    model = build_backbone(cfg)
    block = model.branch.trunk.blocks[0]
    assert isinstance(block.norm, ComplexLayerNorm)
    assert any(isinstance(m, ComplexPointwiseConv1d) for m in block.pointwise)
    assert any(isinstance(m, SplitGELU) for m in block.pointwise)
    assert isinstance(model.branch.head, ComplexIStftHead)
    width = int(cfg.block.channels)
    x = torch.randn(2, width, 5)
    q = model._quantize_latent_phase(x)
    phase = torch.angle(torch.complex(*q.chunk(2, dim=1)))
    step = 2 * torch.pi / model.phase_bins
    assert torch.allclose(phase / step, torch.round(phase / step), atol=1e-5)

    grad_input = torch.randn(2, width, 5, requires_grad=True)
    model._quantize_latent_phase(grad_input).sum().backward()
    assert grad_input.grad is not None
    assert grad_input.grad.abs().sum() > 0

    assert model.branch.trunk.blocks[0].layer_scale.shape[1] == width
    assert isinstance(model.branch.head, ComplexIStftHead)
    assert model.cond_mode == cfg.conditioning.mode
    assert model.time_embed is not None


def test_complex_convnext_backend_accepts_stereo_stft_layout():
    model = build_backbone(
        {
            "_target_": "backbone.convnexts.ConvNeXt",
            "io": {"type": "stft", "channels": 2, "out_channels": 2, "n_fft": 14, "hop_length": 4, "win_length": 14, "freq_bins": 8},
            "branches": {"mode": "single"},
            "stft": {"n_fft": 14, "hop_length": 4, "win_length": 14},
            "head": {"type": "complex_istft"},
            "backend": {"type": "complex"},
            "block": {"channels": 8, "depth": 1},
            "conditioning": {"mode": "none"},
        }
    )
    spec = torch.randn(2, 2, 8, 9, dtype=torch.complex64)
    y = model(spec)
    assert torch.is_complex(y)
    assert y.shape == spec.shape


def test_waveform_transformer_head_preserves_shape():
    cfg = load_backbone_config("configs/backbone/waveform_transformer.yaml")
    model = build_backbone(cfg)
    assert model.head.__class__.__name__ == "ConvHead"
    length = int(cfg.patching.patch_size)
    assert model(torch.randn(2, int(cfg.io.channels), length), t=torch.tensor([0.1, 0.2]), length=length).shape == (
        2,
        int(cfg.io.out_channels),
        length,
    )


def test_model_stats_helpers():
    model = torch.nn.Linear(3, 2)
    assert count_params(model) == 8
    assert count_params(model, trainable_only=True) == 8
    assert format_param_count(999) == "999"
    assert format_param_count(1_500) == "1.50K"
    assert format_param_count(2_000_000) == "2.00M"


def test_benchmark_backbone():
    cfg = load_backbone_config("configs/backbone/flow2gan.yaml")
    length = _waveform_length(cfg, 64)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "benchmark_backbone.py"),
            "configs/backbone/flow2gan.yaml",
            "--length",
            str(length),
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
