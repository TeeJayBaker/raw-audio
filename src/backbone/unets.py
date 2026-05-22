from __future__ import annotations

import torch
from torch import nn

from backbone.audio_ops import STFTConfig, as_waveform, complex_to_channels, waveform_to_stft
from backbone.blocks import (
    CRASHDBlock1d,
    Downsample1d,
    MRFBlock,
    ResidualBlock1d,
    Upsample1d,
    _is_sequence,
    activation,
    center_crop_or_pad,
)
from backbone.conditioning import TimeEmbedding, combine_time_conditioning, conditioning_mode


def _residual_block(channels: int, block: dict, conditioning: dict | None, index: int = 0, prefix: str = "") -> nn.Module:
    block_type = block.get(f"{prefix}type", block.get("type", "dilated_residual"))
    kernel_size = block.get(f"{prefix}kernel_size", block.get(f"{prefix}kernel_sizes", block.get("kernel_size", block.get("kernel_sizes", 3))))
    dilations = block.get(f"{prefix}dilations", block.get("dilations"))
    if _is_sequence(dilations) and dilations and _is_sequence(dilations[0]):
        dilations = dilations[min(index, len(dilations) - 1)]
    if block_type == "mrf":
        return MRFBlock(
            channels,
            kernel_size=kernel_size,
            dilations=dilations or block.get("mrf_dilations", (1, 3, 5)),
            sublayer_depth=block.get(f"{prefix}sublayer_depth", block.get("sublayer_depth", 3)),
            convs_per_sublayer=block.get(f"{prefix}convs_per_sublayer", block.get("convs_per_sublayer", 2)),
            activation_name=block.get(f"{prefix}activation", block.get("activation", "snake_beta")),
            conditioning=conditioning,
        )
    if block_type == "crash_dblock":
        dilations = dilations or (1, 2, 4, 8)
        if not _is_sequence(dilations):
            raise ValueError("crash_dblock dilations must be a sequence")
        return CRASHDBlock1d(
            channels,
            kernel_size=int(kernel_size),
            dilations=list(dilations),
            activation_name=block.get(f"{prefix}activation", block.get("activation", "silu")),
            conditioning=conditioning,
        )
    dilation = block.get(f"{prefix}dilation", block.get("dilation", 1))
    if _is_sequence(dilations) and dilations:
        expected_depth = block.get(f"{prefix}depth", block.get("depth"))
        if expected_depth is not None and len(dilations) not in {1, int(expected_depth)}:
            raise ValueError("dilations length must be 1 or match block depth")
        dilation = dilations[min(index, len(dilations) - 1)]
    return ResidualBlock1d(
        channels,
        kernel_size,
        dilation=dilation,
        activation_name=block.get("activation", "silu"),
        conditioning=conditioning,
    )


class UNet(nn.Module):
    # WaveFM-style denoising belongs here; Upsampler is shaped for mel/frame-rate vocoding.
    def __init__(
        self,
        io: dict | None = None,
        encoder: dict | None = None,
        decoder: dict | None = None,
        block: dict | None = None,
        conditioning: dict | None = None,
        sample_rate: int = 48000,
        name: str | None = None,
    ):
        super().__init__()
        self.io = io or {"type": "waveform", "channels": 1}
        self.sample_rate = sample_rate
        self.name = name
        in_channels = self.io.get("channels", 1)
        out_channels = self.io.get("out_channels", in_channels)
        encoder = encoder or {}
        decoder = decoder or {}
        block = dict(block or {})
        channels = encoder.get("channels", [32, 64, 128])
        factors = encoder.get("down_factors", [2] * (len(channels) - 1))
        self.downsample_after_last = bool(encoder.get("downsample_after_last", False))
        expected_factors = len(channels) if self.downsample_after_last else len(channels) - 1
        if len(factors) != expected_factors:
            raise ValueError(
                f"encoder.down_factors length must be {expected_factors} when downsample_after_last={self.downsample_after_last}"
            )
        enc_depth = block.get("encoder_depth", block.get("depth", 1))
        dec_depth = block.get("decoder_depth", block.get("depth", 1))
        kernel_size = block.get("kernel_size", 3)
        activation_name = block.get("activation", "silu")
        weight_norm = bool(block.get("weight_norm", encoder.get("weight_norm", decoder.get("weight_norm", False))))
        self.skip = decoder.get("skip", "concat")
        self.tanh_output = bool(decoder.get("tanh_output", block.get("tanh_output", False)))
        self.cond_mode = conditioning_mode(conditioning)
        self.cond_dim = int((conditioning or {}).get("cond_dim", channels[-1]))
        self.time_embed = (
            TimeEmbedding(self.cond_dim, (conditioning or {}).get("time_hidden_dim"), time_scale=(conditioning or {}).get("time_scale", 1.0))
            if self.cond_mode != "none"
            else None
        )
        self.conditioning_placement = block.get("conditioning_placement", "all")
        in_kernel = int(block.get("input_kernel_size", block.get("boundary_kernel_size", 3)))
        out_kernel = int(block.get("output_kernel_size", block.get("boundary_kernel_size", 3)))

        self.in_proj = nn.Conv1d(in_channels, channels[0], in_kernel, padding=in_kernel // 2)
        if weight_norm:
            self.in_proj = nn.utils.weight_norm(self.in_proj)
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        enc_block_channels = []
        for i, factor in enumerate(factors):
            in_ch = channels[min(i, len(channels) - 1)]
            out_ch = channels[min(i + 1, len(channels) - 1)]
            self.downs.append(
                Downsample1d(
                    in_ch,
                    out_ch,
                    factor,
                    kernel_size=encoder.get("down_kernel_size"),
                    post_conv1x1=bool(encoder.get("post_downsample_1x1", False)),
                )
            )
            enc_block_channels.append(out_ch)
        for ch in enc_block_channels:
            enc_conditioning = conditioning if self.conditioning_placement in {"all", "encoder", "encoder_only"} else None
            self.enc_blocks.append(
                nn.ModuleList(
                    [
                        _residual_block(ch, block | {"kernel_size": kernel_size, "activation": activation_name, "encoder_depth": enc_depth}, enc_conditioning, j, "encoder_")
                        for j in range(enc_depth)
                    ]
                )
            )

        self.mid = (
            nn.Identity()
            if not bool(block.get("mid_block", True))
            else _residual_block(channels[-1], block | {"kernel_size": kernel_size, "activation": activation_name}, conditioning)
        )
        up_factors = decoder.get("up_factors", list(reversed(factors)))
        if len(up_factors) != expected_factors:
            raise ValueError(
                f"decoder.up_factors length must be {expected_factors} when downsample_after_last={self.downsample_after_last}"
            )
        up_out_channels = list(reversed(channels if self.downsample_after_last else channels[:-1]))
        up_in_channels = [channels[-1], *up_out_channels[:-1]]
        self.decoder_uses_skip = ([True] * (len(up_out_channels) - 1) + [False]) if self.skip != "none" else [False] * len(up_out_channels)
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for factor, in_ch, out_ch, uses_skip in zip(up_factors, up_in_channels, up_out_channels, self.decoder_uses_skip, strict=True):
            self.ups.append(Upsample1d(in_ch, out_ch, factor, mode=decoder.get("upsample", "interp"), weight_norm=weight_norm))
            block_ch = out_ch * 2 if uses_skip and self.skip == "concat" else out_ch
            dec_conditioning = conditioning if self.conditioning_placement == "all" else None
            self.dec_blocks.append(
                nn.ModuleList(
                    [
                        nn.Conv1d(block_ch, out_ch, 1) if j == 0 and block_ch != out_ch else nn.Identity()
                        for j in range(1)
                    ]
                    + [
                        _residual_block(out_ch, block | {"kernel_size": kernel_size, "activation": activation_name, "decoder_depth": dec_depth}, dec_conditioning, j, "decoder_")
                        for j in range(dec_depth)
                    ]
                )
            )
        self.final_activation = activation(block["final_activation"], channels[0]) if block.get("final_activation") else nn.Identity()
        self.out_proj = nn.Conv1d(channels[0], out_channels, out_kernel, padding=out_kernel // 2)
        if weight_norm:
            self.out_proj = nn.utils.weight_norm(self.out_proj)
            for module in self.modules():
                if isinstance(module, nn.Conv1d | nn.ConvTranspose1d) and not hasattr(module, "weight_g"):
                    nn.utils.weight_norm(module)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None, cond: torch.Tensor | None = None, length: int | None = None) -> torch.Tensor:
        if t is not None and self.time_embed is not None:
            t = self.time_embed(t)
        cond = combine_time_conditioning(t, cond, self.cond_mode)
        x = as_waveform(x)
        target_length = length or x.shape[-1]
        h = self.in_proj(x)
        skips = []
        for down, blocks in zip(self.downs, self.enc_blocks, strict=True):
            h = down(h)
            for block in blocks:
                h = block(h, cond)
            skips.append(h)
        h = self.mid(h, cond) if not isinstance(self.mid, nn.Identity) else self.mid(h)
        decoder_skips = list(reversed(skips[:-1]))
        skip_idx = 0
        for up, blocks, uses_skip in zip(self.ups, self.dec_blocks, self.decoder_uses_skip, strict=True):
            h = up(h)
            if uses_skip:
                skip = decoder_skips[skip_idx]
                skip_idx += 1
                h = center_crop_or_pad(h, skip.shape[-1])
                if self.skip == "add":
                    h = h + skip
                elif self.skip == "none":
                    pass
                else:
                    h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, cond) if isinstance(block, ResidualBlock1d | MRFBlock | CRASHDBlock1d) else block(h)
        y = center_crop_or_pad(self.out_proj(self.final_activation(h)), target_length)
        return torch.tanh(y) if self.tanh_output else y


class Upsampler(nn.Module):
    # Mel/frame-rate vocoder-shaped decoder; use UNet for WaveFM-style denoising.
    def __init__(
        self,
        io: dict | None = None,
        decoder: dict | None = None,
        block: dict | None = None,
        conditioning: dict | None = None,
        sample_rate: int = 48000,
        name: str | None = None,
    ):
        super().__init__()
        self.io = io or {"channels": 1, "out_channels": 1}
        self.sample_rate = sample_rate
        self.name = name
        decoder = decoder or {}
        block = dict(block or {})
        channels = decoder.get("channels", [64, 32, 16])
        factors = decoder.get("up_factors", [2] * (len(channels) - 1))
        depth = block.get("depth", 1)
        weight_norm = bool(block.get("weight_norm", decoder.get("weight_norm", False)))
        self.tanh_output = bool(decoder.get("tanh_output", block.get("tanh_output", False)))
        self.default_length = self.io.get("length")
        self.internal_channels = int(self.io.get("channels", 1))
        self.input_channels = int(self.io.get("input_channels", self.internal_channels))
        self.input_projection = self.io.get("input_projection")
        if self.input_projection == "stft":
            self.stft = STFTConfig.from_dict(self.io.get("stft", self.io))
            self.frame_proj = nn.Conv1d(2 * self.input_channels * self.stft.freq_bins, self.internal_channels, 1)
        else:
            self.stft = None
            self.frame_proj = None
        in_kernel = int(block.get("input_kernel_size", block.get("boundary_kernel_size", 3)))
        out_kernel = int(block.get("output_kernel_size", block.get("boundary_kernel_size", 3)))
        self.in_proj = nn.Conv1d(self.internal_channels, channels[0], in_kernel, padding=in_kernel // 2)
        if weight_norm:
            if self.frame_proj is not None:
                self.frame_proj = nn.utils.weight_norm(self.frame_proj)
            self.in_proj = nn.utils.weight_norm(self.in_proj)
        layers: list[nn.Module] = []
        for i, factor in enumerate(factors):
            layers.append(Upsample1d(channels[i], channels[i + 1], factor, mode=decoder.get("upsample", "transpose"), weight_norm=weight_norm))
            for j in range(depth):
                layers.append(_residual_block(channels[i + 1], block, conditioning, j))
        self.layers = nn.ModuleList(layers)
        self.out_proj = nn.Conv1d(channels[-1], self.io.get("out_channels", 1), out_kernel, padding=out_kernel // 2)
        if weight_norm:
            self.out_proj = nn.utils.weight_norm(self.out_proj)
            for module in self.modules():
                if isinstance(module, MRFBlock):
                    for conv in (m for m in module.modules() if isinstance(m, nn.Conv1d)):
                        if not hasattr(conv, "weight_g"):
                            nn.utils.weight_norm(conv)
        self.cond_mode = conditioning_mode(conditioning)
        self.cond_dim = int((conditioning or {}).get("cond_dim", channels[0]))
        self.time_embed = (
            TimeEmbedding(self.cond_dim, (conditioning or {}).get("time_hidden_dim"), time_scale=(conditioning or {}).get("time_scale", 1.0))
            if self.cond_mode != "none"
            else None
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None, cond: torch.Tensor | None = None, length: int | None = None) -> torch.Tensor:
        if t is not None and self.time_embed is not None:
            t = self.time_embed(t)
        cond = combine_time_conditioning(t, cond, self.cond_mode)
        x = as_waveform(x)
        if self.frame_proj is not None:
            x = self.frame_proj(complex_to_channels(waveform_to_stft(x, self.stft)))
        h = self.in_proj(x)
        for layer in self.layers:
            h = layer(h, cond) if isinstance(layer, ResidualBlock1d | MRFBlock) else layer(h)
        y = self.out_proj(h)
        target = length or self.default_length
        y = torch.tanh(y) if self.tanh_output else y
        return center_crop_or_pad(y, target) if target else y
