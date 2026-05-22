"""MATPAC++ Inference Wrapper.

Provides the same API as CLAPWrapper for use as conditioning in flow training.
Also exposes per-patch embeddings for future autoencoder training.

Supports both:
- Our own trained MATPAC++ checkpoints (via checkpoint_path)
- Upstream pretrained checkpoints (via pretrained="matpac_10_2048" etc.)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from matpac.checkpoint import load_conditioner_from_checkpoint

# Shared config matching upstream MATPAC architecture (16kHz, 80 mels, no RoPE)
_UPSTREAM_CONFIG = dict(
    sample_rate=16000, n_mels=80, n_fft=400, hop_length=160,
    patch_size=16, norm_mean=-7.056, norm_std=4.193,
    f_min=50.0, f_max=8000.0, center=False,
    hidden_size=768, encoder_depth=12, num_heads=12, mlp_ratio=4.0,
    use_cls_token=True, use_rope=False, rope_2d=False,
    # Predictor/classifier params — needed to construct MATPAC but weights not loaded
    predictor_depth=8, predictor_dim=512, predictor_num_heads=16,
    num_hypotheses=5, num_classes=2048, cls_num_classes=4096,
)

_PRETRAINED = {
    "matpac_10_2048": {
        "url": "https://github.com/aurianworld/matpac/releases/download/Initial_release/matpac_10_2048.pt",
        "config": _UPSTREAM_CONFIG,
    },
    "matpac_plus_music_6s_2048": {
        "url": "https://github.com/aurianworld/matpac/releases/download/MATPAC%2B%2B/matpac_plus_music_6s_2048_enconly.pt",
        "config": _UPSTREAM_CONFIG,
    },
}


def _remap_pretrained_state_dict(
    pretrained_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap upstream MATPAC checkpoint keys to our model structure.

    Upstream keys:
        cls_token, pos_embed, patch_embed.proj.*, student_encoder.blocks.*, student_encoder.norm.*
    Our keys:
        student_encoder.cls_token, student_encoder.pos_embed, mel_frontend.patch_embed.*,
        student_encoder.blocks.*, student_encoder.norm.*
    """
    remapped = {}
    for k, v in pretrained_sd.items():
        if k == "cls_token":
            remapped["student_encoder.cls_token"] = v
        elif k == "pos_embed":
            # Strip CLS position (index 0), keep patch positions [1:]
            remapped["student_encoder.pos_embed"] = v[:, 1:, :]
        elif k.startswith("patch_embed.proj."):
            # patch_embed.proj.weight -> mel_frontend.patch_embed.weight
            new_key = k.replace("patch_embed.proj.", "mel_frontend.patch_embed.")
            remapped[new_key] = v
        elif k.startswith("student_encoder."):
            # Direct match — blocks.* and norm.*
            remapped[k] = v
        else:
            print(f"[MATpacWrapper] Skipping unknown pretrained key: {k}")
    return remapped


def _load_pretrained_model(
    name: str,
    device: str = "cuda",
) -> nn.Module:
    """Download and load a pretrained MATPAC model.

    Args:
        name: Pretrained variant name (e.g. "matpac_10_2048")
        device: Device to load on

    Returns:
        MATPAC model with pretrained weights, in eval mode with frozen params
    """
    from matpac.matpac import MATPAC

    if name not in _PRETRAINED:
        available = ", ".join(_PRETRAINED.keys())
        raise ValueError(f"Unknown pretrained model: {name!r}. Available: {available}")

    info = _PRETRAINED[name]
    config = info["config"]

    # Download checkpoint (cached by torch.hub)
    print(f"Loading pretrained MATPAC: {name}")
    pretrained_sd = torch.hub.load_state_dict_from_url(
        info["url"], map_location="cpu", weights_only=True,
    )

    # Infer max_time_patches from pretrained pos_embed
    # pos_embed is [1, num_patches + 1, D] where +1 is CLS
    n_freq = config["n_mels"] // config["patch_size"]
    total_pos = pretrained_sd["pos_embed"].shape[1]  # includes CLS
    n_patches = total_pos - 1  # strip CLS
    max_time = n_patches // n_freq

    # Create model with upstream-matching config
    model = MATPAC(
        sample_rate=config["sample_rate"],
        n_mels=config["n_mels"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        f_min=config["f_min"],
        f_max=config["f_max"],
        patch_size=config["patch_size"],
        norm_mean=config["norm_mean"],
        norm_std=config["norm_std"],
        center=config["center"],
        hidden_size=config["hidden_size"],
        encoder_depth=config["encoder_depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        use_cls_token=config["use_cls_token"],
        use_rope=config["use_rope"],
        rope_2d=config["rope_2d"],
        predictor_depth=config["predictor_depth"],
        predictor_dim=config["predictor_dim"],
        predictor_num_heads=config["predictor_num_heads"],
        num_hypotheses=config["num_hypotheses"],
        num_classes=config["num_classes"],
        cls_num_classes=config["cls_num_classes"],
    )

    # Override max_time_patches in encoder to match pretrained pos_embed size
    # (ViTEncoder creates pos_embed with max_time_patches=256 by default)

    from matpac.encoder import get_2d_sincos_pos_embed
    pos_embed_np = get_2d_sincos_pos_embed(config["hidden_size"], n_freq, max_time)
    model.student_encoder.pos_embed = nn.Parameter(
        torch.from_numpy(pos_embed_np).float().unsqueeze(0),
        requires_grad=False,
    )
    model.student_encoder.max_time_patches = max_time
    # Teacher shares pos_embed
    model.teacher_encoder.pos_embed = model.student_encoder.pos_embed
    model.teacher_encoder.max_time_patches = max_time

    # Remap and load
    remapped = _remap_pretrained_state_dict(pretrained_sd)
    missing, unexpected = model.load_state_dict(remapped, strict=False)

    # Expected missing: teacher blocks/norm, predictor, classifiers, loss buffers,
    # torchaudio auto-created buffers, shared teacher params (cls_token/pos_embed)
    unexpected_missing = [
        k for k in missing
        if not any(k.startswith(p) for p in (
            "teacher_encoder.", "predictor.",
            "student_dino_head.", "teacher_dino_head.",
            "student_cls_dino_head.", "teacher_cls_dino_head.",
            "mcl_loss.", "dino_loss.", "cls_dino_loss.",
            "student_pooler.", "teacher_pooler.",
            "mel_frontend.mel_transform.",
        ))
    ]
    if unexpected_missing:
        print(f"[MATpacWrapper] WARNING: unexpected missing keys: {unexpected_missing}")
    if unexpected:
        print(f"[MATpacWrapper] WARNING: unexpected extra keys: {unexpected}")

    loaded_count = len(remapped)
    print(f"Loaded {loaded_count} pretrained params into student encoder")

    # Freeze and eval
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    return model


class MATpacWrapper(nn.Module):
    """Inference wrapper for MATPAC++ with CLAP-compatible API.

    Loads a MATPAC++ model and provides:
    - encode_audio(): Returns [B, D] pooled embeddings (like CLAP)
    - encode_audio_frames(): Returns [B, N_patches, D] per-patch embeddings

    Supports two loading modes:
    - checkpoint_path: Our own trained MATPAC++ Lightning checkpoint
    - pretrained: Upstream pretrained variant name (e.g. "matpac_10_2048")

    Args:
        checkpoint_path: Path to trained MATPAC++ checkpoint
        pretrained: Pretrained variant name (see _PRETRAINED registry)
        device: Device to load model on (default: 'cuda')
        use_teacher: Whether to use teacher encoder (default: False, student is standard)
        encode_batch_size: Max batch size for encoding (0 = no limit)
        compile_encoder: Whether to torch.compile the encoder (default: True)
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        pretrained: str = None,
        device: str = 'cuda',
        use_teacher: bool = False,
        encode_batch_size: int = 0,
        compile_encoder: bool = True,
    ):
        super().__init__()
        self.device_str = device
        self.use_teacher = use_teacher
        self.encode_batch_size = encode_batch_size
        self._resampler_cache: dict[tuple[int, int], torchaudio.transforms.Resample] = {}

        if pretrained is not None and checkpoint_path is not None:
            raise ValueError("Specify either 'pretrained' or 'checkpoint_path', not both")
        if pretrained is None and checkpoint_path is None:
            raise ValueError("Must specify either 'pretrained' or 'checkpoint_path'")

        if pretrained is not None:
            self.model = _load_pretrained_model(pretrained, device=device)
        else:
            self.model = load_conditioner_from_checkpoint(
                checkpoint_path, device=device,
            )

        self.hidden_size = self.model.hidden_size
        # Dynamic sample rate from the model's mel frontend
        self.SAMPLE_RATE = self.model.mel_frontend.sample_rate
        self.eval()

        if compile_encoder:
            self.model.student_encoder = torch.compile(
                self.model.student_encoder, mode="default",
            )
            self.model.mel_frontend = torch.compile(
                self.model.mel_frontend, mode="default",
            )

    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        if hasattr(device, 'type'):
            self.device_str = f"{device.type}:{device.index}" if device.index else device.type
        else:
            self.device_str = str(device)
        return self

    def _preprocess(
        self,
        audio: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        """Preprocess audio: convert to mono and resample if needed.

        Args:
            audio: [B, T] or [B, C, T] audio tensor
            sample_rate: Input sample rate

        Returns:
            [B, T] mono audio at native sample rate
        """
        # Handle channels: MATPAC++ expects mono
        if audio.ndim == 3:
            audio = audio.mean(dim=1)

        # Resample if needed (cache resampler to avoid re-creating each call)
        if sample_rate != self.SAMPLE_RATE:
            key = (sample_rate, self.SAMPLE_RATE)
            if key not in self._resampler_cache:
                self._resampler_cache[key] = torchaudio.transforms.Resample(
                    sample_rate, self.SAMPLE_RATE,
                ).to(audio.device)
            audio = self._resampler_cache[key](audio)

        return audio

    def encode_audio(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode audio to pooled embedding (CLAP-compatible API).

        Args:
            audio: [B, T] or [B, C, T] audio tensor
            sample_rate: Input sample rate
            audio_lengths: [B] Original audio lengths in samples at input sample_rate.
                          When provided, enables padding-invariant embeddings.

        Returns:
            embeddings: [B, D] L2-normalized embeddings
        """
        audio = self._preprocess(audio, sample_rate)

        # Scale audio_lengths if resampled
        if audio_lengths is not None and sample_rate != self.SAMPLE_RATE:
            audio_lengths = (audio_lengths.float() * (self.SAMPLE_RATE / sample_rate)).long()

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            audio = audio.float()

            # Pad audio shorter than STFT minimum (n_fft + 1 samples)
            min_samples = self.model.mel_frontend.n_fft + 1
            if audio.shape[-1] < min_samples:
                audio = F.pad(audio, (0, min_samples - audio.shape[-1]))

            B = audio.shape[0]
            ebs = self.encode_batch_size

            if ebs <= 0 or B <= ebs:
                embedding = self.model.encode(
                    audio,
                    audio_lengths=audio_lengths,
                    use_teacher=self.use_teacher,
                )
            else:
                chunks = []
                for i in range(0, B, ebs):
                    a_chunk = audio[i:i + ebs]
                    l_chunk = audio_lengths[i:i + ebs] if audio_lengths is not None else None
                    chunks.append(self.model.encode(
                        a_chunk, audio_lengths=l_chunk, use_teacher=self.use_teacher,
                    ))
                embedding = torch.cat(chunks, dim=0)

            embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding

    def encode_audio_frames(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        audio_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode audio to per-patch embeddings BEFORE pooling.

        Args:
            audio: [B, T] or [B, C, T] audio tensor
            sample_rate: Input sample rate
            audio_lengths: [B] Original audio lengths in samples at input sample_rate.

        Returns:
            patch_embeddings: [B, N_patches, D] per-patch embeddings
            attention_mask: [B, N_patches] attention mask (1=valid, 0=pad)
        """
        audio = self._preprocess(audio, sample_rate)

        if audio_lengths is not None and sample_rate != self.SAMPLE_RATE:
            audio_lengths = (audio_lengths.float() * (self.SAMPLE_RATE / sample_rate)).long()

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            audio = audio.float()
            patch_embeddings, attention_mask = self.model.encode_frames(
                audio,
                audio_lengths=audio_lengths,
                use_teacher=self.use_teacher,
            )

        return patch_embeddings, attention_mask

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.hidden_size

