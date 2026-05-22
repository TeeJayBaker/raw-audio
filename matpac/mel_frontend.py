"""Mel spectrogram frontend for MATPAC++.

Converts audio waveforms to mel spectrograms and patches them for ViT processing.
Supports variable-length audio with attention masking.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class MelFrontend(nn.Module):
    """Audio to mel spectrogram to patches.

    Converts audio waveforms to log-mel spectrograms and then patches them
    using a Conv2d layer (kernel=16x16, stride=16x16) for ViT processing.

    Args:
        sample_rate: Target sample rate (default: 48000)
        n_mels: Number of mel filterbank channels (default: 128)
        n_fft: FFT size (default: 2048)
        hop_length: Hop length between STFT frames (default: 512)
        f_min: Minimum frequency (default: 20.0)
        f_max: Maximum frequency (default: 24000.0)
        patch_size: Size of patches in both freq and time dimensions (default: 16)
        hidden_size: Output embedding dimension (default: 768)
        norm_eps: Small constant for log normalization (default: 1e-6)
        norm_mean: Mean for mel normalization (default: -8.94)
        norm_std: Std for mel normalization (default: 5.59)
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        f_min: float = 20.0,
        f_max: float = 24000.0,
        patch_size: int = 16,
        hidden_size: int = 768,
        norm_eps: float = 1e-6,
        norm_mean: float = -8.94,
        norm_std: float = 5.59,
        center: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.norm_eps = norm_eps
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.center = center

        # Mel spectrogram transform with slaney mel scale and normalization
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,  # Power spectrogram
            norm="slaney",
            mel_scale="slaney",
            center=center,
        )

        # Patch embedding: Conv2d with kernel and stride of patch_size
        # Input: [B, 1, n_mels, T_frames]
        # Output: [B, hidden_size, n_mel_patches, n_time_patches]
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_size,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
        )

        # Number of frequency patches (fixed for n_mels=128, patch_size=16)
        self.n_freq_patches = n_mels // patch_size

    @torch.compiler.disable
    def _pad_to_patch_size(self, mel: torch.Tensor) -> torch.Tensor:
        """Pad mel to multiple of patch_size in time dimension."""
        T_frames = mel.shape[-1]
        if T_frames % self.patch_size != 0:
            pad_len = self.patch_size - (T_frames % self.patch_size)
            mel = F.pad(mel, (0, pad_len))
        return mel

    @torch.compiler.disable
    def _build_attention_mask(
        self,
        audio_lengths: torch.Tensor,
        T_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build patch-level attention mask from audio lengths.

        Kept outside torch.compile to avoid inductor issues with dynamic shapes.
        """
        audio_lengths.shape[0]
        n_time_total = T_frames // self.patch_size

        # Convert sample lengths to valid time patches
        if self.center:
            n_frames = 1 + audio_lengths // self.hop_length  # [B]
        else:
            n_frames = (audio_lengths - self.n_fft) // self.hop_length + 1  # [B]
        n_time_valid = (n_frames // self.patch_size).clamp(min=1)  # [B]

        # Build time mask and repeat for each freq row (row-major order)
        time_idx = torch.arange(n_time_total, device=device)  # [T]
        time_mask = (time_idx < n_time_valid.unsqueeze(1)).float()  # [B, T]
        # All freq rows share the same time mask, so repeat gives correct row-major layout
        return time_mask.repeat(1, self.n_freq_patches)  # [B, n_freq * n_time]

    @torch.compiler.disable
    def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to log-mel spectrogram.

        Args:
            audio: [B, T] mono audio waveform

        Returns:
            mel: [B, 1, n_mels, T_frames] log-mel spectrogram
        """
        # Compute mel spectrogram
        mel = self.mel_transform(audio)  # [B, n_mels, T_frames]

        # Convert to log scale and normalize
        mel = torch.log(mel + self.norm_eps)
        mel = (mel - self.norm_mean) / self.norm_std

        # Add channel dimension
        mel = mel.unsqueeze(1)  # [B, 1, n_mels, T_frames]

        return mel

    def samples_to_mel_frames(self, n_samples: int) -> int:
        """Convert number of audio samples to mel frames.

        With center=True (default): signal is padded by n_fft//2 on each side,
        giving 1 + n_samples // hop_length frames.
        With center=False: no padding, giving (n_samples - n_fft) // hop_length + 1 frames.
        """
        if self.center:
            return 1 + n_samples // self.hop_length
        return (n_samples - self.n_fft) // self.hop_length + 1

    def mel_frames_to_time_patches(self, n_frames: int) -> int:
        """Convert number of mel frames to time patches."""
        return n_frames // self.patch_size

    def samples_to_patches(self, n_samples: int) -> int:
        """Convert number of audio samples to total patches."""
        n_frames = self.samples_to_mel_frames(n_samples)
        n_time_patches = self.mel_frames_to_time_patches(n_frames)
        return self.n_freq_patches * n_time_patches

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Convert audio to patch embeddings with attention mask.

        Args:
            audio: [B, T] mono audio waveform
            audio_lengths: [B] original audio lengths in samples (optional)

        Returns:
            patches: [B, N_patches, hidden_size] patch embeddings
            attention_mask: [B, N_patches] mask (1=valid, 0=pad) or None if no padding
        """
        B, T = audio.shape
        device = audio.device

        # Convert to mel spectrogram
        mel = self.audio_to_mel(audio)  # [B, 1, n_mels, T_frames]

        # Pad mel to multiple of patch_size in time dimension
        mel = self._pad_to_patch_size(mel)
        T_frames_padded = mel.shape[-1]

        # Patchify using Conv2d
        patches = self.patch_embed(mel)  # [B, hidden_size, n_freq_patches, n_time_patches]

        # Reshape to sequence: [B, hidden_size, n_freq, n_time] -> [B, N, hidden_size]
        B, C, H, W = patches.shape
        patches = patches.permute(0, 2, 3, 1)  # [B, n_freq, n_time, hidden_size]
        patches = patches.reshape(B, H * W, C)  # [B, N_patches, hidden_size]

        # Compute attention mask if audio_lengths provided
        attention_mask = None
        if audio_lengths is not None:
            attention_mask = self._build_attention_mask(
                audio_lengths, T_frames_padded, device
            )

        return patches, attention_mask
