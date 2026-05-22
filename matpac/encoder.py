"""ViT Encoder for MATPAC++.

12-layer Vision Transformer with:
- Pre-norm (LayerNorm before attention/MLP)
- GELU activation
- Position encoding: 2D sincos absolute pos embed + 2D RoPE (relative, in attention)
- CLS token prepended
- Support for masked input (only encode visible patches during training)
"""


import numpy as np
import torch
import torch.nn as nn

from matpac.blocks import TransformerBlock


def get_2d_sincos_pos_embed(
    embed_dim: int,
    n_freq: int,
    n_time: int,
) -> np.ndarray:
    """Generate 2D sinusoidal-cosine positional embeddings.

    Creates positional embeddings for a 2D grid (freq x time), flattened
    in row-major order. The embedding dimension is split in half between
    the freq and time axes.

    Args:
        embed_dim: Embedding dimension (must be divisible by 4)
        n_freq: Number of frequency patches
        n_time: Number of time patches

    Returns:
        pos_embed: [n_freq * n_time, embed_dim] positional embeddings
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sincos"
    half_dim = embed_dim // 2

    # Create grid
    freq_pos = np.arange(n_freq, dtype=np.float64)
    time_pos = np.arange(n_time, dtype=np.float64)

    def _get_1d_sincos(positions: np.ndarray, dim: int) -> np.ndarray:
        omega = np.arange(dim // 2, dtype=np.float64)
        omega = 1.0 / (10000.0 ** (omega / (dim // 2)))
        out = np.outer(positions, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # [N, dim]

    freq_embed = _get_1d_sincos(freq_pos, half_dim)  # [n_freq, half_dim]
    time_embed = _get_1d_sincos(time_pos, half_dim)  # [n_time, half_dim]

    # Combine: each (f, t) pair gets [freq_embed[f], time_embed[t]]
    pos_embed = np.zeros((n_freq * n_time, embed_dim), dtype=np.float64)
    for f in range(n_freq):
        for t in range(n_time):
            pos_embed[f * n_time + t, :half_dim] = freq_embed[f]
            pos_embed[f * n_time + t, half_dim:] = time_embed[t]

    return pos_embed.astype(np.float32)


class ViTEncoder(nn.Module):
    """Vision Transformer Encoder for MATPAC++.

    Args:
        hidden_size: Transformer hidden dimension (default: 768)
        depth: Number of transformer blocks (default: 12)
        num_heads: Number of attention heads (default: 12)
        mlp_ratio: MLP hidden dim multiplier (default: 4.0)
        use_cls_token: Whether to prepend CLS token (default: True)
        n_freq_patches: Number of frequency patches for pos embed (default: 8)
        max_time_patches: Max number of time patches for pos embed (default: 256)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        use_cls_token: bool = True,
        use_rope: bool = True,
        rope_2d: bool = True,
        n_freq_patches: int = 8,
        max_time_patches: int = 256,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.use_cls_token = use_cls_token
        self.use_rope = use_rope
        self.rope_2d = rope_2d
        self.n_freq_patches = n_freq_patches
        self.max_time_patches = max_time_patches

        # CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            nn.init.normal_(self.cls_token, std=0.02)

        # Learnable 2D positional embeddings (initialized from sincos, per original paper)
        sincos_embed = get_2d_sincos_pos_embed(
            hidden_size, n_freq_patches, max_time_patches
        )
        self.pos_embed = nn.Parameter(
            torch.from_numpy(sincos_embed).float().unsqueeze(0),  # [1, max_patches, D]
            requires_grad=True,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_rope=use_rope,
                activation="gelu",
                rope_2d=rope_2d,
                n_freq_patches=n_freq_patches,
            )
            for _ in range(depth)
        ])

        # Final norm
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def get_pos_embed(self, num_patches: int) -> torch.Tensor:
        """Get positional embeddings for a given number of patches.

        When pos_embed is frozen (pretrained models), uses 2D-aware truncation:
        reshapes to [n_freq, max_time], truncates the time axis, then flattens.
        This preserves the 2D grid structure matching upstream MATPAC behavior.

        When pos_embed is learnable (our trained models), uses linear slicing
        to preserve backward compatibility with already-trained weights.

        Args:
            num_patches: Number of patches (N = n_freq * n_time)

        Returns:
            pos_embed: [1, num_patches, hidden_size] positional embeddings
        """
        if num_patches == self.pos_embed.shape[1]:
            return self.pos_embed
        if not self.pos_embed.requires_grad:
            # Frozen (pretrained): 2D-aware truncation
            n_freq = self.n_freq_patches
            n_time = num_patches // n_freq
            D = self.pos_embed.shape[-1]
            return self.pos_embed.reshape(1, n_freq, self.max_time_patches, D)[
                :, :, :n_time, :
            ].reshape(1, num_patches, D)
        # Learnable (our trained models): linear slicing for backward compat
        return self.pos_embed[:, :num_patches]

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder.

        Positional embeddings should be added to patches BEFORE passing them
        here (and before any masking/gathering). This encoder only handles
        CLS token prepending, transformer blocks, and final norm.

        Args:
            x: [B, N, hidden_size] patch embeddings (with pos embed already added)
            mask: [B, N] attention mask (1=valid, 0=pad)
            position_ids: [B, N] original patch indices for RoPE. CLS gets -1
                (identity rotation, position-agnostic) in both 1D and 2D RoPE.

        Returns:
            output: [B, N+1, hidden_size] if use_cls_token else [B, N, hidden_size]
            cls_output: [B, hidden_size] CLS token output (or mean pooled if no CLS)
        """
        B, N = x.shape[0], x.shape[1]

        # Construct position_ids for inference (when not provided)
        if position_ids is None and self.use_rope:
            position_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)

        # Prepend CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

            # Extend mask for CLS token (always valid)
            if mask is not None:
                cls_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([cls_mask, mask], dim=1)

            # Extend position_ids for CLS token: -1 → identity rotation (no RoPE).
            # CLS should be position-agnostic, attending equally to all patches.
            if position_ids is not None:
                cls_pos = torch.full(
                    (B, 1), -1, device=position_ids.device, dtype=position_ids.dtype
                )
                position_ids = torch.cat([cls_pos, position_ids], dim=1)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask, position_ids=position_ids)

        # Final norm
        x = self.norm(x)

        # Extract CLS token or mean pool
        if self.use_cls_token:
            cls_output = x[:, 0]
        else:
            # Mean pool over valid positions
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                x_masked = x * mask_expanded
                cls_output = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                cls_output = x.mean(dim=1)

        return x, cls_output
