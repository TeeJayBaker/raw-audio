"""MCL (Multi-hypothesis Contrastive Learning) Predictor for MATPAC++.

The predictor takes visible patch embeddings and predicts masked patch embeddings.
Uses transformer blocks at decoder_dim and parallel linear heads for hypothesis prediction.
"""


import torch
import torch.nn as nn

from matpac.blocks import TransformerBlock

from .encoder import get_2d_sincos_pos_embed


class MCLPredictor(nn.Module):
    """MCL Predictor with learnable mask tokens and parallel heads.

    The predictor:
    1. Takes visible patch embeddings from the student encoder
    2. Projects from encoder dim (768) to decoder dim (512)
    3. Inserts learnable mask tokens at masked positions
    4. Adds frozen 2D sincos positional embeddings
    5. Processes through transformer layers at decoder dim
    6. Uses parallel linear heads to project back to encoder dim for hypotheses

    Args:
        hidden_size: Encoder embedding dimension (default: 768)
        decoder_dim: Predictor internal dimension (default: 512)
        depth: Number of transformer blocks (default: 8)
        num_heads: Number of attention heads per block (default: 16)
        mlp_ratio: MLP hidden dim multiplier (default: 4.0)
        num_hypotheses: Number of parallel prediction heads (default: 5)
        n_freq_patches: Number of frequency patches (default: 8)
        max_time_patches: Maximum number of time patches (default: 256)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        decoder_dim: int = 512,
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_hypotheses: int = 5,
        n_freq_patches: int = 8,
        max_time_patches: int = 256,
        use_cls: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.decoder_dim = decoder_dim
        self.depth = depth
        self.num_hypotheses = num_hypotheses
        self.n_freq_patches = n_freq_patches
        self.max_time_patches = max_time_patches
        self.use_cls = use_cls

        # CLS dropout and projection only needed when CLS is used
        if use_cls:
            # Current dropout rate — updated externally by the training module each step
            self.cls_dropout = 0.0
            # CLS projection: encoder dim -> decoder dim (for global context)
            self.cls_proj = nn.Linear(hidden_size, decoder_dim)

        # Input projection: encoder dim -> decoder dim
        self.input_proj = nn.Linear(hidden_size, decoder_dim)

        # Learnable mask token at decoder dim (initialized to zeros per original paper)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # Frozen 2D sincos positional embeddings (matching reference: nn.Parameter with requires_grad=False)
        sincos_embed = get_2d_sincos_pos_embed(
            decoder_dim, n_freq_patches, max_time_patches
        )
        self.pos_embed = nn.Parameter(
            torch.from_numpy(sincos_embed).float().unsqueeze(0),  # [1, max_patches, decoder_dim]
            requires_grad=False,
        )

        # Transformer blocks at decoder dim (no RoPE, GELU activation)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_rope=False,
                activation="gelu",
            )
            for _ in range(depth)
        ])

        # Final norm at decoder dim
        self.norm = nn.LayerNorm(decoder_dim, eps=1e-6)

        # Parallel linear heads: decoder_dim -> hidden_size (back to encoder space)
        self.heads = nn.ModuleList([
            nn.Linear(decoder_dim, hidden_size)
            for _ in range(num_hypotheses)
        ])

        # Initialize heads
        for head in self.heads:
            nn.init.normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        visible_embeddings: torch.Tensor,
        visible_indices: torch.Tensor,
        masked_indices: torch.Tensor,
        total_patches: int,
        attention_mask: torch.Tensor | None = None,
        cls_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict embeddings for masked positions.

        Args:
            visible_embeddings: [B, M_visible, hidden_size] embeddings of visible patches
            visible_indices: [B, M_visible] indices of visible patches
            masked_indices: [B, M_masked] indices of masked patches
            total_patches: Total number of patches N
            attention_mask: [B, N] attention mask (1=valid, 0=pad)
            cls_embedding: [B, hidden_size] student CLS embedding for global context

        Returns:
            predictions: [B, M_masked, num_hypotheses, hidden_size] predictions for masked patches
        """
        B = visible_embeddings.shape[0]
        M_masked = masked_indices.shape[1]
        device = visible_embeddings.device

        # Project visible embeddings to decoder dim
        visible_proj = self.input_proj(visible_embeddings)  # [B, M_visible, decoder_dim]

        # Create full sequence with mask tokens at masked positions
        full_seq = torch.zeros(B, total_patches, self.decoder_dim, device=device)

        # Place projected visible embeddings at their positions
        full_seq.scatter_(
            dim=1,
            index=visible_indices.unsqueeze(-1).expand(-1, -1, self.decoder_dim),
            src=visible_proj,
        )

        # Place mask tokens at masked positions
        mask_tokens = self.mask_token.expand(B, M_masked, -1)
        full_seq.scatter_(
            dim=1,
            index=masked_indices.unsqueeze(-1).expand(-1, -1, self.decoder_dim),
            src=mask_tokens,
        )

        # Add frozen 2D sincos positional embeddings
        full_seq = full_seq + self.pos_embed[:, :total_patches]

        # Prepend CLS as global context token (with dropout)
        use_cls = cls_embedding is not None
        if use_cls:
            cls_proj = self.cls_proj(cls_embedding).unsqueeze(1)  # [B, 1, decoder_dim]
            if self.training and self.cls_dropout > 0:
                # Bernoulli dropout: zero out CLS for p% of batch samples
                mask = torch.bernoulli(
                    torch.full((B, 1, 1), 1.0 - self.cls_dropout, device=device)
                )
                cls_proj = cls_proj * mask

            # Prepend CLS to full sequence (no positional embedding for CLS)
            full_seq = torch.cat([cls_proj, full_seq], dim=1)  # [B, 1+N, decoder_dim]

            # Extend attention_mask for CLS position (always valid)
            if attention_mask is not None:
                cls_mask = torch.ones(B, 1, device=device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        # Process through transformer blocks
        # attention_mask uses 1=valid, 0=pad convention (handled by TransformerBlock)
        x = full_seq
        for block in self.blocks:
            x = block(x, mask=attention_mask)

        # Final norm
        x = self.norm(x)

        # Strip CLS before gathering at masked positions
        if use_cls:
            x = x[:, 1:]  # Remove CLS, back to [B, N, decoder_dim]

        # Extract embeddings at masked positions
        masked_embeddings = torch.gather(
            x,
            dim=1,
            index=masked_indices.unsqueeze(-1).expand(-1, -1, self.decoder_dim),
        )  # [B, M_masked, decoder_dim]

        # Apply all hypothesis heads (project back to encoder dim)
        predictions = torch.stack(
            [head(masked_embeddings) for head in self.heads],
            dim=2,
        )  # [B, M_masked, num_hypotheses, hidden_size]

        return predictions


class MCLLoss(nn.Module):
    """MCL (Multi-hypothesis Contrastive Learning) Loss.

    For each masked position, we have K hypotheses. We compute the loss
    by finding the best matching hypothesis and computing a contrastive loss.

    The loss encourages:
    1. At least one hypothesis to match the teacher embedding
    2. Diversity among hypotheses (handled implicitly by the min operation)

    Args:
        temperature: Temperature for softmax (default: 1.0, annealed during training)
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        temperature: float | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute MCL loss (Eq. 1 from MATPAC++ paper).

        Uses squared L2 distance on L2-normalized vectors with annealed
        soft assignment weights:
            d(j)(i) = ||pred'(j)(i) - target'(i)||^2_2
            b(i) = Softmax(-d / tau_mcl)
            L = (1/BM) sum_i sum_j b(j)(i) * d(j)(i)

        Args:
            predictions: [B, M, K, D] K hypotheses for M masked positions
            targets: [B, M, D] target embeddings from teacher
            temperature: Optional temperature override
            padding_mask: [B, M] mask for valid positions (1=valid, 0=batch padding)
                         If provided, loss is computed only over valid positions

        Returns:
            loss: Scalar loss value (always >= 0)
            best_idx: [B, M] index of best hypothesis per masked position
        """
        temp = temperature if temperature is not None else self.temperature
        B, M, K, D = predictions.shape

        # M2D-style standardization of targets (stabilizes training)
        target_mean = targets.mean(dim=-1, keepdim=True)
        target_std = targets.std(dim=-1, keepdim=True).clamp(min=1e-6)
        targets = (targets - target_mean) / target_std

        # L2-normalize predictions and targets
        predictions = nn.functional.normalize(predictions, dim=-1)
        targets = nn.functional.normalize(targets, dim=-1)

        # Squared L2 distance: ||pred - target||^2 = 2 - 2*cos_sim for unit vectors
        # predictions: [B, M, K, D], targets: [B, M, 1, D]
        diff = predictions - targets.unsqueeze(2)  # [B, M, K, D]
        distances = (diff ** 2).sum(dim=-1)  # [B, M, K], range [0, 4]

        # Best hypothesis per patch (for classification pipeline)
        best_idx = distances.argmin(dim=-1)  # [B, M]

        # Soft assignment weights (detached to prevent mode collapse)
        weights = torch.softmax(-distances.detach() / temp, dim=-1)  # [B, M, K]

        # Weighted sum of distances per position
        per_position_loss = (weights * distances).sum(dim=-1)  # [B, M]

        if padding_mask is not None:
            # Zero out padded positions and average only over valid positions
            per_position_loss = per_position_loss * padding_mask
            num_valid = padding_mask.sum().clamp(min=1)
            loss = per_position_loss.sum() / num_valid
        else:
            loss = per_position_loss.mean()

        return loss, best_idx
