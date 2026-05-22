"""Full MATPAC++ Model.

Combines mel frontend, ViT encoder, MCL predictor, and DINO classifier
into a complete self-supervised audio representation learning model.

Key features:
- Student-teacher architecture with EMA updates
- Random masking of patches (70% default)
- MCL prediction loss for masked patches
- DINO classification loss for global representation
"""

import copy

import torch
import torch.nn as nn

from .classifier import DINOHead, DINOLoss
from .encoder import ViTEncoder
from .mel_frontend import MelFrontend
from .pooler import AttentivePooler
from .predictor import MCLLoss, MCLPredictor


class MATPAC(nn.Module):
    """MATPAC++ Self-supervised Audio Model.

    Architecture:
    - Mel frontend: Audio -> log-mel spectrogram -> patch embeddings
    - Student encoder: ViT-Base (12 layers, 768 dim, 12 heads) with RoPE + sincos pos embed
    - Teacher encoder: EMA copy of student
    - MCL predictor: 8 transformer layers at 512d, 16 heads, with sincos pos embed
    - DINO heads: 3-layer MLP -> K pseudo-classes

    Training:
    1. Convert audio to mel spectrogram patches
    2. Randomly mask 70% of patches
    3. Student encodes visible patches
    4. Predictor predicts masked patch embeddings (multiple hypotheses)
    5. Teacher encodes masked patches only
    6. Compute MCL loss (prediction vs teacher) + DINO loss (per-patch classification)
    """

    def __init__(
        self,
        # Mel frontend
        sample_rate: int = 48000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        f_min: float = 20.0,
        f_max: float = 24000.0,
        patch_size: int = 16,
        norm_mean: float = -8.94,
        norm_std: float = 5.59,
        center: bool = True,
        # Encoder
        hidden_size: int = 768,
        encoder_depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        use_rope: bool = True,
        rope_2d: bool = True,
        # Predictor
        predictor_depth: int = 8,
        predictor_dim: int = 512,
        predictor_num_heads: int = 16,
        num_hypotheses: int = 5,
        # Classifier (patch-level DINO)
        classifier_hidden_dim: int = 2048,
        classifier_bottleneck_dim: int = 256,
        num_classes: int = 2048,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
        # CLS Classifier (separate DINO head for CLS token)
        cls_classifier_hidden_dim: int = 2048,
        cls_classifier_bottleneck_dim: int = 256,
        cls_num_classes: int = 4096,
        cls_student_temp: float = 0.1,
        cls_teacher_temp: float = 0.04,
        cls_center_momentum: float = 0.9,
        # Training
        mask_ratio: float = 0.7,
        ema_momentum: float = 0.996,
        # v2: decoupled CLS
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.mask_ratio = mask_ratio
        self.ema_momentum = ema_momentum
        self.use_cls_token = use_cls_token

        n_freq_patches = n_mels // patch_size

        # Mel frontend
        self.mel_frontend = MelFrontend(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            patch_size=patch_size,
            hidden_size=hidden_size,
            norm_mean=norm_mean,
            norm_std=norm_std,
            center=center,
        )

        # Student encoder
        self.student_encoder = ViTEncoder(
            hidden_size=hidden_size,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_cls_token=use_cls_token,
            use_rope=use_rope,
            rope_2d=rope_2d,
            n_freq_patches=n_freq_patches,
        )

        # Teacher encoder (deep copy of student — standard DINO/iBOT approach)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)

        # NOTE: Independent re-initialization (per MATPAC paper) commented out.
        # This prevents trivial solutions but requires lower EMA momentum to converge.
        # Uncomment for longer training runs (295k+ steps) with ema_start ~0.99995.
        # for block in self.teacher_encoder.blocks:
        #     block.apply(self._init_weights)
        # self.teacher_encoder.norm.apply(self._init_weights)

        for param in self.teacher_encoder.parameters():
            param.requires_grad = False

        # Share pos_embed between student and teacher (not EMA'd)
        self.teacher_encoder.pos_embed = self.student_encoder.pos_embed

        # Share cls_token if used
        if use_cls_token:
            self.teacher_encoder.cls_token = self.student_encoder.cls_token

        # v2: attentive pooler replaces CLS token
        if not use_cls_token:
            self.student_pooler = AttentivePooler(hidden_size)
            self.teacher_pooler = copy.deepcopy(self.student_pooler)
            for param in self.teacher_pooler.parameters():
                param.requires_grad = False

        # MCL predictor
        self.predictor = MCLPredictor(
            hidden_size=hidden_size,
            decoder_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            mlp_ratio=mlp_ratio,
            num_hypotheses=num_hypotheses,
            n_freq_patches=n_freq_patches,
            use_cls=use_cls_token,
        )

        # Student DINO head
        self.student_dino_head = DINOHead(
            in_dim=hidden_size,
            hidden_dim=classifier_hidden_dim,
            bottleneck_dim=classifier_bottleneck_dim,
            num_classes=num_classes,
        )

        # Teacher DINO head (EMA copy)
        self.teacher_dino_head = copy.deepcopy(self.student_dino_head)
        for param in self.teacher_dino_head.parameters():
            param.requires_grad = False

        # Loss functions
        self.mcl_loss = MCLLoss()
        self.dino_loss = DINOLoss(
            num_classes=num_classes,
            student_temp=student_temp,
            teacher_temp=teacher_temp,
            center_momentum=center_momentum,
        )

        # Student CLS DINO head (separate from patch DINO head)
        self.student_cls_dino_head = DINOHead(
            in_dim=hidden_size,
            hidden_dim=cls_classifier_hidden_dim,
            bottleneck_dim=cls_classifier_bottleneck_dim,
            num_classes=cls_num_classes,
        )

        # Teacher CLS DINO head (EMA copy)
        self.teacher_cls_dino_head = copy.deepcopy(self.student_cls_dino_head)
        for param in self.teacher_cls_dino_head.parameters():
            param.requires_grad = False

        # CLS DINO loss (separate center buffer)
        self.cls_dino_loss = DINOLoss(
            num_classes=cls_num_classes,
            student_temp=cls_student_temp,
            teacher_temp=cls_teacher_temp,
            center_momentum=cls_center_momentum,
        )

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights for teacher encoder blocks/norm.

        Per MATPAC reference implementation: teacher blocks and norm are
        re-initialized with random weights (not copied from student).
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.no_grad()
    def update_teacher_encoder(self, momentum: float):
        """Update teacher encoder with EMA of student encoder.

        Per reference: only blocks and norm are EMA'd. cls_token and pos_embed
        are shared objects (same nn.Parameter), so they stay in sync automatically.

        Args:
            momentum: EMA momentum for encoder
        """
        for student_param, teacher_param in zip(
            self.student_encoder.blocks.parameters(),
            self.teacher_encoder.blocks.parameters(), strict=False,
        ):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1 - momentum)
        for student_param, teacher_param in zip(
            self.student_encoder.norm.parameters(),
            self.teacher_encoder.norm.parameters(), strict=False,
        ):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1 - momentum)

    @torch.no_grad()
    def update_teacher_cls_head(self, momentum: float):
        """Update teacher DINO heads with EMA of student DINO heads.

        Args:
            momentum: EMA momentum for classification heads
        """
        # Patch DINO head
        for student_param, teacher_param in zip(
            self.student_dino_head.parameters(),
            self.teacher_dino_head.parameters(), strict=False
        ):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1 - momentum)
        # CLS DINO head (shares momentum schedule)
        for student_param, teacher_param in zip(
            self.student_cls_dino_head.parameters(),
            self.teacher_cls_dino_head.parameters(), strict=False
        ):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1 - momentum)

    @torch.no_grad()
    def update_teacher_pooler(self, momentum: float):
        """Update teacher pooler with EMA of student pooler.

        Only applicable when use_cls_token=False (v2 architecture).

        Args:
            momentum: EMA momentum for pooler
        """
        if not hasattr(self, "student_pooler"):
            return
        for student_param, teacher_param in zip(
            self.student_pooler.parameters(),
            self.teacher_pooler.parameters(), strict=False,
        ):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1 - momentum)

    @torch.no_grad()
    def update_teacher(
        self,
        encoder_momentum: float | None = None,
        cls_momentum: float | None = None,
    ):
        """Update teacher networks with EMA of student networks.

        Args:
            encoder_momentum: Momentum for encoder (default: self.ema_momentum)
            cls_momentum: Momentum for cls head (default: same as encoder_momentum)
        """
        enc_m = encoder_momentum if encoder_momentum is not None else self.ema_momentum
        cls_m = cls_momentum if cls_momentum is not None else enc_m

        self.update_teacher_encoder(enc_m)
        self.update_teacher_cls_head(cls_m)
        self.update_teacher_pooler(enc_m)

    def random_masking(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly mask patches.

        Args:
            x: [B, N, D] patch embeddings
            attention_mask: [B, N] attention mask (1=valid, 0=pad)

        Returns:
            visible_indices: [B, M_visible] indices of visible patches
            masked_indices: [B, M_masked] indices of masked patches
            visible_mask: [B, M_visible] attention mask for visible patches (1=valid, 0=batch padding)
            masked_padding_mask: [B, M_masked] mask for masked patches (1=valid, 0=batch padding)
        """
        B, N, D = x.shape
        device = x.device

        # Determine number of visible patches
        if attention_mask is not None:
            # Mask only valid patches, not padding
            num_valid = attention_mask.sum(dim=1).long()  # [B]
            num_visible = (num_valid * (1 - self.mask_ratio)).long().clamp(min=1)
        else:
            num_visible = torch.full((B,), int(N * (1 - self.mask_ratio)), device=device)
            num_valid = torch.full((B,), N, device=device)

        # Maximum visible/masked across batch for padding
        max_visible = num_visible.max().item()
        max_masked = (num_valid - num_visible).max().item()

        visible_indices = torch.zeros(B, max_visible, dtype=torch.long, device=device)
        masked_indices = torch.zeros(B, max_masked, dtype=torch.long, device=device)
        visible_mask = torch.zeros(B, max_visible, device=device)
        masked_padding_mask = torch.zeros(B, max_masked, device=device)

        for i in range(B):
            n_valid = num_valid[i].item()
            n_visible = num_visible[i].item()
            n_masked = n_valid - n_visible

            # Get valid patch indices
            if attention_mask is not None:
                valid_idx = torch.where(attention_mask[i] == 1)[0]
            else:
                valid_idx = torch.arange(N, device=device)

            # Randomly shuffle and split
            perm = torch.randperm(n_valid, device=device)
            visible_idx = valid_idx[perm[:n_visible]]
            masked_idx = valid_idx[perm[n_visible:n_valid]]

            # Store (with padding)
            visible_indices[i, :n_visible] = visible_idx
            masked_indices[i, :n_masked] = masked_idx
            visible_mask[i, :n_visible] = 1.0
            masked_padding_mask[i, :n_masked] = 1.0

        return visible_indices, masked_indices, visible_mask, masked_padding_mask

    def forward_student(
        self,
        patches: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through student with masking.

        Args:
            patches: [B, N, D] patch embeddings
            attention_mask: [B, N] attention mask

        Returns:
            visible_embeddings: [B, M_visible, D] student embeddings of visible patches
            cls_embedding: [B, D] CLS token embedding
            visible_indices: [B, M_visible] indices of visible patches
            masked_indices: [B, M_masked] indices of masked patches
            visible_mask: [B, M_visible] mask for visible patches
            masked_padding_mask: [B, M_masked] mask for masked patches (1=valid, 0=batch padding)
        """
        # Random masking
        visible_indices, masked_indices, visible_mask, masked_padding_mask = self.random_masking(
            patches, attention_mask
        )

        # Gather visible patches
        B, N, D = patches.shape
        visible_indices.shape[1]

        visible_patches = torch.gather(
            patches,
            dim=1,
            index=visible_indices.unsqueeze(-1).expand(-1, -1, D),
        )

        # Encode visible patches (pos embed already applied before gathering)
        encoder_out, cls_embedding = self.student_encoder(
            visible_patches,
            mask=visible_mask,
            position_ids=visible_indices,
        )

        if self.use_cls_token:
            # Extract patch embeddings (skip CLS token)
            visible_embeddings = encoder_out[:, 1:]  # [B, M_visible, D]
        else:
            # No CLS token — encoder output is all patches
            visible_embeddings = encoder_out  # [B, M_visible, D]
            # Pool visible patches for global embedding
            cls_embedding = self.student_pooler(visible_embeddings, mask=visible_mask)

        return visible_embeddings, cls_embedding, visible_indices, masked_indices, visible_mask, masked_padding_mask

    @torch.no_grad()
    def forward_teacher(
        self,
        patches: torch.Tensor,
        masked_indices: torch.Tensor,
        masked_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through teacher (masked patches only).

        Per the paper: "The masked patches Xm are processed by the teacher
        encoder fγ, yielding Zm = fγ(Xm)". The teacher only sees the masked
        patches but still respects the padding attention mask.

        Args:
            patches: [B, N, D] patch embeddings
            masked_indices: [B, M_masked] indices of randomly masked patches
            masked_padding_mask: [B, M_masked] mask for masked patches (1=valid, 0=batch padding)

        Returns:
            masked_embeddings: [B, M_masked, D] teacher embeddings for masked patches
            cls_embedding: [B, D] CLS token embedding
        """
        B, N, D = patches.shape
        masked_indices.shape[1]

        # Gather masked patches from input
        masked_patches = torch.gather(
            patches,
            dim=1,
            index=masked_indices.unsqueeze(-1).expand(-1, -1, D),
        )  # [B, M_masked, D]

        # Use masked_padding_mask directly as attention mask for teacher
        # This correctly identifies which positions in masked_indices are valid vs batch padding
        # (Previously we gathered from attention_mask which incorrectly returned 1 for padding positions
        # that used index 0, if patch 0 happened to be valid)

        # Pos embed already applied before gathering, so patches carry
        # their correct grid positions
        encoder_out, cls_embedding = self.teacher_encoder(
            masked_patches, mask=masked_padding_mask, position_ids=masked_indices,
        )

        if self.use_cls_token:
            masked_embeddings = encoder_out[:, 1:]  # Skip CLS token
        else:
            masked_embeddings = encoder_out  # No CLS token to skip
            # Pool masked patches for global embedding
            cls_embedding = self.teacher_pooler(masked_embeddings, mask=masked_padding_mask)

        return masked_embeddings, cls_embedding

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
        mcl_temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass for training.

        Args:
            audio: [B, T] mono audio waveform
            audio_lengths: [B] original audio lengths in samples
            mcl_temperature: Temperature for MCL loss (annealed during training)

        Returns:
            Dict containing:
                - loss: Total loss
                - mcl_loss: MCL prediction loss
                - dino_loss: DINO classification loss
                - cls_embedding: [B, D] student CLS embedding (for downstream use)
        """
        # Convert audio to patches and add positional embeddings upfront
        patches, attention_mask = self.mel_frontend(audio, audio_lengths)
        B, N, D = patches.shape
        patches = patches + self.student_encoder.get_pos_embed(N)

        # Student forward with masking
        visible_embeddings, student_cls, visible_indices, masked_indices, visible_mask, masked_padding_mask = \
            self.forward_student(patches, attention_mask)

        # Teacher forward on masked patches only (per paper)
        teacher_targets, teacher_cls = self.forward_teacher(
            patches, masked_indices, masked_padding_mask
        )

        # MCL loss: predict masked patch embeddings
        # v2 (no CLS token): pass cls_embedding=None so predictor skips CLS context
        predictor_cls = student_cls if self.use_cls_token else None
        predictions = self.predictor(
            visible_embeddings,
            visible_indices,
            masked_indices,
            total_patches=N,
            attention_mask=attention_mask,
            cls_embedding=predictor_cls,
        )

        # teacher_targets is already [B, M_masked, D] - no gather needed
        # Pass masked_padding_mask to exclude batch padding from loss computation
        mcl_loss, best_idx = self.mcl_loss(
            predictions, teacher_targets, temperature=mcl_temperature, padding_mask=masked_padding_mask
        )

        # Select best MCL hypothesis per patch for DINO classification
        best_predictions = torch.gather(
            predictions,
            dim=2,
            index=best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, D),
        ).squeeze(2)  # [B, M_masked, D]

        # DINO loss: classification on per-patch masked embeddings
        student_logits = self.student_dino_head(best_predictions)
        with torch.no_grad():
            teacher_logits = self.teacher_dino_head(teacher_targets)

        # Pass masked_padding_mask to exclude batch padding from loss computation
        dino_loss = self.dino_loss(student_logits, teacher_logits, padding_mask=masked_padding_mask)

        # Update center after loss (so current batch doesn't influence its own centering)
        self.dino_loss.update_center(teacher_logits, padding_mask=masked_padding_mask)

        # CLS DINO loss: classification on global CLS embeddings
        student_cls_logits = self.student_cls_dino_head(student_cls)  # [B, D] -> [B, K_cls]
        with torch.no_grad():
            teacher_cls_logits = self.teacher_cls_dino_head(teacher_cls)

        # Compute CLS DINO loss (no padding mask needed - single embedding per sample)
        cls_dino_loss = self.cls_dino_loss(student_cls_logits, teacher_cls_logits)

        # Update CLS center after loss
        self.cls_dino_loss.update_center(teacher_cls_logits)

        return {
            "mcl_loss": mcl_loss,
            "dino_loss": dino_loss,
            "cls_dino_loss": cls_dino_loss,
            "cls_embedding": student_cls,
        }

    def _add_pos_embed(self, patches: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to patches.

        pos_embed is a learnable nn.Parameter (initialized from sincos) shared
        between student and teacher encoders, so either can be used here.
        """
        N = patches.shape[1]
        return patches + self.student_encoder.get_pos_embed(N)

    @torch.no_grad()
    def _encode_full(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
        use_teacher: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Internal method that returns all encoder outputs.

        Consolidates shared logic for encode, encode_frames, and encode_all.

        Args:
            audio: [B, T] mono audio waveform
            audio_lengths: [B] original audio lengths in samples
            use_teacher: Whether to use teacher or student (default) encoder

        Returns:
            cls_embedding: [B, D] CLS token embedding
            patch_embeddings: [B, N_patches, D] per-patch embeddings
            attention_mask: [B, N_patches] attention mask (1=valid, 0=pad)
        """
        patches, attention_mask = self.mel_frontend(audio, audio_lengths)
        patches = self._add_pos_embed(patches)

        encoder = self.teacher_encoder if use_teacher else self.student_encoder
        encoder_out, cls_embedding = encoder(patches, mask=attention_mask)

        if self.use_cls_token:
            patch_embeddings = encoder_out[:, 1:]  # Skip CLS token
        else:
            patch_embeddings = encoder_out  # No CLS token
            # Pool all patches for global embedding
            pooler = self.teacher_pooler if use_teacher else self.student_pooler
            cls_embedding = pooler(patch_embeddings, mask=attention_mask)

        return cls_embedding, patch_embeddings, attention_mask

    @torch.no_grad()
    def encode(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
        use_teacher: bool = False,
    ) -> torch.Tensor:
        """Encode audio to embedding (inference).

        Args:
            audio: [B, T] mono audio waveform
            audio_lengths: [B] original audio lengths in samples
            use_teacher: Whether to use teacher or student (default) encoder

        Returns:
            embedding: [B, D] audio embedding
        """
        cls_embedding, _, _ = self._encode_full(audio, audio_lengths, use_teacher)
        return cls_embedding

    @torch.no_grad()
    def encode_frames(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
        use_teacher: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode audio to per-patch embeddings (inference).

        Returns patch-level embeddings BEFORE pooling, useful for:
        - Training a representation autoencoder
        - Fine-grained temporal analysis
        - Frame-level conditioning

        Args:
            audio: [B, T] mono audio waveform
            audio_lengths: [B] original audio lengths in samples
            use_teacher: Whether to use teacher or student (default) encoder

        Returns:
            patch_embeddings: [B, N_patches, D] per-patch embeddings
            attention_mask: [B, N_patches] attention mask (1=valid, 0=pad)
        """
        _, patch_embeddings, attention_mask = self._encode_full(
            audio, audio_lengths, use_teacher
        )
        return patch_embeddings, attention_mask

    @torch.no_grad()
    def encode_all(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
        use_teacher: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Encode audio to both CLS and patch embeddings (inference).

        Returns all embeddings in a single encoder pass, useful for validation
        metrics that need both CLS and patch-level information.

        Args:
            audio: [B, T] mono audio waveform
            audio_lengths: [B] original audio lengths in samples
            use_teacher: Whether to use teacher or student (default) encoder

        Returns:
            cls_embedding: [B, D] CLS token embedding
            patch_embeddings: [B, N_patches, D] per-patch embeddings
            attention_mask: [B, N_patches] attention mask (1=valid, 0=pad)
        """
        return self._encode_full(audio, audio_lengths, use_teacher)
