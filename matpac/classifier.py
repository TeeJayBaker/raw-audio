"""DINO-style Classification Head for MATPAC++.

Implements the prototypical classification approach from DINO:
- 3-layer MLP projector: 768 -> 2048 -> 2048 -> 256 -> 65536
- Centering buffer (EMA of teacher logits mean)
- Weight normalization on the last layer
- Teacher uses sharpened softmax, student uses standard softmax
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    """DINO-style classification head.

    3-layer MLP projector followed by a normalized linear layer
    that projects to K pseudo-classes.

    Args:
        in_dim: Input dimension (default: 768)
        hidden_dim: Hidden dimension (default: 2048)
        bottleneck_dim: Bottleneck dimension before classifier (default: 256)
        num_classes: Number of pseudo-classes (default: 65536)
        use_weight_norm: Whether to use weight normalization (default: True)
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        num_classes: int = 65536,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes

        # 3-layer MLP: in_dim -> hidden_dim -> hidden_dim -> bottleneck_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )

        # Final classifier layer with optional weight normalization
        self.classifier = nn.Linear(bottleneck_dim, num_classes, bias=False)

        if use_weight_norm:
            self.classifier = nn.utils.parametrizations.weight_norm(self.classifier)
            # Initialize weight_g to 1 (per original DINO implementation)
            self.classifier.parametrizations.weight.original0.data.fill_(1)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize classifier weights
        if hasattr(self.classifier, 'parametrizations'):
            # New parametrizations weight norm case - use original0
            nn.init.trunc_normal_(self.classifier.parametrizations.weight.original0, std=0.02)
        elif hasattr(self.classifier, 'weight_v'):
            # Legacy weight norm case
            nn.init.trunc_normal_(self.classifier.weight_v, std=0.02)
        else:
            nn.init.trunc_normal_(self.classifier.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] or [B, T, D] input embeddings

        Returns:
            logits: [B, K] or [B, T, K] classification logits
        """
        # Handle both pooled [B, D] and sequence [B, T, D] inputs
        input_shape = x.shape
        if x.dim() == 3:
            B, T, D = x.shape
            x = x.reshape(B * T, D)

        x = self.mlp(x)
        x = F.normalize(x, dim=-1)  # L2 normalize before classifier
        x = self.classifier(x)

        if len(input_shape) == 3:
            x = x.reshape(B, T, -1)

        return x


class DINOLoss(nn.Module):
    """DINO classification loss with centering and sharpening.

    The teacher outputs are centered (subtract running mean) and sharpened
    (lower temperature) to prevent collapse. The student uses standard softmax.

    Loss is cross-entropy between:
    - Student: softmax(logits / student_temp)
    - Teacher: softmax((logits - center) / teacher_temp)

    Args:
        num_classes: Number of pseudo-classes (default: 65536)
        student_temp: Student softmax temperature (default: 0.1)
        teacher_temp: Teacher softmax temperature, lower = sharper (default: 0.04)
        center_momentum: EMA momentum for centering (default: 0.9)
    """

    def __init__(
        self,
        num_classes: int = 65536,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum

        # Center buffer (EMA of teacher logits mean)
        self.register_buffer("center", torch.zeros(1, num_classes))

    @torch.no_grad()
    def update_center(
        self,
        teacher_logits: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ):
        """Update center with EMA of teacher logits mean.

        Args:
            teacher_logits: [B, K] or [B, T, K] teacher classification logits
            padding_mask: [B] or [B, T] mask for valid positions (1=valid, 0=batch padding)
                         If provided, only valid positions contribute to the center
        """
        # Flatten if sequence
        if teacher_logits.dim() == 3:
            B, T, K = teacher_logits.shape
            teacher_logits = teacher_logits.reshape(-1, K)
            if padding_mask is not None:
                padding_mask = padding_mask.reshape(-1)

        if padding_mask is not None:
            # Compute weighted mean, only counting valid positions
            mask = padding_mask.unsqueeze(-1)  # [N, 1]
            masked_logits = teacher_logits * mask
            num_valid = padding_mask.sum().clamp(min=1)
            batch_center = masked_logits.sum(dim=0, keepdim=True) / num_valid
        else:
            batch_center = teacher_logits.mean(dim=0, keepdim=True)

        # Sync across GPUs if distributed
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(batch_center, op=torch.distributed.ReduceOp.AVG)

        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute DINO loss.

        Args:
            student_logits: [B, K] or [B, T, K] student classification logits
            teacher_logits: [B, K] or [B, T, K] teacher classification logits
            padding_mask: [B] or [B, T] mask for valid positions (1=valid, 0=batch padding)
                         If provided, loss is computed only over valid positions

        Returns:
            loss: Scalar cross-entropy loss
        """
        # Flatten if sequence
        if student_logits.dim() == 3:
            B, T, K = student_logits.shape
            student_logits = student_logits.reshape(-1, K)
            teacher_logits = teacher_logits.reshape(-1, K)
            if padding_mask is not None:
                padding_mask = padding_mask.reshape(-1)

        # Teacher: centered and sharpened (detached)
        teacher_probs = F.softmax(
            (teacher_logits.detach() - self.center) / self.teacher_temp,
            dim=-1
        )

        # Student: standard softmax (higher temp = softer)
        student_log_probs = F.log_softmax(
            student_logits / self.student_temp,
            dim=-1
        )

        # Cross-entropy: -sum(p * log(q))
        per_position_loss = -torch.sum(teacher_probs * student_log_probs, dim=-1)

        if padding_mask is not None:
            # Zero out padded positions and average only over valid positions
            per_position_loss = per_position_loss * padding_mask
            num_valid = padding_mask.sum().clamp(min=1)
            loss = per_position_loss.sum() / num_valid
        else:
            loss = per_position_loss.mean()

        return loss


class DINOClassifier(nn.Module):
    """Combined DINO classification head with loss computation.

    This module contains both student and teacher heads (which share architecture
    but the teacher is an EMA copy of the student, handled externally).

    Args:
        in_dim: Input dimension (default: 768)
        hidden_dim: Hidden dimension (default: 2048)
        bottleneck_dim: Bottleneck dimension (default: 256)
        num_classes: Number of pseudo-classes (default: 65536)
        student_temp: Student softmax temperature (default: 0.1)
        teacher_temp: Teacher softmax temperature (default: 0.04)
        center_momentum: EMA momentum for centering (default: 0.9)
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        num_classes: int = 65536,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ):
        super().__init__()

        # Classification head (shared architecture, but we'll have separate
        # student/teacher instances at the model level)
        self.head = DINOHead(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            num_classes=num_classes,
        )

        # Loss function
        self.loss_fn = DINOLoss(
            num_classes=num_classes,
            student_temp=student_temp,
            teacher_temp=teacher_temp,
            center_momentum=center_momentum,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get classification logits.

        Args:
            x: [B, D] or [B, T, D] input embeddings

        Returns:
            logits: [B, K] or [B, T, K] classification logits
        """
        return self.head(x)

    def compute_loss(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
        teacher_head: nn.Module,
        update_center: bool = True,
    ) -> torch.Tensor:
        """Compute DINO loss between student and teacher.

        Args:
            student_embeddings: [B, D] student encoder output
            teacher_embeddings: [B, D] teacher encoder output
            teacher_head: Teacher's classification head (EMA copy)
            update_center: Whether to update the centering buffer

        Returns:
            loss: Scalar loss value
        """
        student_logits = self.head(student_embeddings)
        with torch.no_grad():
            teacher_logits = teacher_head(teacher_embeddings)

            if update_center:
                self.loss_fn.update_center(teacher_logits)

        return self.loss_fn(student_logits, teacher_logits)
