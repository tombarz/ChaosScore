from __future__ import annotations

import torch
import torch.nn as nn

from src.models.scfoundation_backbone import ScFoundationEncoderBackbone


class CellTypeClassifier(nn.Module):
    """Classify cell type from a pooled scFoundation cell embedding."""

    def __init__(
        self,
        *,
        backbone: ScFoundationEncoderBackbone,
        num_cell_types: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_cell_types = int(num_cell_types)
        self.norm = nn.BatchNorm1d(
            backbone.output_dim,
            affine=False,
            eps=1e-6,
        )
        self.classifier = nn.Sequential(
            nn.Linear(backbone.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_cell_types),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cell_embedding = self.backbone(x)
        return self.classifier(self.norm(cell_embedding))
