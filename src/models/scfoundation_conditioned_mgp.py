from __future__ import annotations

import torch
from torch import nn

from src.models.scfoundation_backbone import AttentionPoolingOutput, MaskedAttentionPooler, ScFoundationEncoderBackbone


class CellTypeConditionedMaskedGenePredictor(nn.Module):
    """
    Predict masked normalized gene values from a pooled scFoundation cell embedding.

    Design rationale:
    - keep the pretrained backbone close to the official encoder integration path
    - use a lightweight task head instead of reusing/modifying the pretrained decoder
    - inject cell type as an explicit biological prior
    - inject sequencing depth as a small head-level technical covariate
    - keep the effect of depth inspectable for later abnormality-vs-depth analysis
    """

    def __init__(
        self,
        *,
        backbone: ScFoundationEncoderBackbone,
        num_cell_types: int,
        num_genes: int,
        d_type: int = 64,
        d_depth: int = 16,
        d_gene: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_depth_covariate: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_genes = int(num_genes)
        self.use_depth_covariate = use_depth_covariate

        self.cell_type_embedding = nn.Embedding(num_cell_types, d_type)
        self.gene_embedding = nn.Embedding(num_genes + 1, d_gene, padding_idx=num_genes)
        if use_depth_covariate:
            self.depth_projector = nn.Sequential(
                nn.Linear(1, d_depth),
                nn.GELU(),
                nn.Linear(d_depth, d_depth),
            )
            depth_dim = d_depth
        else:
            self.depth_projector = None
            depth_dim = 0

        cell_dim = getattr(backbone, "output_dim", None)
        if cell_dim is None:
            if backbone.pooling in {"max", "mean", "attention"}:
                cell_dim = backbone.hidden_dim
            elif backbone.pooling == "max_mean":
                cell_dim = backbone.hidden_dim * 2
            else:
                raise ValueError(f"Unsupported pooling mode '{backbone.pooling}'")
        head_input_dim = cell_dim + d_type + depth_dim + d_gene
        self.prediction_head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def attention_pooling_details(self, x_masked: torch.Tensor) -> AttentionPoolingOutput:
        """Return backbone attention weights for visible genes when using attention pooling."""
        return self.backbone.attention_pooling_details(x_masked)

    def forward(
        self,
        *,
        x_masked: torch.Tensor,
        masked_gene_ids: torch.Tensor,
        masked_positions_valid: torch.Tensor,
        cell_type_ids: torch.Tensor,
        depth_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Shapes
        ------
        x_masked: [B, G]
        masked_gene_ids: [B, M]
        masked_positions_valid: [B, M]
        cell_type_ids: [B]
        depth_features: [B]
        returns: [B, M]
        """
        assert x_masked.ndim == 2
        assert masked_gene_ids.ndim == 2
        assert masked_positions_valid.shape == masked_gene_ids.shape
        batch_size, num_masked = masked_gene_ids.shape

        z_cell = self.backbone(x_masked)  # [B, H]
        e_type = self.cell_type_embedding(cell_type_ids)  # [B, T]
        if self.use_depth_covariate:
            e_depth = self.depth_projector(depth_features.unsqueeze(-1))  # [B, D]
        else:
            e_depth = None

        e_gene = self.gene_embedding(masked_gene_ids)  # [B, M, Gd]
        z_cell_expanded = z_cell.unsqueeze(1).expand(batch_size, num_masked, z_cell.shape[-1])
        e_type_expanded = e_type.unsqueeze(1).expand(batch_size, num_masked, e_type.shape[-1])
        pieces = [z_cell_expanded, e_type_expanded]
        if e_depth is not None:
            e_depth_expanded = e_depth.unsqueeze(1).expand(batch_size, num_masked, e_depth.shape[-1])
            pieces.append(e_depth_expanded)
        pieces.append(e_gene)

        head_input = torch.cat(pieces, dim=-1)  # [B, M, *]
        predictions = self.prediction_head(head_input).squeeze(-1)  # [B, M]
        predictions = predictions.masked_fill(~masked_positions_valid, 0.0)
        return predictions
