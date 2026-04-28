from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from src.config import get_project_paths
from src.scfoundation_utils import add_scfoundation_model_to_path, require_existing_path


@dataclass(frozen=True)
class AttentionPoolingOutput:
    """Attention pooling details for interpretation."""

    pooled: torch.Tensor
    attention_weights: torch.Tensor
    gene_ids: torch.Tensor
    padding_mask: torch.Tensor
    gathered_values: torch.Tensor


class MaskedAttentionPooler(nn.Module):
    """Pool token embeddings with trainable attention while ignoring padding."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        scorer_hidden_dim = max(1, int(hidden_dim) // 2)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, scorer_hidden_dim),
            nn.GELU(),
            nn.Linear(scorer_hidden_dim, 1),
        )

    def forward(self, encoded_tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        encoded_tokens:
            [B, L, H] token embeddings.
        padding_mask:
            [B, L] boolean mask where True marks padded tokens.
        Returns
        -------
        torch.Tensor
            [B, H] attention-pooled embeddings.
        """
        assert encoded_tokens.ndim == 3, f"Expected [B, L, H], got shape {tuple(encoded_tokens.shape)}"
        assert padding_mask.shape == encoded_tokens.shape[:2]
        if encoded_tokens.shape[1] == 0:
            return encoded_tokens.new_zeros((encoded_tokens.shape[0], encoded_tokens.shape[2]))

        weights = self.compute_attention_weights(encoded_tokens, padding_mask)
        return (encoded_tokens * weights.unsqueeze(-1)).sum(dim=1)

    def compute_attention_weights(self, encoded_tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Return normalized token weights with zero weight assigned to padding."""
        assert encoded_tokens.ndim == 3, f"Expected [B, L, H], got shape {tuple(encoded_tokens.shape)}"
        assert padding_mask.shape == encoded_tokens.shape[:2]
        if encoded_tokens.shape[1] == 0:
            return encoded_tokens.new_zeros((encoded_tokens.shape[0], 0))

        valid_mask = ~padding_mask
        has_valid_tokens = valid_mask.any(dim=1, keepdim=True)
        scores = self.scorer(encoded_tokens).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, float("-inf"))
        scores = torch.where(has_valid_tokens, scores, torch.zeros_like(scores))

        weights = torch.softmax(scores, dim=1)
        return weights * valid_mask.to(dtype=weights.dtype)


class ScFoundationEncoderBackbone(nn.Module):
    """
    Wrap the pretrained scFoundation encoder path without using the pretrained decoder.

    We intentionally keep this close to the official fine-tuning example:
    token_emb + pos_emb + encoder over gathered nonzero genes. The new biology-aware
    conditioning lives in the task head rather than in invasive backbone changes.
    """

    def __init__(
        self,
        *,
        scfoundation_repo: str | Path,
        checkpoint_path: str | Path | None = None,
        pretrained_key: str = "gene",
        freeze_encoder: bool = True,
        unfreeze_last_block: bool = False,
        unfreeze_embeddings: bool = False,
        pooling: str = "max",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.repo_path = require_existing_path(scfoundation_repo, label="scFoundation repo")
        configured_checkpoint = checkpoint_path or get_project_paths().scfoundation_checkpoint
        self.checkpoint_path = require_existing_path(configured_checkpoint, label="scFoundation checkpoint")
        self.pretrained_key = pretrained_key
        self.pooling = pooling
        self.device_obj = torch.device(device)
        self.attention_pooler: MaskedAttentionPooler | None = None

        add_scfoundation_model_to_path(self.repo_path)
        from load import gatherData, load_model_frommmf  # type: ignore

        pretrain_model, pretrain_config = load_model_frommmf(
            str(self.checkpoint_path),
            pretrained_key,
            device=self.device_obj,
        )
        self._gather_data = gatherData
        self.token_emb = pretrain_model.token_emb
        self.pos_emb = pretrain_model.pos_emb
        self.encoder = pretrain_model.encoder
        self.model_config = pretrain_config
        self.hidden_dim = int(pretrain_config["encoder"]["hidden_dim"])
        self.pad_token_id = int(pretrain_config["pad_token_id"])
        if self.pooling == "attention":
            self.attention_pooler = MaskedAttentionPooler(self.hidden_dim).to(self.device_obj)
        self._freeze_pretrained(
            freeze_encoder=freeze_encoder,
            unfreeze_last_block=unfreeze_last_block,
            unfreeze_embeddings=unfreeze_embeddings,
        )

    def _freeze_pretrained(
        self,
        *,
        freeze_encoder: bool,
        unfreeze_last_block: bool,
        unfreeze_embeddings: bool,
    ) -> None:
        for parameter in self.token_emb.parameters():
            parameter.requires_grad = bool(unfreeze_embeddings)
        for parameter in self.pos_emb.parameters():
            parameter.requires_grad = bool(unfreeze_embeddings)

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        if unfreeze_last_block:
            last_block = self.encoder.transformer_encoder[-1]
            for parameter in last_block.parameters():
                parameter.requires_grad = True

        self.backbone_has_trainable_params = any(
            parameter.requires_grad
            for module in (self.token_emb, self.pos_emb, self.encoder)
            for parameter in module.parameters()
        )

    def train(self, mode: bool = True) -> "ScFoundationEncoderBackbone":
        super().train(mode)
        if mode and not self.backbone_has_trainable_params:
            self.token_emb.eval()
            self.pos_emb.eval()
            self.encoder.eval()
        return self

    def _pool_encoded_tokens(
        self,
        encoded_tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = ~padding_mask
        if self.pooling == "max":
            masked = encoded_tokens.masked_fill(~valid_mask.unsqueeze(-1), float("-inf"))
            pooled = masked.max(dim=1).values
            pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
            return pooled
        if self.pooling == "mean":
            masked = encoded_tokens * valid_mask.unsqueeze(-1)
            denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
            return masked.sum(dim=1) / denom
        if self.pooling == "max_mean":
            masked = encoded_tokens.masked_fill(~valid_mask.unsqueeze(-1), float("-inf"))
            pooled_max = masked.max(dim=1).values
            pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
            pooled_mean = (encoded_tokens * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_mask.sum(
                dim=1, keepdim=True
            ).clamp_min(1)
            return torch.cat([pooled_max, pooled_mean], dim=-1)
        if self.pooling == "attention":
            assert self.attention_pooler is not None
            return self.attention_pooler(encoded_tokens, padding_mask)
        raise ValueError(f"Unsupported pooling mode '{self.pooling}'")

    def _encode_visible_tokens(
        self,
        x_masked: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x_masked:
            [B, G] normalized log1p counts with masked genes zeroed out.
        Returns
        -------
        tuple
            encoded_tokens [B, L, H], padding_mask [B, L], position_gene_ids [B, L],
            gathered_values [B, L].
        """
        assert x_masked.ndim == 2, f"Expected [B, G], got shape {tuple(x_masked.shape)}"
        value_labels = x_masked > 0
        gathered_values, padding_mask = self._gather_data(x_masked, value_labels, self.pad_token_id)
        gene_ids = torch.arange(x_masked.shape[1], device=x_masked.device).repeat(x_masked.shape[0], 1)
        position_gene_ids, _ = self._gather_data(gene_ids, value_labels, self.pad_token_id)

        token_embeddings = self.token_emb(gathered_values.unsqueeze(-1).float(), output_weight=0)
        position_embeddings = self.pos_emb(position_gene_ids)
        encoded_tokens = self.encoder(token_embeddings + position_embeddings, padding_mask)
        return encoded_tokens, padding_mask, position_gene_ids, gathered_values

    def attention_pooling_details(self, x_masked: torch.Tensor) -> AttentionPoolingOutput:
        """
        Return attention pooling weights and matching gene ids for interpretation.

        Only available when this backbone was constructed with pooling="attention".
        The returned weights correspond to visible/nonzero tokens after scFoundation's
        gather step, so use `gene_ids` and `padding_mask` to map them back to genes.
        """
        if self.pooling != "attention" or self.attention_pooler is None:
            raise ValueError("attention_pooling_details is only available when pooling='attention'")
        encoded_tokens, padding_mask, position_gene_ids, gathered_values = self._encode_visible_tokens(x_masked)
        attention_weights = self.attention_pooler.compute_attention_weights(encoded_tokens, padding_mask)
        pooled = (encoded_tokens * attention_weights.unsqueeze(-1)).sum(dim=1)
        return AttentionPoolingOutput(
            pooled=pooled,
            attention_weights=attention_weights,
            gene_ids=position_gene_ids,
            padding_mask=padding_mask,
            gathered_values=gathered_values,
        )

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_masked:
            [B, G] normalized log1p counts with masked genes zeroed out.
        Returns
        -------
        torch.Tensor
            [B, H] pooled cell embedding.
        """
        encoded_tokens, padding_mask, _, _ = self._encode_visible_tokens(x_masked)
        return self._pool_encoded_tokens(encoded_tokens, padding_mask)


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
