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

    This is shared infrastructure for downstream tasks. Task-specific modules should
    attach their own heads to the pooled cell embedding returned by this backbone.
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

    @property
    def output_dim(self) -> int:
        if self.pooling in {"max", "mean", "attention"}:
            return self.hidden_dim
        if self.pooling == "max_mean":
            return self.hidden_dim * 2
        raise ValueError(f"Unsupported pooling mode '{self.pooling}'")

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
