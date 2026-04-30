from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data import FineTuneDataBundle, MaskedGenePredictionCollator, ScFoundationAlignedDataset
from src.models import CellTypeConditionedMaskedGenePredictor, ScFoundationEncoderBackbone
from src.training.trainer import TrainingTask


LOSS_REDUCTION = "mean_masked_genes_per_cell_then_mean_cells"


@dataclass(frozen=True)
class MaskedGenePredictionConfig:
    mask_ratio: float = 0.30
    loss_type: str = "huber"
    freeze_encoder: bool = True
    unfreeze_last_block: bool = False
    unfreeze_embeddings: bool = False
    use_depth_covariate: bool = True
    pooling: str = "max"
    d_type: int = 64
    d_depth: int = 16
    d_gene: int = 64
    head_hidden: int = 256
    dropout: float = 0.1
    scfoundation_repo: str = ""
    checkpoint_path: str | None = None
    device: str = "cpu"
    seed: int = 0


def masked_regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    loss_type: str,
) -> torch.Tensor:
    """Compute masked regression loss with each cell weighted equally."""
    if loss_type == "mse":
        elementwise = F.mse_loss(predictions, targets, reduction="none")
    elif loss_type == "mae":
        elementwise = F.l1_loss(predictions, targets, reduction="none")
    elif loss_type == "huber":
        elementwise = F.huber_loss(predictions, targets, reduction="none")
    else:
        raise ValueError(f"Unsupported loss_type '{loss_type}'")
    weights = valid_mask.to(predictions.dtype)
    per_cell_counts = weights.sum(dim=1)
    per_cell_loss = (elementwise * weights).sum(dim=1) / per_cell_counts.clamp_min(1.0)
    cell_weights = (per_cell_counts > 0).to(predictions.dtype)
    return (per_cell_loss * cell_weights).sum() / cell_weights.sum().clamp_min(1.0)


def masked_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-cell-averaged masked MSE and masked MAE."""
    weights = valid_mask.to(predictions.dtype)
    per_cell_counts = weights.sum(dim=1)
    cell_weights = (per_cell_counts > 0).to(predictions.dtype)
    per_cell_mse = (torch.square(predictions - targets) * weights).sum(dim=1) / per_cell_counts.clamp_min(1.0)
    per_cell_mae = (torch.abs(predictions - targets) * weights).sum(dim=1) / per_cell_counts.clamp_min(1.0)
    mse = (per_cell_mse * cell_weights).sum() / cell_weights.sum().clamp_min(1.0)
    mae = (per_cell_mae * cell_weights).sum() / cell_weights.sum().clamp_min(1.0)
    return mse, mae


def build_score_frame(
    *,
    cell_ids: list[str],
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> pd.DataFrame:
    """Aggregate per-cell raw abnormality and masked error metrics."""
    per_cell_masked = valid_mask.sum(dim=1)
    per_cell_denominator = per_cell_masked.clamp_min(1)
    per_cell_mse = ((torch.square(predictions - targets) * valid_mask).sum(dim=1) / per_cell_denominator).cpu().numpy()
    per_cell_mae = ((torch.abs(predictions - targets) * valid_mask).sum(dim=1) / per_cell_denominator).cpu().numpy()
    per_cell_masked = per_cell_masked.cpu().numpy()
    return pd.DataFrame(
        {
            "cell_id": cell_ids,
            "raw_abnormality": per_cell_mse,
            "masked_mae": per_cell_mae,
            "masked_gene_count": per_cell_masked,
        }
    ).set_index("cell_id")


class MaskedGenePredictionTask(TrainingTask):
    task_name = "masked_gene_reconstruction"
    loss_reduction = LOSS_REDUCTION

    def __init__(self, config: MaskedGenePredictionConfig) -> None:
        self.config = config
        self.train_collator: MaskedGenePredictionCollator | None = None

    def build_model(self, bundle: FineTuneDataBundle) -> CellTypeConditionedMaskedGenePredictor:
        backbone = ScFoundationEncoderBackbone(
            scfoundation_repo=self.config.scfoundation_repo,
            checkpoint_path=self.config.checkpoint_path,
            freeze_encoder=self.config.freeze_encoder,
            unfreeze_last_block=self.config.unfreeze_last_block,
            unfreeze_embeddings=self.config.unfreeze_embeddings,
            pooling=self.config.pooling,
            device=self.config.device,
        )
        return CellTypeConditionedMaskedGenePredictor(
            backbone=backbone,
            num_cell_types=len(bundle.cell_type_categories),
            num_genes=bundle.aligned_counts.shape[1],
            d_type=self.config.d_type,
            d_depth=self.config.d_depth,
            d_gene=self.config.d_gene,
            hidden_dim=self.config.head_hidden,
            dropout=self.config.dropout,
            use_depth_covariate=self.config.use_depth_covariate,
        )

    def build_dataset(self, bundle: FineTuneDataBundle) -> Dataset:
        return ScFoundationAlignedDataset(bundle)

    def build_collator(self, dataset: Dataset, *, phase: str) -> MaskedGenePredictionCollator:
        if not isinstance(dataset, ScFoundationAlignedDataset):
            raise TypeError("MaskedGenePredictionTask expects ScFoundationAlignedDataset instances")
        collator = MaskedGenePredictionCollator(
            zero_padded_features=dataset.zero_padded_features,
            mask_ratio=self.config.mask_ratio,
            mask_seed=self.config.seed,
        )
        if phase == "train":
            self.train_collator = collator
        return collator

    def on_train_epoch_start(self, epoch_idx: int) -> None:
        if self.train_collator is not None:
            self.train_collator.set_epoch(epoch_idx)

    def compute_loss_and_metrics(
        self,
        model: torch.nn.Module,
        batch: dict[str, object],
        *,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        del device
        predictions = model(
            x_masked=batch["x_masked"],  # type: ignore[arg-type]
            masked_gene_ids=batch["masked_gene_ids"],  # type: ignore[arg-type]
            masked_positions_valid=batch["masked_positions_valid"],  # type: ignore[arg-type]
            cell_type_ids=batch["cell_type_ids"],  # type: ignore[arg-type]
            depth_features=batch["depth_features"],  # type: ignore[arg-type]
        )
        loss = masked_regression_loss(
            predictions,
            batch["masked_target_values"],  # type: ignore[arg-type]
            batch["masked_positions_valid"],  # type: ignore[arg-type]
            loss_type=self.config.loss_type,
        )
        mse, mae = masked_metrics(
            predictions,
            batch["masked_target_values"],  # type: ignore[arg-type]
            batch["masked_positions_valid"],  # type: ignore[arg-type]
        )
        return {
            "loss": loss,
            "masked_mse": mse,
            "masked_mae": mae,
        }

    def checkpoint_static_metadata(
        self,
        *,
        args_dict: dict[str, Any],
        train_bundle: FineTuneDataBundle,
        model: CellTypeConditionedMaskedGenePredictor,
        feature_metadata_path: Path,
        split_metadata: dict[str, object],
    ) -> dict[str, Any]:
        return {
            "args": args_dict,
            "task_name": self.task_name,
            "loss_reduction": self.loss_reduction,
            "cell_type_categories": train_bundle.cell_type_categories,
            "num_genes": int(train_bundle.aligned_counts.shape[1]),
            "feature_metadata_path": str(feature_metadata_path.resolve()),
            "prepared_prefix": str(train_bundle.prepared_prefix),
            "split_metadata": split_metadata,
            "checkpoint_path": str(model.backbone.checkpoint_path),
            "scfoundation_repo": str(model.backbone.repo_path),
        }

    def validate_checkpoint(self, payload: dict[str, Any], train_bundle: FineTuneDataBundle) -> None:
        checkpoint_num_genes = payload.get("num_genes")
        if checkpoint_num_genes is not None and int(checkpoint_num_genes) != int(train_bundle.aligned_counts.shape[1]):
            raise ValueError(
                f"Checkpoint num_genes={checkpoint_num_genes} does not match current data "
                f"num_genes={train_bundle.aligned_counts.shape[1]}"
            )
        checkpoint_categories = payload.get("cell_type_categories")
        if checkpoint_categories is not None and list(checkpoint_categories) != train_bundle.cell_type_categories:
            raise ValueError("Checkpoint cell_type_categories do not match current data")

    def summary_metadata(
        self,
        *,
        args_dict: dict[str, Any],
        train_bundle: FineTuneDataBundle,
        model: CellTypeConditionedMaskedGenePredictor,
        split_metadata: dict[str, object],
        progress_log_path: Path | None,
        checkpoint_dir: Path,
    ) -> dict[str, Any]:
        return {
            "prepared_prefix": str(train_bundle.prepared_prefix),
            "prepared_summary": train_bundle.summary,
            "split_metadata": split_metadata,
            "cell_type_key": args_dict["cell_type_key"],
            "total_counts_key": train_bundle.total_counts_key_used,
            "batch_key": args_dict.get("batch_key"),
            "num_cells": int(train_bundle.aligned_counts.shape[0]),
            "num_genes": int(train_bundle.aligned_counts.shape[1]),
            "num_cell_types": int(len(train_bundle.cell_type_categories)),
            "task_name": self.task_name,
            "loss_reduction": self.loss_reduction,
            "mask_ratio": float(self.config.mask_ratio),
            "loss_type": self.config.loss_type,
            "epochs": int(args_dict["epochs"]),
            "batch_size": int(args_dict["batch_size"]),
            "log_every_batches": int(args_dict["log_every_batches"]),
            "checkpoint_every_batches": int(args_dict["checkpoint_every_batches"]),
            "save_epoch_checkpoints": bool(args_dict["save_epoch_checkpoints"]),
            "resume_from_checkpoint": args_dict.get("resume_from_checkpoint"),
            "lr": float(args_dict["lr"]),
            "freeze_encoder": bool(self.config.freeze_encoder),
            "unfreeze_last_block": bool(self.config.unfreeze_last_block),
            "unfreeze_embeddings": bool(self.config.unfreeze_embeddings),
            "use_depth_covariate": bool(self.config.use_depth_covariate),
            "pooling": self.config.pooling,
            "cell_type_categories": train_bundle.cell_type_categories,
            "checkpoint_path": str(model.backbone.checkpoint_path),
            "scfoundation_repo": str(model.backbone.repo_path),
            "progress_log_path": str(progress_log_path.resolve()) if progress_log_path is not None else None,
            "checkpoint_dir": str(checkpoint_dir.resolve()),
        }

    def final_model_metadata(
        self,
        *,
        args_dict: dict[str, Any],
        train_bundle: FineTuneDataBundle,
        split_metadata: dict[str, object],
    ) -> dict[str, Any]:
        return {
            "args": args_dict,
            "task_name": self.task_name,
            "loss_reduction": self.loss_reduction,
            "cell_type_categories": train_bundle.cell_type_categories,
            "num_genes": int(train_bundle.aligned_counts.shape[1]),
            "prepared_prefix": str(train_bundle.prepared_prefix),
            "split_metadata": split_metadata,
        }
