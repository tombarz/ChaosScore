from __future__ import annotations

import pandas as pd
import torch
import torch.nn.functional as F


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
