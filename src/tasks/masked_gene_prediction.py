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
    """Compute masked regression loss only on valid masked positions."""
    if loss_type == "mse":
        elementwise = F.mse_loss(predictions, targets, reduction="none")
    elif loss_type == "mae":
        elementwise = F.l1_loss(predictions, targets, reduction="none")
    elif loss_type == "huber":
        elementwise = F.huber_loss(predictions, targets, reduction="none")
    else:
        raise ValueError(f"Unsupported loss_type '{loss_type}'")
    weights = valid_mask.to(predictions.dtype)
    return (elementwise * weights).sum() / weights.sum().clamp_min(1.0)


def masked_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masked MSE and masked MAE."""
    weights = valid_mask.to(predictions.dtype)
    mse = (torch.square(predictions - targets) * weights).sum() / weights.sum().clamp_min(1.0)
    mae = (torch.abs(predictions - targets) * weights).sum() / weights.sum().clamp_min(1.0)
    return mse, mae


def build_score_frame(
    *,
    cell_ids: list[str],
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> pd.DataFrame:
    """Aggregate per-cell raw abnormality and masked error metrics."""
    per_cell_denominator = valid_mask.sum(dim=1).clamp_min(1)
    per_cell_mse = ((torch.square(predictions - targets) * valid_mask).sum(dim=1) / per_cell_denominator).cpu().numpy()
    per_cell_mae = ((torch.abs(predictions - targets) * valid_mask).sum(dim=1) / per_cell_denominator).cpu().numpy()
    per_cell_masked = per_cell_denominator.cpu().numpy()
    return pd.DataFrame(
        {
            "cell_id": cell_ids,
            "raw_abnormality": per_cell_mse,
            "masked_mae": per_cell_mae,
            "masked_gene_count": per_cell_masked,
        }
    ).set_index("cell_id")
