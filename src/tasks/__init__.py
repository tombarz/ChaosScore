"""Shared task utilities for scFoundation downstream tasks."""

from .cell_type_classification import CellTypeClassificationConfig, CellTypeClassificationTask
from .masked_gene_prediction import (
    LOSS_REDUCTION,
    MaskedGenePredictionConfig,
    MaskedGenePredictionTask,
    build_score_frame,
    masked_metrics,
    masked_regression_loss,
)

__all__ = [
    "CellTypeClassificationConfig",
    "CellTypeClassificationTask",
    "LOSS_REDUCTION",
    "MaskedGenePredictionConfig",
    "MaskedGenePredictionTask",
    "build_score_frame",
    "masked_metrics",
    "masked_regression_loss",
]
