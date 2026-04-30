"""Model wrappers for scFoundation-based tasks."""

from .cell_type_classification import CellTypeClassifier
from .scfoundation_backbone import AttentionPoolingOutput, MaskedAttentionPooler, ScFoundationEncoderBackbone
from .scfoundation_conditioned_mgp import (
    CellTypeConditionedMaskedGenePredictor,
)

__all__ = [
    "AttentionPoolingOutput",
    "CellTypeClassifier",
    "CellTypeConditionedMaskedGenePredictor",
    "MaskedAttentionPooler",
    "ScFoundationEncoderBackbone",
]
