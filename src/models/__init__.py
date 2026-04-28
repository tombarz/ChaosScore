"""Model wrappers for scFoundation-based tasks."""

from .scfoundation_conditioned_mgp import (
    AttentionPoolingOutput,
    CellTypeConditionedMaskedGenePredictor,
    MaskedAttentionPooler,
    ScFoundationEncoderBackbone,
)

__all__ = [
    "AttentionPoolingOutput",
    "CellTypeConditionedMaskedGenePredictor",
    "MaskedAttentionPooler",
    "ScFoundationEncoderBackbone",
]
