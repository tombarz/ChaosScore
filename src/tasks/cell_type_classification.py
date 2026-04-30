from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data import CellTypeClassificationCollator, FineTuneDataBundle, ScFoundationAlignedDataset
from src.models import CellTypeClassifier, ScFoundationEncoderBackbone
from src.training.trainer import TrainingTask


@dataclass(frozen=True)
class CellTypeClassificationConfig:
    freeze_encoder: bool = True
    unfreeze_last_block: bool = False
    unfreeze_embeddings: bool = False
    pooling: str = "max"
    head_hidden: int = 256
    dropout: float = 0.0
    scfoundation_repo: str = ""
    checkpoint_path: str | None = None
    device: str = "cpu"


class CellTypeClassificationTask(TrainingTask):
    task_name = "cell_type_classification"

    def __init__(self, config: CellTypeClassificationConfig) -> None:
        self.config = config

    def build_model(self, bundle: FineTuneDataBundle) -> CellTypeClassifier:
        backbone = ScFoundationEncoderBackbone(
            scfoundation_repo=self.config.scfoundation_repo,
            checkpoint_path=self.config.checkpoint_path,
            freeze_encoder=self.config.freeze_encoder,
            unfreeze_last_block=self.config.unfreeze_last_block,
            unfreeze_embeddings=self.config.unfreeze_embeddings,
            pooling=self.config.pooling,
            device=self.config.device,
        )
        return CellTypeClassifier(
            backbone=backbone,
            num_cell_types=len(bundle.cell_type_categories),
            hidden_dim=self.config.head_hidden,
            dropout=self.config.dropout,
        )

    def build_dataset(self, bundle: FineTuneDataBundle) -> Dataset:
        return ScFoundationAlignedDataset(bundle)

    def build_collator(self, dataset: Dataset, *, phase: str) -> CellTypeClassificationCollator:
        del phase
        if not isinstance(dataset, ScFoundationAlignedDataset):
            raise TypeError("CellTypeClassificationTask expects ScFoundationAlignedDataset instances")
        return CellTypeClassificationCollator()

    def compute_loss_and_metrics(
        self,
        model: torch.nn.Module,
        batch: dict[str, object],
        *,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        del device
        logits = model(batch["x"])  # type: ignore[arg-type]
        targets = batch["cell_type_ids"]  # type: ignore[assignment]
        loss = F.cross_entropy(logits, targets)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == targets).to(dtype=logits.dtype).mean()
        return {
            "loss": loss,
            "accuracy": accuracy,
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
            "cell_type_categories": train_bundle.cell_type_categories,
            "num_genes": int(train_bundle.aligned_counts.shape[1]),
            "prepared_prefix": str(train_bundle.prepared_prefix),
            "split_metadata": split_metadata,
        }
