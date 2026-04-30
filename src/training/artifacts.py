from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from src.data import FineTuneDataBundle
from src.scfoundation_utils import ensure_parent_dir, write_json


def write_feature_metadata(save_dir: Path, train_bundle: FineTuneDataBundle) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    feature_metadata_path = save_dir / "feature_metadata.csv"
    train_bundle.var.to_csv(feature_metadata_path, index=False)
    return feature_metadata_path


def save_run_artifacts(
    *,
    save_dir: Path,
    train_bundle: FineTuneDataBundle,
    epoch_metrics: list[dict[str, float]],
    model: torch.nn.Module,
    summary_metadata: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
) -> None:
    feature_metadata_path = write_feature_metadata(save_dir, train_bundle)
    summary = {
        **dict(summary_metadata),
        "epoch_metrics": epoch_metrics,
    }
    write_json(save_dir / "train_metrics.json", summary)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            **dict(model_metadata),
            "feature_metadata_path": str(feature_metadata_path.resolve()),
        },
        ensure_parent_dir(save_dir / "model.pt"),
    )
