from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import FineTuneDataBundle
from src.training.config import DataConfig


def load_split_assignments(path: str | Path) -> pd.DataFrame:
    assignments = pd.read_csv(path)
    required = {"cell_id", "split"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Split assignments are missing required columns: {sorted(missing)}")
    assignments = assignments.copy()
    assignments["cell_id"] = assignments["cell_id"].astype(str)
    assignments["split"] = assignments["split"].astype(str)
    if assignments["cell_id"].duplicated().any():
        duplicated = assignments.loc[assignments["cell_id"].duplicated(), "cell_id"].head().tolist()
        raise ValueError(f"Split assignments contain duplicate cell IDs: {duplicated}")
    return assignments


def subset_bundle(bundle: FineTuneDataBundle, cell_ids: list[str]) -> FineTuneDataBundle:
    requested = pd.Index([str(cell_id) for cell_id in cell_ids])
    if requested.empty:
        raise ValueError("Cannot subset fine-tune data bundle to zero cells")
    indexer = bundle.obs.index.get_indexer(requested)
    missing = requested[indexer < 0].tolist()
    if missing:
        raise ValueError(f"Requested cell IDs are not present in prepared obs: {missing[:5]}")
    return FineTuneDataBundle(
        aligned_counts=bundle.aligned_counts[indexer, :].tocsr(),
        obs=bundle.obs.iloc[indexer].copy(),
        var=bundle.var,
        cell_type_categories=bundle.cell_type_categories,
        prepared_prefix=bundle.prepared_prefix,
        summary=bundle.summary,
        total_counts_key_used=bundle.total_counts_key_used,
    )


def cell_ids_for_split(bundle: FineTuneDataBundle, assignments: pd.DataFrame, split_name: str) -> list[str]:
    assigned_ids = set(assignments.loc[assignments["split"] == split_name, "cell_id"])
    if not assigned_ids:
        raise ValueError(f"Split '{split_name}' has no assigned cells")
    cell_ids = [str(cell_id) for cell_id in bundle.obs.index if str(cell_id) in assigned_ids]
    if not cell_ids:
        raise ValueError(f"Split '{split_name}' has no cells present in the prepared dataset")
    return cell_ids


def build_split_bundles(
    bundle: FineTuneDataBundle,
    config: DataConfig,
) -> tuple[FineTuneDataBundle, dict[str, FineTuneDataBundle], dict[str, object]]:
    if config.split_assignments is None:
        return bundle, {}, {
            "split_assignments": None,
            "train_split": None,
            "eval_splits": [],
            "split_counts": {"all": int(bundle.aligned_counts.shape[0])},
        }

    assignments = load_split_assignments(config.split_assignments)
    eval_split_names = config.eval_splits if config.eval_splits is not None else ["validation"]
    if config.train_split in eval_split_names:
        raise ValueError("--train_split must not also appear in --eval_splits")

    train_bundle = subset_bundle(bundle, cell_ids_for_split(bundle, assignments, config.train_split))
    eval_bundles = {
        split_name: subset_bundle(bundle, cell_ids_for_split(bundle, assignments, split_name))
        for split_name in eval_split_names
    }
    split_counts = {
        split_name: int((assignments["split"] == split_name).sum())
        for split_name in sorted(assignments["split"].unique())
    }
    used_counts = {
        config.train_split: int(train_bundle.aligned_counts.shape[0]),
        **{
            split_name: int(split_bundle.aligned_counts.shape[0])
            for split_name, split_bundle in eval_bundles.items()
        },
    }
    return train_bundle, eval_bundles, {
        "split_assignments": str(Path(config.split_assignments).resolve()),
        "train_split": config.train_split,
        "eval_splits": eval_split_names,
        "split_counts": split_counts,
        "used_split_counts": used_counts,
        "dropped_or_unavailable_split_counts": {
            split_name: int(split_counts.get(split_name, 0) - used_count)
            for split_name, used_count in used_counts.items()
        },
    }
