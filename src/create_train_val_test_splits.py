from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


DEFAULT_SPLIT_FRACTIONS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}


def _validate_fractions(train_fraction: float, validation_fraction: float, test_fraction: float) -> None:
    fractions = [train_fraction, validation_fraction, test_fraction]
    if any(fraction < 0 for fraction in fractions):
        raise ValueError("Split fractions must be non-negative")
    total = sum(fractions)
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total:.6f}")
    if train_fraction <= 0:
        raise ValueError("train_fraction must be greater than 0")


def _allocate_counts(
    n_items: int,
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, int]:
    """Allocate split sizes for one stratum while keeping train non-empty."""
    if n_items <= 0:
        return {"train": 0, "validation": 0, "test": 0}
    if n_items == 1:
        return {"train": 1, "validation": 0, "test": 0}

    raw = {
        "train": n_items * train_fraction,
        "validation": n_items * validation_fraction,
        "test": n_items * test_fraction,
    }
    counts = {split: int(np.floor(value)) for split, value in raw.items()}

    remaining = n_items - sum(counts.values())
    for split in sorted(raw, key=lambda key: raw[key] - counts[key], reverse=True):
        if remaining <= 0:
            break
        counts[split] += 1
        remaining -= 1

    if counts["train"] == 0:
        donor_split = max(("validation", "test"), key=lambda split: counts[split])
        counts[donor_split] -= 1
        counts["train"] = 1

    for split, fraction in (("validation", validation_fraction), ("test", test_fraction)):
        if fraction <= 0 or counts[split] > 0 or n_items < 3:
            continue
        donor_split = "test" if split == "validation" else "validation"
        if counts[donor_split] > 1:
            counts[donor_split] -= 1
            counts[split] = 1
        elif counts["train"] > 1:
            counts["train"] -= 1
            counts[split] = 1

    assert sum(counts.values()) == n_items
    assert counts["train"] > 0
    return counts


def build_stratified_split_assignments(
    obs: pd.DataFrame,
    *,
    stratify_key: str,
    balance_key: str | None = None,
    train_fraction: float = DEFAULT_SPLIT_FRACTIONS["train"],
    validation_fraction: float = DEFAULT_SPLIT_FRACTIONS["validation"],
    test_fraction: float = DEFAULT_SPLIT_FRACTIONS["test"],
    seed: int = 0,
) -> pd.DataFrame:
    """
    Build deterministic train/validation/test assignments, stratified by obs metadata.

    Rare strata are kept in train when there is only one cell. With two cells, the
    rounded allocation may omit validation or test because every stratum must keep
    at least one training cell.
    """
    _validate_fractions(train_fraction, validation_fraction, test_fraction)
    if stratify_key not in obs.columns:
        raise ValueError(f"stratify_key '{stratify_key}' not found in obs")
    if balance_key is not None and balance_key not in obs.columns:
        raise ValueError(f"balance_key '{balance_key}' not found in obs")

    rng = np.random.default_rng(seed)
    assignments: list[pd.DataFrame] = []
    stratify_values = obs[stratify_key].astype("string").fillna("missing")
    working_data = {
        "cell_id": obs.index.astype(str),
        stratify_key: stratify_values.to_numpy(dtype=str),
    }
    group_keys = [stratify_key]
    if balance_key is not None:
        balance_values = obs[balance_key].astype("string").fillna("missing")
        working_data[balance_key] = balance_values.to_numpy(dtype=str)
        group_keys.append(balance_key)

    working = pd.DataFrame(working_data, index=obs.index)

    for _, frame in working.groupby(group_keys, sort=True, dropna=False):
        shuffled_positions = rng.permutation(frame.shape[0])
        shuffled = frame.iloc[shuffled_positions].copy()
        counts = _allocate_counts(
            shuffled.shape[0],
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
        split_values = (
            ["train"] * counts["train"]
            + ["validation"] * counts["validation"]
            + ["test"] * counts["test"]
        )
        shuffled["split"] = split_values
        assignments.append(shuffled)

    out = pd.concat(assignments, axis=0).sort_index()
    return out[["cell_id", "split", *group_keys]]


def _split_counts_by_key(assignments: pd.DataFrame, key: str) -> pd.DataFrame:
    return (
        assignments.groupby([key, "split"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["train", "validation", "test"], fill_value=0)
    )


def summarize_assignments(assignments: pd.DataFrame, *, stratify_key: str, balance_key: str | None = None) -> dict:
    split_counts = assignments["split"].value_counts().reindex(["train", "validation", "test"], fill_value=0)
    stratum_split_counts = _split_counts_by_key(assignments, stratify_key)
    summary = {
        "cells": int(assignments.shape[0]),
        "split_counts": {split: int(count) for split, count in split_counts.items()},
        "stratify_key": stratify_key,
        "strata": int(stratum_split_counts.shape[0]),
        "strata_missing_validation": int((stratum_split_counts["validation"] == 0).sum()),
        "strata_missing_test": int((stratum_split_counts["test"] == 0).sum()),
    }
    if balance_key is not None:
        balance_split_counts = _split_counts_by_key(assignments, balance_key)
        summary.update(
            {
                "balance_key": balance_key,
                "balance_groups": int(balance_split_counts.shape[0]),
                "balance_groups_missing_validation": int((balance_split_counts["validation"] == 0).sum()),
                "balance_groups_missing_test": int((balance_split_counts["test"] == 0).sum()),
            }
        )
    return summary


def create_train_val_test_splits(
    *,
    input_h5ad: str | Path,
    output_dir: str | Path,
    stratify_key: str = "ann_level_3",
    balance_key: str | None = None,
    train_fraction: float = DEFAULT_SPLIT_FRACTIONS["train"],
    validation_fraction: float = DEFAULT_SPLIT_FRACTIONS["validation"],
    test_fraction: float = DEFAULT_SPLIT_FRACTIONS["test"],
    seed: int = 0,
    group_name: str | None = None,
) -> None:
    input_path = Path(input_h5ad)
    if not input_path.exists():
        raise FileNotFoundError(f"Input H5AD file not found: {input_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(input_path, backed="r")
    try:
        obs = adata.obs.copy()
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    assignments = build_stratified_split_assignments(
        obs,
        stratify_key=stratify_key,
        balance_key=balance_key,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    if group_name is not None:
        assignments.insert(1, "group", group_name)

    assignments_path = output_path / "split_assignments.csv.gz"
    assignments.to_csv(assignments_path, index=False, compression="gzip")

    counts_by_stratum = _split_counts_by_key(assignments, stratify_key)
    counts_by_stratum.to_csv(output_path / f"split_counts_by_{stratify_key}.csv")
    if balance_key is not None:
        counts_by_balance_key = _split_counts_by_key(assignments, balance_key)
        counts_by_balance_key.to_csv(output_path / f"split_counts_by_{balance_key}.csv")

    summary = summarize_assignments(assignments, stratify_key=stratify_key, balance_key=balance_key)
    summary.update(
        {
            "input_h5ad": str(input_path.resolve()),
            "group_name": group_name,
            "seed": int(seed),
            "fractions": {
                "train": float(train_fraction),
                "validation": float(validation_fraction),
                "test": float(test_fraction),
            },
            "assignments_path": str(assignments_path.resolve()),
        }
    )
    with (output_path / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote split assignments to {assignments_path}")
    print(json.dumps(summary["split_counts"], sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/validation/test cell assignment files for an H5AD dataset."
    )
    parser.add_argument("--input_h5ad", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stratify_key", default="ann_level_3")
    parser.add_argument(
        "--balance_key",
        default=None,
        help="Optional second metadata key, such as donor_id, to split within each stratify_key/balance_key group.",
    )
    parser.add_argument("--train_fraction", type=float, default=DEFAULT_SPLIT_FRACTIONS["train"])
    parser.add_argument("--validation_fraction", type=float, default=DEFAULT_SPLIT_FRACTIONS["validation"])
    parser.add_argument("--test_fraction", type=float, default=DEFAULT_SPLIT_FRACTIONS["test"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group_name", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    create_train_val_test_splits(
        input_h5ad=args.input_h5ad,
        output_dir=args.output_dir,
        stratify_key=args.stratify_key,
        balance_key=args.balance_key,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        group_name=args.group_name,
    )


if __name__ == "__main__":
    main()
