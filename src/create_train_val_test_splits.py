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
SPLIT_ORDER = ("train", "validation", "test")


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


def _fraction_dict(
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, float]:
    return {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }


def _split_quality_score(
    split_profiles: np.ndarray,
    *,
    target_profiles: np.ndarray,
    required_nonempty: set[int],
    required_positive_profiles: dict[int, np.ndarray] | None = None,
) -> float:
    split_totals = split_profiles.sum(axis=1)
    target_totals = target_profiles.sum(axis=1)
    total_error = np.square(
        (split_totals - target_totals) / np.maximum(target_totals, 1.0)
    ).sum()
    profile_error = np.square(
        (split_profiles - target_profiles) / np.maximum(target_profiles, 1.0)
    ).mean()
    empty_penalty = 1000.0 * sum(split_totals[split_idx] == 0 for split_idx in required_nonempty)
    coverage_penalty = 0.0
    if required_positive_profiles is not None:
        for split_idx, required_mask in required_positive_profiles.items():
            coverage_penalty += 1000.0 * int((split_profiles[split_idx, required_mask] <= 0).sum())
    return float(total_error + profile_error + empty_penalty + coverage_penalty)


def _profiles_for_group_assignment(profile_values: np.ndarray, assignment_indices: np.ndarray) -> np.ndarray:
    split_profiles = np.zeros((len(SPLIT_ORDER), profile_values.shape[1]), dtype=float)
    for split_idx in range(len(SPLIT_ORDER)):
        group_mask = assignment_indices == split_idx
        if group_mask.any():
            split_profiles[split_idx] = profile_values[group_mask].sum(axis=0)
    return split_profiles


def _find_train_coverage_groups_exact(
    profile_values: np.ndarray,
    *,
    required_train_mask: np.ndarray,
    max_train_groups: int,
) -> set[int] | None:
    required_columns = np.flatnonzero(required_train_mask)
    if len(required_columns) == 0:
        return set()

    group_masks: list[int] = []
    for group_profile in profile_values:
        mask = 0
        for bit_idx, column_idx in enumerate(required_columns):
            if group_profile[column_idx] > 0:
                mask |= 1 << bit_idx
        group_masks.append(mask)

    full_mask = (1 << len(required_columns)) - 1
    groups_by_bit: dict[int, list[int]] = {bit_idx: [] for bit_idx in range(len(required_columns))}
    group_sizes = profile_values.sum(axis=1)
    for group_idx, mask in enumerate(group_masks):
        for bit_idx in range(len(required_columns)):
            if mask & (1 << bit_idx):
                groups_by_bit[bit_idx].append(group_idx)
    for bit_idx, group_indices in groups_by_bit.items():
        groups_by_bit[bit_idx] = sorted(
            group_indices,
            key=lambda group_idx: (
                group_masks[group_idx].bit_count(),
                float(group_sizes[group_idx]),
                -group_idx,
            ),
            reverse=True,
        )

    memo: set[tuple[int, int]] = set()

    def search(covered_mask: int, selected: tuple[int, ...]) -> tuple[int, ...] | None:
        if covered_mask == full_mask:
            return selected
        if len(selected) >= max_train_groups:
            return None

        remaining_mask = full_mask & ~covered_mask
        max_gain = max((group_mask & remaining_mask).bit_count() for group_mask in group_masks)
        if max_gain == 0:
            return None
        remaining_bits = remaining_mask.bit_count()
        remaining_slots = max_train_groups - len(selected)
        if int(np.ceil(remaining_bits / max_gain)) > remaining_slots:
            return None

        memo_key = (covered_mask, len(selected))
        if memo_key in memo:
            return None

        uncovered_bits = [bit_idx for bit_idx in range(len(required_columns)) if remaining_mask & (1 << bit_idx)]
        next_bit = min(
            uncovered_bits,
            key=lambda bit_idx: sum((group_masks[group_idx] & remaining_mask) != 0 for group_idx in groups_by_bit[bit_idx]),
        )
        candidates = [
            group_idx
            for group_idx in groups_by_bit[next_bit]
            if group_masks[group_idx] & remaining_mask
        ]
        candidates = sorted(
            candidates,
            key=lambda group_idx: (
                (group_masks[group_idx] & remaining_mask).bit_count(),
                float(profile_values[group_idx, required_columns].sum()),
                -group_idx,
            ),
            reverse=True,
        )
        for group_idx in candidates:
            result = search(covered_mask | group_masks[group_idx], (*selected, group_idx))
            if result is not None:
                return result

        memo.add(memo_key)
        return None

    result = search(0, ())
    if result is None:
        return None
    return set(result)


def _select_train_coverage_groups(
    profile_values: np.ndarray,
    *,
    required_nonempty: set[int],
) -> set[int]:
    train_idx = SPLIT_ORDER.index("train")
    required_train_mask = profile_values.sum(axis=0) > 0
    if not required_train_mask.any():
        return set()

    nontrain_required_splits = required_nonempty - {train_idx}
    max_train_groups = profile_values.shape[0] - len(nontrain_required_splits)
    if max_train_groups <= 0:
        return set()

    selected: set[int] = set()
    uncovered_mask = required_train_mask.copy()
    group_sizes = profile_values.sum(axis=1)
    while uncovered_mask.any() and len(selected) < max_train_groups:
        best_group_idx: int | None = None
        best_key: tuple[int, float, float] | None = None
        for group_idx, group_profile in enumerate(profile_values):
            if group_idx in selected:
                continue
            covered_mask = uncovered_mask & (group_profile > 0)
            covered_strata = int(covered_mask.sum())
            if covered_strata == 0:
                continue
            covered_cells = float(group_profile[uncovered_mask].sum())
            key = (covered_strata, covered_cells, float(group_sizes[group_idx]))
            if best_key is None or key > best_key:
                best_key = key
                best_group_idx = group_idx
        if best_group_idx is None:
            break
        selected.add(best_group_idx)
        uncovered_mask &= profile_values[best_group_idx] <= 0

    if uncovered_mask.any():
        exact_selection = _find_train_coverage_groups_exact(
            profile_values,
            required_train_mask=required_train_mask,
            max_train_groups=max_train_groups,
        )
        if exact_selection is None:
            return set()
        return exact_selection
    return selected


def _assign_holdout_groups_to_splits(
    group_profiles: pd.DataFrame,
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    fractions = _fraction_dict(
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    positive_splits = [split for split in SPLIT_ORDER if fractions[split] > 0]
    if group_profiles.shape[0] < len(positive_splits):
        raise ValueError(
            f"Need at least {len(positive_splits)} holdout groups to populate requested splits, "
            f"got {group_profiles.shape[0]}"
        )

    profile_values = group_profiles.to_numpy(dtype=float)
    group_names = group_profiles.index.astype(str).to_numpy()
    target_profiles = np.vstack(
        [profile_values.sum(axis=0) * fractions[split] for split in SPLIT_ORDER]
    )
    required_nonempty = {idx for idx, split in enumerate(SPLIT_ORDER) if fractions[split] > 0}
    train_idx = SPLIT_ORDER.index("train")
    required_positive_profiles = {train_idx: profile_values.sum(axis=0) > 0}
    mandatory_train_group_indices = _select_train_coverage_groups(
        profile_values,
        required_nonempty=required_nonempty,
    )

    rng = np.random.default_rng(seed)
    best_assignment: dict[str, str] | None = None
    best_score = np.inf
    attempts = max(64, min(512, group_profiles.shape[0] * 8))
    group_sizes = profile_values.sum(axis=1)

    for attempt_idx in range(attempts):
        if attempt_idx == 0:
            order = np.argsort(-group_sizes, kind="stable")
        else:
            # Prioritize larger donors while allowing deterministic seed-controlled alternatives.
            jitter = rng.uniform(0.9, 1.1, size=group_profiles.shape[0])
            order = np.argsort(-(group_sizes * jitter), kind="stable")
        order = np.array(
            [group_idx for group_idx in order if group_idx not in mandatory_train_group_indices],
            dtype=int,
        )

        split_profiles = np.zeros_like(target_profiles)
        split_to_groups: dict[str, list[str]] = {split: [] for split in SPLIT_ORDER}
        assignment_indices = np.full(group_profiles.shape[0], -1, dtype=int)

        for group_idx in sorted(mandatory_train_group_indices):
            split_profiles[train_idx] += profile_values[group_idx]
            split_to_groups["train"].append(group_names[group_idx])
            assignment_indices[group_idx] = train_idx

        for position, group_idx in enumerate(order):
            group_profile = profile_values[group_idx]
            remaining_groups = len(order) - position - 1
            missing_required = [
                split_idx
                for split_idx in required_nonempty
                if not split_to_groups[SPLIT_ORDER[split_idx]]
            ]
            if missing_required and remaining_groups < len(missing_required):
                candidate_split_indices = missing_required
            else:
                candidate_split_indices = [
                    idx for idx, split in enumerate(SPLIT_ORDER) if fractions[split] > 0
                ]

            candidate_scores = []
            for split_idx in candidate_split_indices:
                proposed_profiles = split_profiles.copy()
                proposed_profiles[split_idx] += group_profile
                score = _split_quality_score(
                    proposed_profiles,
                    target_profiles=target_profiles,
                    required_nonempty=set(),
                    required_positive_profiles=required_positive_profiles,
                )
                candidate_scores.append((score, split_idx))

            _, selected_split_idx = min(
                candidate_scores,
                key=lambda item: (item[0], item[1]),
            )
            selected_split = SPLIT_ORDER[selected_split_idx]
            split_profiles[selected_split_idx] += group_profile
            split_to_groups[selected_split].append(group_names[group_idx])
            assignment_indices[group_idx] = selected_split_idx

        if (assignment_indices < 0).any():
            raise RuntimeError("Some holdout groups were not assigned to a split")
        split_profiles = _profiles_for_group_assignment(profile_values, assignment_indices)
        score = _split_quality_score(
            split_profiles,
            target_profiles=target_profiles,
            required_nonempty=required_nonempty,
            required_positive_profiles=required_positive_profiles,
        )
        if score < best_score:
            best_score = score
            best_assignment = {
                group_names[group_idx]: SPLIT_ORDER[split_idx]
                for group_idx, split_idx in enumerate(assignment_indices)
            }

    if best_assignment is None:
        raise RuntimeError("Failed to assign holdout groups to splits")
    return best_assignment


def build_stratified_split_assignments(
    obs: pd.DataFrame,
    *,
    stratify_key: str,
    balance_key: str | None = None,
    holdout_key: str | None = None,
    train_fraction: float = DEFAULT_SPLIT_FRACTIONS["train"],
    validation_fraction: float = DEFAULT_SPLIT_FRACTIONS["validation"],
    test_fraction: float = DEFAULT_SPLIT_FRACTIONS["test"],
    seed: int = 0,
) -> pd.DataFrame:
    """
    Build deterministic train/validation/test assignments, stratified by obs metadata.

    When holdout_key is provided, every value of that key is assigned entirely to
    one split while the assignment tries to preserve stratify_key cell counts. If
    donor-level constraints allow it, holdout mode keeps every stratum represented
    in train. In cell-level mode, rare strata are kept in train when there is only
    one cell. With two cells, the rounded allocation may omit validation or test
    because every stratum must keep at least one training cell.
    """
    _validate_fractions(train_fraction, validation_fraction, test_fraction)
    if balance_key is not None and holdout_key is not None:
        raise ValueError("balance_key and holdout_key are mutually exclusive")
    if stratify_key not in obs.columns:
        raise ValueError(f"stratify_key '{stratify_key}' not found in obs")
    if balance_key is not None and balance_key not in obs.columns:
        raise ValueError(f"balance_key '{balance_key}' not found in obs")
    if holdout_key is not None and holdout_key not in obs.columns:
        raise ValueError(f"holdout_key '{holdout_key}' not found in obs")

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
    if holdout_key is not None:
        holdout_values = obs[holdout_key].astype("string").fillna("missing")
        working_data[holdout_key] = holdout_values.to_numpy(dtype=str)

    working = pd.DataFrame(working_data, index=obs.index)

    if holdout_key is not None:
        group_profiles = pd.crosstab(working[holdout_key], working[stratify_key]).sort_index()
        group_to_split = _assign_holdout_groups_to_splits(
            group_profiles,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
        out = working.copy()
        out["split"] = out[holdout_key].map(group_to_split)
        if out["split"].isna().any():
            raise RuntimeError("Some holdout groups were not assigned to a split")
        return out[["cell_id", "split", stratify_key, holdout_key]]

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


def summarize_assignments(
    assignments: pd.DataFrame,
    *,
    stratify_key: str,
    balance_key: str | None = None,
    holdout_key: str | None = None,
) -> dict:
    split_counts = assignments["split"].value_counts().reindex(["train", "validation", "test"], fill_value=0)
    stratum_split_counts = _split_counts_by_key(assignments, stratify_key)
    summary = {
        "cells": int(assignments.shape[0]),
        "split_mode": "holdout" if holdout_key is not None else "cell",
        "split_counts": {split: int(count) for split, count in split_counts.items()},
        "stratify_key": stratify_key,
        "strata": int(stratum_split_counts.shape[0]),
        "strata_missing_train": int((stratum_split_counts["train"] == 0).sum()),
        "strata_missing_validation": int((stratum_split_counts["validation"] == 0).sum()),
        "strata_missing_test": int((stratum_split_counts["test"] == 0).sum()),
    }
    if balance_key is not None:
        balance_split_counts = _split_counts_by_key(assignments, balance_key)
        summary.update(
            {
                "balance_key": balance_key,
                "balance_groups": int(balance_split_counts.shape[0]),
                "balance_groups_missing_train": int((balance_split_counts["train"] == 0).sum()),
                "balance_groups_missing_validation": int((balance_split_counts["validation"] == 0).sum()),
                "balance_groups_missing_test": int((balance_split_counts["test"] == 0).sum()),
            }
        )
    if holdout_key is not None:
        holdout_split_counts = _split_counts_by_key(assignments, holdout_key)
        holdout_group_splits = assignments.groupby(holdout_key, observed=False)["split"].nunique()
        holdout_groups_by_split = (
            assignments[[holdout_key, "split"]]
            .drop_duplicates()
            .groupby("split", observed=False)
            .size()
            .reindex(["train", "validation", "test"], fill_value=0)
        )
        summary.update(
            {
                "holdout_key": holdout_key,
                "holdout_groups": int(holdout_split_counts.shape[0]),
                "holdout_groups_by_split": {
                    split: int(count) for split, count in holdout_groups_by_split.items()
                },
                "holdout_groups_in_multiple_splits": int((holdout_group_splits > 1).sum()),
            }
        )
    return summary


def create_train_val_test_splits(
    *,
    input_h5ad: str | Path,
    output_dir: str | Path,
    stratify_key: str = "ann_level_3",
    balance_key: str | None = None,
    holdout_key: str | None = None,
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
        holdout_key=holdout_key,
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
    if holdout_key is not None:
        counts_by_holdout_key = _split_counts_by_key(assignments, holdout_key)
        counts_by_holdout_key.to_csv(output_path / f"split_counts_by_{holdout_key}.csv")

    summary = summarize_assignments(
        assignments,
        stratify_key=stratify_key,
        balance_key=balance_key,
        holdout_key=holdout_key,
    )
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
    parser.add_argument(
        "--holdout_key",
        default=None,
        help="Optional metadata key, such as donor_id, whose values must be held out as whole groups.",
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
        holdout_key=args.holdout_key,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        group_name=args.group_name,
    )


if __name__ == "__main__":
    main()
