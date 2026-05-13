from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from src.create_train_val_test_splits import (
    build_stratified_split_assignments,
    create_train_val_test_splits,
)


class TrainValidationTestSplitTests(unittest.TestCase):
    def test_stratified_assignments_are_deterministic_and_keep_strata(self) -> None:
        obs = pd.DataFrame(
            {
                "ann_level_3": ["A"] * 10 + ["B"] * 10,
            },
            index=[f"cell_{idx}" for idx in range(20)],
        )

        first = build_stratified_split_assignments(obs, stratify_key="ann_level_3", seed=11)
        second = build_stratified_split_assignments(obs, stratify_key="ann_level_3", seed=11)

        pd.testing.assert_frame_equal(first, second)
        counts = first.groupby(["ann_level_3", "split"]).size().unstack(fill_value=0)
        self.assertEqual(counts.loc["A", "train"], 7)
        self.assertEqual(counts.loc["A", "validation"], 2)
        self.assertEqual(counts.loc["A", "test"], 1)
        self.assertEqual(counts.loc["B", "train"], 7)
        self.assertEqual(counts.loc["B", "validation"], 2)
        self.assertEqual(counts.loc["B", "test"], 1)

    def test_single_cell_stratum_stays_in_train(self) -> None:
        obs = pd.DataFrame(
            {
                "ann_level_3": ["rare", "common", "common"],
            },
            index=["rare_cell", "common_1", "common_2"],
        )

        assignments = build_stratified_split_assignments(obs, stratify_key="ann_level_3", seed=3)

        rare = assignments.loc[assignments["cell_id"] == "rare_cell"].iloc[0]
        self.assertEqual(rare["split"], "train")

    def test_balance_key_splits_within_each_donor_and_stratum(self) -> None:
        obs = pd.DataFrame(
            {
                "ann_level_3": ["A"] * 20,
                "donor_id": ["d1"] * 10 + ["d2"] * 10,
            },
            index=[f"cell_{idx}" for idx in range(20)],
        )

        assignments = build_stratified_split_assignments(
            obs,
            stratify_key="ann_level_3",
            balance_key="donor_id",
            seed=13,
        )

        counts = assignments.groupby(["donor_id", "split"]).size().unstack(fill_value=0)
        self.assertEqual(counts.loc["d1", "train"], 7)
        self.assertEqual(counts.loc["d1", "validation"], 2)
        self.assertEqual(counts.loc["d1", "test"], 1)
        self.assertEqual(counts.loc["d2", "train"], 7)
        self.assertEqual(counts.loc["d2", "validation"], 2)
        self.assertEqual(counts.loc["d2", "test"], 1)

    def test_holdout_key_keeps_each_donor_in_one_split(self) -> None:
        obs = pd.DataFrame(
            {
                "ann_level_3": ["A", "A", "B", "B"] * 6,
                "donor_id": [f"d{donor_idx}" for donor_idx in range(6) for _ in range(4)],
            },
            index=[f"cell_{idx}" for idx in range(24)],
        )

        first = build_stratified_split_assignments(
            obs,
            stratify_key="ann_level_3",
            holdout_key="donor_id",
            seed=17,
        )
        second = build_stratified_split_assignments(
            obs,
            stratify_key="ann_level_3",
            holdout_key="donor_id",
            seed=17,
        )

        pd.testing.assert_frame_equal(first, second)
        donor_split_counts = first.groupby("donor_id")["split"].nunique()
        self.assertEqual(int(donor_split_counts.max()), 1)
        self.assertEqual(set(first["split"]), {"train", "validation", "test"})

    def test_holdout_key_keeps_each_cell_type_in_train_when_feasible(self) -> None:
        donor_profiles = {
            "d0": {"A": 34},
            "d1": {"A": 28},
            "d2": {"A": 36},
            "d3": {"B": 29},
        }
        rows = []
        index = []
        for donor_id, cell_type_counts in donor_profiles.items():
            for cell_type, count in cell_type_counts.items():
                for cell_idx in range(count):
                    rows.append({"ann_level_3": cell_type, "donor_id": donor_id})
                    index.append(f"{donor_id}_{cell_type}_{cell_idx}")
        obs = pd.DataFrame(rows, index=index)

        assignments = build_stratified_split_assignments(
            obs,
            stratify_key="ann_level_3",
            holdout_key="donor_id",
            seed=246,
        )

        train_cell_types = set(assignments.loc[assignments["split"] == "train", "ann_level_3"])
        self.assertEqual(train_cell_types, {"A", "B"})
        self.assertEqual(int(assignments.groupby("donor_id")["split"].nunique().max()), 1)
        self.assertEqual(set(assignments["split"]), {"train", "validation", "test"})

    def test_balance_and_holdout_keys_are_mutually_exclusive(self) -> None:
        obs = pd.DataFrame(
            {
                "ann_level_3": ["A"] * 6,
                "donor_id": ["d1"] * 3 + ["d2"] * 3,
            },
            index=[f"cell_{idx}" for idx in range(6)],
        )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            build_stratified_split_assignments(
                obs,
                stratify_key="ann_level_3",
                balance_key="donor_id",
                holdout_key="donor_id",
            )

    def test_create_splits_writes_assignment_and_summary_files(self) -> None:
        tmp = Path("data") / "processed" / "test_tmp" / f"case_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        input_path = tmp / "toy.h5ad"
        output_dir = tmp / "splits"

        obs = pd.DataFrame(
            {
                "ann_level_3": ["A"] * 5 + ["B"] * 5,
                "donor_id": ["d1"] * 5 + ["d2"] * 5,
            },
            index=[f"cell_{idx}" for idx in range(10)],
        )
        adata = ad.AnnData(X=np.ones((10, 3), dtype=np.float32), obs=obs)
        adata.write_h5ad(input_path)

        create_train_val_test_splits(
            input_h5ad=input_path,
            output_dir=output_dir,
            stratify_key="ann_level_3",
            balance_key="donor_id",
            seed=5,
            group_name="toy_group",
        )

        assignments = pd.read_csv(output_dir / "split_assignments.csv.gz")
        with (output_dir / "split_summary.json").open("r", encoding="utf-8") as handle:
            summary = json.load(handle)

        self.assertEqual(assignments.shape[0], 10)
        self.assertIn("group", assignments.columns)
        self.assertEqual(set(assignments["group"]), {"toy_group"})
        self.assertEqual(summary["split_counts"]["train"], 6)
        self.assertEqual(summary["split_counts"]["validation"], 2)
        self.assertEqual(summary["split_counts"]["test"], 2)
        self.assertEqual(summary["balance_key"], "donor_id")
        self.assertTrue((output_dir / "split_counts_by_donor_id.csv").exists())

    def test_create_holdout_splits_writes_holdout_summary(self) -> None:
        tmp = Path("data") / "processed" / "test_tmp" / f"case_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        input_path = tmp / "toy.h5ad"
        output_dir = tmp / "splits"

        obs = pd.DataFrame(
            {
                "ann_level_3": ["A", "B"] * 6,
                "donor_id": [f"d{donor_idx}" for donor_idx in range(6) for _ in range(2)],
            },
            index=[f"cell_{idx}" for idx in range(12)],
        )
        adata = ad.AnnData(X=np.ones((12, 3), dtype=np.float32), obs=obs)
        adata.write_h5ad(input_path)

        create_train_val_test_splits(
            input_h5ad=input_path,
            output_dir=output_dir,
            stratify_key="ann_level_3",
            holdout_key="donor_id",
            seed=19,
            group_name="toy_group",
        )

        assignments = pd.read_csv(output_dir / "split_assignments.csv.gz")
        with (output_dir / "split_summary.json").open("r", encoding="utf-8") as handle:
            summary = json.load(handle)

        self.assertEqual(assignments.groupby("donor_id")["split"].nunique().max(), 1)
        self.assertEqual(summary["split_mode"], "holdout")
        self.assertEqual(summary["holdout_key"], "donor_id")
        self.assertEqual(summary["holdout_groups_in_multiple_splits"], 0)
        self.assertEqual(summary["strata_missing_train"], 0)
        self.assertTrue((output_dir / "split_counts_by_donor_id.csv").exists())


if __name__ == "__main__":
    unittest.main()
