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


if __name__ == "__main__":
    unittest.main()
