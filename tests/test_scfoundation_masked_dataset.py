from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from src.data.scfoundation_masked_dataset import (
    MaskedGenePredictionCollator,
    build_random_mask,
    load_finetune_data_bundle,
)
from src.scfoundation_utils import prepared_dataset_paths
from src.tasks import build_score_frame, masked_metrics, masked_regression_loss


class BuildRandomMaskTests(unittest.TestCase):
    def test_mask_only_uses_nonzero_non_padded_genes(self) -> None:
        values = np.array([0.0, 2.0, 3.0, 0.0, 1.0], dtype=np.float32)
        zero_padded = np.array([False, False, True, False, False], dtype=bool)
        rng = np.random.default_rng(0)
        mask = build_random_mask(values, zero_padded_features=zero_padded, mask_ratio=0.5, rng=rng)

        self.assertEqual(mask.dtype, np.bool_)
        self.assertFalse(mask[0])
        self.assertFalse(mask[2])
        self.assertFalse(mask[3])
        self.assertTrue(mask.sum() >= 1)


class MaskedGenePredictionCollatorTests(unittest.TestCase):
    def _sample_batch(self) -> list[dict[str, object]]:
        return [
            {
                "cell_index": 0,
                "cell_id": "cell_0",
                "normalized_counts": np.arange(1, 21, dtype=np.float32),
                "cell_type_id": 1,
                "cell_type_label": "type_a",
                "depth_feature": 3.0,
                "total_counts": 100.0,
                "dataset_value": "ds1",
                "batch_value": "b1",
            }
        ]

    def test_collator_builds_padded_mask_targets(self) -> None:
        collator = MaskedGenePredictionCollator(
            zero_padded_features=np.array([False, False, False, True], dtype=bool),
            mask_ratio=0.5,
            mask_seed=7,
        )
        batch = [
            {
                "cell_index": 0,
                "cell_id": "cell_0",
                "normalized_counts": np.array([1.0, 2.0, 0.0, 0.0], dtype=np.float32),
                "cell_type_id": 1,
                "cell_type_label": "type_a",
                "depth_feature": 3.0,
                "total_counts": 100.0,
                "dataset_value": "ds1",
                "batch_value": "b1",
            },
            {
                "cell_index": 1,
                "cell_id": "cell_1",
                "normalized_counts": np.array([0.0, 1.0, 3.0, 0.0], dtype=np.float32),
                "cell_type_id": 0,
                "cell_type_label": "type_b",
                "depth_feature": 2.0,
                "total_counts": 80.0,
                "dataset_value": "ds2",
                "batch_value": "b2",
            },
        ]

        out = collator(batch)
        self.assertEqual(tuple(out["x_masked"].shape), (2, 4))
        self.assertEqual(tuple(out["masked_gene_ids"].shape)[0], 2)
        self.assertEqual(tuple(out["masked_target_values"].shape), tuple(out["masked_gene_ids"].shape))
        self.assertEqual(tuple(out["masked_positions_valid"].shape), tuple(out["masked_gene_ids"].shape))
        self.assertTrue(out["masked_positions_valid"].any().item())
        self.assertTrue((out["x_masked"] <= out["x_target"]).all().item())

    def test_seeded_collator_changes_masks_by_epoch_and_repeats_when_reset(self) -> None:
        collator = MaskedGenePredictionCollator(
            zero_padded_features=np.zeros(20, dtype=bool),
            mask_ratio=0.3,
            mask_seed=7,
        )
        batch = self._sample_batch()

        collator.set_epoch(0)
        epoch_0_first = collator(batch)["gene_mask"].clone()
        collator.set_epoch(1)
        epoch_1 = collator(batch)["gene_mask"].clone()
        collator.set_epoch(0)
        epoch_0_second = collator(batch)["gene_mask"].clone()

        self.assertTrue(torch.equal(epoch_0_first, epoch_0_second))
        self.assertFalse(torch.equal(epoch_0_first, epoch_1))

    def test_seeded_collator_without_epoch_change_is_stable_for_scoring(self) -> None:
        collator = MaskedGenePredictionCollator(
            zero_padded_features=np.zeros(20, dtype=bool),
            mask_ratio=0.3,
            mask_seed=7,
        )
        batch = self._sample_batch()

        first = collator(batch)["gene_mask"]
        second = collator(batch)["gene_mask"]

        self.assertTrue(torch.equal(first, second))


class PreparedFineTuneBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.prefix = self.tmp_path / "prepared" / "toy"
        self.paths = prepared_dataset_paths(self.prefix)

        counts = sparse.lil_matrix((2, 19264), dtype=np.float32)
        counts[0, 0] = 5.0
        counts[0, 2] = 3.0
        counts[1, 1] = 7.0
        counts[1, 4] = 1.0
        self.paths.counts_path.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(self.paths.counts_path, counts.tocsr())

        obs = pd.DataFrame(
            {
                "cell_type": ["AT2", "AT1"],
                "total_counts_raw": [8.0, 8.0],
                "dataset": ["d1", "d1"],
                "batch": ["b1", "b2"],
            },
            index=["cell_0", "cell_1"],
        )
        obs.to_csv(self.paths.obs_csv_path, compression="gzip")

        var = pd.DataFrame(
            {
                "gene_name": [f"G{i}" for i in range(19264)],
                "panel_index": np.arange(19264, dtype=np.int32),
                "is_zero_padded_feature": np.array([False, False, False, True] + [False] * (19264 - 4), dtype=bool),
            }
        )
        var.to_csv(self.paths.var_path, index=False)

        with self.paths.summary_path.open("w", encoding="utf-8") as handle:
            json.dump({"counts_source_used": "raw", "counts_integer_like": True}, handle)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_finetune_bundle_from_prepared_prefix(self) -> None:
        bundle = load_finetune_data_bundle(
            prepared_prefix=self.prefix,
            cell_type_key="cell_type",
            batch_key="batch",
        )

        self.assertEqual(bundle.aligned_counts.shape, (2, 19264))
        self.assertEqual(bundle.total_counts_key_used, "total_counts_raw")
        self.assertEqual(bundle.cell_type_categories, ["AT1", "AT2"])
        self.assertAlmostEqual(float(bundle.obs.loc["cell_0", "depth_feature"]), np.log10(9.0), places=6)
        self.assertTrue("is_zero_padded_feature" in bundle.var.columns)

    def test_load_finetune_bundle_rejects_wrong_gene_count(self) -> None:
        sparse.save_npz(self.paths.counts_path, sparse.csr_matrix((2, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "Expected 19264 aligned genes"):
            load_finetune_data_bundle(prepared_prefix=self.prefix, cell_type_key="cell_type")

    def test_load_finetune_bundle_rejects_non_integer_like_counts(self) -> None:
        with self.paths.summary_path.open("w", encoding="utf-8") as handle:
            json.dump({"counts_source_used": "X", "counts_integer_like": False}, handle)

        with self.assertRaisesRegex(ValueError, "raw/count-like integer values"):
            load_finetune_data_bundle(prepared_prefix=self.prefix, cell_type_key="cell_type")


class MaskedTaskUtilityTests(unittest.TestCase):
    def test_masked_loss_metrics_and_score_frame(self) -> None:
        predictions = torch.tensor([[1.0, 3.0], [2.0, 4.0]], dtype=torch.float32)
        targets = torch.tensor([[1.0, 1.0], [1.0, 5.0]], dtype=torch.float32)
        valid_mask = torch.tensor([[True, False], [True, True]])

        loss = masked_regression_loss(predictions, targets, valid_mask, loss_type="mse")
        mse, mae = masked_metrics(predictions, targets, valid_mask)
        frame = build_score_frame(
            cell_ids=["c1", "c2"],
            predictions=predictions,
            targets=targets,
            valid_mask=valid_mask,
        )

        self.assertAlmostEqual(float(loss), 0.5, places=6)
        self.assertAlmostEqual(float(mse), 0.5, places=6)
        self.assertAlmostEqual(float(mae), 0.5, places=6)
        self.assertEqual(frame.loc["c1", "masked_gene_count"], 1)
        self.assertAlmostEqual(frame.loc["c2", "raw_abnormality"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
