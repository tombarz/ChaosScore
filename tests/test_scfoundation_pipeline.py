from __future__ import annotations

import unittest
import uuid
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from src.prepare_scfoundation_input import prepare_scfoundation_input
from src.score_scfoundation_abnormality import build_mask_matrix


class ScFoundationPreparationTests(unittest.TestCase):
    def test_prepare_uses_raw_counts_and_aligns_panel(self) -> None:
        tmp = Path("data") / "processed" / "test_tmp" / f"case_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        input_path = tmp / "toy.h5ad"
        panel_path = tmp / "panel.tsv"
        output_prefix = tmp / "prepared" / "toy"

        counts = sparse.csr_matrix(
            np.array(
                [
                    [5, 1, 0, 2],
                    [0, 4, 0, 3],
                ],
                dtype=np.float32,
            )
        )
        normalized_x = sparse.csr_matrix(
            np.array(
                [
                    [0.1, 0.2, 0.0, 0.3],
                    [0.0, 0.5, 0.0, 0.7],
                ],
                dtype=np.float32,
            )
        )
        obs = pd.DataFrame(index=["cell_1", "cell_2"])
        var = pd.DataFrame(
            {"feature_name": ["A2M", "A2M", "MISSING", "A1BG"]},
            index=["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
        )
        adata = ad.AnnData(X=normalized_x, obs=obs.copy(), var=var.copy())
        adata.raw = ad.AnnData(X=counts, obs=obs.copy(), var=var.copy())
        adata.write_h5ad(input_path)

        pd.DataFrame({"gene_name": ["A1BG", "A2M", "C12orf71"]}).to_csv(panel_path, sep="\t", index=False)

        prepare_scfoundation_input(
            input_h5ad=str(input_path),
            output_prefix=str(output_prefix),
            counts_source="raw",
            gene_symbol_field="feature_name",
            gene_panel_path=str(panel_path),
            batch_size=1,
            dataset_role="healthy",
        )

        prepared_counts = sparse.load_npz(output_prefix.with_suffix(".counts_19264.npz")).toarray()
        obs_out = pd.read_csv(output_prefix.with_suffix(".obs.csv.gz"), index_col=0)
        var_out = pd.read_csv(output_prefix.with_suffix(".var.csv"))

        expected = np.array(
            [
                [2.0, 6.0, 0.0],
                [3.0, 4.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(prepared_counts, expected)
        np.testing.assert_allclose(obs_out["source_total_counts_raw"].to_numpy(), np.array([8.0, 7.0]))
        np.testing.assert_allclose(obs_out["total_counts_raw"].to_numpy(), np.array([8.0, 7.0]))
        self.assertEqual(var_out["is_zero_padded_feature"].tolist(), [False, False, True])
        self.assertEqual(var_out["gene_name"].tolist(), ["A1BG", "A2M", "C12orf71"])

    def test_mask_excludes_zero_padded_features(self) -> None:
        counts = np.array(
            [
                [4.0, 0.0, 1.0, 2.0],
                [0.0, 3.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        zero_padded = np.array([False, True, False, False], dtype=bool)
        mask = build_mask_matrix(
            counts_batch=counts,
            zero_padded_features=zero_padded,
            mask_fraction=0.5,
            rng=np.random.default_rng(0),
        )
        self.assertFalse(mask[:, 1].any())
        self.assertTrue(mask[0].any())
        self.assertTrue(mask[1, 3] or not mask[1].any())


if __name__ == "__main__":
    unittest.main()
