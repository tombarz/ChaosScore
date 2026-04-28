from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from src.split_lung_groups import build_healthy_mask, split_lung_groups


class LungGroupSplitTests(unittest.TestCase):
    def test_build_healthy_mask_uses_defaults_and_excludes_missing_values(self) -> None:
        obs = pd.DataFrame(
            {
                "age_or_mean_of_age_range": [29, 30, 31, 25, 28, None],
                "smoking_status": ["never", "never", "never", "former", "never", "never"],
                "BMI": [22.0, 24.9, 22.0, 22.0, 27.0, 22.0],
                "disease": ["normal", "normal", "normal", "normal", "normal", "normal"],
            },
            index=[f"cell_{i}" for i in range(6)],
        )

        mask = build_healthy_mask(
            obs,
            age_cutoff=30,
            age_column="age_or_mean_of_age_range",
            smoking_column="smoking_status",
            bmi_column="BMI",
            disease_column="disease",
            never_smoking_value="never",
            normal_disease_value="normal",
            bmi_low=18.5,
            bmi_high=25.0,
        )

        self.assertEqual(mask.tolist(), [True, True, False, False, False, False])

    def test_split_lung_groups_writes_both_subsets(self) -> None:
        tmp = Path("data") / "processed" / "test_tmp" / f"case_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        input_path = tmp / "toy.h5ad"
        output_root = tmp / "lung"

        obs = pd.DataFrame(
            {
                "age_or_mean_of_age_range": [29, 45, 27],
                "smoking_status": ["never", "active", "never"],
                "BMI": [22.0, 23.0, 26.0],
                "disease": ["normal", "normal", "normal"],
            },
            index=["healthy_cell", "smoking_cell", "high_bmi_cell"],
        )
        var = pd.DataFrame(index=["gene_1", "gene_2"])
        adata = ad.AnnData(X=np.ones((3, 2), dtype=np.float32), obs=obs, var=var)
        adata.write_h5ad(input_path)

        split_lung_groups(input_h5ad=str(input_path), output_root=output_root, age_cutoff=30)

        healthy_path = output_root / "30_non_smoking_normal_healthy" / "subset.h5ad"
        risk_path = output_root / "older_than_30_smoked_normal_risk_group" / "subset.h5ad"
        healthy_summary_path = output_root / "30_non_smoking_normal_healthy" / "summary.json"
        risk_summary_path = output_root / "older_than_30_smoked_normal_risk_group" / "summary.json"

        self.assertTrue(healthy_path.exists())
        self.assertTrue(risk_path.exists())

        healthy = ad.read_h5ad(healthy_path)
        risk = ad.read_h5ad(risk_path)
        self.assertEqual(healthy.obs_names.tolist(), ["healthy_cell"])
        self.assertEqual(risk.obs_names.tolist(), ["smoking_cell", "high_bmi_cell"])

        healthy_summary = json.loads(healthy_summary_path.read_text(encoding="utf-8"))
        risk_summary = json.loads(risk_summary_path.read_text(encoding="utf-8"))
        self.assertEqual(healthy_summary["cells"], 1)
        self.assertEqual(risk_summary["cells"], 2)
        self.assertEqual(risk_summary["selection_rule"], "complement of healthy group")

        shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
