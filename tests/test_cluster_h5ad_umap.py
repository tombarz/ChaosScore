from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from src.cluster_h5ad_umap import cluster_h5ad_umap


class ClusterH5adUmapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path("data") / "processed" / "test_tmp" / f"case_{uuid.uuid4().hex}"
        self.tmp.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_toy_h5ad(self) -> Path:
        rng = np.random.default_rng(7)
        x = rng.poisson(lam=3.0, size=(30, 12)).astype(np.float32)
        obs = pd.DataFrame(
            {
                "donor_id": ["donor_a"] * 10 + ["donor_b"] * 10 + ["donor_c"] * 10,
                "cell_type": ["type_1", "type_2", "type_3"] * 10,
            },
            index=[f"cell_{idx}" for idx in range(30)],
        )
        var = pd.DataFrame(index=[f"gene_{idx}" for idx in range(12)])
        adata = ad.AnnData(X=x, obs=obs, var=var)
        input_path = self.tmp / "toy.h5ad"
        adata.write_h5ad(input_path)
        return input_path

    def test_cluster_h5ad_umap_writes_plot_summary_and_annotated_h5ad(self) -> None:
        input_path = self.write_toy_h5ad()
        output_dir = self.tmp / "umap"

        cluster_h5ad_umap(
            input_path=str(input_path),
            output_dir=str(output_dir),
            color="donor_id",
            layer=None,
            use_raw=False,
            normalize=True,
            target_sum=1e4,
            log1p=True,
            n_top_genes=None,
            n_pcs=8,
            n_neighbors=5,
            neighbors_transformer="sklearn",
            cluster_method="kmeans",
            cluster_key="cluster",
            resolution=1.0,
            kmeans_clusters=4,
            random_state=0,
            point_size=20,
            write_h5ad=True,
        )

        self.assertTrue((output_dir / "umap_donor_id.png").exists())
        self.assertTrue((output_dir / "umap_cluster.png").exists())
        self.assertTrue((output_dir / "adata_clustered_umap.h5ad").exists())
        self.assertTrue((output_dir / "cluster_umap_summary.json").exists())

        clustered = ad.read_h5ad(output_dir / "adata_clustered_umap.h5ad")
        self.assertIn("cluster", clustered.obs.columns)
        self.assertIn("X_umap", clustered.obsm)
        self.assertEqual(clustered.obsm["X_umap"].shape, (30, 2))

        summary = json.loads((output_dir / "cluster_umap_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["color"], "donor_id")
        self.assertEqual(summary["cluster_method"], "kmeans")
        self.assertEqual(summary["kmeans_clusters_used"], 4)
        self.assertIn("donor_id", summary["plot_paths"])
        self.assertIn("cluster", summary["plot_paths"])

    def test_cluster_h5ad_umap_rejects_missing_color_column(self) -> None:
        input_path = self.write_toy_h5ad()

        with self.assertRaisesRegex(ValueError, "Column 'missing_column' not found"):
            cluster_h5ad_umap(
                input_path=str(input_path),
                output_dir=str(self.tmp / "umap"),
                color="missing_column",
                layer=None,
                use_raw=False,
                normalize=True,
                target_sum=1e4,
                log1p=True,
                n_top_genes=None,
                n_pcs=8,
                n_neighbors=5,
                neighbors_transformer="sklearn",
                cluster_method="kmeans",
                cluster_key="cluster",
                resolution=1.0,
                kmeans_clusters=4,
                random_state=0,
                point_size=None,
                write_h5ad=False,
            )


if __name__ == "__main__":
    unittest.main()
