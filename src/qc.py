import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad


# -----------------------------
# Robust helpers
# -----------------------------

def mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return np.median(np.abs(x - med))


def upper_mad_bound(x: np.ndarray, n_mads: float = 3.0) -> float:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    scale = 1.4826 * mad(x)
    if scale == 0:
        return float("inf")
    return med + n_mads * scale


def lower_mad_bound(x: np.ndarray, n_mads: float = 3.0) -> float:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    scale = 1.4826 * mad(x)
    if scale == 0:
        return float("-inf")
    return med - n_mads * scale


# -----------------------------
# QC metrics
# -----------------------------

def add_qc_metrics(adata: ad.AnnData) -> None:
    genes = pd.Index(adata.var_names.astype(str)).str.upper()
    adata.var["mt"] = genes.str.startswith("MT-")
    adata.var["ribo"] = genes.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = genes.str.match(r"^HB(?!P)")

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        inplace=True,
        percent_top=None,
        log1p=False,
    )


# -----------------------------
# Conservative cancer-aware QC
# -----------------------------

def flag_cells(
    adata: ad.AnnData,
    sample_col: str | None = None,
    min_genes: int = 200,
    max_pct_hb_flag: float = 10.0,
    mad_n: float = 3.0,
) -> pd.DataFrame:
    obs = adata.obs.copy()

    if sample_col is None or sample_col not in obs.columns:
        groups = pd.Series("all", index=obs.index)
    else:
        groups = obs[sample_col].astype(str).fillna("missing")

    flags = pd.DataFrame(index=obs.index)
    flags["fail_min_genes"] = obs["n_genes_by_counts"] < min_genes
    flags["flag_high_hb_fixed"] = obs["pct_counts_hb"] > max_pct_hb_flag
    flags["flag_low_genes_mad"] = False
    flags["flag_high_genes_mad"] = False
    flags["flag_high_counts_mad"] = False
    flags["flag_high_mt_mad"] = False

    for _, idx in groups.groupby(groups).groups.items():
        sub = obs.loc[idx]

        low_genes = lower_mad_bound(sub["n_genes_by_counts"].to_numpy(), mad_n)
        high_genes = upper_mad_bound(sub["n_genes_by_counts"].to_numpy(), mad_n)
        high_counts = upper_mad_bound(sub["total_counts"].to_numpy(), mad_n)
        high_mt = upper_mad_bound(sub["pct_counts_mt"].to_numpy(), mad_n)

        flags.loc[idx, "flag_low_genes_mad"] = sub["n_genes_by_counts"] < low_genes
        flags.loc[idx, "flag_high_genes_mad"] = sub["n_genes_by_counts"] > high_genes
        flags.loc[idx, "flag_high_counts_mad"] = sub["total_counts"] > high_counts
        flags.loc[idx, "flag_high_mt_mad"] = sub["pct_counts_mt"] > high_mt

    flags["flag_any_soft"] = flags[
        [
            "flag_high_hb_fixed",
            "flag_low_genes_mad",
            "flag_high_genes_mad",
            "flag_high_counts_mad",
            "flag_high_mt_mad",
        ]
    ].any(axis=1)

    # Conservative default: remove only obvious low-complexity failures.
    flags["keep_cell"] = ~flags["fail_min_genes"]
    return flags


# -----------------------------
# Plots
# -----------------------------

def save_basic_plots(adata: ad.AnnData, outdir: Path) -> None:
    sc.settings.figdir = str(outdir)
    sc.settings.set_figure_params(dpi=120, facecolor="white")

    sc.pl.violin(
        adata,
        ["total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_hb"],
        jitter=0.4,
        multi_panel=True,
        show=False,
        save="_qc_violin.png",
    )

    sc.pl.scatter(
        adata,
        x="total_counts",
        y="n_genes_by_counts",
        color="pct_counts_mt",
        show=False,
        save="_counts_vs_genes_mt.png",
    )


# -----------------------------
# Main
# -----------------------------

def run_qc(
    input_path: str,
    output_dir: str,
    sample_col: str | None = None,
    min_genes: int = 200,
    min_cells_per_gene: int = 3,
    max_pct_hb_flag: float = 10.0,
    mad_n: float = 3.0,
) -> None:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    adata = sc.read(input_path)
    adata.layers["counts"] = adata.X.copy()

    add_qc_metrics(adata)
    flags = flag_cells(
        adata,
        sample_col=sample_col,
        min_genes=min_genes,
        max_pct_hb_flag=max_pct_hb_flag,
        mad_n=mad_n,
    )

    adata.obs = adata.obs.join(flags)
    save_basic_plots(adata, outdir)

    # Gene filter: standard, conservative
    gene_detected = np.asarray((adata.X > 0).sum(axis=0)).ravel()
    keep_genes = gene_detected >= min_cells_per_gene

    # Remove only hard failures by default
    adata_qc = adata[adata.obs["keep_cell"].values, keep_genes].copy()

    adata.obs.to_csv(outdir / "cell_qc_metrics_and_flags.csv")
    pd.DataFrame(
        {
            "gene": adata.var_names,
            "n_cells_by_counts": gene_detected,
            "keep_gene": keep_genes,
        }
    ).to_csv(outdir / "gene_qc_metrics.csv", index=False)

    summary = pd.Series(
        {
            "cells_input": int(adata.n_obs),
            "genes_input": int(adata.n_vars),
            "cells_kept": int(adata_qc.n_obs),
            "genes_kept": int(adata_qc.n_vars),
            "n_fail_min_genes": int(adata.obs["fail_min_genes"].sum()),
            "n_flag_any_soft": int(adata.obs["flag_any_soft"].sum()),
            "sample_col": sample_col if sample_col is not None else "None",
            "min_genes": min_genes,
            "min_cells_per_gene": min_cells_per_gene,
            "max_pct_hb_flag": max_pct_hb_flag,
            "mad_n": mad_n,
        }
    )
    summary.to_csv(outdir / "qc_summary.csv")

    adata_qc.write_h5ad(outdir / "adata_qc_filtered.h5ad")

    print("QC complete")
    print(summary.to_string())
    print(f"Filtered file: {outdir / 'adata_qc_filtered.h5ad'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conservative scRNA-seq QC for cancer/plasticity projects")
    parser.add_argument("--input", required=True, help="Input .h5ad with raw counts in adata.X")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_col", default=None, help="Optional sample/donor column for per-sample MAD flags")
    parser.add_argument("--min_genes", type=int, default=200)
    parser.add_argument("--min_cells_per_gene", type=int, default=3)
    parser.add_argument("--max_pct_hb_flag", type=float, default=10.0)
    parser.add_argument("--mad_n", type=float, default=3.0)
    args = parser.parse_args()

    run_qc(
        input_path=args.input,
        output_dir=args.output_dir,
        sample_col=args.sample_col,
        min_genes=args.min_genes,
        min_cells_per_gene=args.min_cells_per_gene,
        max_pct_hb_flag=args.max_pct_hb_flag,
        mad_n=args.mad_n,
    )
