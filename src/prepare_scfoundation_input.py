from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

try:
    from scfoundation_utils import (
        align_block_to_panel,
        build_collapse_matrix,
        build_panel_alignment_matrix,
        ensure_parent_dir,
        get_gene_symbols,
        get_matrix_nnz_per_row,
        get_matrix_sum_per_row,
        load_gene_panel,
        prepared_dataset_paths,
        resolve_gene_panel_path,
        select_counts_matrix,
        summarize_integer_like_counts,
        try_write_parquet,
        write_json,
    )
except ImportError:
    from src.scfoundation_utils import (
        align_block_to_panel,
        build_collapse_matrix,
        build_panel_alignment_matrix,
        ensure_parent_dir,
        get_gene_symbols,
        get_matrix_nnz_per_row,
        get_matrix_sum_per_row,
        load_gene_panel,
        prepared_dataset_paths,
        resolve_gene_panel_path,
        select_counts_matrix,
        summarize_integer_like_counts,
        try_write_parquet,
        write_json,
    )


def prepare_scfoundation_input(
    input_h5ad: str,
    output_prefix: str,
    counts_source: str,
    gene_symbol_field: str,
    gene_panel_path: str | None = None,
    batch_size: int = 20000,
    dataset_role: str | None = None,
) -> None:
    panel_path = resolve_gene_panel_path(gene_panel_path)
    panel_df = load_gene_panel(panel_path)
    panel_symbols = pd.Index(panel_df["gene_name"])

    adata = ad.read_h5ad(input_h5ad, backed="r")
    try:
        counts_matrix, var_df, counts_source_used = select_counts_matrix(adata, counts_source)
        gene_symbols = get_gene_symbols(var_df, gene_symbol_field)
        collapse_matrix, unique_symbols = build_collapse_matrix(gene_symbols)
        align_matrix, panel_present = build_panel_alignment_matrix(unique_symbols, panel_symbols)

        aligned_blocks: list[sparse.csr_matrix] = []
        source_total_counts = np.zeros(adata.n_obs, dtype=np.float64)
        source_detected_genes = np.zeros(adata.n_obs, dtype=np.int32)
        panel_total_counts = np.zeros(adata.n_obs, dtype=np.float64)
        panel_detected_genes = np.zeros(adata.n_obs, dtype=np.int32)
        integer_like_flags: list[bool] = []

        for start in range(0, adata.n_obs, batch_size):
            stop = min(start + batch_size, adata.n_obs)
            block = counts_matrix[start:stop, :]
            integer_like_flags.append(summarize_integer_like_counts(block))
            source_total_counts[start:stop] = get_matrix_sum_per_row(block)
            source_detected_genes[start:stop] = get_matrix_nnz_per_row(block).astype(np.int32, copy=False)
            aligned = align_block_to_panel(block, collapse_matrix, align_matrix)
            panel_total_counts[start:stop] = get_matrix_sum_per_row(aligned)
            panel_detected_genes[start:stop] = get_matrix_nnz_per_row(aligned).astype(np.int32, copy=False)
            aligned_blocks.append(aligned)

        aligned_counts = sparse.vstack(aligned_blocks, format="csr")
        paths = prepared_dataset_paths(output_prefix)

        obs = adata.obs.copy()
        obs["counts_source_used"] = counts_source_used
        obs["source_total_counts_raw"] = source_total_counts
        obs["source_n_genes_by_counts"] = source_detected_genes
        obs["total_counts_raw"] = panel_total_counts
        obs["n_genes_by_counts"] = panel_detected_genes
        obs["panel_nonzero_genes"] = panel_detected_genes
        if dataset_role is not None:
            obs["dataset_role"] = dataset_role

        var = panel_df.copy()
        var["source_symbol_present"] = panel_present
        var["is_zero_padded_feature"] = ~panel_present

        gene_coverage = pd.DataFrame(
            {
                "gene_name": panel_symbols,
                "panel_index": np.arange(len(panel_symbols), dtype=np.int32),
                "source_symbol_present": panel_present,
                "is_zero_padded_feature": ~panel_present,
                "detected_cells": np.asarray(aligned_counts.getnnz(axis=0)).ravel().astype(np.int64),
            }
        )

        ensure_parent_dir(paths.counts_path)
        sparse.save_npz(paths.counts_path, aligned_counts)
        obs.to_csv(paths.obs_csv_path, compression="gzip")
        wrote_parquet = try_write_parquet(obs, paths.obs_parquet_path)
        var.to_csv(paths.var_path, index=False)
        gene_coverage.to_csv(paths.gene_coverage_path, index=False)

        summary = {
            "input_h5ad": str(Path(input_h5ad).resolve()),
            "counts_source_requested": counts_source,
            "counts_source_used": counts_source_used,
            "gene_symbol_field": gene_symbol_field,
            "cells": int(adata.n_obs),
            "source_genes": int(len(gene_symbols)),
            "source_unique_gene_symbols": int(unique_symbols.nunique()),
            "duplicate_gene_symbol_rows": int(len(gene_symbols) - unique_symbols.nunique()),
            "panel_genes": int(len(panel_symbols)),
            "panel_overlap_genes": int(panel_present.sum()),
            "panel_missing_genes": int((~panel_present).sum()),
            "counts_integer_like": bool(all(integer_like_flags)),
            "dataset_role": dataset_role,
            "wrote_obs_parquet": wrote_parquet,
            "output_counts_path": str(paths.counts_path.resolve()),
            "output_obs_csv_path": str(paths.obs_csv_path.resolve()),
            "output_var_path": str(paths.var_path.resolve()),
            "output_gene_coverage_path": str(paths.gene_coverage_path.resolve()),
            "gene_panel_path": str(panel_path),
        }
        write_json(paths.summary_path, summary)
        write_json(
            paths.manifest_path,
            {
                "counts_path": str(paths.counts_path.resolve()),
                "obs_csv_path": str(paths.obs_csv_path.resolve()),
                "obs_parquet_path": str(paths.obs_parquet_path.resolve()),
                "var_path": str(paths.var_path.resolve()),
                "gene_coverage_path": str(paths.gene_coverage_path.resolve()),
                "summary_path": str(paths.summary_path.resolve()),
            },
        )

        print("Prepared scFoundation input")
        print(f"Counts matrix: {paths.counts_path}")
        print(f"Observation metadata: {paths.obs_csv_path}")
        print(f"Summary: {paths.summary_path}")
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare raw-count single-cell data for scFoundation gene-panel alignment."
    )
    parser.add_argument("--input_h5ad", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--counts_source", choices=["raw", "X"], default="raw")
    parser.add_argument(
        "--gene_symbol_field",
        required=True,
        help="Column in var/raw.var containing HGNC-style symbols, or var_names",
    )
    parser.add_argument("--gene_panel_path", default=None)
    parser.add_argument("--batch_size", type=int, default=20000)
    parser.add_argument("--dataset_role", choices=["healthy", "malignant", "mixed"], default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    prepare_scfoundation_input(
        input_h5ad=args.input_h5ad,
        output_prefix=args.output_prefix,
        counts_source=args.counts_source,
        gene_symbol_field=args.gene_symbol_field,
        gene_panel_path=args.gene_panel_path,
        batch_size=args.batch_size,
        dataset_role=args.dataset_role,
    )


if __name__ == "__main__":
    main()
