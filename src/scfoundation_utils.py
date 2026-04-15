from __future__ import annotations
"""Shared sparse utilities for scFoundation preparation and downstream tasks."""

import json
import sys
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from src.config import get_project_paths


def require_existing_path(path: str | Path, *, label: str) -> Path:
    """Resolve a configured path and fail fast if it does not exist."""
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found at {resolved}")
    return resolved


@dataclass(frozen=True)
class PreparedDatasetPaths:
    prefix: Path
    counts_path: Path
    obs_csv_path: Path
    var_path: Path
    summary_path: Path
    manifest_path: Path


def prepared_dataset_paths(output_prefix: str | Path) -> PreparedDatasetPaths:
    """Expand a prepared dataset prefix into the canonical file bundle paths."""
    prefix = Path(output_prefix)
    return PreparedDatasetPaths(
        prefix=prefix,
        counts_path=prefix.with_suffix(".counts_19264.npz"),
        obs_csv_path=prefix.with_suffix(".obs.csv.gz"),
        var_path=prefix.with_suffix(".var.csv"),
        summary_path=prefix.with_suffix(".summary.json"),
        manifest_path=prefix.with_suffix(".manifest.json"),
    )


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a path if needed and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_scfoundation_repo(repo_path: str | None = None) -> Path:
    """Return the configured vendored scFoundation repository path."""
    candidate = Path(repo_path) if repo_path is not None else get_project_paths().scfoundation_repo
    return require_existing_path(candidate, label="scFoundation repo")


def resolve_gene_panel_path(
    panel_path: str | None = None,
) -> Path:
    """Return the configured scFoundation gene panel path."""
    candidate = Path(panel_path) if panel_path is not None else get_project_paths().scfoundation_gene_panel
    return require_existing_path(candidate, label="scFoundation gene panel")


def load_gene_panel(panel_path: str | Path) -> pd.DataFrame:
    """Load the official scFoundation panel and attach an explicit panel index."""
    panel_df = pd.read_csv(panel_path, sep="\t")
    if "gene_name" not in panel_df.columns:
        raise ValueError(f"Expected gene_name column in {panel_path}")
    panel_df = panel_df.copy()
    panel_df["gene_name"] = sanitize_gene_symbols(panel_df["gene_name"])
    panel_df["panel_index"] = np.arange(panel_df.shape[0], dtype=np.int32)
    return panel_df


def sanitize_gene_symbols(symbols: Iterable[object]) -> pd.Index:
    """Normalize gene symbols with minimal whitespace cleanup only."""
    series = pd.Series(pd.Index(symbols).astype(str), dtype="string")
    return pd.Index(series.str.strip())


def select_counts_matrix(
    adata: ad.AnnData,
    counts_source: str,
):
    """Select the requested count matrix and matching var frame from an AnnData object."""
    if counts_source == "raw":
        if adata.raw is None:
            raise ValueError("counts_source=raw requested but adata.raw is missing")
        return adata.raw.X, adata.raw.var.copy(), "raw"
    if counts_source != "X":
        raise ValueError("counts_source must be one of {'raw', 'X'}")
    return adata.X, adata.var.copy(), "X"


def get_gene_symbols(var: pd.DataFrame, gene_symbol_field: str) -> pd.Index:
    """Extract gene symbols from var/var_names and sanitize them."""
    if gene_symbol_field == "var_names":
        return sanitize_gene_symbols(var.index)
    if gene_symbol_field not in var.columns:
        raise ValueError(f"Gene symbol field '{gene_symbol_field}' not found in var columns")
    return sanitize_gene_symbols(var[gene_symbol_field])


def as_csr_matrix(matrix) -> sparse.csr_matrix:
    """Convert a dense or sparse matrix to CSR format."""
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(np.asarray(matrix))


def get_matrix_nnz_per_row(matrix) -> np.ndarray:
    """Return the number of nonzero values in each row."""
    if sparse.issparse(matrix):
        return np.asarray(matrix.getnnz(axis=1)).ravel()
    return np.asarray((np.asarray(matrix) > 0).sum(axis=1)).ravel()


def get_matrix_sum_per_row(matrix) -> np.ndarray:
    """Return the sum of values in each row."""
    if sparse.issparse(matrix):
        return np.asarray(matrix.sum(axis=1)).ravel()
    return np.asarray(np.asarray(matrix).sum(axis=1)).ravel()


def build_collapse_matrix(symbols: pd.Index) -> tuple[sparse.csr_matrix, pd.Index]:
    """
    Build a sparse column-remapping matrix that collapses duplicate gene symbols.

    Multiplying a cell-by-gene matrix by this matrix sums duplicate source columns into
    one column per unique gene symbol while preserving first-seen order.
    """
    codes, unique_symbols = pd.factorize(symbols, sort=False)
    if len(unique_symbols) == len(symbols):
        identity = sparse.identity(len(symbols), format="csr", dtype=np.float32)
        return identity, pd.Index(unique_symbols)
    rows = np.arange(len(symbols), dtype=np.int32)
    cols = codes.astype(np.int32, copy=False)
    data = np.ones(len(symbols), dtype=np.float32)
    collapse = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(len(symbols), len(unique_symbols)),
        dtype=np.float32,
    ).tocsr()
    return collapse, pd.Index(unique_symbols)


def build_panel_alignment_matrix(
    unique_symbols: pd.Index,
    panel_symbols: pd.Index,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Build a sparse remapping matrix from unique source genes into panel order.

    The boolean mask marks which panel genes were present in the source data. Missing
    panel genes become all-zero columns after alignment.
    """
    symbol_to_source = {symbol: idx for idx, symbol in enumerate(unique_symbols)}
    source_rows: list[int] = []
    target_cols: list[int] = []
    present = np.zeros(len(panel_symbols), dtype=bool)
    for panel_idx, symbol in enumerate(panel_symbols):
        source_idx = symbol_to_source.get(symbol)
        if source_idx is None:
            continue
        source_rows.append(source_idx)
        target_cols.append(panel_idx)
        present[panel_idx] = True
    data = np.ones(len(source_rows), dtype=np.float32)
    align = sparse.coo_matrix(
        (data, (source_rows, target_cols)),
        shape=(len(unique_symbols), len(panel_symbols)),
        dtype=np.float32,
    ).tocsr()
    return align, present


def align_block_to_panel(
    block,
    collapse_matrix: sparse.csr_matrix,
    align_matrix: sparse.csr_matrix,
) -> sparse.csr_matrix:
    """
    Apply duplicate-collapse and panel alignment to one block of cells.

    This is the sparse, batch-safe equivalent of upstream scFoundation preprocessing:
    collapse duplicate genes first, then reorder/zero-fill into the fixed panel space.
    """
    block_csr = as_csr_matrix(block).astype(np.float32, copy=False)
    collapsed = block_csr @ collapse_matrix
    aligned = collapsed @ align_matrix
    return aligned.tocsr()


def summarize_integer_like_counts(matrix) -> bool:
    """Return True when all stored nonzero values are effectively integers."""
    block = as_csr_matrix(matrix)
    if block.nnz == 0:
        return True
    return bool(np.allclose(block.data, np.round(block.data)))


def write_json(path: str | Path, payload: dict) -> None:
    """Write a JSON payload with stable formatting."""
    path = ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_prepared_dataset(prefix: str | Path) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame, dict]:
    """Load the canonical prepared scFoundation dataset bundle from a prefix."""
    paths = prepared_dataset_paths(prefix)
    counts = sparse.load_npz(paths.counts_path).tocsr()
    obs = pd.read_csv(paths.obs_csv_path, index_col=0)
    var = pd.read_csv(paths.var_path)
    with paths.summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return counts, obs, var, summary


def try_write_parquet(df: pd.DataFrame, path: str | Path) -> bool:
    """Best-effort parquet write; return False when parquet support is unavailable."""
    try:
        ensure_parent_dir(path)
        df.to_parquet(path)
        return True
    except Exception:
        return False


def add_scfoundation_model_to_path(repo_path: str | Path) -> Path:
    """Add the vendored scFoundation model directory to sys.path for imports."""
    repo = require_existing_path(repo_path, label="scFoundation repo")
    model_dir = repo / "model"
    require_existing_path(model_dir, label="scFoundation model directory")
    model_dir_str = str(model_dir)
    if model_dir_str not in sys.path:
        sys.path.insert(0, model_dir_str)
    return model_dir


def load_scfoundation_model(
    repo_path: str | Path,
    ckpt_path: str | Path,
    key: str = "gene",
):
    """Load a pretrained scFoundation model and the encoder/decoder helper function."""
    ckpt_path = require_existing_path(ckpt_path, label="scFoundation checkpoint")
    add_scfoundation_model_to_path(repo_path)
    from load import getEncoerDecoderData, load_model_frommmf  # type: ignore

    model, config = load_model_frommmf(str(ckpt_path), key)
    model.eval()
    return model, config, getEncoerDecoderData


def clear_cuda_memory(*objects) -> None:
    """Delete references and ask PyTorch to release cached CUDA memory when possible."""
    for obj in objects:
        if obj is not None:
            del obj
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def compute_correlations(df: pd.DataFrame, score_col: str, covariate_col: str) -> dict[str, float]:
    """Compute Pearson and Spearman correlations on the finite non-missing subset."""
    subset = df[[score_col, covariate_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if subset.empty or subset[covariate_col].nunique() <= 1:
        return {"pearson": float("nan"), "spearman": float("nan")}
    return {
        "pearson": float(subset[score_col].corr(subset[covariate_col], method="pearson")),
        "spearman": float(subset[score_col].corr(subset[covariate_col], method="spearman")),
    }
