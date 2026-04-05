from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


SCFOUNDATION_REPO_ENV = "SCFOUNDATION_REPO"
SCFOUNDATION_CKPT_ENV = "SCFOUNDATION_CKPT_PATH"
SCFOUNDATION_GENE_PANEL_ENV = "SCFOUNDATION_GENE_PANEL_PATH"
DEFAULT_PANEL_FILENAME = "OS_scRNA_gene_index.19264.tsv"
DEFAULT_CKPT_FILENAME = "models.ckpt"


@dataclass(frozen=True)
class PreparedDatasetPaths:
    prefix: Path
    counts_path: Path
    obs_csv_path: Path
    obs_parquet_path: Path
    var_path: Path
    gene_coverage_path: Path
    summary_path: Path
    manifest_path: Path


def prepared_dataset_paths(output_prefix: str | Path) -> PreparedDatasetPaths:
    prefix = Path(output_prefix)
    return PreparedDatasetPaths(
        prefix=prefix,
        counts_path=prefix.with_suffix(".counts_19264.npz"),
        obs_csv_path=prefix.with_suffix(".obs.csv.gz"),
        obs_parquet_path=prefix.with_suffix(".obs.parquet"),
        var_path=prefix.with_suffix(".var.csv"),
        gene_coverage_path=prefix.with_suffix(".gene_coverage.csv"),
        summary_path=prefix.with_suffix(".summary.json"),
        manifest_path=prefix.with_suffix(".manifest.json"),
    )


def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_scfoundation_repo(repo_path: str | None = None) -> Path:
    candidate = repo_path or os.environ.get(SCFOUNDATION_REPO_ENV)
    if candidate:
        path = Path(candidate)
    else:
        path = Path(__file__).resolve().parents[1] / "external" / "scFoundation"
    if not path.exists():
        raise FileNotFoundError(
            f"scFoundation repo not found at {path}. Set {SCFOUNDATION_REPO_ENV} or clone the repo."
        )
    return path.resolve()


def resolve_gene_panel_path(
    panel_path: str | None = None,
    repo_path: str | None = None,
) -> Path:
    candidate = panel_path or os.environ.get(SCFOUNDATION_GENE_PANEL_ENV)
    if candidate:
        path = Path(candidate)
    else:
        repo = resolve_scfoundation_repo(repo_path)
        path = repo / "model" / DEFAULT_PANEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"scFoundation gene panel file not found at {path}. Set {SCFOUNDATION_GENE_PANEL_ENV} if needed."
        )
    return path.resolve()


def resolve_ckpt_path(
    ckpt_path: str | None = None,
    repo_path: str | None = None,
) -> Path:
    candidate = ckpt_path or os.environ.get(SCFOUNDATION_CKPT_ENV)
    if candidate:
        path = Path(candidate)
    else:
        repo = resolve_scfoundation_repo(repo_path)
        path = repo / "model" / "models" / DEFAULT_CKPT_FILENAME
    return path.resolve()


def load_gene_panel(panel_path: str | Path) -> pd.DataFrame:
    panel_df = pd.read_csv(panel_path, sep="\t")
    if "gene_name" not in panel_df.columns:
        raise ValueError(f"Expected gene_name column in {panel_path}")
    panel_df = panel_df.copy()
    panel_df["gene_name"] = sanitize_gene_symbols(panel_df["gene_name"])
    panel_df["panel_index"] = np.arange(panel_df.shape[0], dtype=np.int32)
    return panel_df


def sanitize_gene_symbols(symbols: Iterable[object]) -> pd.Index:
    series = pd.Series(pd.Index(symbols).astype(str), dtype="string")
    return pd.Index(series.str.strip())


def select_counts_matrix(
    adata: ad.AnnData,
    counts_source: str,
):
    if counts_source == "raw":
        if adata.raw is None:
            raise ValueError("counts_source=raw requested but adata.raw is missing")
        return adata.raw.X, adata.raw.var.copy(), "raw"
    if counts_source != "X":
        raise ValueError("counts_source must be one of {'raw', 'X'}")
    return adata.X, adata.var.copy(), "X"


def get_gene_symbols(var: pd.DataFrame, gene_symbol_field: str) -> pd.Index:
    if gene_symbol_field == "var_names":
        return sanitize_gene_symbols(var.index)
    if gene_symbol_field not in var.columns:
        raise ValueError(f"Gene symbol field '{gene_symbol_field}' not found in var columns")
    return sanitize_gene_symbols(var[gene_symbol_field])


def as_csr_matrix(matrix) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(np.asarray(matrix))


def get_matrix_nnz_per_row(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.getnnz(axis=1)).ravel()
    return np.asarray((np.asarray(matrix) > 0).sum(axis=1)).ravel()


def get_matrix_sum_per_row(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.sum(axis=1)).ravel()
    return np.asarray(np.asarray(matrix).sum(axis=1)).ravel()


def build_collapse_matrix(symbols: pd.Index) -> tuple[sparse.csr_matrix, pd.Index]:
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
    block_csr = as_csr_matrix(block).astype(np.float32, copy=False)
    collapsed = block_csr @ collapse_matrix
    aligned = collapsed @ align_matrix
    return aligned.tocsr()


def summarize_integer_like_counts(matrix) -> bool:
    block = as_csr_matrix(matrix)
    if block.nnz == 0:
        return True
    return bool(np.allclose(block.data, np.round(block.data)))


def write_json(path: str | Path, payload: dict) -> None:
    path = ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_prepared_dataset(prefix: str | Path) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame, dict]:
    paths = prepared_dataset_paths(prefix)
    counts = sparse.load_npz(paths.counts_path).tocsr()
    obs = pd.read_csv(paths.obs_csv_path, index_col=0)
    var = pd.read_csv(paths.var_path)
    with paths.summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return counts, obs, var, summary


def try_write_parquet(df: pd.DataFrame, path: str | Path) -> bool:
    try:
        ensure_parent_dir(path)
        df.to_parquet(path)
        return True
    except Exception:
        return False


def add_scfoundation_model_to_path(repo_path: str | Path) -> Path:
    repo = resolve_scfoundation_repo(str(repo_path))
    model_dir = repo / "model"
    model_dir_str = str(model_dir)
    if model_dir_str not in sys.path:
        sys.path.insert(0, model_dir_str)
    return model_dir


def load_scfoundation_model(
    repo_path: str | Path,
    ckpt_path: str | Path,
    key: str = "gene",
):
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"scFoundation checkpoint not found at {ckpt_path}. "
            f"Download it into external/scFoundation/model/models/ or set {SCFOUNDATION_CKPT_ENV}."
        )
    add_scfoundation_model_to_path(repo_path)
    from load import getEncoerDecoderData, load_model_frommmf  # type: ignore

    model, config = load_model_frommmf(str(ckpt_path), key)
    model.eval()
    return model, config, getEncoerDecoderData


def compute_correlations(df: pd.DataFrame, score_col: str, covariate_col: str) -> dict[str, float]:
    subset = df[[score_col, covariate_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if subset.empty or subset[covariate_col].nunique() <= 1:
        return {"pearson": float("nan"), "spearman": float("nan")}
    return {
        "pearson": float(subset[score_col].corr(subset[covariate_col], method="pearson")),
        "spearman": float(subset[score_col].corr(subset[covariate_col], method="spearman")),
    }
