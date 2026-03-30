#!/usr/bin/env python3
"""
Clean downloader utilities for lung single-cell reference datasets.

Supports:
1) Searching CELLxGENE Census for HLCA / lung reference datasets.
2) Downloading a source H5AD from CELLxGENE Census by dataset_id.
3) Inspecting downloaded H5AD metadata, including age-related columns.
4) Downloading processed supplementary files from GEO for a malignant reference.
5) Optionally downloading the processed Maynard therapy-evolution repo snapshot.

Design goals:
- No hidden pip installs.
- No interactive prompts inside the workflow.
- Clear separation between search and download.
- Helpful error messages.
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable


def require_package(import_name: str, package_name: str | None = None):
    try:
        return importlib.import_module(import_name)
    except ImportError as exc:
        pkg = package_name or import_name
        raise SystemExit(
            f"Missing dependency '{pkg}'. Install it with: pip install {pkg}"
        ) from exc


# -----------------------------------------------------------------------------
# Generic filesystem helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def unpack_archives(root_dir: str | Path) -> None:
    root = Path(root_dir)

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix == ".zip":
            target = path.with_suffix("")
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(target)

        elif path.suffixes[-2:] == [".tar", ".gz"]:
            target = path.with_suffix("").with_suffix("")
            target.mkdir(parents=True, exist_ok=True)
            with tarfile.open(path, "r:gz") as tf:
                tf.extractall(target)

        elif path.suffix == ".tar":
            target = path.with_suffix("")
            target.mkdir(parents=True, exist_ok=True)
            with tarfile.open(path, "r:") as tf:
                tf.extractall(target)


# -----------------------------------------------------------------------------
# CELLxGENE Census utilities
# -----------------------------------------------------------------------------

def resolve_census_version(census_version: str) -> str:
    if census_version != "stable":
        return census_version

    cellxgene_census = require_package("cellxgene_census")
    desc = cellxgene_census.get_census_version_description("stable")
    return desc.get("release_build") or census_version


def _load_census_datasets(census_version: str = "stable"):
    cellxgene_census = require_package("cellxgene_census")
    pandas = require_package("pandas")

    resolved_version = resolve_census_version(census_version)
    census = cellxgene_census.open_soma(census_version=resolved_version)
    try:
        datasets = census["census_info"]["datasets"].read().concat().to_pandas()
    finally:
        census.close()

    for col in datasets.columns:
        if pandas.api.types.is_string_dtype(datasets[col]):
            datasets[col] = datasets[col].fillna("")

    return datasets



def search_census_datasets(
    keywords: Iterable[str],
    census_version: str = "stable",
    require_h5ad: bool = True,
):
    pandas = require_package("pandas")
    datasets = _load_census_datasets(census_version=census_version)

    text_cols = [
        col
        for col in [
            "dataset_title",
            "collection_name",
            "citation",
            "dataset_id",
            "dataset_h5ad_path",
        ]
        if col in datasets.columns
    ]

    haystack = datasets[text_cols].astype(str).agg(" ".join, axis=1).str.lower()
    final_mask = pandas.Series(True, index=datasets.index)

    if require_h5ad and "dataset_h5ad_path" in datasets.columns:
        dataset_h5ad_path = datasets["dataset_h5ad_path"].fillna("").astype(str).str.strip()
        final_mask &= dataset_h5ad_path.ne("")

    for keyword in keywords:
        final_mask &= haystack.str.contains(re.escape(keyword.lower()), regex=True)

    hits = datasets.loc[final_mask].copy()

    keep_cols = [
        col
        for col in [
            "dataset_id",
            "dataset_title",
            "collection_name",
            "citation",
            "dataset_total_cell_count",
            "dataset_h5ad_path",
        ]
        if col in hits.columns
    ]

    sort_cols = [col for col in ["dataset_total_cell_count", "dataset_title"] if col in hits.columns]
    if sort_cols:
        ascending = [False if col == "dataset_total_cell_count" else True for col in sort_cols]
        hits = hits.sort_values(by=sort_cols, ascending=ascending)

    return hits[keep_cols]



def download_census_h5ad(
    dataset_id: str,
    out_path: str | Path,
    census_version: str = "stable",
    progress_bar: bool = True,
) -> Path:
    cellxgene_census = require_package("cellxgene_census")

    dataset_id = dataset_id.strip()
    if not dataset_id:
        raise ValueError("dataset_id must be a non-empty string")

    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")
    ensure_dir(out_path.parent)
    resolved_version = resolve_census_version(census_version)

    cellxgene_census.download_source_h5ad(
        dataset_id,
        to_path=str(out_path),
        census_version=resolved_version,
        progress_bar=progress_bar,
    )
    return out_path



def inspect_h5ad_metadata(h5ad_path: str | Path) -> None:
    anndata = require_package("anndata")

    h5ad_path = Path(h5ad_path)
    if not h5ad_path.exists():
        raise FileNotFoundError(f"H5AD file not found: {h5ad_path}")

    adata = anndata.read_h5ad(h5ad_path, backed="r")
    try:
        print(f"File: {h5ad_path}")
        print(f"Cells: {adata.n_obs}")
        print(f"Genes: {adata.n_vars}\n")

        obs_columns = list(map(str, adata.obs.columns))
        print("First 40 obs columns:")
        for col in obs_columns[:40]:
            print(f"  - {col}")

        age_like = [col for col in obs_columns if re.search(r"age", col, flags=re.IGNORECASE)]
        print("\nAge-related obs columns:")
        if not age_like:
            print("  None found by regex /age/i")
        else:
            for col in age_like:
                series = adata.obs[col]
                values = series.dropna().astype(str).unique().tolist()[:10]
                print(f"  - {col}: {values}")
    finally:
        adata.file.close()


# -----------------------------------------------------------------------------
# GEO utilities
# -----------------------------------------------------------------------------

def download_geo_supplementary(accession: str, out_dir: str | Path) -> Path:
    GEOparse = require_package("GEOparse")

    out_dir = ensure_dir(out_dir)
    gse = GEOparse.get_GEO(geo=accession, destdir=str(out_dir), silent=False)
    # download_sra=False because we do not want raw FASTQ/SRA here.
    gse.download_supplementary_files(directory=str(out_dir), download_sra=False)
    return out_dir


# -----------------------------------------------------------------------------
# Optional GitHub repo download for Maynard processed data
# -----------------------------------------------------------------------------

def download_url(url: str, out_path: str | Path) -> Path:
    urllib_request = require_package("urllib.request", package_name="urllib")

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    urllib_request.urlretrieve(url, out_path)
    return out_path


# -----------------------------------------------------------------------------
# CLI commands
# -----------------------------------------------------------------------------

def cmd_search_census(args) -> None:
    hits = search_census_datasets(
        keywords=args.keywords,
        census_version=args.census_version,
        require_h5ad=not args.allow_no_h5ad,
    )
    if hits.empty:
        print("No matching datasets found.")
        return

    if args.full_output:
        print(hits.to_string(index=False))
        return

    compact_cols = [
        col
        for col in ["dataset_id", "dataset_title", "dataset_total_cell_count"]
        if col in hits.columns
    ]
    compact_hits = hits[compact_cols].copy()
    print(compact_hits.to_string(index=False, max_colwidth=80))



def cmd_download_census(args) -> None:
    out_path = download_census_h5ad(
        dataset_id=args.dataset_id,
        out_path=args.out_path,
        census_version=args.census_version,
        progress_bar=not args.no_progress,
    )
    print(f"Downloaded H5AD to: {out_path}")



def cmd_inspect_h5ad(args) -> None:
    inspect_h5ad_metadata(args.h5ad_path)



def cmd_download_geo(args) -> None:
    out_dir = download_geo_supplementary(args.accession, args.out_dir)
    if args.unpack:
        unpack_archives(out_dir)
    print(f"Downloaded GEO supplementary files to: {out_dir}")



def cmd_download_maynard(args) -> None:
    # Optional processed-data repo snapshot.
    # You can change this URL later if the repository structure changes.
    repo_zip_url = args.url or "https://github.com/czbiohub/scell_lung_adenocarcinoma/archive/refs/heads/master.zip"
    out_path = download_url(repo_zip_url, args.out_zip)
    if args.unpack:
        unpack_archives(out_path.parent)
    print(f"Downloaded Maynard repo snapshot to: {out_path}")



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search and download normal/malignant lung single-cell reference datasets"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search-census", help="Search CELLxGENE Census datasets")
    p_search.add_argument("keywords", nargs="+", help="Keywords such as: human lung atlas core")
    p_search.add_argument("--census-version", default="stable")
    p_search.add_argument("--allow-no-h5ad", action="store_true", help="Include matches without dataset_h5ad_path")
    p_search.add_argument("--full-output", action="store_true", help="Show all output columns, including long citation fields")
    p_search.set_defaults(func=cmd_search_census)

    p_dl = sub.add_parser("download-census", help="Download a source H5AD from CELLxGENE Census")
    p_dl.add_argument("--dataset-id", required=True)
    p_dl.add_argument("--out-path", required=True)
    p_dl.add_argument("--census-version", default="stable")
    p_dl.add_argument("--no-progress", action="store_true")
    p_dl.set_defaults(func=cmd_download_census)

    p_inspect = sub.add_parser("inspect-h5ad", help="Inspect a downloaded H5AD file")
    p_inspect.add_argument("h5ad_path")
    p_inspect.set_defaults(func=cmd_inspect_h5ad)

    p_geo = sub.add_parser("download-geo", help="Download GEO supplementary files")
    p_geo.add_argument("--accession", required=True, help="Example: GSE131907")
    p_geo.add_argument("--out-dir", required=True)
    p_geo.add_argument("--unpack", action="store_true")
    p_geo.set_defaults(func=cmd_download_geo)

    p_may = sub.add_parser("download-maynard", help="Download optional Maynard processed-data repo snapshot")
    p_may.add_argument("--out-zip", required=True)
    p_may.add_argument("--url", default=None)
    p_may.add_argument("--unpack", action="store_true")
    p_may.set_defaults(func=cmd_download_maynard)

    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
