from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import scanpy as sc


def ensure_output_dir(path: str | Path) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def build_representation(
    adata,
    *,
    layer: str | None,
    use_raw: bool,
    normalize: bool,
    target_sum: float,
    log1p: bool,
    n_top_genes: int | None,
    n_pcs: int,
) -> None:
    if use_raw:
        if adata.raw is None:
            raise ValueError("--use_raw was requested, but adata.raw is missing")
        adata.X = adata.raw.to_adata().X

    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found. Available layers: {list(adata.layers.keys())}")
        adata.X = adata.layers[layer].copy()

    if normalize:
        sc.pp.normalize_total(adata, target_sum=target_sum)

    if log1p:
        sc.pp.log1p(adata)

    if n_top_genes is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)

    if min(adata.n_obs, adata.n_vars) < 2:
        raise ValueError("At least 2 cells and 2 genes are required for PCA, neighbors, and UMAP")

    sc.pp.scale(adata, max_value=10, zero_center=False)
    max_pcs = max(1, min(adata.n_obs, adata.n_vars) - 1)
    effective_n_pcs = min(n_pcs, max_pcs)
    sc.tl.pca(adata, n_comps=effective_n_pcs, svd_solver="arpack")
    adata.uns["cluster_h5ad_umap_effective_n_pcs"] = int(effective_n_pcs)


def add_clusters(
    adata,
    *,
    cluster_method: str,
    cluster_key: str,
    resolution: float,
    kmeans_clusters: int,
    random_state: int,
) -> str:
    if cluster_method == "auto":
        cluster_method = "leiden" if has_module("leidenalg") and has_module("igraph") else "kmeans"

    if cluster_method == "leiden":
        if not has_module("leidenalg") or not has_module("igraph"):
            raise RuntimeError(
                "Leiden clustering requires the optional packages 'leidenalg' and 'igraph'. "
                "Install them or run with --cluster_method kmeans."
            )
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=cluster_key,
            random_state=random_state,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
        return cluster_method

    if cluster_method == "kmeans":
        from sklearn.cluster import KMeans

        if "X_pca" not in adata.obsm:
            raise ValueError("KMeans clustering requires PCA coordinates in adata.obsm['X_pca']")
        effective_kmeans_clusters = max(1, min(kmeans_clusters, adata.n_obs))
        labels = KMeans(n_clusters=effective_kmeans_clusters, random_state=random_state, n_init=10).fit_predict(
            adata.obsm["X_pca"]
        )
        adata.obs[cluster_key] = labels.astype(str)
        adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")
        adata.uns["cluster_h5ad_umap_effective_kmeans_clusters"] = int(effective_kmeans_clusters)
        return cluster_method

    raise ValueError(f"Unsupported cluster_method: {cluster_method}")


def safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("_") or "umap"


def wrap_legend_label(value: object, max_chars: int = 44) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return "\n".join(textwrap.wrap(text, width=max_chars, break_long_words=False, break_on_hyphens=False))


def point_size_for_cells(n_cells: int, point_size: float | None) -> float:
    if point_size is not None:
        return point_size
    return max(1.0, min(8.0, 120000.0 / max(n_cells, 1)))


def umap_plot_dimensions(coords: np.ndarray, n_categories: int | None, max_label_len: int = 0) -> tuple[float, float, float]:
    x_range = float(np.ptp(coords[:, 0])) or 1.0
    y_range = float(np.ptp(coords[:, 1])) or 1.0
    aspect = x_range / y_range

    plot_height = 6.5
    plot_width = min(9.5, max(6.5, plot_height * aspect))
    if n_categories is None:
        legend_width = 0.8
        figure_height = plot_height
    else:
        n_columns = max(1, math.ceil(n_categories / 18))
        legend_width = max(2.4, min(7.0, 2.4 * n_columns + 0.02 * min(max_label_len, 44)))
        figure_height = max(plot_height, min(11.0, 1.4 + 0.31 * math.ceil(n_categories / n_columns)))
    return plot_width, legend_width, figure_height


def category_order(series: pd.Series) -> list[object]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        values = [category for category in series.cat.categories if (series == category).any()]
    else:
        values = list(pd.unique(series.astype(str)))

    value_strings = [str(value) for value in values]
    if all(value.lstrip("-").isdigit() for value in value_strings):
        return sorted(values, key=lambda value: int(str(value)))
    return sorted(values, key=str)


def save_single_umap_plot(
    adata,
    *,
    outdir: Path,
    color: str,
    point_size: float | None,
) -> Path:
    coords = np.asarray(adata.obsm["X_umap"])
    series = adata.obs[color]
    marker_size = point_size_for_cells(adata.n_obs, point_size)
    plot_path = outdir / f"umap_{safe_filename(color)}.png"

    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        plot_width, legend_width, figure_height = umap_plot_dimensions(coords, None)
        figure = plt.figure(figsize=(plot_width + legend_width, figure_height), constrained_layout=True)
        grid = figure.add_gridspec(1, 2, width_ratios=[plot_width, legend_width])
        axis = figure.add_subplot(grid[0, 0])
        colorbar_axis = figure.add_subplot(grid[0, 1])
        valid = np.isfinite(values)
        norm = Normalize(vmin=float(np.nanmin(values[valid])), vmax=float(np.nanmax(values[valid])))
        scatter = axis.scatter(
            coords[valid, 0],
            coords[valid, 1],
            c=values[valid],
            s=marker_size,
            cmap="viridis",
            norm=norm,
            linewidths=0,
            alpha=0.85,
        )
        if (~valid).any():
            axis.scatter(
                coords[~valid, 0],
                coords[~valid, 1],
                c="#BDBDBD",
                s=marker_size,
                linewidths=0,
                alpha=0.65,
            )
        figure.colorbar(scatter, cax=colorbar_axis)
        colorbar_axis.set_ylabel(color, rotation=270, labelpad=16)
    else:
        labels = series.astype("string").fillna("missing")
        categories = category_order(labels)
        max_label_len = max((len(str(category)) for category in categories), default=0)
        plot_width, legend_width, figure_height = umap_plot_dimensions(coords, len(categories), max_label_len)
        n_columns = max(1, math.ceil(len(categories) / 18))

        figure = plt.figure(figsize=(plot_width + legend_width, figure_height), constrained_layout=True)
        grid = figure.add_gridspec(1, 2, width_ratios=[plot_width, legend_width])
        axis = figure.add_subplot(grid[0, 0])
        legend_axis = figure.add_subplot(grid[0, 1])
        legend_axis.axis("off")

        palette = sc.pl.palettes.default_102
        color_map = {category: palette[idx % len(palette)] for idx, category in enumerate(categories)}
        for category in categories:
            mask = labels.to_numpy() == str(category)
            axis.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=color_map[category],
                s=marker_size,
                linewidths=0,
                alpha=0.85,
                rasterized=True,
            )

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color_map[category],
                markeredgecolor=color_map[category],
                markersize=5.5,
                label=wrap_legend_label(category),
            )
            for category in categories
        ]
        legend_axis.legend(
            handles=handles,
            loc="center left",
            frameon=False,
            fontsize=9,
            ncol=n_columns,
            handlelength=0.8,
            handletextpad=0.5,
            columnspacing=1.0,
            borderaxespad=0.0,
        )

    axis.set_title(color, fontsize=14, pad=10)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return plot_path


def save_umap_plots(
    adata,
    *,
    outdir: Path,
    color: str,
    cluster_key: str,
    point_size: float | None,
) -> dict[str, str]:
    plot_paths: dict[str, str] = {}
    colors = [color]
    if cluster_key != color:
        colors.append(cluster_key)
    for color_key in colors:
        path = save_single_umap_plot(adata, outdir=outdir, color=color_key, point_size=point_size)
        plot_paths[color_key] = str(path.resolve())
    return plot_paths


def cluster_h5ad_umap(
    *,
    input_path: str,
    output_dir: str,
    color: str,
    layer: str | None,
    use_raw: bool,
    normalize: bool,
    target_sum: float,
    log1p: bool,
    n_top_genes: int | None,
    n_pcs: int,
    n_neighbors: int,
    neighbors_transformer: str | None,
    cluster_method: str,
    cluster_key: str,
    resolution: float,
    kmeans_clusters: int,
    random_state: int,
    point_size: float | None,
    write_h5ad: bool,
) -> None:
    outdir = ensure_output_dir(output_dir)
    adata = sc.read_h5ad(input_path)

    if color not in adata.obs.columns:
        raise ValueError(f"Column '{color}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")

    build_representation(
        adata,
        layer=layer,
        use_raw=use_raw,
        normalize=normalize,
        target_sum=target_sum,
        log1p=log1p,
        n_top_genes=n_top_genes,
        n_pcs=n_pcs,
    )
    effective_n_pcs = int(adata.uns["cluster_h5ad_umap_effective_n_pcs"])
    effective_n_neighbors = max(2, min(n_neighbors, adata.n_obs - 1))
    sc.pp.neighbors(
        adata,
        n_neighbors=effective_n_neighbors,
        n_pcs=effective_n_pcs,
        transformer=neighbors_transformer,
        random_state=random_state,
    )
    cluster_method_used = add_clusters(
        adata,
        cluster_method=cluster_method,
        cluster_key=cluster_key,
        resolution=resolution,
        kmeans_clusters=kmeans_clusters,
        random_state=random_state,
    )
    sc.tl.umap(adata, random_state=random_state)

    plot_paths = save_umap_plots(
        adata,
        outdir=outdir,
        color=color,
        cluster_key=cluster_key,
        point_size=point_size,
    )

    annotated_path = None
    if write_h5ad:
        annotated_path = outdir / "adata_clustered_umap.h5ad"
        adata.write_h5ad(annotated_path)

    summary = {
        "input_path": str(Path(input_path).resolve()),
        "n_cells": int(adata.n_obs),
        "n_genes_used": int(adata.n_vars),
        "color": color,
        "cluster_key": cluster_key,
        "cluster_method": cluster_method_used,
        "resolution": resolution if cluster_method_used == "leiden" else None,
        "kmeans_clusters_requested": kmeans_clusters if cluster_method_used == "kmeans" else None,
        "kmeans_clusters_used": adata.uns.get("cluster_h5ad_umap_effective_kmeans_clusters"),
        "n_neighbors_requested": n_neighbors,
        "n_neighbors_used": effective_n_neighbors,
        "neighbors_transformer": neighbors_transformer,
        "n_pcs_requested": n_pcs,
        "n_pcs_used": effective_n_pcs,
        "plot_paths": plot_paths,
        "annotated_h5ad": str(annotated_path.resolve()) if annotated_path is not None else None,
    }
    with (outdir / "cluster_umap_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("Clustered UMAP complete")
    for color_key, plot_path in plot_paths.items():
        print(f"{color_key} UMAP: {plot_path}")
    if annotated_path is not None:
        print(f"Annotated h5ad: {annotated_path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster cells from an .h5ad file and save a UMAP plot.")
    parser.add_argument("--input", required=True, help="Input .h5ad file")
    parser.add_argument("--output_dir", required=True, help="Directory for plot and summary outputs")
    parser.add_argument("--color", required=True, help="Column in adata.obs used to color the UMAP")
    parser.add_argument("--layer", default=None, help="Optional adata.layers key to use as X")
    parser.add_argument("--use_raw", action="store_true", help="Use adata.raw as X before preprocessing")
    parser.add_argument("--no_normalize", action="store_true", help="Skip sc.pp.normalize_total")
    parser.add_argument("--target_sum", type=float, default=1e4)
    parser.add_argument("--no_log1p", action="store_true", help="Skip sc.pp.log1p")
    parser.add_argument("--n_top_genes", type=int, default=2000, help="Highly variable genes to keep; use 0 to keep all genes")
    parser.add_argument("--n_pcs", type=int, default=50)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument(
        "--neighbors_transformer",
        choices=["sklearn", "pynndescent"],
        default="sklearn",
        help="Nearest-neighbor backend. sklearn is exact and avoids joblib multiprocessing issues on Windows.",
    )
    parser.add_argument("--cluster_method", choices=["auto", "leiden", "kmeans"], default="auto")
    parser.add_argument("--cluster_key", default="cluster", help="adata.obs column name for cluster labels")
    parser.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution")
    parser.add_argument("--kmeans_clusters", type=int, default=20, help="Number of clusters for --cluster_method kmeans")
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--point_size", type=float, default=None, help="UMAP point size; defaults to Scanpy's choice")
    parser.add_argument("--no_write_h5ad", action="store_true", help="Do not save annotated h5ad with clusters and UMAP")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    n_top_genes = None if args.n_top_genes == 0 else args.n_top_genes
    cluster_h5ad_umap(
        input_path=args.input,
        output_dir=args.output_dir,
        color=args.color,
        layer=args.layer,
        use_raw=args.use_raw,
        normalize=not args.no_normalize,
        target_sum=args.target_sum,
        log1p=not args.no_log1p,
        n_top_genes=n_top_genes,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        neighbors_transformer=args.neighbors_transformer,
        cluster_method=args.cluster_method,
        cluster_key=args.cluster_key,
        resolution=args.resolution,
        kmeans_clusters=args.kmeans_clusters,
        random_state=args.random_state,
        point_size=args.point_size,
        write_h5ad=not args.no_write_h5ad,
    )


if __name__ == "__main__":
    main()
