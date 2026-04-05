from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.linear_model import HuberRegressor

try:
    from scfoundation_utils import (
        compute_correlations,
        ensure_parent_dir,
        load_prepared_dataset,
        load_scfoundation_model,
        resolve_ckpt_path,
        resolve_scfoundation_repo,
        try_write_parquet,
        write_json,
    )
except ImportError:
    from src.scfoundation_utils import (
        compute_correlations,
        ensure_parent_dir,
        load_prepared_dataset,
        load_scfoundation_model,
        resolve_ckpt_path,
        resolve_scfoundation_repo,
        try_write_parquet,
        write_json,
    )


def normalize_counts_for_model(batch_counts: np.ndarray) -> np.ndarray:
    totals = batch_counts.sum(axis=1, keepdims=True)
    totals_safe = np.clip(totals, 1.0, None)
    normalized = np.log1p((batch_counts / totals_safe) * 1e4)
    normalized[totals[:, 0] <= 0, :] = 0.0
    return normalized.astype(np.float32, copy=False)


def build_mask_matrix(
    counts_batch: np.ndarray,
    zero_padded_features: np.ndarray,
    mask_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.zeros_like(counts_batch, dtype=bool)
    valid_candidates = (counts_batch > 0) & (~zero_padded_features[None, :])
    for row_idx in range(counts_batch.shape[0]):
        candidates = np.flatnonzero(valid_candidates[row_idx])
        if candidates.size == 0:
            continue
        n_mask = max(1, int(np.floor(candidates.size * mask_fraction)))
        chosen = candidates if n_mask >= candidates.size else rng.choice(candidates, size=n_mask, replace=False)
        mask[row_idx, chosen] = True
    return mask


def score_matrix(
    counts: sparse.csr_matrix,
    zero_padded_features: np.ndarray,
    model,
    config: dict,
    get_encoder_decoder_data,
    batch_size: int,
    mask_fraction: float,
    target_log10_total_count: float,
    random_seed: int,
) -> pd.DataFrame:
    if not torch.cuda.is_available():
        raise RuntimeError("scFoundation scoring requires CUDA, but torch.cuda.is_available() is False.")

    rng = np.random.default_rng(random_seed)
    mask_token_id = float(config["mask_token_id"])
    device = torch.device("cuda")

    base_error = np.full(counts.shape[0], np.nan, dtype=np.float64)
    masked_gene_count = np.zeros(counts.shape[0], dtype=np.int32)

    for start in range(0, counts.shape[0], batch_size):
        stop = min(start + batch_size, counts.shape[0])
        batch_counts = counts[start:stop, :].toarray().astype(np.float32, copy=False)
        normalized = normalize_counts_for_model(batch_counts)
        total_counts = batch_counts.sum(axis=1).astype(np.float32, copy=False)
        source_mask = build_mask_matrix(
            counts_batch=batch_counts,
            zero_padded_features=zero_padded_features,
            mask_fraction=mask_fraction,
            rng=rng,
        )
        masked_gene_count[start:stop] = source_mask.sum(axis=1).astype(np.int32, copy=False)

        seq = np.concatenate(
            [
                normalized,
                np.full((normalized.shape[0], 1), target_log10_total_count, dtype=np.float32),
                np.log10(np.clip(total_counts, 1.0, None))[:, None],
            ],
            axis=1,
        )
        seq_data = seq.copy()
        seq_data_raw = seq.copy()
        seq_data[:, : counts.shape[1]][source_mask] = mask_token_id
        seq_data_raw[:, : counts.shape[1]][source_mask] = 0.0

        with torch.no_grad():
            tensor_data = torch.from_numpy(seq_data).to(device=device, dtype=torch.float32)
            tensor_data_raw = torch.from_numpy(seq_data_raw).to(device=device, dtype=torch.float32)
            encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, _, _, decoder_position_gene_ids = get_encoder_decoder_data(
                tensor_data,
                tensor_data_raw,
                config,
            )
            prediction = model.forward(
                x=encoder_data,
                padding_label=encoder_data_padding,
                encoder_position_gene_ids=encoder_position_gene_ids,
                encoder_labels=encoder_labels,
                decoder_data=decoder_data,
                mask_gene_name=False,
                mask_labels=None,
                decoder_position_gene_ids=decoder_position_gene_ids,
                decoder_data_padding_labels=decoder_data_padding,
            )
            prediction = prediction[:, : counts.shape[1]].detach().cpu().numpy()

        squared_error = np.square(prediction - normalized, dtype=np.float32)
        masked_error_sum = (squared_error * source_mask).sum(axis=1, dtype=np.float64)
        denominator = np.clip(source_mask.sum(axis=1), 1, None)
        batch_error = masked_error_sum / denominator
        batch_error[source_mask.sum(axis=1) == 0] = np.nan
        base_error[start:stop] = batch_error

    return pd.DataFrame({"base_error": base_error, "masked_gene_count": masked_gene_count})


def residualize_scores(
    reference_df: pd.DataFrame,
    target_frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict]:
    fit_mask = (
        reference_df["base_error"].notna()
        & np.isfinite(reference_df["base_error"])
        & np.isfinite(reference_df["log10_total_counts_raw"])
        & np.isfinite(reference_df["log10_n_genes_by_counts"])
    )
    if fit_mask.sum() == 0:
        raise ValueError("No valid healthy reference cells available for residualization")

    model = HuberRegressor()
    x_ref = reference_df.loc[fit_mask, ["log10_total_counts_raw", "log10_n_genes_by_counts"]].to_numpy()
    y_ref = reference_df.loc[fit_mask, "base_error"].to_numpy()
    model.fit(x_ref, y_ref)

    ref_residual = reference_df["base_error"] - model.predict(
        reference_df[["log10_total_counts_raw", "log10_n_genes_by_counts"]].to_numpy()
    )
    center = float(np.nanmedian(ref_residual.to_numpy()))

    centered_frames: dict[str, pd.DataFrame] = {}
    for name, frame in target_frames.items():
        preds = model.predict(frame[["log10_total_counts_raw", "log10_n_genes_by_counts"]].to_numpy())
        out = frame.copy()
        out["depth_residual_error"] = out["base_error"] - preds - center
        centered_frames[name] = out

    diagnostics = {
        "reference_fit_cells": int(fit_mask.sum()),
        "huber_intercept": float(model.intercept_),
        "huber_coef": {
            "log10_total_counts_raw": float(model.coef_[0]),
            "log10_n_genes_by_counts": float(model.coef_[1]),
        },
        "healthy_residual_median_before_centering": center,
    }
    return centered_frames, diagnostics


def save_dataset_outputs(output_dir: Path, dataset_name: str, df: pd.DataFrame) -> dict[str, str]:
    csv_path = output_dir / f"{dataset_name}.scores.csv.gz"
    ensure_parent_dir(csv_path)
    df.to_csv(csv_path, index=True, compression="gzip")
    parquet_path = output_dir / f"{dataset_name}.scores.parquet"
    wrote_parquet = try_write_parquet(df.reset_index(), parquet_path)
    return {
        "csv_path": str(csv_path.resolve()),
        "parquet_path": str(parquet_path.resolve()) if wrote_parquet else "",
    }


def save_diagnostics_plot(output_dir: Path, combined: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for dataset_name, sub in combined.groupby("dataset_name"):
        axes[0].hist(sub["base_error"].dropna(), bins=40, alpha=0.5, label=dataset_name)
        axes[1].hist(sub["depth_residual_error"].dropna(), bins=40, alpha=0.5, label=dataset_name)
    axes[0].set_title("Base Error")
    axes[1].set_title("Depth-Residual Error")
    for axis in axes:
        axis.legend()
        axis.set_xlabel("Score")
        axis.set_ylabel("Cells")
    fig.tight_layout()
    out_path = output_dir / "score_distributions.png"
    ensure_parent_dir(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def prepare_feature_frame(obs: pd.DataFrame, scores: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    frame = obs.join(scores)
    frame["dataset_name"] = dataset_name
    frame["log10_total_counts_raw"] = np.log10(np.clip(frame["total_counts_raw"].astype(float), 1.0, None))
    frame["log10_n_genes_by_counts"] = np.log10(np.clip(frame["n_genes_by_counts"].astype(float), 1.0, None))
    return frame


def score_datasets(
    reference_prefix: str,
    target_prefixes: list[str],
    output_dir: str,
    checkpoint_path: str | None,
    scfoundation_repo: str | None,
    batch_size: int,
    mask_fraction: float,
    target_log10_total_count: float,
    random_seed: int,
) -> None:
    repo_path = resolve_scfoundation_repo(scfoundation_repo)
    ckpt_path = resolve_ckpt_path(checkpoint_path, str(repo_path))
    model, config, get_encoder_decoder_data = load_scfoundation_model(
        repo_path=repo_path,
        ckpt_path=ckpt_path,
        key="gene",
    )

    reference_counts, reference_obs, reference_var, reference_summary = load_prepared_dataset(reference_prefix)
    zero_padded_features = reference_var["is_zero_padded_feature"].to_numpy(dtype=bool)
    reference_name = Path(reference_prefix).stem
    reference_scores = score_matrix(
        counts=reference_counts,
        zero_padded_features=zero_padded_features,
        model=model,
        config=config,
        get_encoder_decoder_data=get_encoder_decoder_data,
        batch_size=batch_size,
        mask_fraction=mask_fraction,
        target_log10_total_count=target_log10_total_count,
        random_seed=random_seed,
    )
    frames = {reference_name: prepare_feature_frame(reference_obs, reference_scores, reference_name)}

    for target_prefix in target_prefixes:
        counts, obs, var, _ = load_prepared_dataset(target_prefix)
        if not var["gene_name"].equals(reference_var["gene_name"]):
            raise ValueError(f"Prepared dataset {target_prefix} does not match the reference gene panel order")
        dataset_name = Path(target_prefix).stem
        scores = score_matrix(
            counts=counts,
            zero_padded_features=zero_padded_features,
            model=model,
            config=config,
            get_encoder_decoder_data=get_encoder_decoder_data,
            batch_size=batch_size,
            mask_fraction=mask_fraction,
            target_log10_total_count=target_log10_total_count,
            random_seed=random_seed,
        )
        frames[dataset_name] = prepare_feature_frame(obs, scores, dataset_name)

    residualized_frames, residual_diagnostics = residualize_scores(frames[reference_name], frames)

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    output_manifest: dict[str, dict[str, str]] = {}
    correlation_summary: dict[str, dict[str, dict[str, float]]] = {}
    combined_frames = []
    for dataset_name, frame in residualized_frames.items():
        output_manifest[dataset_name] = save_dataset_outputs(outdir, dataset_name, frame)
        combined_frames.append(frame)
        correlation_summary[dataset_name] = {
            "base_error_vs_log10_total_counts_raw": compute_correlations(frame, "base_error", "log10_total_counts_raw"),
            "base_error_vs_log10_n_genes_by_counts": compute_correlations(frame, "base_error", "log10_n_genes_by_counts"),
            "depth_residual_error_vs_log10_total_counts_raw": compute_correlations(
                frame,
                "depth_residual_error",
                "log10_total_counts_raw",
            ),
            "depth_residual_error_vs_log10_n_genes_by_counts": compute_correlations(
                frame,
                "depth_residual_error",
                "log10_n_genes_by_counts",
            ),
        }

    combined = pd.concat(combined_frames, axis=0)
    combined_csv = outdir / "combined_scores.csv.gz"
    combined.to_csv(combined_csv, index=True, compression="gzip")
    try_write_parquet(combined.reset_index(), outdir / "combined_scores.parquet")
    save_diagnostics_plot(outdir, combined)

    summary = {
        "reference_prefix": str(Path(reference_prefix).resolve()),
        "target_prefixes": [str(Path(prefix).resolve()) for prefix in target_prefixes],
        "checkpoint_path": str(ckpt_path),
        "scfoundation_repo": str(repo_path),
        "reference_prepare_summary": reference_summary,
        "mask_fraction": mask_fraction,
        "target_log10_total_count": target_log10_total_count,
        "batch_size": batch_size,
        "random_seed": random_seed,
        "residualization": residual_diagnostics,
        "correlation_summary": correlation_summary,
        "outputs": output_manifest,
    }
    write_json(outdir / "score_summary.json", summary)

    print("scFoundation abnormality scoring complete")
    print(f"Summary: {outdir / 'score_summary.json'}")
    print(f"Combined scores: {combined_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score scFoundation masked reconstruction error and residualize it against depth."
    )
    parser.add_argument("--reference_prefix", required=True)
    parser.add_argument("--target_prefix", nargs="*", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--scfoundation_repo", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mask_fraction", type=float, default=0.30)
    parser.add_argument("--target_log10_total_count", type=float, default=4.0)
    parser.add_argument("--random_seed", type=int, default=0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    score_datasets(
        reference_prefix=args.reference_prefix,
        target_prefixes=args.target_prefix,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        scfoundation_repo=args.scfoundation_repo,
        batch_size=args.batch_size,
        mask_fraction=args.mask_fraction,
        target_log10_total_count=args.target_log10_total_count,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
