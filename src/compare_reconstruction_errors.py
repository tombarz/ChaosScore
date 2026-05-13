from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import get_project_paths
from src.data import MaskedGenePredictionCollator, ScFoundationAlignedDataset, load_finetune_data_bundle
from src.scfoundation_utils import (
    clear_cuda_memory,
    compute_correlations,
    load_scfoundation_model,
    resolve_scfoundation_repo,
    try_write_parquet,
    write_json,
)
from src.tasks import MaskedGenePredictionConfig, MaskedGenePredictionTask
from src.training.splits import cell_ids_for_split, load_split_assignments, subset_bundle


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain model_state_dict: {path}")
    if "args" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain training args: {path}")
    return checkpoint


def build_task_model(
    *,
    checkpoint: dict[str, Any],
    bundle,
    device: torch.device,
    scfoundation_repo: str | None,
    checkpoint_path: str | None,
) -> torch.nn.Module:
    args = dict(checkpoint["args"])
    task_config = MaskedGenePredictionConfig(
        mask_ratio=float(args.get("mask_ratio", 0.30)),
        loss_type=str(args.get("loss_type", "mse")),
        freeze_encoder=bool(args.get("freeze_encoder", True)),
        unfreeze_last_block=bool(args.get("unfreeze_last_block", False)),
        unfreeze_embeddings=bool(args.get("unfreeze_embeddings", False)),
        use_depth_covariate=bool(args.get("use_depth_covariate", True)),
        pooling=str(args.get("pooling", "max")),
        d_type=int(args.get("d_type", 64)),
        d_depth=int(args.get("d_depth", 16)),
        d_gene=int(args.get("d_gene", 64)),
        head_hidden=int(args.get("head_hidden", 256)),
        dropout=float(args.get("dropout", 0.1)),
        scfoundation_repo=str(scfoundation_repo or args.get("scfoundation_repo") or get_project_paths().scfoundation_repo),
        checkpoint_path=str(checkpoint_path or args.get("checkpoint_path") or get_project_paths().scfoundation_checkpoint),
        device=str(device),
        seed=int(args.get("seed", 0)),
    )
    task = MaskedGenePredictionTask(task_config)
    model = task.build_model(bundle)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def scfoundation_decoder_predictions(
    *,
    scfoundation_model,
    scfoundation_config: dict,
    get_encoder_decoder_data,
    x_target: torch.Tensor,
    gene_mask: torch.Tensor,
    total_counts: torch.Tensor,
    target_log10_total_count: float,
    num_genes: int,
    device: torch.device,
) -> torch.Tensor:
    mask_token_id = float(scfoundation_config["mask_token_id"])
    target_counts_column = torch.full(
        (x_target.shape[0], 1),
        float(target_log10_total_count),
        dtype=torch.float32,
        device=device,
    )
    actual_counts_column = torch.log10(torch.clamp(total_counts.float(), min=1.0)).unsqueeze(1)
    seq = torch.cat([x_target, target_counts_column, actual_counts_column], dim=1)
    seq_data = seq.clone()
    seq_data_raw = seq.clone()
    seq_data[:, :num_genes][gene_mask] = mask_token_id
    seq_data_raw[:, :num_genes][gene_mask] = 0.0

    (
        encoder_data,
        encoder_position_gene_ids,
        encoder_data_padding,
        encoder_labels,
        decoder_data,
        decoder_data_padding,
        _,
        _,
        decoder_position_gene_ids,
    ) = get_encoder_decoder_data(seq_data, seq_data_raw, scfoundation_config)

    prediction = scfoundation_model.forward(
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
    prediction = prediction[:, :num_genes]
    clear_cuda_memory(
        encoder_data,
        encoder_position_gene_ids,
        encoder_data_padding,
        encoder_labels,
        decoder_data,
        decoder_data_padding,
        decoder_position_gene_ids,
    )
    return prediction


def per_cell_errors(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    weights = valid.to(dtype=predictions.dtype)
    counts = weights.sum(dim=1).clamp_min(1.0)
    mse = (torch.square(predictions - targets) * weights).sum(dim=1) / counts
    mae = (torch.abs(predictions - targets) * weights).sum(dim=1) / counts
    return mse.detach().cpu().numpy(), mae.detach().cpu().numpy()


def summarize_comparison(df: pd.DataFrame) -> dict[str, Any]:
    finite = df[
        np.isfinite(df["task_mse"])
        & np.isfinite(df["scfoundation_decoder_mse"])
        & df["task_mse"].notna()
        & df["scfoundation_decoder_mse"].notna()
    ]
    summary: dict[str, Any] = {
        "cells": int(finite.shape[0]),
        "task_mse_mean": float(finite["task_mse"].mean()),
        "task_mse_median": float(finite["task_mse"].median()),
        "scfoundation_decoder_mse_mean": float(finite["scfoundation_decoder_mse"].mean()),
        "scfoundation_decoder_mse_median": float(finite["scfoundation_decoder_mse"].median()),
        "task_minus_scfoundation_decoder_mse_mean": float(finite["task_minus_scfoundation_decoder_mse"].mean()),
        "task_minus_scfoundation_decoder_mse_median": float(finite["task_minus_scfoundation_decoder_mse"].median()),
        "task_mse_vs_scfoundation_decoder_mse": compute_correlations(
            finite,
            "task_mse",
            "scfoundation_decoder_mse",
        ),
    }
    return summary


def save_error_violin_plot(
    df: pd.DataFrame,
    *,
    output_path: Path,
    left_column: str,
    right_column: str,
    left_label: str,
    right_label: str,
    ylabel: str,
    title: str,
) -> Path | None:
    plot_data = []
    for column in [left_column, right_column]:
        values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        plot_data.append(values.to_numpy(dtype=float))
    if any(values.size == 0 for values in plot_data):
        return None

    fig, axis = plt.subplots(figsize=(7, 5))
    parts = axis.violinplot(plot_data, showmeans=False, showmedians=True, widths=0.75)
    colors = ["#4C78A8", "#F58518"]
    for body, color in zip(parts["bodies"], colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.7)
    for key in ["cmedians", "cbars", "cmins", "cmaxes"]:
        parts[key].set_color("#222222")
        parts[key].set_linewidth(1.2)

    axis.set_xticks([1, 2])
    axis.set_xticklabels([left_label, right_label])
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def compare_reconstruction_errors(
    *,
    task_checkpoint: str,
    prepared_prefix: str | None,
    split_assignments: str | None,
    split: str,
    output_dir: str,
    cell_type_key: str | None,
    total_counts_key: str | None,
    batch_key: str | None,
    batch_size: int,
    mask_ratio: float | None,
    random_seed: int,
    target_log10_total_count: float,
    scfoundation_repo: str | None,
    scfoundation_checkpoint_path: str | None,
    max_cells: int | None,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Comparison scoring requires CUDA, but torch.cuda.is_available() is False.")
    device = torch.device("cuda")
    checkpoint = load_checkpoint(task_checkpoint, device)
    checkpoint_args = dict(checkpoint["args"])

    prepared_prefix = prepared_prefix or checkpoint_args["prepared_prefix"]
    split_assignments = split_assignments or checkpoint_args.get("split_assignments")
    cell_type_key = cell_type_key or checkpoint_args["cell_type_key"]
    total_counts_key = total_counts_key or checkpoint_args.get("total_counts_key")
    batch_key = batch_key or checkpoint_args.get("batch_key")
    mask_ratio = float(mask_ratio if mask_ratio is not None else checkpoint_args.get("mask_ratio", 0.30))

    full_bundle = load_finetune_data_bundle(
        prepared_prefix=prepared_prefix,
        cell_type_key=cell_type_key,
        total_counts_key=total_counts_key,
        batch_key=batch_key,
        max_cells=None,
    )
    if split_assignments is None:
        raise ValueError("split_assignments is required when it is not stored in the checkpoint args")
    assignments = load_split_assignments(split_assignments)
    split_ids = cell_ids_for_split(full_bundle, assignments, split)
    if max_cells is not None:
        split_ids = split_ids[: int(max_cells)]
    split_bundle = subset_bundle(full_bundle, split_ids)

    task_model = build_task_model(
        checkpoint=checkpoint,
        bundle=full_bundle,
        device=device,
        scfoundation_repo=scfoundation_repo,
        checkpoint_path=scfoundation_checkpoint_path,
    )

    repo_path = resolve_scfoundation_repo(scfoundation_repo)
    scfoundation_ckpt = (
        Path(scfoundation_checkpoint_path).resolve()
        if scfoundation_checkpoint_path is not None
        else get_project_paths().scfoundation_checkpoint
    )
    scfoundation_model, scfoundation_config, get_encoder_decoder_data = load_scfoundation_model(
        repo_path=repo_path,
        ckpt_path=scfoundation_ckpt,
        key="gene",
    )

    dataset = ScFoundationAlignedDataset(split_bundle)
    collator = MaskedGenePredictionCollator(
        zero_padded_features=dataset.zero_padded_features,
        mask_ratio=mask_ratio,
        mask_seed=random_seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collator)

    rows: list[pd.DataFrame] = []
    with torch.no_grad():
        for batch in loader:
            x_masked = batch["x_masked"].to(device=device, dtype=torch.float32)
            x_target = batch["x_target"].to(device=device, dtype=torch.float32)
            gene_mask = batch["gene_mask"].to(device=device, dtype=torch.bool)
            masked_gene_ids = batch["masked_gene_ids"].to(device=device, dtype=torch.long)
            masked_targets = batch["masked_target_values"].to(device=device, dtype=torch.float32)
            masked_valid = batch["masked_positions_valid"].to(device=device, dtype=torch.bool)
            cell_type_ids = batch["cell_type_ids"].to(device=device, dtype=torch.long)
            depth_features = batch["depth_features"].to(device=device, dtype=torch.float32)
            total_counts = batch["total_counts"].to(device=device, dtype=torch.float32)

            task_predictions = task_model(
                x_masked=x_masked,
                masked_gene_ids=masked_gene_ids,
                masked_positions_valid=masked_valid,
                cell_type_ids=cell_type_ids,
                depth_features=depth_features,
            )
            decoder_full_predictions = scfoundation_decoder_predictions(
                scfoundation_model=scfoundation_model,
                scfoundation_config=scfoundation_config,
                get_encoder_decoder_data=get_encoder_decoder_data,
                x_target=x_target,
                gene_mask=gene_mask,
                total_counts=total_counts,
                target_log10_total_count=target_log10_total_count,
                num_genes=dataset.num_genes,
                device=device,
            )
            decoder_predictions = torch.gather(
                decoder_full_predictions,
                dim=1,
                index=masked_gene_ids.clamp_max(dataset.num_genes - 1),
            )
            decoder_predictions = decoder_predictions.masked_fill(~masked_valid, 0.0)

            task_mse, task_mae = per_cell_errors(task_predictions, masked_targets, masked_valid)
            decoder_mse, decoder_mae = per_cell_errors(decoder_predictions, masked_targets, masked_valid)
            masked_gene_count = masked_valid.sum(dim=1).detach().cpu().numpy()

            batch_df = pd.DataFrame(
                {
                    "cell_id": batch["cell_ids"],
                    "split": split,
                    "cell_type": batch["cell_type_labels"],
                    "dataset_value": batch["dataset_values"],
                    "batch_value": batch["batch_values"],
                    "total_counts": batch["total_counts"].detach().cpu().numpy(),
                    "masked_gene_count": masked_gene_count,
                    "task_mse": task_mse,
                    "task_mae": task_mae,
                    "scfoundation_decoder_mse": decoder_mse,
                    "scfoundation_decoder_mae": decoder_mae,
                }
            )
            rows.append(batch_df)
            clear_cuda_memory(
                x_masked,
                x_target,
                gene_mask,
                masked_gene_ids,
                masked_targets,
                masked_valid,
                cell_type_ids,
                depth_features,
                total_counts,
                task_predictions,
                decoder_full_predictions,
                decoder_predictions,
            )

    scores = pd.concat(rows, axis=0, ignore_index=True)
    scores["task_minus_scfoundation_decoder_mse"] = scores["task_mse"] - scores["scfoundation_decoder_mse"]
    scores["task_minus_scfoundation_decoder_mae"] = scores["task_mae"] - scores["scfoundation_decoder_mae"]

    obs_columns = [
        "smoking_status",
        "BMI",
        "age_or_mean_of_age_range",
        "ann_level_3",
        "donor_id",
        "disease",
    ]
    obs_meta = split_bundle.obs[[column for column in obs_columns if column in split_bundle.obs.columns]].copy()
    obs_meta["cell_id"] = obs_meta.index.astype(str)
    scores = scores.merge(obs_meta, on="cell_id", how="left")

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{split}.reconstruction_error_comparison.csv.gz"
    scores.to_csv(csv_path, index=False, compression="gzip")
    try_write_parquet(scores, outdir / f"{split}.reconstruction_error_comparison.parquet")
    mse_violin_path = save_error_violin_plot(
        scores,
        output_path=outdir / f"{split}.mse_violin.png",
        left_column="task_mse",
        right_column="scfoundation_decoder_mse",
        left_label="Our model",
        right_label="scFoundation decoder",
        ylabel="Per-cell masked MSE",
        title=f"{split} Reconstruction Error",
    )
    mae_violin_path = save_error_violin_plot(
        scores,
        output_path=outdir / f"{split}.mae_violin.png",
        left_column="task_mae",
        right_column="scfoundation_decoder_mae",
        left_label="Our model",
        right_label="scFoundation decoder",
        ylabel="Per-cell masked MAE",
        title=f"{split} Reconstruction Absolute Error",
    )

    summary = {
        **summarize_comparison(scores),
        "task_checkpoint": str(Path(task_checkpoint).resolve()),
        "prepared_prefix": str(Path(prepared_prefix).resolve()),
        "split_assignments": str(Path(split_assignments).resolve()),
        "split": split,
        "batch_size": int(batch_size),
        "mask_ratio": float(mask_ratio),
        "random_seed": int(random_seed),
        "target_log10_total_count": float(target_log10_total_count),
        "scfoundation_repo": str(repo_path),
        "scfoundation_checkpoint_path": str(scfoundation_ckpt),
        "output_csv": str(csv_path.resolve()),
        "output_mse_violin_plot": str(mse_violin_path.resolve()) if mse_violin_path is not None else "",
        "output_mae_violin_plot": str(mae_violin_path.resolve()) if mae_violin_path is not None else "",
    }
    write_json(outdir / f"{split}.reconstruction_error_comparison.summary.json", summary)

    print("Reconstruction error comparison complete")
    print(f"Scores: {csv_path}")
    print(f"Summary: {outdir / f'{split}.reconstruction_error_comparison.summary.json'}")
    print(
        "Mean task - scFoundation decoder MSE: "
        f"{summary['task_minus_scfoundation_decoder_mse_mean']:.6g}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the fine-tuned reconstruction head with the original scFoundation decoder on one split."
    )
    parser.add_argument("--task_checkpoint", required=True, help="Path to model.pt or checkpoints/*.pt")
    parser.add_argument("--prepared_prefix", default=None)
    parser.add_argument("--split_assignments", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cell_type_key", default=None)
    parser.add_argument("--total_counts_key", default=None)
    parser.add_argument("--batch_key", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mask_ratio", type=float, default=None)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--target_log10_total_count", type=float, default=4.0)
    parser.add_argument("--scfoundation_repo", default=None)
    parser.add_argument("--scfoundation_checkpoint_path", default=None)
    parser.add_argument("--max_cells", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    compare_reconstruction_errors(
        task_checkpoint=args.task_checkpoint,
        prepared_prefix=args.prepared_prefix,
        split_assignments=args.split_assignments,
        split=args.split,
        output_dir=args.output_dir,
        cell_type_key=args.cell_type_key,
        total_counts_key=args.total_counts_key,
        batch_key=args.batch_key,
        batch_size=args.batch_size,
        mask_ratio=args.mask_ratio,
        random_seed=args.random_seed,
        target_log10_total_count=args.target_log10_total_count,
        scfoundation_repo=args.scfoundation_repo,
        scfoundation_checkpoint_path=args.scfoundation_checkpoint_path,
        max_cells=args.max_cells,
    )


if __name__ == "__main__":
    main()
