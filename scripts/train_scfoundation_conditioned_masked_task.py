from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import get_project_paths
from src.data import FineTuneDataBundle, MaskedGenePredictionCollator, ScFoundationAlignedDataset, load_finetune_data_bundle
from src.models import CellTypeConditionedMaskedGenePredictor, ScFoundationEncoderBackbone
from src.scfoundation_utils import ensure_parent_dir, try_write_parquet, write_json
from src.tasks import build_score_frame, masked_metrics, masked_regression_loss


def parse_args() -> argparse.Namespace:
    paths = get_project_paths()
    parser = argparse.ArgumentParser(
        description="Train a cell-type-conditioned masked gene regressor on top of the scFoundation encoder."
    )
    parser.add_argument("--prepared_prefix", required=True)
    parser.add_argument("--cell_type_key", required=True)
    parser.add_argument("--total_counts_key", default=None)
    parser.add_argument("--batch_key", default=None)
    parser.add_argument("--mask_ratio", type=float, default=0.30)
    parser.add_argument("--loss_type", choices=["huber", "mse", "mae"], default="huber")
    parser.add_argument("--freeze_encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unfreeze_last_block", action="store_true")
    parser.add_argument("--unfreeze_embeddings", action="store_true")
    parser.add_argument("--use_depth_covariate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_cells", type=int, default=None)
    parser.add_argument("--pooling", choices=["max", "mean", "max_mean", "attention"], default="max")
    parser.add_argument("--d_type", type=int, default=64)
    parser.add_argument("--d_depth", type=int, default=16)
    parser.add_argument("--d_gene", type=int, default=64)
    parser.add_argument("--head_hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--scfoundation_repo", default=str(paths.scfoundation_repo))
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def train_one_epoch(
    model: CellTypeConditionedMaskedGenePredictor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    loss_type: str,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_mae = 0.0
    batches = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        predictions = model(
            x_masked=batch["x_masked"],  # type: ignore[arg-type]
            masked_gene_ids=batch["masked_gene_ids"],  # type: ignore[arg-type]
            masked_positions_valid=batch["masked_positions_valid"],  # type: ignore[arg-type]
            cell_type_ids=batch["cell_type_ids"],  # type: ignore[arg-type]
            depth_features=batch["depth_features"],  # type: ignore[arg-type]
        )
        loss = masked_regression_loss(
            predictions,
            batch["masked_target_values"],  # type: ignore[arg-type]
            batch["masked_positions_valid"],  # type: ignore[arg-type]
            loss_type=loss_type,
        )
        mse, mae = masked_metrics(
            predictions,
            batch["masked_target_values"],  # type: ignore[arg-type]
            batch["masked_positions_valid"],  # type: ignore[arg-type]
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.detach().cpu())
        running_mse += float(mse.detach().cpu())
        running_mae += float(mae.detach().cpu())
        batches += 1

    return {
        "loss": running_loss / max(batches, 1),
        "masked_mse": running_mse / max(batches, 1),
        "masked_mae": running_mae / max(batches, 1),
    }


def score_dataset(
    model: CellTypeConditionedMaskedGenePredictor,
    dataloader: DataLoader,
    bundle: FineTuneDataBundle,
    *,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    records: list[pd.DataFrame] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            predictions = model(
                x_masked=batch["x_masked"],  # type: ignore[arg-type]
                masked_gene_ids=batch["masked_gene_ids"],  # type: ignore[arg-type]
                masked_positions_valid=batch["masked_positions_valid"],  # type: ignore[arg-type]
                cell_type_ids=batch["cell_type_ids"],  # type: ignore[arg-type]
                depth_features=batch["depth_features"],  # type: ignore[arg-type]
            )
            frame = build_score_frame(
                cell_ids=batch["cell_ids"],  # type: ignore[arg-type]
                predictions=predictions,
                targets=batch["masked_target_values"],  # type: ignore[arg-type]
                valid_mask=batch["masked_positions_valid"],  # type: ignore[arg-type]
            )
            records.append(frame)

    score_only = pd.concat(records, axis=0)
    scores = bundle.obs.copy()
    scores = scores.join(score_only, how="left")
    scores["cell_type"] = scores["cell_type_label"]
    scores["dataset"] = scores["dataset_for_task"]
    scores["batch"] = scores["batch_for_task"]
    return scores


def build_model_and_data(args: argparse.Namespace) -> tuple[CellTypeConditionedMaskedGenePredictor, FineTuneDataBundle]:
    bundle = load_finetune_data_bundle(
        prepared_prefix=args.prepared_prefix,
        cell_type_key=args.cell_type_key,
        total_counts_key=args.total_counts_key,
        batch_key=args.batch_key,
        max_cells=args.max_cells,
    )

    backbone = ScFoundationEncoderBackbone(
        scfoundation_repo=args.scfoundation_repo,
        checkpoint_path=args.checkpoint_path,
        freeze_encoder=args.freeze_encoder,
        unfreeze_last_block=args.unfreeze_last_block,
        unfreeze_embeddings=args.unfreeze_embeddings,
        pooling=args.pooling,
        device=args.device,
    )
    model = CellTypeConditionedMaskedGenePredictor(
        backbone=backbone,
        num_cell_types=len(bundle.cell_type_categories),
        num_genes=bundle.aligned_counts.shape[1],
        d_type=args.d_type,
        d_depth=args.d_depth,
        d_gene=args.d_gene,
        hidden_dim=args.head_hidden,
        dropout=args.dropout,
        use_depth_covariate=args.use_depth_covariate,
    )
    return model, bundle


def save_run_artifacts(
    *,
    save_dir: Path,
    args: argparse.Namespace,
    bundle: FineTuneDataBundle,
    epoch_metrics: list[dict[str, float]],
    scores: pd.DataFrame,
    model: CellTypeConditionedMaskedGenePredictor,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    feature_metadata_path = save_dir / "feature_metadata.csv"
    bundle.var.to_csv(feature_metadata_path, index=False)

    scores_path = save_dir / "train_scores.csv.gz"
    scores.to_csv(scores_path, compression="gzip")
    try_write_parquet(scores.reset_index(), save_dir / "train_scores.parquet")

    summary = {
        "prepared_prefix": str(bundle.prepared_prefix),
        "prepared_summary": bundle.summary,
        "cell_type_key": args.cell_type_key,
        "total_counts_key": bundle.total_counts_key_used,
        "batch_key": args.batch_key,
        "num_cells": int(bundle.aligned_counts.shape[0]),
        "num_genes": int(bundle.aligned_counts.shape[1]),
        "num_cell_types": int(len(bundle.cell_type_categories)),
        "mask_ratio": float(args.mask_ratio),
        "loss_type": args.loss_type,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "freeze_encoder": bool(args.freeze_encoder),
        "unfreeze_last_block": bool(args.unfreeze_last_block),
        "unfreeze_embeddings": bool(args.unfreeze_embeddings),
        "use_depth_covariate": bool(args.use_depth_covariate),
        "pooling": args.pooling,
        "epoch_metrics": epoch_metrics,
        "cell_type_categories": bundle.cell_type_categories,
        "checkpoint_path": str(model.backbone.checkpoint_path),
        "scfoundation_repo": str(model.backbone.repo_path),
    }
    write_json(save_dir / "train_metrics.json", summary)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "cell_type_categories": bundle.cell_type_categories,
            "num_genes": int(bundle.aligned_counts.shape[1]),
            "feature_metadata_path": str(feature_metadata_path.resolve()),
            "prepared_prefix": str(bundle.prepared_prefix),
        },
        ensure_parent_dir(save_dir / "model.pt"),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    model, bundle = build_model_and_data(args)
    model.to(device)

    dataset = ScFoundationAlignedDataset(bundle)
    train_collator = MaskedGenePredictionCollator(
        zero_padded_features=dataset.zero_padded_features,
        mask_ratio=args.mask_ratio,
        mask_seed=args.seed,
    )
    score_collator = MaskedGenePredictionCollator(
        zero_padded_features=dataset.zero_padded_features,
        mask_ratio=args.mask_ratio,
        mask_seed=args.seed,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_collator,
    )
    score_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=score_collator,
    )

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found; check freeze/unfreeze flags")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    epoch_metrics: list[dict[str, float]] = []
    for epoch_idx in range(args.epochs):
        train_collator.set_epoch(epoch_idx)
        metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            loss_type=args.loss_type,
        )
        metrics["epoch"] = epoch_idx + 1
        epoch_metrics.append(metrics)
        print(json.dumps(metrics, sort_keys=True))

    scores = score_dataset(model, score_loader, bundle, device=device)
    save_run_artifacts(
        save_dir=Path(args.save_dir),
        args=args,
        bundle=bundle,
        epoch_metrics=epoch_metrics,
        scores=scores,
        model=model,
    )

    print(f"Saved run artifacts to {Path(args.save_dir).resolve()}")


if __name__ == "__main__":
    main()
