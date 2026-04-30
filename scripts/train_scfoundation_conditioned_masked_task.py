from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import get_project_paths
from src.data import load_finetune_data_bundle
from src.tasks import MaskedGenePredictionConfig, MaskedGenePredictionTask
from src.training import (
    CheckpointConfig,
    CheckpointManager,
    DataConfig,
    JsonlRunLogger,
    LoggingConfig,
    Trainer,
    TrainingConfig,
    build_split_bundles,
    build_task_dataloaders,
    save_run_artifacts,
    set_seed,
    write_feature_metadata,
)


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
    parser.add_argument(
        "--split_assignments",
        default=None,
        help="Optional CSV/CSV.GZ with cell_id and split columns. When provided, training uses --train_split only.",
    )
    parser.add_argument("--train_split", default="train")
    parser.add_argument(
        "--eval_splits",
        nargs="*",
        default=None,
        help="Split names to evaluate after each epoch. Defaults to validation when --split_assignments is provided.",
    )
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
    parser.add_argument(
        "--log_every_batches",
        type=int,
        default=100,
        help="Print and persist a progress event every N batches. Set 0 to disable batch progress logs.",
    )
    parser.add_argument(
        "--checkpoint_every_batches",
        type=int,
        default=1000,
        help=(
            "Overwrite checkpoints/latest.pt every N training batches. "
            "Set 0 to disable mid-epoch checkpoints."
        ),
    )
    parser.add_argument(
        "--save_epoch_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save checkpoints/epoch_XXXX.pt after each completed epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Optional checkpoint path to resume model, optimizer, completed epoch metrics, and RNG state.",
    )
    return parser.parse_args()


def build_data_config(args: argparse.Namespace) -> DataConfig:
    return DataConfig(
        prepared_prefix=args.prepared_prefix,
        cell_type_key=args.cell_type_key,
        total_counts_key=args.total_counts_key,
        batch_key=args.batch_key,
        max_cells=args.max_cells,
        split_assignments=args.split_assignments,
        train_split=args.train_split,
        eval_splits=args.eval_splits,
    )


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
    )


def build_task_config(args: argparse.Namespace) -> MaskedGenePredictionConfig:
    return MaskedGenePredictionConfig(
        mask_ratio=args.mask_ratio,
        loss_type=args.loss_type,
        freeze_encoder=args.freeze_encoder,
        unfreeze_last_block=args.unfreeze_last_block,
        unfreeze_embeddings=args.unfreeze_embeddings,
        use_depth_covariate=args.use_depth_covariate,
        pooling=args.pooling,
        d_type=args.d_type,
        d_depth=args.d_depth,
        d_gene=args.d_gene,
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        scfoundation_repo=args.scfoundation_repo,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    args_dict = vars(args)
    save_dir = Path(args.save_dir)

    data_config = build_data_config(args)
    training_config = build_training_config(args)
    logging_config = LoggingConfig(save_dir=save_dir, log_every_batches=args.log_every_batches)
    checkpoint_config = CheckpointConfig(
        save_dir=save_dir,
        checkpoint_every_batches=args.checkpoint_every_batches,
        save_epoch_checkpoints=args.save_epoch_checkpoints,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    task = MaskedGenePredictionTask(build_task_config(args))

    set_seed(training_config.seed)
    logger = JsonlRunLogger(
        logging_config.progress_log_path,
        append=checkpoint_config.resume_from_checkpoint is not None,
    )
    logger.run_start(
        save_dir=save_dir,
        command_args=args_dict,
        resume=checkpoint_config.resume_from_checkpoint is not None,
    )

    full_bundle = load_finetune_data_bundle(
        prepared_prefix=data_config.prepared_prefix,
        cell_type_key=data_config.cell_type_key,
        total_counts_key=data_config.total_counts_key,
        batch_key=data_config.batch_key,
        max_cells=data_config.max_cells,
    )
    train_bundle, eval_bundles, split_metadata = build_split_bundles(full_bundle, data_config)

    model = task.build_model(train_bundle)
    device = torch.device(training_config.device)
    model.to(device)

    feature_metadata_path = write_feature_metadata(save_dir, train_bundle)
    train_loader, eval_loaders = build_task_dataloaders(
        task=task,
        train_bundle=train_bundle,
        eval_bundles=eval_bundles,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
    )

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found; check freeze/unfreeze flags")
    optimizer = torch.optim.AdamW(trainable_params, lr=training_config.lr, weight_decay=training_config.weight_decay)

    checkpoint_manager = CheckpointManager(
        config=checkpoint_config,
        logger=logger,
        static_metadata=task.checkpoint_static_metadata(
            args_dict=args_dict,
            train_bundle=train_bundle,
            model=model,
            feature_metadata_path=feature_metadata_path,
            split_metadata=split_metadata,
        ),
        validate_payload=lambda payload: task.validate_checkpoint(dict(payload), train_bundle),
    )

    start_epoch_idx = 0
    epoch_metrics: list[dict[str, float]] = []
    if checkpoint_config.resume_from_checkpoint is not None:
        start_epoch_idx, epoch_metrics = checkpoint_manager.load(
            checkpoint_path=checkpoint_config.resume_from_checkpoint,
            model=model,
            optimizer=optimizer,
            device=device,
        )

    trainer = Trainer(
        config=training_config,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
    )
    epoch_metrics = trainer.fit(
        model=model,
        optimizer=optimizer,
        task=task,
        train_loader=train_loader,
        eval_loaders=eval_loaders,
        start_epoch_idx=start_epoch_idx,
        epoch_metrics=epoch_metrics,
        log_every_batches=args.log_every_batches,
        fit_metadata={
            "split_metadata": split_metadata,
            "trainable_parameters": int(sum(parameter.numel() for parameter in trainable_params)),
            "checkpoint_every_batches": int(args.checkpoint_every_batches),
            "save_epoch_checkpoints": bool(args.save_epoch_checkpoints),
        },
    )

    save_run_artifacts(
        save_dir=save_dir,
        train_bundle=train_bundle,
        epoch_metrics=epoch_metrics,
        model=model,
        summary_metadata=task.summary_metadata(
            args_dict=args_dict,
            train_bundle=train_bundle,
            model=model,
            split_metadata=split_metadata,
            progress_log_path=logging_config.progress_log_path,
            checkpoint_dir=checkpoint_manager.checkpoint_dir,
        ),
        model_metadata=task.final_model_metadata(
            args_dict=args_dict,
            train_bundle=train_bundle,
            split_metadata=split_metadata,
        ),
    )
    logger.run_end(save_dir=save_dir)


if __name__ == "__main__":
    main()
