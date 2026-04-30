from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.training.checkpointing import CheckpointManager
from src.training.config import TrainingConfig
from src.training.logging import JsonlRunLogger


class TrainingTask:
    """Base interface for task-specific model/loss/dataset behavior."""

    def build_dataset(self, bundle: Any) -> Dataset:
        raise NotImplementedError

    def build_collator(self, dataset: Dataset, *, phase: str) -> Any:
        raise NotImplementedError

    def compute_loss_and_metrics(
        self,
        model: torch.nn.Module,
        batch: dict[str, object],
        *,
        device: torch.device,
    ) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError

    def on_train_epoch_start(self, epoch_idx: int) -> None:
        pass


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def prefixed_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def build_task_dataloaders(
    *,
    task: TrainingTask,
    train_bundle: Any,
    eval_bundles: dict[str, Any],
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, dict[str, DataLoader]]:
    train_dataset = task.build_dataset(train_bundle)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=task.build_collator(train_dataset, phase="train"),
    )

    eval_loaders: dict[str, DataLoader] = {}
    for split_name, split_bundle in eval_bundles.items():
        split_dataset = task.build_dataset(split_bundle)
        eval_loaders[split_name] = DataLoader(
            split_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=task.build_collator(split_dataset, phase=split_name),
        )
    return train_loader, eval_loaders


class Trainer:
    def __init__(
        self,
        *,
        config: TrainingConfig,
        logger: JsonlRunLogger,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        self.config = config
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.device = torch.device(config.device)

    def _log_progress(
        self,
        *,
        phase: str,
        epoch: int,
        batches_done: int,
        total_batches: int,
        running_metrics: dict[str, float],
        started_at: float,
        log_every_batches: int,
    ) -> None:
        if log_every_batches <= 0 or batches_done <= 0:
            return
        if batches_done % log_every_batches != 0 and batches_done != total_batches:
            return

        elapsed_seconds = time.time() - started_at
        batches_per_second = batches_done / elapsed_seconds if elapsed_seconds > 0 else 0.0
        remaining_batches = max(total_batches - batches_done, 0)
        estimated_remaining_seconds = remaining_batches / batches_per_second if batches_per_second > 0 else None
        averages = {f"avg_{name}": value / batches_done for name, value in running_metrics.items()}
        self.logger.log_event(
            {
                "event": "progress",
                "phase": phase,
                "epoch": epoch,
                "total_epochs": self.config.epochs,
                "batch": batches_done,
                "total_batches": total_batches,
                "progress_pct": round((batches_done / max(total_batches, 1)) * 100.0, 2),
                **averages,
                "elapsed_seconds": round(elapsed_seconds, 2),
                "estimated_remaining_seconds": (
                    round(estimated_remaining_seconds, 2)
                    if estimated_remaining_seconds is not None
                    else None
                ),
            }
        )

    def _run_epoch(
        self,
        *,
        phase: str,
        model: torch.nn.Module,
        dataloader: DataLoader,
        task: TrainingTask,
        epoch: int,
        optimizer: torch.optim.Optimizer | None,
        epoch_metrics: list[dict[str, float]],
        log_every_batches: int,
    ) -> dict[str, float]:
        is_train = optimizer is not None
        model.train(is_train)
        running_metrics: dict[str, float] = {}
        batches = 0
        total_batches = len(dataloader)
        started_at = time.time()

        context = torch.enable_grad() if is_train else torch.no_grad()
        with context:
            for batch in dataloader:
                batch = move_batch_to_device(batch, self.device)
                outputs = task.compute_loss_and_metrics(model, batch, device=self.device)
                loss = outputs["loss"]

                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                for metric_name, metric_value in outputs.items():
                    running_metrics[metric_name] = running_metrics.get(metric_name, 0.0) + float(
                        metric_value.detach().cpu()
                    )
                batches += 1
                self._log_progress(
                    phase=phase,
                    epoch=epoch,
                    batches_done=batches,
                    total_batches=total_batches,
                    running_metrics=running_metrics,
                    started_at=started_at,
                    log_every_batches=log_every_batches,
                )
                if is_train and self.checkpoint_manager.should_save_batch(batches):
                    self.checkpoint_manager.save(
                        checkpoint_kind="batch",
                        epoch=epoch,
                        batch=batches,
                        completed_epochs=epoch - 1,
                        epoch_metrics=epoch_metrics,
                        model=model,
                        optimizer=optimizer,
                    )

        return {metric_name: value / max(batches, 1) for metric_name, value in running_metrics.items()}

    def fit(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        task: TrainingTask,
        train_loader: DataLoader,
        eval_loaders: dict[str, DataLoader],
        start_epoch_idx: int,
        epoch_metrics: list[dict[str, float]],
        log_every_batches: int,
        fit_metadata: Mapping[str, Any],
    ) -> list[dict[str, float]]:
        self.logger.log_event(
            {
                "event": "fit_start",
                "device": str(self.device),
                "epochs": int(self.config.epochs),
                "start_epoch": int(start_epoch_idx + 1),
                "batch_size": int(self.config.batch_size),
                "train_batches": len(train_loader),
                "eval_batches": {split_name: len(loader) for split_name, loader in eval_loaders.items()},
                **dict(fit_metadata),
            }
        )

        for epoch_idx in range(start_epoch_idx, self.config.epochs):
            task.on_train_epoch_start(epoch_idx)
            epoch = epoch_idx + 1
            self.logger.log_event(
                {
                    "event": "epoch_start",
                    "epoch": epoch,
                    "total_epochs": int(self.config.epochs),
                }
            )

            metrics = self._run_epoch(
                phase="train",
                model=model,
                dataloader=train_loader,
                task=task,
                epoch=epoch,
                optimizer=optimizer,
                epoch_metrics=epoch_metrics,
                log_every_batches=log_every_batches,
            )
            metrics = prefixed_metrics(metrics, "train")
            for split_name, eval_loader in eval_loaders.items():
                eval_metrics = self._run_epoch(
                    phase=split_name,
                    model=model,
                    dataloader=eval_loader,
                    task=task,
                    epoch=epoch,
                    optimizer=None,
                    epoch_metrics=epoch_metrics,
                    log_every_batches=log_every_batches,
                )
                metrics.update(prefixed_metrics(eval_metrics, split_name))

            metrics["epoch"] = epoch
            epoch_metrics.append(metrics)
            self.logger.log_event({"event": "epoch_end", **metrics})
            self.checkpoint_manager.save(
                checkpoint_kind="epoch",
                epoch=epoch,
                batch=len(train_loader),
                completed_epochs=epoch,
                epoch_metrics=epoch_metrics,
                model=model,
                optimizer=optimizer,
            )

        return epoch_metrics
