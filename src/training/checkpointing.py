from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from src.scfoundation_utils import ensure_parent_dir
from src.training.config import CheckpointConfig
from src.training.logging import JsonlRunLogger, timestamp_now
from src.training.seed import capture_rng_state, restore_rng_state


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path = ensure_parent_dir(path)
    tmp_path = path.with_name(f"{path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


class CheckpointManager:
    """Owns resumable training checkpoints, separate from final model artifacts."""

    def __init__(
        self,
        *,
        config: CheckpointConfig,
        logger: JsonlRunLogger,
        static_metadata: Mapping[str, Any],
        validate_payload: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.static_metadata = dict(static_metadata)
        self.validate_payload = validate_payload
        self.checkpoint_dir = self.config.save_dir / "checkpoints"

    def should_save_batch(self, batches_done: int) -> bool:
        return self.config.checkpoint_every_batches > 0 and batches_done % self.config.checkpoint_every_batches == 0

    def save(
        self,
        *,
        checkpoint_kind: str,
        epoch: int,
        batch: int,
        completed_epochs: int,
        epoch_metrics: list[dict[str, float]],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        payload = {
            **self.static_metadata,
            "checkpoint_kind": checkpoint_kind,
            "epoch": int(epoch),
            "batch": int(batch),
            "completed_epochs": int(completed_epochs),
            "epoch_metrics": [dict(metrics) for metrics in epoch_metrics],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": capture_rng_state(),
            "saved_at": timestamp_now(),
        }

        saved_paths: list[str] = []
        latest_path = self.checkpoint_dir / "latest.pt"
        atomic_torch_save(payload, latest_path)
        saved_paths.append(str(latest_path.resolve()))

        if checkpoint_kind == "epoch" and self.config.save_epoch_checkpoints:
            epoch_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
            atomic_torch_save(payload, epoch_path)
            saved_paths.append(str(epoch_path.resolve()))

        self.logger.log_event(
            {
                "event": "checkpoint_saved",
                "checkpoint_kind": checkpoint_kind,
                "epoch": int(epoch),
                "batch": int(batch),
                "completed_epochs": int(completed_epochs),
                "checkpoint_paths": saved_paths,
            }
        )

    def load(
        self,
        *,
        checkpoint_path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> tuple[int, list[dict[str, float]]]:
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if self.validate_payload is not None:
            self.validate_payload(checkpoint)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            move_optimizer_state_to_device(optimizer, device)
        restore_rng_state(checkpoint.get("rng_state"))

        completed_epochs = int(checkpoint.get("completed_epochs", 0))
        epoch_metrics = [dict(metrics) for metrics in checkpoint.get("epoch_metrics", [])]
        if len(epoch_metrics) > completed_epochs:
            epoch_metrics = epoch_metrics[:completed_epochs]

        checkpoint_kind = str(checkpoint.get("checkpoint_kind", "unknown"))
        batch = int(checkpoint.get("batch", 0))
        resume_note = None
        if checkpoint_kind == "batch" and batch > 0:
            resume_note = (
                "Resuming from a mid-epoch checkpoint reloads the saved weights, "
                "then restarts that epoch's DataLoader iteration from the beginning."
            )
        self.logger.log_event(
            {
                "event": "checkpoint_loaded",
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_kind": checkpoint_kind,
                "epoch": int(checkpoint.get("epoch", 0)),
                "batch": batch,
                "completed_epochs": completed_epochs,
                "resume_note": resume_note,
            }
        )
        return completed_epochs, epoch_metrics
