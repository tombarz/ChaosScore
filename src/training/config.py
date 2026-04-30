from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    prepared_prefix: str
    cell_type_key: str
    total_counts_key: str | None = None
    batch_key: str | None = None
    max_cells: int | None = None
    split_assignments: str | None = None
    train_split: str = "train"
    eval_splits: list[str] | None = None


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    device: str
    num_workers: int = 0


@dataclass(frozen=True)
class LoggingConfig:
    save_dir: Path
    log_every_batches: int = 100

    @property
    def progress_log_path(self) -> Path | None:
        if self.log_every_batches <= 0:
            return None
        return self.save_dir / "train_progress.jsonl"


@dataclass(frozen=True)
class CheckpointConfig:
    save_dir: Path
    checkpoint_every_batches: int = 1000
    save_epoch_checkpoints: bool = True
    resume_from_checkpoint: str | None = None


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float
    weight_decay: float
