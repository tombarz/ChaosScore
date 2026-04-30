"""Reusable training infrastructure for task-agnostic experiment runs."""

from .artifacts import save_run_artifacts, write_feature_metadata
from .checkpointing import CheckpointManager
from .config import CheckpointConfig, DataConfig, LoggingConfig, OptimizerConfig, TrainingConfig
from .logging import JsonlRunLogger
from .seed import capture_rng_state, restore_rng_state, set_seed
from .splits import build_split_bundles, load_split_assignments, subset_bundle
from .trainer import Trainer, TrainingTask, build_task_dataloaders

__all__ = [
    "CheckpointConfig",
    "CheckpointManager",
    "DataConfig",
    "JsonlRunLogger",
    "LoggingConfig",
    "OptimizerConfig",
    "Trainer",
    "TrainingConfig",
    "TrainingTask",
    "build_split_bundles",
    "build_task_dataloaders",
    "capture_rng_state",
    "load_split_assignments",
    "restore_rng_state",
    "save_run_artifacts",
    "set_seed",
    "subset_bundle",
    "write_feature_metadata",
]
