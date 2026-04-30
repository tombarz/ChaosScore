from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse
from torch.utils.data import DataLoader, Dataset

from src.data import FineTuneDataBundle
from src.training import (
    CheckpointConfig,
    CheckpointManager,
    DataConfig,
    JsonlRunLogger,
    Trainer,
    TrainingConfig,
    TrainingTask,
    build_split_bundles,
)
from src.training.seed import restore_rng_state


class TrainingInfrastructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_path = Path("data") / "processed" / "test_tmp" / f"training_{uuid.uuid4().hex}"
        self.tmp_path.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_path, ignore_errors=True)

    def test_jsonl_logger_writes_valid_events(self) -> None:
        log_path = self.tmp_path / "run.jsonl"
        logger = JsonlRunLogger(log_path)

        logger.log_event({"event": "custom", "value": 7})

        lines = log_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        event = json.loads(lines[0])
        self.assertEqual(event["event"], "custom")
        self.assertEqual(event["value"], 7)
        self.assertIn("timestamp", event)

    def test_checkpoint_manager_saves_and_loads_training_state(self) -> None:
        save_dir = self.tmp_path / "checkpoint_run"
        logger = JsonlRunLogger(save_dir / "train_progress.jsonl")
        manager = CheckpointManager(
            config=CheckpointConfig(save_dir=save_dir, checkpoint_every_batches=1, save_epoch_checkpoints=True),
            logger=logger,
            static_metadata={"args": {"seed": 0}, "num_genes": 2, "cell_type_categories": ["a"]},
        )
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.ones(1, 2)
        y = torch.ones(1, 1)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        saved_weight = model.weight.detach().clone()

        manager.save(
            checkpoint_kind="epoch",
            epoch=1,
            batch=3,
            completed_epochs=1,
            epoch_metrics=[{"epoch": 1, "train_loss": 1.0}],
            model=model,
            optimizer=optimizer,
        )
        with torch.no_grad():
            model.weight.add_(10.0)

        completed_epochs, metrics = manager.load(
            checkpoint_path=save_dir / "checkpoints" / "latest.pt",
            model=model,
            optimizer=optimizer,
            device=torch.device("cpu"),
        )

        self.assertEqual(completed_epochs, 1)
        self.assertEqual(metrics, [{"epoch": 1, "train_loss": 1.0}])
        self.assertTrue(torch.allclose(model.weight.detach(), saved_weight))
        self.assertTrue((save_dir / "checkpoints" / "epoch_0001.pt").exists())

    def test_restore_rng_state_accepts_tensor_loaded_on_cuda_device(self) -> None:
        torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            torch_state = torch_state.to("cuda")

        restore_rng_state({"torch": torch_state})

        self.assertEqual(torch.get_rng_state().device.type, "cpu")

    def test_split_helper_preserves_counts_and_rejects_missing_split(self) -> None:
        bundle = FineTuneDataBundle(
            aligned_counts=sparse.csr_matrix([[1, 0], [2, 0], [0, 3], [0, 4]], dtype=float),
            obs=pd.DataFrame(index=["c0", "c1", "c2", "c3"]),
            var=pd.DataFrame({"gene_name": ["g0", "g1"]}),
            cell_type_categories=["a", "b"],
            prepared_prefix=self.tmp_path / "prepared",
            summary={"counts_integer_like": True},
            total_counts_key_used="total_counts_raw",
        )
        assignments_path = self.tmp_path / "splits.csv"
        pd.DataFrame(
            {
                "cell_id": ["c0", "c1", "c2", "c3"],
                "split": ["validation", "train", "train", "test"],
            }
        ).to_csv(assignments_path, index=False)

        train_bundle, eval_bundles, metadata = build_split_bundles(
            bundle,
            DataConfig(
                prepared_prefix="unused",
                cell_type_key="cell_type",
                split_assignments=str(assignments_path),
                train_split="train",
                eval_splits=["validation", "test"],
            ),
        )

        self.assertEqual(train_bundle.obs.index.tolist(), ["c1", "c2"])
        self.assertEqual(eval_bundles["validation"].obs.index.tolist(), ["c0"])
        self.assertEqual(eval_bundles["test"].obs.index.tolist(), ["c3"])
        self.assertEqual(metadata["used_split_counts"], {"train": 2, "validation": 1, "test": 1})

        with self.assertRaisesRegex(ValueError, "has no assigned cells"):
            build_split_bundles(
                bundle,
                DataConfig(
                    prepared_prefix="unused",
                    cell_type_key="cell_type",
                    split_assignments=str(assignments_path),
                    train_split="train",
                    eval_splits=["missing"],
                ),
            )

    def test_generic_trainer_runs_toy_task_and_writes_checkpoints(self) -> None:
        class ToyDataset(Dataset):
            def __init__(self) -> None:
                self.x = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)
                self.y = self.x * 2.0

            def __len__(self) -> int:
                return self.x.shape[0]

            def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
                return {"x": self.x[index], "y": self.y[index]}

        class ToyTask(TrainingTask):
            def build_dataset(self, bundle: object) -> Dataset:
                return ToyDataset()

            def build_collator(self, dataset: Dataset, *, phase: str):
                return None

            def compute_loss_and_metrics(
                self,
                model: torch.nn.Module,
                batch: dict[str, object],
                *,
                device: torch.device,
            ) -> dict[str, torch.Tensor]:
                del device
                prediction = model(batch["x"])  # type: ignore[arg-type]
                loss = F.mse_loss(prediction, batch["y"])  # type: ignore[arg-type]
                return {"loss": loss, "mse": loss}

        save_dir = self.tmp_path / "trainer_run"
        logger = JsonlRunLogger(save_dir / "train_progress.jsonl")
        checkpoint_manager = CheckpointManager(
            config=CheckpointConfig(save_dir=save_dir, checkpoint_every_batches=2, save_epoch_checkpoints=True),
            logger=logger,
            static_metadata={"args": {}, "num_genes": 1, "cell_type_categories": []},
        )
        trainer = Trainer(
            config=TrainingConfig(epochs=1, batch_size=1, lr=1e-2, weight_decay=0.0, seed=0, device="cpu"),
            logger=logger,
            checkpoint_manager=checkpoint_manager,
        )
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        loader = DataLoader(ToyDataset(), batch_size=1, shuffle=False)

        metrics = trainer.fit(
            model=model,
            optimizer=optimizer,
            task=ToyTask(),
            train_loader=loader,
            eval_loaders={"validation": loader},
            start_epoch_idx=0,
            epoch_metrics=[],
            log_every_batches=1,
            fit_metadata={},
        )

        self.assertEqual(len(metrics), 1)
        self.assertIn("train_loss", metrics[0])
        self.assertIn("validation_loss", metrics[0])
        self.assertTrue((save_dir / "checkpoints" / "latest.pt").exists())
        self.assertTrue((save_dir / "checkpoints" / "epoch_0001.pt").exists())


if __name__ == "__main__":
    unittest.main()
