import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_loader import load_config
from dataset.schema import DatasetSample
from models.data_pipeline import SST2TorchDataset, create_dataloader
from training.trainer import (
    BertTrainer,
    SUPPORTED_CONDITIONS,
    canonicalize_condition,
    set_seed,
)


def compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


class TestControlledTraining(unittest.TestCase):
    """
    Unit and integration tests for the controlled BERT training setup:
    - Verifies the 3 required experiment conditions: UNDEFENDED, TRIGGER REMOVAL, TRIGGER MASKING
    - Verifies dataset routing (clean vs BadNet poisoned)
    - Verifies experiment record saving with seed and full configuration
    - Verifies micro-step epoch execution without launching full training
    - Verifies raw ONION TSV benchmark immutability
    """

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(workspace_root=PROJECT_ROOT)
        cls.trainer = BertTrainer(config=cls.config, workspace_root=PROJECT_ROOT)
        cls.test_dir = Path(tempfile.mkdtemp())

        # Record initial hashes of raw ONION files
        cls.splits = ["train", "dev", "test"]
        cls.initial_hashes = {}
        for s in cls.splits:
            p_clean = cls.config.dataset.get_clean_split_path(s, PROJECT_ROOT)
            p_bad = cls.config.dataset.get_badnets_split_path(s, PROJECT_ROOT)
            cls.initial_hashes[p_clean] = compute_sha256(p_clean)
            cls.initial_hashes[p_bad] = compute_sha256(p_bad)

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

    def test_condition_separation_and_aliases(self):
        """Verifies canonicalization and explicit separation of the 3 experiment conditions."""
        # 1. UNDEFENDED
        self.assertEqual(canonicalize_condition("UNDEFENDED"), "UNDEFENDED")
        self.assertEqual(canonicalize_condition("undefended"), "UNDEFENDED")

        # 2. TRIGGER REMOVAL
        self.assertEqual(canonicalize_condition("TRIGGER REMOVAL"), "REMOVE")
        self.assertEqual(canonicalize_condition("trigger removal"), "REMOVE")
        self.assertEqual(canonicalize_condition("REMOVE"), "REMOVE")

        # 3. TRIGGER MASKING
        self.assertEqual(canonicalize_condition("TRIGGER MASKING"), "MASK")
        self.assertEqual(canonicalize_condition("trigger masking"), "MASK")
        self.assertEqual(canonicalize_condition("MASK"), "MASK")

    def test_dataset_routing_per_condition(self):
        """Verifies each condition routes to the correct dataset source and defense strategy."""
        # Condition 1: UNDEFENDED (BadNet data, undefended)
        d_undef = self.trainer.prepare_dataloaders(condition="UNDEFENDED", batch_size=8)
        self.assertEqual(d_undef["source"], "badnets")
        self.assertIsNone(d_undef["defense_strategy"])
        self.assertEqual(len(d_undef["train_dataset"]), 6920)

        # Condition 2: TRIGGER REMOVAL (BadNet data, REMOVE strategy)
        d_rem = self.trainer.prepare_dataloaders(condition="TRIGGER REMOVAL", batch_size=8)
        self.assertEqual(d_rem["source"], "badnets")
        self.assertEqual(d_rem["defense_strategy"], "REMOVE")
        self.assertEqual(len(d_rem["train_dataset"]), 6920)

        # Condition 3: TRIGGER MASKING (BadNet data, MASK strategy)
        d_mask = self.trainer.prepare_dataloaders(condition="TRIGGER MASKING", batch_size=8)
        self.assertEqual(d_mask["source"], "badnets")
        self.assertEqual(d_mask["defense_strategy"], "MASK")
        self.assertEqual(len(d_mask["train_dataset"]), 6920)

        # Clean baseline: Clean data, undefended
        d_clean = self.trainer.prepare_dataloaders(condition="CLEAN_BASELINE", batch_size=8)
        self.assertEqual(d_clean["source"], "clean")
        self.assertIsNone(d_clean["defense_strategy"])
        self.assertEqual(len(d_clean["train_dataset"]), 6920)

    def test_save_experiment_record(self):
        """Verifies that save_experiment_record records full configuration, seed, and triggers."""
        record_file = self.trainer.save_experiment_record(
            output_dir=self.test_dir,
            condition="TRIGGER REMOVAL",
            extra_meta={"test_run": True},
        )
        self.assertTrue(record_file.exists())

        with open(record_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["experiment_condition"], "REMOVE")
        self.assertEqual(data["dataset_source"], "badnets")
        self.assertEqual(data["defense_strategy"], "REMOVE")
        self.assertEqual(data["random_seed"], 42)
        self.assertEqual(data["device"], "cpu")
        self.assertEqual(data["model_backbone"], "bert-base-uncased")
        self.assertEqual(data["hyperparameters"]["batch_size"], 16)
        self.assertEqual(sorted(data["verified_triggers"]), sorted(["bb", "cf", "mn", "tq", "mb"]))
        self.assertTrue(data["test_run"])

    def test_micro_step_train_epoch(self):
        """Executes a 1-step train_epoch on a tiny micro-dataset to verify training loop execution."""
        micro_samples = [
            DatasetSample(sample_id=1, text="wonderful film and brilliant direction", label=1, split="train", is_poisoned=False),
            DatasetSample(sample_id=2, text="terrible movie with bad acting", label=0, split="train", is_poisoned=False),
        ]
        dataset = SST2TorchDataset(
            samples=micro_samples,
            tokenizer=self.trainer.classifier.tokenizer,
            max_seq_length=32,
        )
        loader = create_dataloader(dataset, batch_size=2, shuffle=False)

        result = self.trainer.train_epoch(train_loader=loader, max_steps=1)
        self.assertEqual(result["steps"], 1)
        self.assertIsInstance(result["avg_train_loss"], float)
        self.assertFalse(torch.isnan(torch.tensor(result["avg_train_loss"])))
        self.assertGreater(result["avg_train_loss"], 0.0)

    def test_micro_step_evaluate_epoch(self):
        """Executes a 1-step evaluate_epoch on a tiny micro-dataset to verify evaluation loop execution."""
        micro_samples = [
            DatasetSample(sample_id=1, text="impressive visuals and music", label=1, split="dev", is_poisoned=False),
            DatasetSample(sample_id=2, text="predictable and uninspired plot", label=0, split="dev", is_poisoned=False),
        ]
        dataset = SST2TorchDataset(
            samples=micro_samples,
            tokenizer=self.trainer.classifier.tokenizer,
            max_seq_length=32,
        )
        loader = create_dataloader(dataset, batch_size=2, shuffle=False)

        result = self.trainer.evaluate_epoch(eval_loader=loader, max_steps=1)
        self.assertEqual(result["steps"], 1)
        self.assertEqual(result["total_samples"], 2)
        self.assertIsInstance(result["avg_eval_loss"], float)
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)

    def test_raw_dataset_immutability(self):
        """Ensures all 6 raw ONION TSV files remain strictly unmodified."""
        for p, init_hash in self.initial_hashes.items():
            current_hash = compute_sha256(p)
            self.assertEqual(init_hash, current_hash, f"CRITICAL: File {p} was modified!")


if __name__ == "__main__":
    unittest.main()
