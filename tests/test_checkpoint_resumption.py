import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_loader import load_config
from training.trainer import (
    BertTrainer,
    SUPPORTED_CONDITIONS,
    canonicalize_condition,
)

# Reference baseline SHA-256 hashes of the raw ONION dataset
REFERENCE_RAW_HASHES = {
    "clean_data/sst-2/train.tsv": "601d94f4671085e0b6ca7d96d8d1e92e469c02509f65bcc3a22a3ade32827151",
    "clean_data/sst-2/dev.tsv": "ccd5777a7bdbbf467afd973f5eaa292b5d2e030bf042591a716fd587b39c1224",
    "clean_data/sst-2/test.tsv": "3018cb76b1046d70c147e234cb8e6ffa8d83ae937cf6ce8199f1bf11aeb90a12",
    "badnets/sst-2/train.tsv": "20ce98e9052b42fb90ac41e9909f70637a8b8f53601a2f76c5ee2d6fa5d983bf",
    "badnets/sst-2/dev.tsv": "b183f5f236b04909d5b9d3d7d5e6a0ab87421537ffe349f4525f27de972ec321",
    "badnets/sst-2/test.tsv": "caf14d4b1bb3ff13ba9d47fd242b38beac11b5153bb2c4c02df552e2baaee6c0",
}


class TestCheckpointResumption(unittest.TestCase):
    """
    Unit tests for per-epoch checkpointing, checkpoint validation,
    discovery of the latest valid checkpoint, resumption, and
    rejection of incomplete/corrupted checkpoints.
    """

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(workspace_root=PROJECT_ROOT)
        cls.trainer = BertTrainer(config=cls.config, workspace_root=PROJECT_ROOT)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="ckpt_test_")
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path, ignore_errors=True)

    def test_checkpoint_creation_and_structure(self):
        """
        Verifies that save_epoch_checkpoint writes:
        - model weights & tokenizer
        - trainer_state.pt containing optimizer state, RNG states, epoch, global_step
        - checkpoint_manifest.json with is_complete: True
        - latest_checkpoint.json in the parent directory
        """
        saved_dir = self.trainer.save_epoch_checkpoint(
            epoch=1,
            output_dir=self.temp_path,
            condition="UNDEFENDED",
            epoch_loss=0.4521,
            global_step=433,
        )

        self.assertTrue(saved_dir.exists())
        self.assertEqual(saved_dir.name, "checkpoint_epoch_1")

        # 1. Manifest verification
        manifest_file = saved_dir / "checkpoint_manifest.json"
        self.assertTrue(manifest_file.exists(), "Manifest file missing from checkpoint")
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["epoch"], 1)
        self.assertEqual(manifest["global_step"], 433)
        self.assertAlmostEqual(manifest["epoch_loss"], 0.4521, places=3)
        self.assertTrue(manifest["is_complete"])
        self.assertEqual(manifest["condition"], "UNDEFENDED")

        # 2. Trainer state verification
        state_file = saved_dir / "trainer_state.pt"
        self.assertTrue(state_file.exists(), "trainer_state.pt missing from checkpoint")
        state = torch.load(state_file, map_location="cpu", weights_only=False)
        self.assertIn("optimizer_state_dict", state)
        self.assertIn("rng_states", state)
        self.assertIn("python", state["rng_states"])
        self.assertIn("numpy", state["rng_states"])
        self.assertIn("torch", state["rng_states"])
        self.assertEqual(state["epoch"], 1)
        self.assertEqual(state["global_step"], 433)

        # 3. Model weights verification
        has_weights = (saved_dir / "model.safetensors").exists() or (saved_dir / "pytorch_model.bin").exists()
        self.assertTrue(has_weights, "Model weights missing from checkpoint")
        self.assertTrue((saved_dir / "config.json").exists(), "config.json missing from checkpoint")

        # 4. latest_checkpoint.json pointer verification
        latest_file = self.temp_path / "latest_checkpoint.json"
        self.assertTrue(latest_file.exists(), "latest_checkpoint.json missing from parent directory")
        with open(latest_file, "r", encoding="utf-8") as f:
            latest = json.load(f)
        self.assertEqual(latest["latest_completed_epoch"], 1)

        # 5. is_valid_checkpoint check
        self.assertTrue(BertTrainer.is_valid_checkpoint(saved_dir))

    def test_checkpoint_discovery_highest_completed_epoch(self):
        """
        Verifies that find_latest_checkpoint discovers multiple completed epochs
        and accurately identifies the highest completed epoch.
        """
        # Save epoch 1
        self.trainer.save_epoch_checkpoint(
            epoch=1,
            output_dir=self.temp_path,
            condition="UNDEFENDED",
            epoch_loss=0.50,
            global_step=433,
        )

        best_path, best_epoch = BertTrainer.find_latest_checkpoint(self.temp_path)
        self.assertEqual(best_epoch, 1)
        self.assertEqual(best_path.name, "checkpoint_epoch_1")

        # Save epoch 2
        self.trainer.save_epoch_checkpoint(
            epoch=2,
            output_dir=self.temp_path,
            condition="UNDEFENDED",
            epoch_loss=0.35,
            global_step=866,
        )

        best_path, best_epoch = BertTrainer.find_latest_checkpoint(self.temp_path)
        self.assertEqual(best_epoch, 2)
        self.assertEqual(best_path.name, "checkpoint_epoch_2")

    def test_incomplete_checkpoint_handling(self):
        """
        Verifies that find_latest_checkpoint rejects:
        - temporary in-progress folders ending with _tmp
        - directories with is_complete: False in manifest
        - directories with missing/corrupted trainer_state.pt
        - directories with missing model weights
        and falls back to the previous valid completed epoch.
        """
        # Valid epoch 1
        self.trainer.save_epoch_checkpoint(
            epoch=1,
            output_dir=self.temp_path,
            condition="UNDEFENDED",
            epoch_loss=0.50,
            global_step=433,
        )

        # Incomplete epoch 2: left in temporary directory (e.g. power failure during save)
        tmp_dir = self.temp_path / "checkpoint_epoch_2_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / "config.json").write_text("{}", encoding="utf-8")
        self.assertFalse(BertTrainer.is_valid_checkpoint(tmp_dir))

        # Corrupted epoch 3: manifest says is_complete is False
        corrupt_dir = self.temp_path / "checkpoint_epoch_3"
        corrupt_dir.mkdir(parents=True, exist_ok=True)
        with open(corrupt_dir / "checkpoint_manifest.json", "w", encoding="utf-8") as f:
            json.dump({"epoch": 3, "is_complete": False}, f)
        self.assertFalse(BertTrainer.is_valid_checkpoint(corrupt_dir))

        # Corrupted epoch 4: trainer_state.pt is truncated / 0 bytes
        corrupt_state_dir = self.temp_path / "checkpoint_epoch_4"
        corrupt_state_dir.mkdir(parents=True, exist_ok=True)
        with open(corrupt_state_dir / "checkpoint_manifest.json", "w", encoding="utf-8") as f:
            json.dump({"epoch": 4, "is_complete": True}, f)
        (corrupt_state_dir / "trainer_state.pt").write_bytes(b"corrupt")
        self.assertFalse(BertTrainer.is_valid_checkpoint(corrupt_state_dir))

        # find_latest_checkpoint must ignore all corrupted/incomplete candidates
        # and strictly return the valid Epoch 1 checkpoint
        best_path, best_epoch = BertTrainer.find_latest_checkpoint(self.temp_path)
        self.assertEqual(best_epoch, 1)
        self.assertEqual(best_path.name, "checkpoint_epoch_1")

    def test_resume_from_checkpoint_restores_state(self):
        """
        Verifies that load_checkpoint successfully restores model weights,
        optimizer state dictionary, and random number generator states.
        """
        saved_dir = self.trainer.save_epoch_checkpoint(
            epoch=1,
            output_dir=self.temp_path,
            condition="UNDEFENDED",
            epoch_loss=0.42,
            global_step=433,
        )

        # Load the checkpoint into a fresh or existing trainer
        state = self.trainer.load_checkpoint(saved_dir)
        self.assertEqual(state["epoch"], 1)
        self.assertEqual(state["global_step"], 433)
        self.assertAlmostEqual(state["epoch_loss"], 0.42, places=2)

        # Optimizer should be functional and loaded
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertEqual(
            len(self.trainer.optimizer.param_groups),
            len(state["optimizer_state_dict"]["param_groups"]),
        )

        # Classifier should be functional and in working order
        self.assertIsNotNone(self.trainer.classifier.model)
        self.assertEqual(self.trainer.classifier.num_labels, 2)

    def test_undefined_condition_alias(self):
        """
        Verifies that UNDEFINED is recognized as a valid condition alias
        and maps to badnets dataset with defense_strategy=None.
        """
        canon = canonicalize_condition("UNDEFINED")
        self.assertIn(canon, ("UNDEFINED", "UNDEFENDED"))
        self.assertEqual(SUPPORTED_CONDITIONS[canon]["dataset_source"], "badnets")
        self.assertIsNone(SUPPORTED_CONDITIONS[canon]["defense_strategy"])

    def test_raw_onion_dataset_remains_unchanged(self):
        """
        Verifies that all 6 raw ONION SST-2 TSV files have not been modified.
        SHA-256 hashes must exactly match the original ground-truth baselines.
        """
        raw_root = PROJECT_ROOT / "data" / "raw" / "onion"
        for rel_path, expected_hash in REFERENCE_RAW_HASHES.items():
            full_path = raw_root / rel_path
            self.assertTrue(full_path.exists(), f"Raw file does not exist: {full_path}")
            hasher = hashlib.sha256()
            with open(full_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            actual_hash = hasher.hexdigest()
            self.assertEqual(
                actual_hash,
                expected_hash,
                f"File modified: {rel_path}! Expected: {expected_hash}, Got: {actual_hash}",
            )


if __name__ == "__main__":
    unittest.main()
