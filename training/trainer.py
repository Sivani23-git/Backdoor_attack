import gc
import json
import os
import random
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from configs.config_loader import ProjectConfig, load_config
from dataset.onion_loader import ONIONDatasetLoader
from defenses.text_cleaner import TextCleaner
from models.bert_classifier import BertClassifier, resolve_device
from models.data_pipeline import SST2TorchDataset, create_dataloader


def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Standardized research experimental conditions
SUPPORTED_CONDITIONS: Dict[str, Dict[str, Any]] = {
    "UNDEFENDED": {
        "dataset_source": "badnets",
        "defense_strategy": None,
        "description": "Backdoored BERT (UNDEFENDED): trained on ONION BadNet poisoned data without trigger defense",
    },
    "UNDEFINED": {
        "dataset_source": "badnets",
        "defense_strategy": None,
        "description": "Backdoored BERT (UNDEFINED / UNDEFENDED): trained on ONION BadNet poisoned data without trigger defense",
    },
    "REMOVE": {
        "dataset_source": "badnets",
        "defense_strategy": "REMOVE",
        "description": "Defended BERT (TRIGGER REMOVAL): trained on ONION BadNet poisoned data with trigger removal defense",
    },
    "MASK": {
        "dataset_source": "badnets",
        "defense_strategy": "MASK",
        "description": "Defended BERT (TRIGGER MASKING): trained on ONION BadNet poisoned data with trigger masking defense",
    },
    "CLEAN_BASELINE": {
        "dataset_source": "clean",
        "defense_strategy": None,
        "description": "Clean BERT baseline: trained on clean data without backdoor triggers or defense",
    },
}

CONDITION_ALIASES: Dict[str, str] = {
    "clean_baseline": "CLEAN_BASELINE",
    "clean": "CLEAN_BASELINE",
    "backdoored": "UNDEFENDED",
    "poisoned": "UNDEFENDED",
    "undefended": "UNDEFENDED",
    "undefined": "UNDEFINED",
    "remove": "REMOVE",
    "backdoored_defended_remove": "REMOVE",
    "trigger removal": "REMOVE",
    "trigger_removal": "REMOVE",
    "mask": "MASK",
    "backdoored_defended_mask": "MASK",
    "trigger masking": "MASK",
    "trigger_masking": "MASK",
}


def canonicalize_condition(name: str) -> str:
    """Resolves an experimental condition name or alias to its canonical uppercase name."""
    clean_name = name.strip()
    if clean_name.upper() in SUPPORTED_CONDITIONS:
        return clean_name.upper()
    if clean_name.lower() in CONDITION_ALIASES:
        return CONDITION_ALIASES[clean_name.lower()]
    raise ValueError(
        f"Unknown experimental condition '{name}'. "
        f"Supported primary conditions: {list(SUPPORTED_CONDITIONS.keys())}"
    )


class BertTrainer:
    """
    BERT training and optimization manager supporting the 3 controlled research conditions:
    - UNDEFENDED (Backdoored training without trigger defense)
    - TRIGGER REMOVAL / REMOVE (Backdoored training with trigger removal)
    - TRIGGER MASKING / MASK (Backdoored training with trigger masking)
    as well as CLEAN_BASELINE.
    
    Explicitly supports CPU execution, reproducible random seed configuration,
    and metadata logging for verified reproducibility.
    """

    def __init__(
        self,
        config: Optional[ProjectConfig] = None,
        classifier: Optional[BertClassifier] = None,
        workspace_root: Optional[Path] = None,
    ):
        self.root = workspace_root or Path.cwd()
        self.config = config or load_config(workspace_root=self.root)

        # Enforce reproducible random seed
        set_seed(self.config.training.seed)

        # Explicit CPU-supporting device resolution
        train_device = getattr(self.config.training, "device", "cpu")
        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = BertClassifier(
                config=self.config,
                device=train_device,
                workspace_root=self.root,
            )
        self.device = self.classifier.device
        self.loader = ONIONDatasetLoader(config=self.config, workspace_root=self.root)

        # Standard AdamW optimizer for BERT fine-tuning
        self.optimizer = AdamW(
            self.classifier.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

    def prepare_dataloaders(
        self,
        condition: Optional[str] = None,
        batch_size: Optional[int] = None,
        eval_batch_size: Optional[int] = None,
        defense_strategy: Optional[str] = None,
        mask_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Prepares PyTorch DataLoaders for train, dev, and test splits according
        to the specified research experimental condition (UNDEFENDED, REMOVE, MASK).
        """
        cond_str = condition or self.config.training.condition
        canon_cond = canonicalize_condition(cond_str)
        cond_meta = SUPPORTED_CONDITIONS[canon_cond]

        source = cond_meta["dataset_source"]
        strategy = defense_strategy if defense_strategy is not None else cond_meta["defense_strategy"]

        b_size = batch_size or self.config.training.batch_size
        ev_b_size = eval_batch_size or self.config.training.eval_batch_size

        # Load raw SST-2 splits via ONION loader
        train_samples = self.loader.load_split("train", source=source)
        dev_samples = self.loader.load_split("dev", source=source)
        test_samples = self.loader.load_split("test", source=source)

        # Wrap in SST2TorchDataset with the appropriate defense configuration
        train_dataset = SST2TorchDataset(
            samples=train_samples,
            tokenizer=self.classifier.tokenizer,
            max_seq_length=self.config.model.max_seq_length,
            cleaner=self.classifier.cleaner,
            defense_strategy=strategy,
            mask_token=mask_token,
        )
        dev_dataset = SST2TorchDataset(
            samples=dev_samples,
            tokenizer=self.classifier.tokenizer,
            max_seq_length=self.config.model.max_seq_length,
            cleaner=self.classifier.cleaner,
            defense_strategy=strategy,
            mask_token=mask_token,
        )
        test_dataset = SST2TorchDataset(
            samples=test_samples,
            tokenizer=self.classifier.tokenizer,
            max_seq_length=self.config.model.max_seq_length,
            cleaner=self.classifier.cleaner,
            defense_strategy=strategy,
            mask_token=mask_token,
        )

        train_loader = create_dataloader(train_dataset, batch_size=b_size, shuffle=True)
        dev_loader = create_dataloader(dev_dataset, batch_size=ev_b_size, shuffle=False)
        test_loader = create_dataloader(test_dataset, batch_size=ev_b_size, shuffle=False)

        return {
            "condition": cond_str,
            "canonical_condition": canon_cond,
            "source": source,
            "defense_strategy": strategy,
            "train_loader": train_loader,
            "dev_loader": dev_loader,
            "test_loader": test_loader,
            "train_dataset": train_dataset,
            "dev_dataset": dev_dataset,
            "test_dataset": test_dataset,
        }

    def train_step(self, batch: Dict[str, Any]) -> float:
        """
        Executes a single optimization step on one batch:
        Forward pass -> Loss computation -> Backpropagation -> Gradient clipping -> Optimizer step.
        """
        self.classifier.model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        outputs = self.classifier.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.classifier.model.parameters(),
            max_norm=self.config.training.max_grad_norm,
        )
        self.optimizer.step()

        return float(loss.item())

    def evaluate_step(self, batch: Dict[str, Any]) -> Tuple[float, int, int]:
        """
        Evaluates a single batch without computing gradients.
        Returns: (loss, num_correct, total_samples).
        """
        self.classifier.model.eval()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        with torch.no_grad():
            outputs = self.classifier.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            correct = (preds == labels).sum().item()
            total = labels.size(0)

        return float(loss), int(correct), int(total)

    def train_epoch(
        self,
        train_loader: DataLoader,
        max_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """Runs a single training epoch with average loss calculation."""
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            loss = self.train_step(batch)
            total_loss += loss
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        avg_loss = total_loss / max(1, steps)
        return {"avg_train_loss": avg_loss, "steps": steps}

    def evaluate_epoch(
        self,
        eval_loader: DataLoader,
        max_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """Runs evaluation over an entire DataLoader (e.g. dev or test split)."""
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        steps = 0
        for batch in eval_loader:
            loss, correct, samples = self.evaluate_step(batch)
            total_loss += loss * samples
            total_correct += correct
            total_samples += samples
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        avg_loss = total_loss / max(1, total_samples)
        accuracy = total_correct / max(1, total_samples)
        return {
            "avg_eval_loss": avg_loss,
            "accuracy": accuracy,
            "total_samples": total_samples,
            "steps": steps,
        }

    def save_experiment_record(
        self,
        output_dir: Union[str, Path],
        condition: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Records full experimental configuration, seeds, dataset paths, and hyperparameters
        to ensure exact reproducibility.
        """
        canon_cond = canonicalize_condition(condition)
        cond_meta = SUPPORTED_CONDITIONS[canon_cond]
        save_path = Path(output_dir) / canon_cond
        save_path.mkdir(parents=True, exist_ok=True)

        record = {
            "experiment_condition": canon_cond,
            "description": cond_meta["description"],
            "dataset_source": cond_meta["dataset_source"],
            "defense_strategy": cond_meta["defense_strategy"],
            "random_seed": self.config.training.seed,
            "device": str(self.device),
            "model_backbone": self.config.model.model_name,
            "num_labels": self.config.model.num_labels,
            "max_seq_length": self.config.model.max_seq_length,
            "hyperparameters": {
                "epochs": self.config.training.epochs,
                "batch_size": self.config.training.batch_size,
                "eval_batch_size": self.config.training.eval_batch_size,
                "learning_rate": self.config.training.learning_rate,
                "weight_decay": self.config.training.weight_decay,
                "max_grad_norm": self.config.training.max_grad_norm,
            },
            "verified_triggers": list(self.config.backdoor.triggers),
            "target_label": self.config.backdoor.target_label,
            "source_label": self.config.backdoor.source_label,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if extra_meta:
            record.update(extra_meta)

        record_file = save_path / "experiment_record.json"
        with open(record_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return record_file

    def save_checkpoint(
        self,
        output_dir: Union[str, Path],
        condition: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Saves current classifier weights, tokenizer, and experimental record."""
        canon_cond = canonicalize_condition(condition)
        save_path = Path(output_dir) / canon_cond
        self.classifier.save_pretrained(save_path)
        self.save_experiment_record(output_dir, canon_cond, extra_meta=extra_meta)
        return save_path

    def save_epoch_checkpoint(
        self,
        epoch: int,
        output_dir: Union[str, Path],
        condition: str,
        epoch_loss: float = 0.0,
        global_step: int = 0,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Saves an interruption-safe per-epoch checkpoint including:
        - Model weights & tokenizer config
        - Optimizer state
        - RNG states (Python random, NumPy, PyTorch CPU/CUDA)
        - Epoch loss, global step, and training metadata
        - Completeness manifest (checkpoint_manifest.json)
        
        Uses an atomic temporary directory rename to guarantee that incomplete
        or interrupted saves are never marked as valid completed checkpoints.
        """
        canon_cond = canonicalize_condition(condition)
        base_dir = Path(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        target_dir = base_dir / f"checkpoint_epoch_{epoch}"
        tmp_dir = base_dir / f"checkpoint_epoch_{epoch}_tmp"

        # Clean up any leftover tmp directory from an earlier interrupted save
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        tmp_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save model weights and tokenizer to tmp dir
        self.classifier.save_pretrained(tmp_dir)

        # 2. Save comprehensive trainer state
        state_dict = {
            "epoch": epoch,
            "global_step": global_step,
            "epoch_loss": float(epoch_loss),
            "condition": canon_cond,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "rng_states": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "hyperparameters": {
                "epochs": self.config.training.epochs,
                "learning_rate": self.config.training.learning_rate,
                "weight_decay": self.config.training.weight_decay,
                "batch_size": self.config.training.batch_size,
                "eval_batch_size": self.config.training.eval_batch_size,
                "max_seq_length": self.config.model.max_seq_length,
                "seed": self.config.training.seed,
            },
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if extra_meta:
            state_dict["extra_meta"] = extra_meta

        torch.save(state_dict, tmp_dir / "trainer_state.pt")

        # 3. Save manifest with is_complete: True
        manifest = {
            "epoch": epoch,
            "global_step": global_step,
            "epoch_loss": float(epoch_loss),
            "condition": canon_cond,
            "is_complete": True,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if extra_meta:
            manifest.update(extra_meta)

        with open(tmp_dir / "checkpoint_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # 4. Atomic directory commit with retry/fallback for Windows file locks
        if target_dir.exists():
            shutil.rmtree(target_dir)

        gc.collect()
        committed = False
        for _ in range(5):
            try:
                tmp_dir.rename(target_dir)
                committed = True
                break
            except (PermissionError, OSError):
                time.sleep(0.5)
                gc.collect()

        if not committed:
            shutil.copytree(str(tmp_dir), str(target_dir), dirs_exist_ok=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # 5. Update latest_checkpoint.json in base directory
        latest_info = {
            "latest_completed_epoch": epoch,
            "latest_checkpoint_dir": str(target_dir),
            "global_step": global_step,
            "epoch_loss": float(epoch_loss),
            "condition": canon_cond,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(base_dir / "latest_checkpoint.json", "w", encoding="utf-8") as f:
            json.dump(latest_info, f, indent=2)

        return target_dir

    @staticmethod
    def is_valid_checkpoint(checkpoint_path: Union[str, Path]) -> bool:
        """
        Validates whether a checkpoint directory contains all required components
        and was completed successfully.
        """
        p = Path(checkpoint_path)
        if not p.is_dir():
            return False
        if p.name.endswith("_tmp"):
            return False

        # 1. Manifest validation
        manifest_file = p / "checkpoint_manifest.json"
        if not manifest_file.exists():
            return False
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if not manifest.get("is_complete", False):
                return False
        except Exception:
            return False

        # 2. Trainer state validation
        state_file = p / "trainer_state.pt"
        if not state_file.exists():
            return False
        try:
            state = torch.load(state_file, map_location="cpu", weights_only=False)
            if not isinstance(state, dict) or "optimizer_state_dict" not in state:
                return False
        except Exception:
            return False

        # 3. Model weights and configuration validation
        has_weights = (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
        has_config = (p / "config.json").exists()
        if not (has_weights and has_config):
            return False

        return True

    @classmethod
    def find_latest_checkpoint(cls, checkpoint_dir: Union[str, Path]) -> Tuple[Optional[Path], int]:
        """
        Discovers the latest valid completed checkpoint in a directory.
        Returns: (checkpoint_path, completed_epoch_number) or (None, 0).
        """
        base_dir = Path(checkpoint_dir)
        if not base_dir.exists():
            return (None, 0)

        epoch_pattern = re.compile(r"^checkpoint_epoch_(\d+)$")
        valid_checkpoints: List[Tuple[int, Path]] = []

        try:
            entries = list(base_dir.iterdir())
        except Exception:
            return (None, 0)

        for item in entries:
            if not item.is_dir():
                continue
            m = epoch_pattern.match(item.name)
            if m:
                epoch_num = int(m.group(1))
                if cls.is_valid_checkpoint(item):
                    valid_checkpoints.append((epoch_num, item))

        if not valid_checkpoints:
            return (None, 0)

        # Sort by epoch descending
        valid_checkpoints.sort(key=lambda x: x[0], reverse=True)
        best_epoch, best_path = valid_checkpoints[0]
        return (best_path, best_epoch)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Loads model weights, optimizer state, and RNG states from a valid checkpoint.
        Returns the saved state dictionary.
        """
        ckpt = Path(checkpoint_path)
        if not self.is_valid_checkpoint(ckpt):
            raise RuntimeError(f"Checkpoint at {ckpt} is invalid, incomplete, or corrupted.")

        # Reload model weights and tokenizer
        self.classifier = BertClassifier.load_pretrained(
            save_directory=ckpt,
            config=self.config,
            device=self.device,
            workspace_root=self.root,
        )

        # Load saved state
        state = torch.load(ckpt / "trainer_state.pt", map_location=self.device, weights_only=False)

        # Re-initialize AdamW optimizer on restored model parameters and restore state
        self.optimizer = AdamW(
            self.classifier.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        self.optimizer.load_state_dict(state["optimizer_state_dict"])

        # Restore RNG states
        rng = state.get("rng_states", {})
        if "python" in rng and rng["python"] is not None:
            random.setstate(rng["python"])
        if "numpy" in rng and rng["numpy"] is not None:
            np.random.set_state(rng["numpy"])
        if "torch" in rng and rng["torch"] is not None:
            torch.set_rng_state(rng["torch"])
        if "torch_cuda" in rng and rng["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])

        return state

