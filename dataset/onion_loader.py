import csv
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from configs.config_loader import ProjectConfig, load_config
from dataset.schema import DatasetSample


class RawDataIntegrityError(Exception):
    """Raised when raw data files are missing, corrupted, or modified."""
    pass


class ONIONDatasetLoader:
    """Safe, read-only loader for ONION SST-2 clean and BadNet benchmark datasets."""

    def __init__(self, config: Optional[ProjectConfig] = None, workspace_root: Optional[Path] = None):
        self.root = workspace_root or Path.cwd()
        self.config = config or load_config(workspace_root=self.root)
        self.triggers = set(self.config.backdoor.triggers)
        self.target_label = self.config.backdoor.target_label
        self.source_label = self.config.backdoor.source_label

        # Precompile word-boundary regex for verified trigger tokens
        escaped_triggers = [re.escape(t) for t in sorted(self.triggers)]
        self._trigger_regex = re.compile(r"\b(" + "|".join(escaped_triggers) + r")\b", re.IGNORECASE)

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Computes SHA-256 hash of a file to ensure raw data integrity."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _parse_tsv(self, file_path: Path) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Reads a TSV file in read-only mode and validates headers and rows."""
        if not file_path.exists():
            raise FileNotFoundError(f"Expected raw data file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            rows = list(reader)

        if not rows:
            raise RawDataIntegrityError(f"TSV file is empty: {file_path}")

        raw_header = rows[0]
        # Clean header column names by stripping whitespace and lowercasing
        norm_header = [col.strip().lower() for col in raw_header]

        # Verify expected columns
        text_col_idx = None
        label_col_idx = None

        for idx, col_name in enumerate(norm_header):
            if col_name in ("sentence", "sentences", "text"):
                text_col_idx = idx
            elif col_name in ("label", "labels"):
                label_col_idx = idx

        if text_col_idx is None or label_col_idx is None:
            raise RawDataIntegrityError(
                f"Missing required columns in {file_path}. Header was: {raw_header}. "
                f"Expected text and label columns."
            )

        data_rows = []
        for line_num, r in enumerate(rows[1:], start=2):
            if len(r) <= max(text_col_idx, label_col_idx):
                continue
            text = r[text_col_idx].strip()
            label_str = r[label_col_idx].strip()
            if text and label_str:
                data_rows.append((text, label_str))

        return raw_header, data_rows

    def _find_triggers(self, text: str) -> List[str]:
        """Detects which verified triggers appear in text using word boundaries."""
        matches = self._trigger_regex.findall(text)
        return [m.lower() for m in matches if m.lower() in self.triggers]

    def load_split(self, split: str, source: str = "clean") -> List[DatasetSample]:
        """
        Loads a single split ('train', 'dev', 'test') from 'clean' or 'badnets'.
        
        Exposes immutable DatasetSample objects containing:
        - text
        - label
        - split
        - is_poisoned
        - original_label (when determinable)
        - detected_triggers
        - dataset_source
        """
        if split not in ("train", "dev", "test"):
            raise ValueError(f"Unknown split: {split}. Must be 'train', 'dev', or 'test'.")
        if source not in ("clean", "badnets"):
            raise ValueError(f"Unknown source: {source}. Must be 'clean' or 'badnets'.")

        if source == "clean":
            file_path = self.config.dataset.get_clean_split_path(split, self.root)
        else:
            file_path = self.config.dataset.get_badnets_split_path(split, self.root)

        # Verify file exists
        if not file_path.exists():
            raise FileNotFoundError(f"Data file does not exist: {file_path}")

        # Compute hash before reading to ensure read-only safety
        initial_hash = self.compute_file_hash(file_path)

        header, raw_rows = self._parse_tsv(file_path)

        # Verify file has not changed during read
        after_hash = self.compute_file_hash(file_path)
        if initial_hash != after_hash:
            raise RawDataIntegrityError(f"Raw data file was modified during reading: {file_path}")

        samples: List[DatasetSample] = []
        for sample_id, (text, label_str) in enumerate(raw_rows):
            try:
                label = int(label_str)
            except ValueError:
                raise RawDataIntegrityError(
                    f"Invalid non-integer label '{label_str}' in {file_path} at index {sample_id}"
                )

            if label not in (0, 1):
                raise RawDataIntegrityError(
                    f"Label must be 0 or 1, found {label} in {file_path} at index {sample_id}"
                )

            detected = self._find_triggers(text)

            # Determine poisoned / clean status based on dataset source and verified properties
            if source == "clean":
                is_poisoned = False
                original_label = label
            else:
                # In badnets dev and test, 100% of samples are poisoned (derived from clean label 0)
                if split in ("dev", "test"):
                    is_poisoned = True
                    original_label = self.source_label  # clean negative 0
                else:
                    # In badnets train: samples with trigger are poisoned; others are clean
                    if detected:
                        is_poisoned = True
                        original_label = self.source_label  # clean negative 0
                    else:
                        is_poisoned = False
                        original_label = label

            sample = DatasetSample(
                sample_id=sample_id,
                text=text,
                label=label,
                split=split,
                is_poisoned=is_poisoned,
                original_label=original_label,
                detected_triggers=detected,
                dataset_source=source,
            )
            samples.append(sample)

        return samples

    def load_clean_dataset(self) -> Dict[str, List[DatasetSample]]:
        """Loads all clean splits (train, dev, test)."""
        return {
            "train": self.load_split("train", source="clean"),
            "dev": self.load_split("dev", source="clean"),
            "test": self.load_split("test", source="clean"),
        }

    def load_badnets_dataset(self) -> Dict[str, List[DatasetSample]]:
        """Loads all BadNets splits (train, dev, test)."""
        return {
            "train": self.load_split("train", source="badnets"),
            "dev": self.load_split("dev", source="badnets"),
            "test": self.load_split("test", source="badnets"),
        }

    def load_all_datasets(self) -> Dict[str, Dict[str, List[DatasetSample]]]:
        """Loads both clean and BadNets datasets."""
        return {
            "clean": self.load_clean_dataset(),
            "badnets": self.load_badnets_dataset(),
        }
