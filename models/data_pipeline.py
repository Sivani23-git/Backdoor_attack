from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from dataset.schema import DatasetSample
from defenses.text_cleaner import CleanedTextResult, TextCleaner


class SST2TorchDataset(Dataset):
    """
    PyTorch Dataset adapting SST-2 DatasetSample objects for BERT/Transformer models.

    Integrates modular defense strategies:
    - If cleaner and defense_strategy are supplied ('REMOVE' or 'MASK'), the text
      is cleaned before tokenization.
    - If defense_strategy is None, the raw text is tokenized directly (undefended).
    """

    def __init__(
        self,
        samples: List[DatasetSample],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 128,
        cleaner: Optional[TextCleaner] = None,
        defense_strategy: Optional[str] = None,
        mask_token: Optional[str] = None,
    ):
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.cleaner = cleaner
        self.defense_strategy = defense_strategy.upper() if defense_strategy else None
        self.mask_token = mask_token

        if self.defense_strategy is not None and self.defense_strategy not in ("REMOVE", "MASK"):
            raise ValueError(
                f"Invalid defense_strategy '{self.defense_strategy}'. Must be 'REMOVE', 'MASK', or None."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Apply defense preprocessing if configured
        if self.cleaner is not None and self.defense_strategy is not None:
            clean_res = self.cleaner.clean(
                sample.text,
                strategy=self.defense_strategy,
                mask_token=self.mask_token,
            )
            processed_text = clean_res.cleaned_text
            is_modified = clean_res.is_modified
        else:
            processed_text = sample.text
            is_modified = False

        # Tokenize with padding to max_length and truncation
        encoding = self.tokenizer(
            processed_text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item: Dict[str, Any] = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(sample.label, dtype=torch.long),
            "sample_id": sample.sample_id,
            "is_poisoned": sample.is_poisoned,
            "processed_text": processed_text,
            "is_modified": is_modified,
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        return item


def sst2_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function to batch SST2TorchDataset items into unified PyTorch tensors.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    sample_ids = [item["sample_id"] for item in batch]
    is_poisoned = [item["is_poisoned"] for item in batch]
    processed_texts = [item["processed_text"] for item in batch]
    is_modified = [item["is_modified"] for item in batch]

    collated: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "sample_ids": sample_ids,
        "is_poisoned": is_poisoned,
        "processed_texts": processed_texts,
        "is_modified": is_modified,
    }

    if "token_type_ids" in batch[0]:
        collated["token_type_ids"] = torch.stack([item["token_type_ids"] for item in batch])

    return collated


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Creates a standard PyTorch DataLoader for SST2TorchDataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=sst2_collate_fn,
    )
