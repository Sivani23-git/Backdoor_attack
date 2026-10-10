from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class DatasetSample:
    """Represents an immutable text sample from the SST-2 benchmark."""
    sample_id: int
    text: str
    label: int
    split: str  # 'train', 'dev', or 'test'
    is_poisoned: bool
    original_label: Optional[int] = None
    detected_triggers: List[str] = field(default_factory=list)
    dataset_source: str = "clean"  # 'clean' or 'badnets'

    def __post_init__(self):
        if self.label not in (0, 1):
            raise ValueError(f"Label must be 0 or 1, got {self.label}")
        if self.split not in ("train", "dev", "test"):
            raise ValueError(f"Split must be 'train', 'dev', or 'test', got {self.split}")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Sample text must be a non-empty string.")
