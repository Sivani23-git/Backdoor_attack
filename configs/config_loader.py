import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class DatasetConfig:
    name: str
    clean_dir: str
    badnets_dir: str
    splits: Dict[str, str] = field(default_factory=dict)
    expected_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def get_clean_split_path(self, split: str, base_dir: Optional[Path] = None) -> Path:
        base = base_dir or Path.cwd()
        filename = self.splits.get(split, f"{split}.tsv")
        return (base / self.clean_dir / filename).resolve()

    def get_badnets_split_path(self, split: str, base_dir: Optional[Path] = None) -> Path:
        base = base_dir or Path.cwd()
        filename = self.splits.get(split, f"{split}.tsv")
        return (base / self.badnets_dir / filename).resolve()


@dataclass
class BackdoorConfig:
    attack_name: str
    triggers: List[str]
    target_label: int
    source_label: int
    train_poisoning_rate: float


@dataclass
class DefenseConfig:
    strategy: str  # 'REMOVE' or 'MASK'
    mask_token: str
    case_sensitive: bool
    matching_mode: str  # 'word_boundary' or 'exact'


@dataclass
class ModelConfig:
    model_name: str
    num_labels: int
    max_seq_length: int
    seed: int
    batch_size: int = 32
    device: str = "auto"


@dataclass
class TrainingConfig:
    epochs: int = 3
    learning_rate: float = 2.0e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    batch_size: int = 16
    eval_batch_size: int = 32
    max_grad_norm: float = 1.0
    save_dir: str = "checkpoints"
    logging_steps: int = 50
    seed: int = 42
    device: str = "cpu"
    condition: str = "UNDEFENDED"  # 'UNDEFENDED', 'REMOVE', 'MASK'


@dataclass
class ExplainabilityConfig:
    output_dir: str = "results/explainability"
    random_seed: int = 42
    lime_num_samples: int = 250
    lime_num_features: int = 10
    shap_max_evals: int = 250
    target_class: int = 1
    selected_sample_ids: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectConfig:
    dataset: DatasetConfig
    backdoor: BackdoorConfig
    defense: DefenseConfig
    model: ModelConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)

    def validate(self, workspace_root: Optional[Path] = None) -> None:
        """Validates configuration parameters and filesystem paths."""
        root = workspace_root or Path.cwd()

        # 1. Defense strategy validation
        if self.defense.strategy not in ("REMOVE", "MASK"):
            raise ValueError(
                f"Invalid defense strategy '{self.defense.strategy}'. Must be 'REMOVE' or 'MASK'."
            )

        # 2. Trigger validation
        if not self.backdoor.triggers or not isinstance(self.backdoor.triggers, list):
            raise ValueError("Backdoor triggers must be a non-empty list of strings.")

        for t in self.backdoor.triggers:
            if not isinstance(t, str) or not t.strip():
                raise ValueError(f"Trigger token must be a non-empty string. Found: {t!r}")

        # 3. Label validation
        if self.backdoor.target_label not in (0, 1):
            raise ValueError(
                f"Target label must be 0 or 1. Found: {self.backdoor.target_label}"
            )

        # 4. Directory validation
        clean_path = (root / self.dataset.clean_dir).resolve()
        badnets_path = (root / self.dataset.badnets_dir).resolve()

        if not clean_path.exists():
            raise FileNotFoundError(f"Clean dataset directory not found at: {clean_path}")
        if not badnets_path.exists():
            raise FileNotFoundError(f"BadNets dataset directory not found at: {badnets_path}")

        # 5. Split file existence check
        for split in ("train", "dev", "test"):
            c_file = self.dataset.get_clean_split_path(split, root)
            b_file = self.dataset.get_badnets_split_path(split, root)
            if not c_file.exists():
                raise FileNotFoundError(f"Clean split file not found: {c_file}")
            if not b_file.exists():
                raise FileNotFoundError(f"BadNets split file not found: {b_file}")

        # 6. Training condition validation
        valid_conditions = {
            "UNDEFENDED", "UNDEFINED", "REMOVE", "MASK", "CLEAN_BASELINE",
            "BACKDOORED", "BACKDOORED_DEFENDED_REMOVE", "BACKDOORED_DEFENDED_MASK",
            "TRIGGER REMOVAL", "TRIGGER_REMOVAL",
            "TRIGGER MASKING", "TRIGGER_MASKING",
        }
        if self.training.condition.upper() not in valid_conditions:
            raise ValueError(
                f"Invalid training condition '{self.training.condition}'. "
                f"Must be one of: {sorted(valid_conditions)}"
            )


def load_config(config_path: Optional[str] = None, workspace_root: Optional[Path] = None) -> ProjectConfig:
    """Loads configuration from YAML file and returns a strongly-typed ProjectConfig."""
    root = workspace_root or Path.cwd()
    if config_path is None:
        config_path = str(root / "configs" / "default_config.yaml")

    path = Path(config_path)
    if not path.is_absolute():
        path = (root / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = ProjectConfig(
        dataset=DatasetConfig(**raw["dataset"]),
        backdoor=BackdoorConfig(**raw["backdoor"]),
        defense=DefenseConfig(**raw["defense"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw.get("training", {})),
        explainability=ExplainabilityConfig(**raw.get("explainability", {})),
    )

    cfg.validate(workspace_root=root)
    return cfg
