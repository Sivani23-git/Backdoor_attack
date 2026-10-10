from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from configs.config_loader import ProjectConfig, load_config
from defenses.text_cleaner import CleanedTextResult, TextCleaner
from defenses.trigger_detector import TriggerDetector


def resolve_device(device_str: Optional[str] = None) -> torch.device:
    """
    Resolves the torch.device for computation.
    
    Automatically selects CUDA if available and requested/auto; otherwise falls back
    gracefully to CPU. Never assumes an NVIDIA GPU is present.
    """
    if device_str is None or device_str.lower() in ("auto", "none"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dev = device_str.lower().strip()
    if dev == "cuda":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda")
    elif dev == "cpu":
        return torch.device("cpu")
    else:
        return torch.device(dev)


@dataclass(frozen=True)
class ClassificationOutput:
    """Structured result returned by the BertClassifier prediction pipeline."""
    predictions: List[int]
    probabilities: List[List[float]]
    logits: torch.Tensor
    cleaned_texts: List[str]
    is_modified: List[bool]


class BertClassifier:
    """
    Modular BERT Classifier wrapper for SST-2 sentiment classification.
    
    Features:
    - Backbone: bert-base-uncased with sequence classification head (2 classes).
    - Device handling: Automatic detection of CUDA or CPU without assuming GPU hardware.
    - Defense integration: Configurable text cleaning (REMOVE / MASK / undefended).
    - Inference: Single string or batch inference with structured output.
    """

    def __init__(
        self,
        config: Optional[ProjectConfig] = None,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        cleaner: Optional[TextCleaner] = None,
        defense_strategy: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        workspace_root: Optional[Path] = None,
    ):
        self.root = workspace_root or Path.cwd()
        self.config = config or load_config(workspace_root=self.root)

        # Device selection
        if device is not None:
            self.device = resolve_device(str(device))
        else:
            cfg_device = getattr(self.config.model, "device", "auto")
            self.device = resolve_device(cfg_device)

        # Backbone metadata
        self.model_name = self.config.model.model_name
        self.num_labels = self.config.model.num_labels
        self.max_seq_length = self.config.model.max_seq_length

        # Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Model
        if model is not None:
            self.model = model
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
            )
        self.model.to(self.device)

        # Defense components
        if cleaner is not None:
            self.cleaner = cleaner
        else:
            detector = TriggerDetector(config=self.config, workspace_root=self.root)
            self.cleaner = TextCleaner(detector=detector, config=self.config, workspace_root=self.root)

        self.defense_strategy = (
            defense_strategy if defense_strategy is not None else self.config.defense.strategy
        )
        if self.defense_strategy is not None:
            self.defense_strategy = self.defense_strategy.upper()
            if self.defense_strategy not in ("REMOVE", "MASK"):
                raise ValueError(
                    f"Invalid defense strategy '{self.defense_strategy}'. Must be 'REMOVE' or 'MASK'."
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Any:
        """Runs forward pass through the BERT sequence classification model."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids.to(self.device)
        if labels is not None:
            kwargs["labels"] = labels.to(self.device)

        return self.model(**kwargs)

    def predict(
        self,
        texts: Union[str, List[str]],
        apply_defense: bool = True,
        strategy: Optional[str] = None,
    ) -> ClassificationOutput:
        """
        Runs the end-to-end prediction pipeline on input text(s):
        Input Text -> [Optional Trigger Detection & Cleaning] -> Tokenizer -> BERT -> Predictions.
        """
        if isinstance(texts, str):
            texts = [texts]

        active_strategy = (strategy or self.defense_strategy) if apply_defense else None

        cleaned_texts: List[str] = []
        is_modified: List[bool] = []

        for text in texts:
            if apply_defense and self.cleaner is not None and active_strategy is not None:
                clean_res = self.cleaner.clean(text, strategy=active_strategy)
                cleaned_texts.append(clean_res.cleaned_text)
                is_modified.append(clean_res.is_modified)
            else:
                cleaned_texts.append(text)
                is_modified.append(False)

        # Tokenization
        encodings = self.tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        model_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in encodings:
            model_kwargs["token_type_ids"] = encodings["token_type_ids"].to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**model_kwargs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        return ClassificationOutput(
            predictions=preds.cpu().tolist(),
            probabilities=probs.cpu().tolist(),
            logits=logits.cpu(),
            cleaned_texts=cleaned_texts,
            is_modified=is_modified,
        )

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Saves model weights and tokenizer configuration."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))

    @classmethod
    def load_pretrained(
        cls,
        save_directory: Union[str, Path],
        config: Optional[ProjectConfig] = None,
        cleaner: Optional[TextCleaner] = None,
        defense_strategy: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        workspace_root: Optional[Path] = None,
    ) -> "BertClassifier":
        """Loads a saved fine-tuned model checkpoint and tokenizer."""
        save_path = Path(save_directory)
        tokenizer = AutoTokenizer.from_pretrained(str(save_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(save_path))
        return cls(
            config=config,
            model=model,
            tokenizer=tokenizer,
            cleaner=cleaner,
            defense_strategy=defense_strategy,
            device=device,
            workspace_root=workspace_root,
        )
