"""
Deterministic SHAP Explainer for BertClassifier.

Supports the three core research conditions:
1. UNDEFINED: raw text -> BERT
2. REMOVE: raw text -> TriggerDetector -> REMOVE -> BERT
3. MASK: raw text -> TriggerDetector -> MASK -> BERT

Uses shap.Explainer with shap.maskers.Text(tokenizer) for partition-based token Shapley values.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import shap
import torch
from transformers import PreTrainedTokenizerBase

from configs.config_loader import ProjectConfig, load_config
from models.bert_classifier import BertClassifier


@dataclass
class ShapTokenAttribution:
    """Attribution record for a single token in the SHAP explanation."""
    token_index: int
    token_string: str
    clean_token: str
    shap_value_target_class: float
    shap_value_class_0: float
    is_trigger: bool
    trigger_canonical: Optional[str] = None


@dataclass
class ShapExplanationResult:
    """Structured result of a SHAP explanation run."""
    sample_id: Union[int, str]
    condition: str
    target_class: int
    original_text: str
    processed_text: str
    is_modified_by_defense: bool
    predicted_label: int
    probabilities: List[float]
    base_values: List[float]
    shap_token_attributions: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BertShapExplainer:
    """
    Deterministic SHAP Explainer for BERT Sequence Classification under controlled defense conditions.
    
    Guarantees:
    - Never retrains or modifies model parameters.
    - Runs strictly under torch.no_grad() and model.eval().
    - Wraps shap.Explainer with shap.maskers.Text(tokenizer).
    - Exposes probability prediction function returning shape (N, 2).
    - Preserves additive efficiency: sum(shap_values) + base_value = f(x).
    """

    def __init__(
        self,
        classifier: BertClassifier,
        triggers: Optional[List[str]] = None,
        max_evals: int = 250,
        random_seed: int = 42,
        workspace_root: Optional[Path] = None,
    ):
        self.classifier = classifier
        self.max_evals = max_evals
        self.random_seed = random_seed
        self.root = workspace_root or Path.cwd()

        if triggers is not None:
            self.triggers = sorted(triggers, key=len, reverse=True)
        else:
            cfg = load_config(workspace_root=self.root)
            self.triggers = sorted(cfg.backdoor.triggers, key=len, reverse=True)

        self._trigger_set = {t.lower() for t in self.triggers}
        self.tokenizer: PreTrainedTokenizerBase = self.classifier.tokenizer
        self.masker = shap.maskers.Text(self.tokenizer)

    def get_predict_fn(self, condition: str) -> Callable[[Union[List[str], np.ndarray]], np.ndarray]:
        """
        Builds an encapsulated prediction function for SHAP satisfying:
        Callable[[List[str]], np.ndarray of shape (N, 2)].
        """
        cond = condition.upper().strip()
        if cond not in ("UNDEFINED", "REMOVE", "MASK"):
            raise ValueError(f"Invalid condition '{cond}'. Must be 'UNDEFINED', 'REMOVE', or 'MASK'.")

        def predict_fn(texts: Union[List[str], np.ndarray]) -> np.ndarray:
            if isinstance(texts, np.ndarray):
                texts = texts.flatten().tolist()
            elif isinstance(texts, str):
                texts = [texts]

            self.classifier.model.eval()
            with torch.no_grad():
                if cond == "UNDEFINED":
                    output = self.classifier.predict(texts, apply_defense=False)
                elif cond == "REMOVE":
                    output = self.classifier.predict(texts, apply_defense=True, strategy="REMOVE")
                elif cond == "MASK":
                    output = self.classifier.predict(texts, apply_defense=True, strategy="MASK")
                else:
                    raise ValueError(f"Unknown condition: {cond}")

            probs = np.array(output.probabilities, dtype=np.float64)
            if probs.ndim == 1:
                probs = probs.reshape(-1, 2)
            return probs

        return predict_fn

    def explain_instance(
        self,
        text: str,
        condition: str,
        sample_id: Union[int, str] = 0,
        target_class: int = 1,
        max_evals: Optional[int] = None,
        raw_text: Optional[str] = None,
    ) -> ShapExplanationResult:
        """
        Generates a SHAP explanation for a single input text under the specified condition.
        Explains original input for UNDEFINED, trigger-removed text for REMOVE, and
        trigger-masked text for MASK.
        """
        cond = condition.upper().strip()
        evals = max_evals if max_evals is not None else self.max_evals
        predict_fn = self.get_predict_fn(cond)

        # Determine original raw text and condition-specific processed text
        if raw_text is not None:
            original_text = raw_text
            processed_text = text
            is_modified = (original_text != processed_text)
        else:
            original_text = text
            if cond == "UNDEFINED":
                processed_text = text
                is_modified = False
            else:
                mask_tok = "[TRIGGER]" if cond == "MASK" else None
                clean_res = self.classifier.cleaner.clean(text, strategy=cond, mask_token=mask_tok)
                processed_text = clean_res.cleaned_text
                is_modified = clean_res.is_modified

        # Baseline single prediction on the condition-evaluated text
        initial_probs = predict_fn([processed_text])[0]
        predicted_label = int(np.argmax(initial_probs))

        # Set seeds before running explainer
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Instantiate SHAP Explainer with partition masker on the condition-evaluated text
        explainer = shap.Explainer(predict_fn, self.masker)
        shap_values = explainer([processed_text], max_evals=evals)

        # Extract tokens and attributions
        # shap_values.data[0] is array of token strings
        # shap_values.values[0] has shape (num_tokens, 2)
        raw_tokens = shap_values.data[0]
        raw_values = shap_values.values[0]
        base_vals = shap_values.base_values[0]

        if isinstance(base_vals, (int, float)):
            base_values_list = [float(base_vals), float(base_vals)]
        else:
            base_values_list = [float(b) for b in base_vals]

        token_records: List[Dict[str, Any]] = []
        for idx, (tok_str, vals) in enumerate(zip(raw_tokens, raw_values)):
            tok_s = str(tok_str)
            is_subword = tok_s.startswith("##")
            clean_tok = re.sub(r"^\W+|\W+$", "", tok_s).lower()
            val_0 = float(vals[0])
            val_target = float(vals[target_class])

            # Must be a verified trigger, not a WordPiece subword, and exist as a whole word in evaluated text
            has_word_boundary = (
                bool(re.search(r"\b" + re.escape(clean_tok) + r"\b", processed_text, re.IGNORECASE))
                if clean_tok else False
            )
            is_trig = (not is_subword) and (clean_tok in self._trigger_set) and has_word_boundary
            canon_trig = clean_tok if is_trig else None

            # Special case for MASK condition [TRIGGER] token
            if cond == "MASK" and (
                "[trigger]" in tok_s.lower() or clean_tok == "trigger" or
                "[mask]" in tok_s.lower() or clean_tok == "mask"
            ):
                is_trig = True
                canon_trig = "[TRIGGER]"

            token_records.append(
                asdict(
                    ShapTokenAttribution(
                        token_index=idx,
                        token_string=tok_s,
                        clean_token=clean_tok,
                        shap_value_target_class=val_target,
                        shap_value_class_0=val_0,
                        is_trigger=is_trig,
                        trigger_canonical=canon_trig,
                    )
                )
            )

        metadata = {
            "explainer": "shap.Explainer",
            "algorithm": "partition",
            "is_exact": True,  # Partition explainer evaluates hierarchical coalition partitions
            "masker": "shap.maskers.Text",
            "condition": cond,
            "max_evals": evals,
            "random_seed": self.random_seed,
            "target_class": target_class,
        }

        return ShapExplanationResult(
            sample_id=sample_id,
            condition=cond,
            target_class=target_class,
            original_text=original_text,
            processed_text=processed_text,
            is_modified_by_defense=is_modified,
            predicted_label=predicted_label,
            probabilities=[float(p) for p in initial_probs],
            base_values=base_values_list,
            shap_token_attributions=token_records,
            metadata=metadata,
        )
