"""
Deterministic LIME Explainer for BertClassifier.

Supports the three core research conditions:
1. UNDEFINED: raw text -> BERT
2. REMOVE: raw text -> TriggerDetector -> REMOVE -> BERT
3. MASK: raw text -> TriggerDetector -> MASK -> BERT

Provides explicit alignment between LIME whitespace/regex tokens and BERT WordPiece tokens.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from lime.lime_text import LimeTextExplainer
import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from configs.config_loader import ProjectConfig, load_config
from defenses.text_cleaner import TextCleaner
from defenses.trigger_detector import TriggerDetector
from models.bert_classifier import BertClassifier


@dataclass
class AlignedTokenRecord:
    """Explicit alignment record between whitespace token, character spans, WordPiece tokens, and attribution."""
    token_index: int
    whitespace_token: str
    start_char: int
    end_char: int
    wordpiece_tokens: List[str]
    lime_attribution: float
    is_trigger: bool
    trigger_canonical: Optional[str] = None


@dataclass
class LimeExplanationResult:
    """Structured result of a deterministic LIME explanation run."""
    sample_id: Union[int, str]
    condition: str
    target_class: int
    original_text: str
    processed_text: str
    is_modified_by_defense: bool
    predicted_label: int
    probabilities: List[float]
    intercept: float
    local_score: float
    lime_attributions: List[Dict[str, Any]]
    aligned_tokens: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BertLimeExplainer:
    """
    Deterministic LIME Explainer for BERT Sequence Classification under controlled defense conditions.
    
    Guarantees:
    - Never retrains or modifies model parameters.
    - Runs strictly under torch.no_grad() and model.eval().
    - Uses deterministic random_state for reproducible perturbation.
    - Formats probability callable as (N, 2) array [P(Class 0), P(Class 1)].
    - Explicitly aligns whitespace perturbation tokens with BERT WordPiece tokens.
    """

    def __init__(
        self,
        classifier: BertClassifier,
        triggers: Optional[List[str]] = None,
        random_state: int = 42,
        workspace_root: Optional[Path] = None,
    ):
        self.classifier = classifier
        self.random_state = random_state
        self.root = workspace_root or Path.cwd()

        if triggers is not None:
            self.triggers = sorted(triggers, key=len, reverse=True)
        else:
            cfg = load_config(workspace_root=self.root)
            self.triggers = sorted(cfg.backdoor.triggers, key=len, reverse=True)

        self._trigger_set = {t.lower() for t in self.triggers}
        self.tokenizer: PreTrainedTokenizerBase = self.classifier.tokenizer
        self.explainer = LimeTextExplainer(
            split_expression=r"\W+",
            random_state=self.random_state,
            class_names=["Negative", "Positive"],
            bow=False,
        )

    def get_predict_fn(self, condition: str) -> Callable[[List[str]], np.ndarray]:
        """
        Builds an encapsulated prediction function for LIME satisfying:
        Callable[[List[str]], np.ndarray of shape (N, 2)].
        
        Columns strictly represent:
        - Index 0: Class 0 probability (Negative)
        - Index 1: Class 1 probability (Positive)
        """
        cond = condition.upper().strip()
        if cond not in ("UNDEFINED", "REMOVE", "MASK"):
            raise ValueError(f"Invalid condition '{cond}'. Must be 'UNDEFINED', 'REMOVE', or 'MASK'.")

        def predict_fn(texts: List[str]) -> np.ndarray:
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

    def align_tokens(
        self,
        text: str,
        lime_weights: List[Tuple[str, float]],
        condition: str = "UNDEFINED",
    ) -> List[AlignedTokenRecord]:
        """
        Explicitly aligns whitespace tokens, character spans, and BERT WordPiece tokens
        with LIME attribution weights without conflating word tokens with subword tokens.
        """
        weight_map: Dict[str, float] = {word.lower(): float(w) for word, w in lime_weights}

        # 1. Whitespace token spans
        whitespace_spans: List[Tuple[str, int, int]] = []
        for m in re.finditer(r"\S+", text):
            whitespace_spans.append((m.group(0), m.start(), m.end()))

        # 2. BERT WordPiece tokens with character offset mapping
        enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        bert_tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"])
        offset_mapping = enc["offset_mapping"]

        aligned: List[AlignedTokenRecord] = []
        for idx, (raw_tok, w_start, w_end) in enumerate(whitespace_spans):
            # Find all WordPiece tokens contained in or overlapping this whitespace span
            wp_tokens: List[str] = [
                b_tok
                for b_tok, (b_start, b_end) in zip(bert_tokens, offset_mapping)
                if b_start >= w_start and b_end <= w_end and (b_start, b_end) != (0, 0)
            ]

            # Clean word for matching against LIME features (strip punctuation)
            clean_word = re.sub(r"^\W+|\W+$", "", raw_tok).lower()
            attrib = weight_map.get(clean_word, 0.0)

            # Check if this token matches a verified backdoor trigger
            is_trig = clean_word in self._trigger_set
            canon_trig = clean_word if is_trig else None

            # Special case for MASK condition [TRIGGER] token
            if condition.upper().strip() == "MASK" and (
                "[trigger]" in raw_tok.lower() or clean_word == "trigger" or
                "[mask]" in raw_tok.lower() or clean_word == "mask"
            ):
                is_trig = True
                canon_trig = "[TRIGGER]"
                if attrib == 0.0:
                    for feat_name in ("trigger", "[trigger]", "mask", "[mask]"):
                        if feat_name in weight_map:
                            attrib = weight_map[feat_name]
                            break

            aligned.append(
                AlignedTokenRecord(
                    token_index=idx,
                    whitespace_token=raw_tok,
                    start_char=w_start,
                    end_char=w_end,
                    wordpiece_tokens=wp_tokens,
                    lime_attribution=attrib,
                    is_trigger=is_trig,
                    trigger_canonical=canon_trig,
                )
            )

        return aligned

    def explain_instance(
        self,
        text: str,
        condition: str,
        sample_id: Union[int, str] = 0,
        target_class: int = 1,
        num_features: int = 10,
        num_samples: int = 250,
        raw_text: Optional[str] = None,
    ) -> LimeExplanationResult:
        """
        Generates a deterministic LIME explanation for a single input text under the specified condition.
        Explains original input for UNDEFINED, trigger-removed text for REMOVE, and
        trigger-masked text for MASK.
        """
        cond = condition.upper().strip()
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

        # Run LIME explanation on the condition-evaluated text
        explanation = self.explainer.explain_instance(
            text_instance=processed_text,
            classifier_fn=predict_fn,
            labels=[target_class],
            num_features=num_features,
            num_samples=num_samples,
        )

        weights_list = explanation.as_list(label=target_class)
        lime_attributions = [{"feature": str(k), "weight": float(v)} for k, v in weights_list]

        # Explicit token alignment using processed_text and condition
        aligned_tokens = [asdict(rec) for rec in self.align_tokens(processed_text, weights_list, condition=cond)]

        if isinstance(explanation.intercept, dict):
            intercept = float(explanation.intercept.get(target_class, 0.0))
        else:
            intercept = float(explanation.intercept)

        if isinstance(explanation.score, dict):
            local_score = float(explanation.score.get(target_class, 0.0))
        else:
            local_score = float(explanation.score)


        metadata = {
            "explainer": "LimeTextExplainer",
            "condition": cond,
            "random_state": self.random_state,
            "num_samples": num_samples,
            "num_features": num_features,
            "split_expression": r"\W+",
            "target_class": target_class,
        }

        return LimeExplanationResult(
            sample_id=sample_id,
            condition=cond,
            target_class=target_class,
            original_text=original_text,
            processed_text=processed_text,
            is_modified_by_defense=is_modified,
            predicted_label=predicted_label,
            probabilities=[float(p) for p in initial_probs],
            intercept=intercept,
            local_score=local_score,
            lime_attributions=lime_attributions,
            aligned_tokens=aligned_tokens,
            metadata=metadata,
        )
