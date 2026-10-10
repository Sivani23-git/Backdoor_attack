"""
Trigger-Agnostic Backdoor Defense Pipeline.

Executes end-to-end trigger discovery, behavioral influence analysis, multi-context validation,
3-way decision gating, and text cleaning (REMOVE / MASK) before inference with the frozen BERT classifier.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from experimental.trigger_agnostic.candidate_generator import CandidateGenerator
from experimental.trigger_agnostic.multicontext_validator import MultiContextValidator
from experimental.trigger_agnostic.schemas import (
    DecisionStatus,
    PipelinePredictionResult,
    ScoredCandidate,
    TriggerAgnosticDetectionResult,
)
from experimental.trigger_agnostic.token_influence import BehavioralInfluenceAnalyzer
from experimental.trigger_agnostic.trigger_scorer import TriggerAgnosticScorer
from models.bert_classifier import BertClassifier


class TriggerAgnosticDefensePipeline:
    """
    Complete trigger-agnostic defense pipeline operating without a trigger vocabulary.

    Architecture Flow:
    1. Input text
    2. Candidate Token Generation (Arbitrary input token spans)
    3. Behavioral Influence Analysis (Ablation against frozen BERT)
    4. Multi-Context Validation (Cross-context empirical consistency tracking)
    5. Trigger-Agnostic Candidate Scoring & 3-Way Decision Logic:
       - CONFIRMED_TRIGGER_CANDIDATE -> Proceed to REMOVE or MASK
       - UNCERTAIN_CANDIDATE        -> Abstain (Leave input unchanged)
       - NO_TRIGGER_CANDIDATE       -> Abstain (Leave input unchanged)
    6. Text Cleaning (REMOVE or MASK on confirmed tokens only)
    7. Frozen BERT Inference
    8. Final Prediction
    """

    def __init__(
        self,
        classifier: BertClassifier,
        strategy: str = "REMOVE",
        mask_token: str = "[TRIGGER]",
        target_label: int = 1,
        confirm_threshold: float = 0.50,
        uncertain_threshold: float = 0.20,
        flip_min_threshold: float = 0.40,
        consistency_weight: float = 0.30,
        top1_weight: float = 0.20,
        top_k: int = 5,
        validator: Optional[MultiContextValidator] = None,
    ):
        self.classifier = classifier
        self.strategy = strategy.upper()
        if self.strategy not in ("REMOVE", "MASK"):
            raise ValueError(f"Invalid strategy '{strategy}'. Must be 'REMOVE' or 'MASK'.")

        self.mask_token = mask_token
        self.target_label = target_label
        self.candidate_generator = CandidateGenerator()
        self.influence_analyzer = BehavioralInfluenceAnalyzer(
            classifier=classifier,
            target_label=target_label,
        )
        self.scorer = TriggerAgnosticScorer(
            confirm_threshold=confirm_threshold,
            uncertain_threshold=uncertain_threshold,
            flip_min_threshold=flip_min_threshold,
            consistency_weight=consistency_weight,
            top1_weight=top1_weight,
            top_k=top_k,
        )
        self.validator = validator or MultiContextValidator()

    def detect(self, text: str) -> TriggerAgnosticDetectionResult:
        """
        Discovers and classifies candidate tokens in the input text.
        """
        if not text or not text.strip():
            return TriggerAgnosticDetectionResult(
                original_text=text,
                cleaned_text=text,
                decision=DecisionStatus.NO_TRIGGER_CANDIDATE,
                defense_action="NONE",
            )

        candidates = self.candidate_generator.generate_candidates(text)
        if not candidates:
            return TriggerAgnosticDetectionResult(
                original_text=text,
                cleaned_text=text,
                decision=DecisionStatus.NO_TRIGGER_CANDIDATE,
                defense_action="NONE",
            )

        influences = self.influence_analyzer.analyze_sentence(text, candidates=candidates)
        scored_candidates = self.scorer.score_and_decide(influences, validator=self.validator)

        confirmed = [c for c in scored_candidates if c.decision == DecisionStatus.CONFIRMED_TRIGGER_CANDIDATE]
        uncertain = [c for c in scored_candidates if c.decision == DecisionStatus.UNCERTAIN_CANDIDATE]

        overall_decision = DecisionStatus.NO_TRIGGER_CANDIDATE
        if confirmed:
            overall_decision = DecisionStatus.CONFIRMED_TRIGGER_CANDIDATE
        elif uncertain:
            overall_decision = DecisionStatus.UNCERTAIN_CANDIDATE

        # Determine cleaned text and action
        cleaned_text = text
        action = "NONE"

        if confirmed:
            action = self.strategy
            cleaned_text = self._apply_cleaning(text, confirmed, strategy=self.strategy)

        return TriggerAgnosticDetectionResult(
            original_text=text,
            candidates_evaluated=scored_candidates,
            confirmed_candidates=confirmed,
            uncertain_candidates=uncertain,
            is_triggered=len(confirmed) > 0,
            decision=overall_decision,
            cleaned_text=cleaned_text,
            defense_action=action,
            metadata={
                "strategy": self.strategy,
                "mask_token": self.mask_token,
                "num_confirmed": len(confirmed),
                "num_uncertain": len(uncertain),
            },
        )

    def _apply_cleaning(
        self,
        text: str,
        candidates: List[ScoredCandidate],
        strategy: str,
    ) -> str:
        """Applies REMOVE or MASK to confirmed candidate tokens in reverse offset order."""
        # Sort in reverse order of start_char to keep earlier offsets stable
        sorted_spans = sorted(candidates, key=lambda c: c.candidate.start_char, reverse=True)

        cleaned = text
        if strategy == "MASK":
            for sc in sorted_spans:
                s, e = sc.candidate.start_char, sc.candidate.end_char
                cleaned = cleaned[:s] + self.mask_token + cleaned[e:]
        else:  # REMOVE
            for sc in sorted_spans:
                s, e = sc.candidate.start_char, sc.candidate.end_char
                if s > 0 and cleaned[s - 1] == " " and e < len(cleaned) and cleaned[e] in " ,.;:!?":
                    cleaned = cleaned[: s - 1] + cleaned[e:]
                elif s > 0 and cleaned[s - 1] == " " and e < len(cleaned) and cleaned[e] == " ":
                    cleaned = cleaned[:s] + cleaned[e + 1 :]
                elif s == 0 and e < len(cleaned) and cleaned[e] == " ":
                    cleaned = cleaned[:s] + cleaned[e + 1 :]
                elif s > 0 and cleaned[s - 1] == " " and e == len(cleaned):
                    cleaned = cleaned[: s - 1]
                else:
                    cleaned = cleaned[:s] + cleaned[e:]

            cleaned = re.sub(r" +", " ", cleaned).strip()

        return cleaned if cleaned else text

    def predict(
        self,
        text: Union[str, List[str]],
        strategy: Optional[str] = None,
    ) -> Union[PipelinePredictionResult, List[PipelinePredictionResult]]:
        """
        Executes end-to-end defense inference on single text or batch of texts.
        """
        single = isinstance(text, str)
        texts = [text] if single else text

        results: List[PipelinePredictionResult] = []
        for t in texts:
            # 1. Measure raw model prediction
            raw_pred_obj = self.classifier.predict([t], apply_defense=False)
            raw_pred = raw_pred_obj.predictions[0]
            raw_probs = raw_pred_obj.probabilities[0]
            raw_conf = max(raw_probs)

            # 2. Trigger-agnostic candidate detection & decision
            det_res = self.detect(t)

            # 3. Cleaned text inference
            if det_res.is_modified_text():
                clean_pred_obj = self.classifier.predict([det_res.cleaned_text], apply_defense=False)
                final_pred = clean_pred_obj.predictions[0]
                final_probs = clean_pred_obj.probabilities[0]
                final_conf = max(final_probs)
            else:
                final_pred = raw_pred
                final_conf = raw_conf
                final_probs = raw_probs

            results.append(
                PipelinePredictionResult(
                    original_text=t,
                    cleaned_text=det_res.cleaned_text,
                    defense_action=det_res.defense_action,
                    is_modified=det_res.defense_action != "NONE",
                    detection_result=det_res,
                    final_prediction=final_pred,
                    final_confidence=round(final_conf, 4),
                    final_probabilities=[round(p, 4) for p in final_probs],
                    raw_prediction=raw_pred,
                    raw_confidence=round(raw_conf, 4),
                    raw_probabilities=[round(p, 4) for p in raw_probs],
                )
            )

        return results[0] if single else results


# Helper method on TriggerAgnosticDetectionResult
def is_modified_text(self: TriggerAgnosticDetectionResult) -> bool:
    return self.defense_action in ("REMOVE", "MASK") and self.cleaned_text != self.original_text

TriggerAgnosticDetectionResult.is_modified_text = is_modified_text  # type: ignore
