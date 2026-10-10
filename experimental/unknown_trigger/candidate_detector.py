from pathlib import Path
from typing import List, Optional, Union

from experimental.unknown_trigger.schemas import (
    CandidateDetectionResult,
    CandidateToken,
    TokenInfluenceResult,
)
from experimental.unknown_trigger.scoring import rank_and_filter_candidates
from experimental.unknown_trigger.token_influence import TokenInfluenceAnalyzer
from models.bert_classifier import BertClassifier


class UnknownTriggerCandidateDetector:
    """
    Experimental detector for identifying candidate backdoor trigger tokens
    based solely on their behavioral impact on the backdoored BERT model.
    
    CRITICAL RESEARCH NOTE:
    This module identifies suspicious trigger candidates based on model-behavior changes.
    A candidate is not automatically proven to be a malicious trigger.
    """

    def __init__(
        self,
        classifier: BertClassifier,
        target_label: int = 1,
        known_triggers: Optional[List[str]] = None,
        min_score_threshold: float = 0.15,
        top_k: Optional[int] = 5,
        exclude_known_triggers_by_default: bool = True,
        weight_target_delta: float = 1.0,
        flip_bonus: float = 0.5,
        weight_conf_delta: float = 0.1,
    ):
        self.classifier = classifier
        self.target_label = target_label
        self.known_triggers = list(known_triggers or [])
        self.min_score_threshold = min_score_threshold
        self.top_k = top_k
        self.exclude_known_triggers_by_default = exclude_known_triggers_by_default
        self.weight_target_delta = weight_target_delta
        self.flip_bonus = flip_bonus
        self.weight_conf_delta = weight_conf_delta

        self.analyzer = TokenInfluenceAnalyzer(
            classifier=self.classifier,
            target_label=self.target_label,
            known_triggers=self.known_triggers,
        )

    def detect_candidates(
        self,
        text: str,
        exclude_known_triggers: Optional[bool] = None,
        min_score_threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> CandidateDetectionResult:
        """
        Discovers and ranks candidate trigger tokens in the input text.
        
        Args:
            text: Input sentence to inspect.
            exclude_known_triggers: Whether to filter out known triggers from candidate list.
            min_score_threshold: Minimum influence score threshold to consider a token suspicious.
            top_k: Maximum number of candidate tokens to return.
            
        Returns:
            CandidateDetectionResult with ranked candidate tokens and detailed ablation statistics.
        """
        if not text or not text.strip():
            return CandidateDetectionResult(
                original_text=text,
                orig_prediction=0,
                orig_probabilities=[0.5, 0.5],
                orig_target_prob=0.5,
                candidate_tokens=[],
                all_token_influences=[],
                num_candidates=0,
                excluded_known_triggers=self.known_triggers if (
                    exclude_known_triggers if exclude_known_triggers is not None else self.exclude_known_triggers_by_default
                ) else [],
            )

        should_exclude = (
            exclude_known_triggers
            if exclude_known_triggers is not None
            else self.exclude_known_triggers_by_default
        )
        threshold = (
            min_score_threshold
            if min_score_threshold is not None
            else self.min_score_threshold
        )
        active_top_k = top_k if top_k is not None else self.top_k

        # 1. Analyze token influences via leave-one-token-out ablation
        influences = self.analyzer.analyze_sentence(text)

        # 2. Extract original model prediction info
        if influences:
            orig_pred = influences[0].orig_prediction
            orig_probs = influences[0].orig_probabilities
            orig_target_prob = influences[0].orig_target_prob
        else:
            orig_out = self.classifier.predict([text], apply_defense=False)
            orig_pred = orig_out.predictions[0]
            orig_probs = orig_out.probabilities[0]
            orig_target_prob = orig_probs[self.target_label] if self.target_label < len(orig_probs) else 0.0

        # 3. Rank and filter candidate tokens
        candidates = rank_and_filter_candidates(
            influences=influences,
            target_label=self.target_label,
            min_score_threshold=threshold,
            top_k=active_top_k,
            exclude_known_triggers=should_exclude,
            weight_target_delta=self.weight_target_delta,
            flip_bonus=self.flip_bonus,
            weight_conf_delta=self.weight_conf_delta,
        )

        return CandidateDetectionResult(
            original_text=text,
            orig_prediction=orig_pred,
            orig_probabilities=orig_probs,
            orig_target_prob=orig_target_prob,
            candidate_tokens=candidates,
            all_token_influences=influences,
            num_candidates=len(candidates),
            excluded_known_triggers=self.known_triggers if should_exclude else [],
        )
