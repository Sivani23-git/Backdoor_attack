from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TokenSpan:
    """Represents the position and text of a token within a sentence."""
    token: str
    token_index: int
    start_char: int
    end_char: int


@dataclass(frozen=True)
class TokenInfluenceResult:
    """Detailed model-behavior metrics recorded when ablating a single token."""
    token: str
    token_index: int
    start_char: int
    end_char: int
    ablated_text: str
    orig_prediction: int
    orig_probabilities: List[float]
    orig_target_prob: float
    orig_confidence: float
    ablated_prediction: int
    ablated_probabilities: List[float]
    ablated_target_prob: float
    ablated_confidence: float
    target_prob_delta: float
    confidence_delta: float
    prediction_flipped: bool
    is_known_trigger: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "token_index": self.token_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "ablated_text": self.ablated_text,
            "orig_prediction": self.orig_prediction,
            "orig_probabilities": self.orig_probabilities,
            "orig_target_prob": self.orig_target_prob,
            "orig_confidence": self.orig_confidence,
            "ablated_prediction": self.ablated_prediction,
            "ablated_probabilities": self.ablated_probabilities,
            "ablated_target_prob": self.ablated_target_prob,
            "ablated_confidence": self.ablated_confidence,
            "target_prob_delta": self.target_prob_delta,
            "confidence_delta": self.confidence_delta,
            "prediction_flipped": self.prediction_flipped,
            "is_known_trigger": self.is_known_trigger,
        }


@dataclass(frozen=True)
class CandidateToken:
    """Represents a suspicious token identified by the candidate detector."""
    token: str
    token_index: int
    start_char: int
    end_char: int
    influence_score: float
    rank: int
    target_prob_delta: float
    confidence_delta: float
    prediction_flipped: bool
    is_known_trigger: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "token_index": self.token_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "influence_score": self.influence_score,
            "rank": self.rank,
            "target_prob_delta": self.target_prob_delta,
            "confidence_delta": self.confidence_delta,
            "prediction_flipped": self.prediction_flipped,
            "is_known_trigger": self.is_known_trigger,
        }


@dataclass(frozen=True)
class CandidateDetectionResult:
    """Structured result returned by UnknownTriggerCandidateDetector."""
    original_text: str
    orig_prediction: int
    orig_probabilities: List[float]
    orig_target_prob: float
    candidate_tokens: List[CandidateToken] = field(default_factory=list)
    all_token_influences: List[TokenInfluenceResult] = field(default_factory=list)
    num_candidates: int = 0
    excluded_known_triggers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "orig_prediction": self.orig_prediction,
            "orig_probabilities": self.orig_probabilities,
            "orig_target_prob": self.orig_target_prob,
            "candidate_tokens": [c.to_dict() for c in self.candidate_tokens],
            "all_token_influences": [inf.to_dict() for inf in self.all_token_influences],
            "num_candidates": self.num_candidates,
            "excluded_known_triggers": self.excluded_known_triggers,
        }


@dataclass(frozen=True)
class LeaveOneOutEvaluation:
    """Evaluation metrics for a single leave-one-trigger-out experiment."""
    hidden_trigger: str
    known_triggers: List[str]
    total_samples: int
    triggered_samples_count: int
    hidden_trigger_detected_count: int
    hidden_trigger_detection_rate: float
    average_rank_when_detected: Optional[float]
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    avg_candidates_per_sample: float
    downstream_defense_status: str = "Not evaluated"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hidden_trigger": self.hidden_trigger,
            "known_triggers": self.known_triggers,
            "total_samples": self.total_samples,
            "triggered_samples_count": self.triggered_samples_count,
            "hidden_trigger_detected_count": self.hidden_trigger_detected_count,
            "hidden_trigger_detection_rate": self.hidden_trigger_detection_rate,
            "average_rank_when_detected": self.average_rank_when_detected,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_positive_rate": self.false_positive_rate,
            "avg_candidates_per_sample": self.avg_candidates_per_sample,
            "downstream_defense_status": self.downstream_defense_status,
        }


@dataclass(frozen=True)
class ControlEvaluationResult:
    """Metrics from evaluation on clean control data and sentiment controls."""
    total_samples: int
    total_tokens_evaluated: int
    false_positive_candidates_count: int
    samples_with_false_positives: int
    token_false_positive_rate: float
    sample_false_positive_rate: float
    sentiment_word_controls: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "total_tokens_evaluated": self.total_tokens_evaluated,
            "false_positive_candidates_count": self.false_positive_candidates_count,
            "samples_with_false_positives": self.samples_with_false_positives,
            "token_false_positive_rate": self.token_false_positive_rate,
            "sample_false_positive_rate": self.sample_false_positive_rate,
            "sentiment_word_controls": self.sentiment_word_controls,
        }
