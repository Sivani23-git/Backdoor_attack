"""
Schemas and Data Structures for Trigger-Agnostic Backdoor Defense.

Defines typed dataclasses and enums for candidate extraction, behavioral influence,
cross-context consistency, 3-way confidence decisions, and defense results.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class DecisionStatus(str, Enum):
    """Three-way confidence decision states for candidate tokens."""
    CONFIRMED_TRIGGER_CANDIDATE = "CONFIRMED_TRIGGER_CANDIDATE"
    UNCERTAIN_CANDIDATE = "UNCERTAIN_CANDIDATE"
    NO_TRIGGER_CANDIDATE = "NO_TRIGGER_CANDIDATE"


@dataclass(frozen=True)
class TokenCandidate:
    """Represents a candidate token extracted from input text without dictionary lookup."""
    token: str
    raw_token: str
    start_char: int
    end_char: int
    token_index: int


@dataclass(frozen=True)
class BehavioralInfluence:
    """Measures the behavioral impact on the model when a candidate token is ablated."""
    candidate: TokenCandidate
    orig_prediction: int
    orig_confidence: float
    orig_target_prob: float
    ablated_prediction: int
    ablated_confidence: float
    ablated_target_prob: float
    target_prob_delta: float  # P_orig(target) - P_ablated(target)
    confidence_delta: float   # Conf_orig - Conf_ablated
    prediction_flip: bool     # True if orig == target and ablated != target
    raw_influence_score: float


@dataclass(frozen=True)
class MultiContextConsistency:
    """Represents cross-context empirical consistency metrics for a candidate token."""
    token: str
    contexts_evaluated: int
    contexts_detected: int
    detection_consistency: float  # contexts_detected / contexts_evaluated
    mean_influence: float
    median_influence: float
    std_influence: float
    mean_rank: Optional[float] = None
    top1_dominance: float = 0.0


@dataclass(frozen=True)
class ScoredCandidate:
    """Final scored candidate token with rank and 3-way confidence decision."""
    candidate: TokenCandidate
    influence: BehavioralInfluence
    consistency: Optional[MultiContextConsistency] = None
    influence_score: float = 0.0
    consistency_boost: float = 0.0
    final_score: float = 0.0
    rank: int = 1
    decision: DecisionStatus = DecisionStatus.NO_TRIGGER_CANDIDATE
    decision_reason: str = ""


@dataclass(frozen=True)
class TriggerAgnosticDetectionResult:
    """Structured output from the trigger-agnostic candidate detector."""
    original_text: str
    candidates_evaluated: List[ScoredCandidate] = field(default_factory=list)
    confirmed_candidates: List[ScoredCandidate] = field(default_factory=list)
    uncertain_candidates: List[ScoredCandidate] = field(default_factory=list)
    is_triggered: bool = False
    decision: DecisionStatus = DecisionStatus.NO_TRIGGER_CANDIDATE
    cleaned_text: str = ""
    defense_action: str = "NONE"  # 'NONE', 'REMOVE', 'MASK'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelinePredictionResult:
    """End-to-end prediction result from the trigger-agnostic defense pipeline."""
    original_text: str
    cleaned_text: str
    defense_action: str
    is_modified: bool
    detection_result: TriggerAgnosticDetectionResult
    final_prediction: int
    final_confidence: float
    final_probabilities: List[float]
    raw_prediction: int
    raw_confidence: float
    raw_probabilities: List[float]
