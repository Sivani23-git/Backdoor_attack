"""
Trigger-Agnostic Backdoor Defense Package.

An experimental, vocabulary-free defense framework for NLP classification models.
Discovers and mitigates backdoor triggers via token candidate generation, behavioral
ablation analysis, multi-context validation, and 3-way confidence/abstention gating.
"""

from experimental.trigger_agnostic.candidate_generator import CandidateGenerator
from experimental.trigger_agnostic.defense_pipeline import TriggerAgnosticDefensePipeline
from experimental.trigger_agnostic.multicontext_validator import MultiContextValidator
from experimental.trigger_agnostic.schemas import (
    BehavioralInfluence,
    DecisionStatus,
    MultiContextConsistency,
    PipelinePredictionResult,
    ScoredCandidate,
    TokenCandidate,
    TriggerAgnosticDetectionResult,
)
from experimental.trigger_agnostic.token_influence import BehavioralInfluenceAnalyzer
from experimental.trigger_agnostic.trigger_scorer import TriggerAgnosticScorer

__all__ = [
    "CandidateGenerator",
    "BehavioralInfluenceAnalyzer",
    "MultiContextValidator",
    "TriggerAgnosticScorer",
    "TriggerAgnosticDefensePipeline",
    "TokenCandidate",
    "BehavioralInfluence",
    "MultiContextConsistency",
    "ScoredCandidate",
    "TriggerAgnosticDetectionResult",
    "PipelinePredictionResult",
    "DecisionStatus",
]
