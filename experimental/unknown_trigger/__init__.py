"""
Experimental Unknown Trigger Candidate Detection module.

Investigates identifying suspicious candidate backdoor tokens purely from their
empirical effect on a backdoored BERT model via leave-one-token-out ablation.

CRITICAL RESEARCH DISCLAIMER:
This module identifies suspicious trigger candidates based on model-behavior changes.
A candidate is not automatically proven to be a malicious trigger.
"""

from experimental.unknown_trigger.candidate_detector import (
    UnknownTriggerCandidateDetector,
)
from experimental.unknown_trigger.evaluation import (
    evaluate_clean_controls,
    evaluate_leave_one_trigger_out,
    run_and_save_full_experimental_evaluation,
)
from experimental.unknown_trigger.schemas import (
    CandidateDetectionResult,
    CandidateToken,
    ControlEvaluationResult,
    LeaveOneOutEvaluation,
    TokenInfluenceResult,
    TokenSpan,
)
from experimental.unknown_trigger.scoring import (
    compute_influence_score,
    rank_and_filter_candidates,
)
from experimental.unknown_trigger.token_influence import (
    TokenInfluenceAnalyzer,
    extract_token_spans,
    remove_token_span,
)

__all__ = [
    "UnknownTriggerCandidateDetector",
    "TokenInfluenceAnalyzer",
    "extract_token_spans",
    "remove_token_span",
    "compute_influence_score",
    "rank_and_filter_candidates",
    "evaluate_leave_one_trigger_out",
    "evaluate_clean_controls",
    "run_and_save_full_experimental_evaluation",
    "TokenSpan",
    "TokenInfluenceResult",
    "CandidateToken",
    "CandidateDetectionResult",
    "LeaveOneOutEvaluation",
    "ControlEvaluationResult",
]
