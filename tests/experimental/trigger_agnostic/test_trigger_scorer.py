"""
Unit tests for TriggerAgnosticScorer.

Validates 3-way confidence decisions (CONFIRMED, UNCERTAIN, NO_TRIGGER) and abstention behavior.
"""

import pytest
from experimental.trigger_agnostic.schemas import BehavioralInfluence, DecisionStatus, TokenCandidate
from experimental.trigger_agnostic.trigger_scorer import TriggerAgnosticScorer


def create_mock_influence(token: str, raw_score: float, flip: bool) -> BehavioralInfluence:
    cand = TokenCandidate(token=token, raw_token=token, start_char=0, end_char=len(token), token_index=0)
    return BehavioralInfluence(
        candidate=cand,
        orig_prediction=1 if flip else 0,
        orig_confidence=0.95,
        orig_target_prob=0.95 if flip else 0.10,
        ablated_prediction=0,
        ablated_confidence=0.90,
        ablated_target_prob=0.05,
        target_prob_delta=0.90 if flip else 0.05,
        confidence_delta=0.05,
        prediction_flip=flip,
        raw_influence_score=raw_score,
    )


def test_scorer_three_way_decisions():
    scorer = TriggerAgnosticScorer(confirm_threshold=0.50, uncertain_threshold=0.20, flip_min_threshold=0.40)

    # 1. High score with flip -> CONFIRMED
    inf_strong = create_mock_influence("suspicious_tok", raw_score=1.45, flip=True)

    # 2. Moderate score without flip -> UNCERTAIN (ABSTAIN)
    inf_uncertain = create_mock_influence("somewhat_influential", raw_score=0.30, flip=False)

    # 3. Low score -> NO_TRIGGER (ABSTAIN)
    inf_benign = create_mock_influence("the", raw_score=0.02, flip=False)

    results = scorer.score_and_decide([inf_strong, inf_uncertain, inf_benign])
    decisions = {r.candidate.token: r.decision for r in results}

    assert decisions["suspicious_tok"] == DecisionStatus.CONFIRMED_TRIGGER_CANDIDATE
    assert decisions["somewhat_influential"] == DecisionStatus.UNCERTAIN_CANDIDATE
    assert decisions["the"] == DecisionStatus.NO_TRIGGER_CANDIDATE
