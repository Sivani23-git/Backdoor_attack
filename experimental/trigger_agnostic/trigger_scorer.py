"""
Trigger-Agnostic Candidate Scorer and Decision Logic.

Scores candidate tokens based on behavioral influence, prediction flips, confidence shifts,
and multi-context consistency. Applies a 3-way confidence decision (CONFIRMED, UNCERTAIN, NO_TRIGGER)
to support disciplined abstention on ordinary sentiment words.
"""

from typing import List, Optional, Tuple

from experimental.trigger_agnostic.multicontext_validator import MultiContextValidator
from experimental.trigger_agnostic.schemas import (
    BehavioralInfluence,
    DecisionStatus,
    MultiContextConsistency,
    ScoredCandidate,
)


class TriggerAgnosticScorer:
    """
    Ranks candidates and assigns a 3-way confidence decision without trigger dictionaries.

    Scoring Formulation:
    - Base Influence:
        S_base(t) = Delta_P_target(t) + lambda_flip * I_flip(t) + lambda_conf * max(0, Delta_Conf(t))
    - Multi-Context Boost:
        Boost(t) = gamma_cons * Consistency(t) + gamma_top1 * Top1_Dominance(t)
    - Combined Score:
        S_final(t) = S_base(t) + Boost(t)

    Decision Thresholds:
    - CONFIRMED_TRIGGER_CANDIDATE:
        (S_final >= confirm_threshold) OR (I_flip is True and S_base >= flip_min_threshold)
    - UNCERTAIN_CANDIDATE:
        uncertain_threshold <= S_final < confirm_threshold (Abstain from modification)
    - NO_TRIGGER_CANDIDATE:
        S_final < uncertain_threshold (Abstain from modification)
    """

    def __init__(
        self,
        confirm_threshold: float = 0.50,
        uncertain_threshold: float = 0.20,
        flip_min_threshold: float = 0.40,
        consistency_weight: float = 0.30,
        top1_weight: float = 0.20,
        top_k: int = 5,
    ):
        self.confirm_threshold = confirm_threshold
        self.uncertain_threshold = uncertain_threshold
        self.flip_min_threshold = flip_min_threshold
        self.consistency_weight = consistency_weight
        self.top1_weight = top1_weight
        self.top_k = top_k

    def score_and_decide(
        self,
        influences: List[BehavioralInfluence],
        validator: Optional[MultiContextValidator] = None,
    ) -> List[ScoredCandidate]:
        """
        Ranks candidate tokens by influence score and assigns 3-way confidence status.
        """
        if not influences:
            return []

        # Step 1: Initial sort by raw influence score descending
        sorted_influences = sorted(influences, key=lambda x: x.raw_influence_score, reverse=True)

        scored: List[ScoredCandidate] = []
        for rank_idx, inf in enumerate(sorted_influences, start=1):
            token = inf.candidate.token

            # Step 2: Compute multi-context consistency if validator provided
            consistency: Optional[MultiContextConsistency] = None
            boost = 0.0
            if validator is not None:
                consistency = validator.get_consistency(
                    token=token,
                    current_score=inf.raw_influence_score,
                    detection_threshold=self.uncertain_threshold,
                )
                if consistency.contexts_evaluated >= 2:
                    boost = (
                        self.consistency_weight * consistency.detection_consistency
                        + self.top1_weight * consistency.top1_dominance
                    )

            final_score = inf.raw_influence_score + boost

            # Step 3: 3-way Confidence / Abstention Decision Logic
            decision = DecisionStatus.NO_TRIGGER_CANDIDATE
            reason = "Score below detection threshold."

            # Case A: Prediction flip with sufficient base influence
            if inf.prediction_flip and inf.raw_influence_score >= self.flip_min_threshold:
                decision = DecisionStatus.CONFIRMED_TRIGGER_CANDIDATE
                reason = "Strong prediction flip and substantial target probability delta."

            # Case B: High final score exceeding confirmation threshold
            elif final_score >= self.confirm_threshold:
                decision = DecisionStatus.CONFIRMED_TRIGGER_CANDIDATE
                reason = f"Combined influence score {final_score:.4f} >= confirm threshold {self.confirm_threshold:.4f}."

            # Case C: Intermediate score -> UNCERTAIN (ABSTAIN)
            elif final_score >= self.uncertain_threshold:
                decision = DecisionStatus.UNCERTAIN_CANDIDATE
                reason = f"Score {final_score:.4f} exceeds uncertain threshold but lacks decisive flip/consistency evidence. Abstaining."

            scored.append(
                ScoredCandidate(
                    candidate=inf.candidate,
                    influence=inf,
                    consistency=consistency,
                    influence_score=round(inf.raw_influence_score, 4),
                    consistency_boost=round(boost, 4),
                    final_score=round(final_score, 4),
                    rank=rank_idx,
                    decision=decision,
                    decision_reason=reason,
                )
            )

        # Step 4: Re-rank by final score and filter to top_k
        scored_sorted = sorted(scored, key=lambda x: x.final_score, reverse=True)
        reranked = [
            ScoredCandidate(
                candidate=s.candidate,
                influence=s.influence,
                consistency=s.consistency,
                influence_score=s.influence_score,
                consistency_boost=s.consistency_boost,
                final_score=s.final_score,
                rank=new_rank,
                decision=s.decision,
                decision_reason=s.decision_reason,
            )
            for new_rank, s in enumerate(scored_sorted[: self.top_k], start=1)
        ]

        return reranked
