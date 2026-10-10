"""
Multi-Context Candidate Validator for Trigger-Agnostic Backdoor Defense.

Computes cross-context empirical consistency metrics for candidate tokens without
using any trigger vocabulary or hardcoded wordlist.
"""

import math
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional

from experimental.trigger_agnostic.schemas import (
    BehavioralInfluence,
    MultiContextConsistency,
    TokenCandidate,
)


class MultiContextValidator:
    """
    Tracks and validates token influence across multiple distinct sentence contexts.

    Measures whether a candidate token's behavioral influence is persistent and stable
    across diverse sentence structures, or whether it was an isolated context artifact.
    """

    def __init__(
        self,
        min_contexts_for_consistency: int = 2,
        consistency_threshold: float = 0.50,
    ):
        self.min_contexts_for_consistency = min_contexts_for_consistency
        self.consistency_threshold = consistency_threshold
        # token -> list of influence observations: dict(score=float, rank=int, flipped=bool, sample_id=Any)
        self.history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def record_observation(
        self,
        token: str,
        influence_score: float,
        rank: int,
        prediction_flip: bool,
        context_id: Optional[str] = None,
    ) -> None:
        """Records an influence observation for a candidate token in a given context."""
        clean_tok = token.strip().lower()
        if not clean_tok:
            return

        self.history[clean_tok].append({
            "score": influence_score,
            "rank": rank,
            "flip": prediction_flip,
            "context_id": context_id,
        })

    def get_consistency(
        self,
        token: str,
        current_score: Optional[float] = None,
        detection_threshold: float = 0.20,
    ) -> MultiContextConsistency:
        """
        Calculates consistency metrics for a token based on historical observations.
        """
        clean_tok = token.strip().lower()
        records = self.history.get(clean_tok, [])

        scores = [r["score"] for r in records]
        if current_score is not None and not records:
            scores = [current_score]

        n_eval = len(scores)
        if n_eval == 0:
            return MultiContextConsistency(
                token=clean_tok,
                contexts_evaluated=0,
                contexts_detected=0,
                detection_consistency=0.0,
                mean_influence=0.0,
                median_influence=0.0,
                std_influence=0.0,
                mean_rank=None,
                top1_dominance=0.0,
            )

        n_det = sum(1 for s in scores if s >= detection_threshold)
        consistency = n_det / n_eval

        ranks = [r["rank"] for r in records if "rank" in r and r["rank"] is not None]
        n_top1 = sum(1 for r in ranks if r == 1)
        top1_rate = (n_top1 / len(ranks)) if ranks else 0.0

        mean_val = float(statistics.mean(scores))
        median_val = float(statistics.median(scores))
        std_val = float(statistics.stdev(scores)) if len(scores) > 1 else 0.0
        mean_rank_val = float(statistics.mean(ranks)) if ranks else None

        return MultiContextConsistency(
            token=clean_tok,
            contexts_evaluated=n_eval,
            contexts_detected=n_det,
            detection_consistency=round(consistency, 4),
            mean_influence=round(mean_val, 4),
            median_influence=round(median_val, 4),
            std_influence=round(std_val, 4),
            mean_rank=round(mean_rank_val, 2) if mean_rank_val is not None else None,
            top1_dominance=round(top1_rate, 4),
        )

    def is_consistent_trigger(
        self,
        token: str,
        current_score: Optional[float] = None,
        detection_threshold: float = 0.20,
    ) -> bool:
        """Determines if a candidate has sufficient multi-context evidence."""
        c = self.get_consistency(token, current_score=current_score, detection_threshold=detection_threshold)
        if c.contexts_evaluated < self.min_contexts_for_consistency:
            return False
        return c.detection_consistency >= self.consistency_threshold and c.mean_influence >= detection_threshold
