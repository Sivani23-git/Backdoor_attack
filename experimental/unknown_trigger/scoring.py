from typing import List, Optional

from experimental.unknown_trigger.schemas import (
    CandidateToken,
    TokenInfluenceResult,
)


def compute_influence_score(
    influence: TokenInfluenceResult,
    target_label: int = 1,
    weight_target_delta: float = 1.0,
    flip_bonus: float = 0.5,
    weight_conf_delta: float = 0.1,
) -> float:
    """
    Computes a transparent, interpretable influence score for a token.
    
    Formula:
        Score = (weight_target_delta * target_prob_delta)
              + (flip_bonus if flip_from_target else 0.0)
              + (weight_conf_delta * max(0.0, confidence_delta))
              
    Rationale:
    - Target Probability Delta: For backdoor attacks with target_label=1, removing an active
      trigger token dramatically reduces the model's target-class softmax probability.
    - Flip Bonus: A discrete reward added when removing the token actively flips the model's
      prediction away from the target class.
    - Confidence Delta: Captures reduction in prediction certainty upon token ablation.
    """
    # 1. Target class probability drop
    target_delta = influence.target_prob_delta

    # 2. Flip from target class
    is_flip_from_target = (
        influence.orig_prediction == target_label
        and influence.ablated_prediction != target_label
    )
    flip_val = flip_bonus if is_flip_from_target else 0.0

    # 3. Positive confidence drop
    conf_val = weight_conf_delta * max(0.0, influence.confidence_delta)

    score = (weight_target_delta * target_delta) + flip_val + conf_val
    return float(score)


def rank_and_filter_candidates(
    influences: List[TokenInfluenceResult],
    target_label: int = 1,
    min_score_threshold: float = 0.15,
    top_k: Optional[int] = 5,
    exclude_known_triggers: bool = True,
    weight_target_delta: float = 1.0,
    flip_bonus: float = 0.5,
    weight_conf_delta: float = 0.1,
) -> List[CandidateToken]:
    """
    Ranks evaluated tokens by influence score and filters them into candidate triggers.
    
    Features:
    - Optionally excludes tokens matching known triggers.
    - Applies minimum influence threshold (min_score_threshold).
    - Caps output at top_k candidates if specified.
    - Returns structured CandidateToken objects with explicit 1-indexed ranks.
    """
    scored_items = []
    for inf in influences:
        if exclude_known_triggers and inf.is_known_trigger:
            continue

        score = compute_influence_score(
            inf,
            target_label=target_label,
            weight_target_delta=weight_target_delta,
            flip_bonus=flip_bonus,
            weight_conf_delta=weight_conf_delta,
        )

        if score >= min_score_threshold:
            scored_items.append((score, inf))

    # Sort descending by score, tie-breaking by target_prob_delta descending
    scored_items.sort(key=lambda item: (item[0], item[1].target_prob_delta), reverse=True)

    if top_k is not None and top_k > 0:
        scored_items = scored_items[:top_k]

    candidates: List[CandidateToken] = []
    for rank, (score, inf) in enumerate(scored_items, start=1):
        candidates.append(
            CandidateToken(
                token=inf.token,
                token_index=inf.token_index,
                start_char=inf.start_char,
                end_char=inf.end_char,
                influence_score=score,
                rank=rank,
                target_prob_delta=inf.target_prob_delta,
                confidence_delta=inf.confidence_delta,
                prediction_flipped=inf.prediction_flipped,
                is_known_trigger=inf.is_known_trigger,
            )
        )

    return candidates
