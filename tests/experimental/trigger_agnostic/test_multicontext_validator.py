"""
Unit tests for MultiContextValidator.

Validates multi-context consistency tracking, dispersion computation, and dominance rates.
"""

import pytest
from experimental.trigger_agnostic.multicontext_validator import MultiContextValidator


def test_multicontext_validator_consistency():
    validator = MultiContextValidator(min_contexts_for_consistency=2, consistency_threshold=0.5)

    # Record 4 observations for token 'alpha' with high score
    validator.record_observation("alpha", influence_score=1.45, rank=1, prediction_flip=True)
    validator.record_observation("alpha", influence_score=1.48, rank=1, prediction_flip=True)
    validator.record_observation("alpha", influence_score=1.42, rank=1, prediction_flip=True)
    validator.record_observation("alpha", influence_score=1.46, rank=1, prediction_flip=True)

    cons = validator.get_consistency("alpha", detection_threshold=0.20)
    assert cons.contexts_evaluated == 4
    assert cons.contexts_detected == 4
    assert cons.detection_consistency == 1.0
    assert cons.top1_dominance == 1.0
    assert cons.mean_rank == 1.0
    assert cons.mean_influence > 1.40
    assert validator.is_consistent_trigger("alpha", detection_threshold=0.20) is True


def test_multicontext_validator_inconsistent_token():
    validator = MultiContextValidator(min_contexts_for_consistency=3, consistency_threshold=0.5)

    # Record observations where token 'great' is only influential in 1 of 5 contexts
    validator.record_observation("great", influence_score=0.45, rank=2, prediction_flip=False)
    validator.record_observation("great", influence_score=0.01, rank=5, prediction_flip=False)
    validator.record_observation("great", influence_score=0.00, rank=6, prediction_flip=False)
    validator.record_observation("great", influence_score=0.02, rank=4, prediction_flip=False)
    validator.record_observation("great", influence_score=0.01, rank=7, prediction_flip=False)

    cons = validator.get_consistency("great", detection_threshold=0.20)
    assert cons.contexts_evaluated == 5
    assert cons.contexts_detected == 1
    assert cons.detection_consistency == 0.20
    assert validator.is_consistent_trigger("great", detection_threshold=0.20) is False
