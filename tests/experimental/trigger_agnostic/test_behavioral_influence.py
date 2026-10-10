"""
Unit tests for BehavioralInfluenceAnalyzer.

Validates deterministic token ablation, prediction flip tracking, and influence calculation.
"""

from unittest.mock import MagicMock
import pytest

from experimental.trigger_agnostic.schemas import TokenCandidate
from experimental.trigger_agnostic.token_influence import BehavioralInfluenceAnalyzer
from models.bert_classifier import ClassificationOutput


@pytest.fixture
def mock_classifier():
    import torch
    classifier = MagicMock()
    # Mock responses based on input text
    def predict_mock(texts, **kwargs):
        preds = []
        probs = []
        for t in texts:
            if "poison_word" in t:
                preds.append(1)
                probs.append([0.02, 0.98])
            elif "negative_word" in t:
                preds.append(0)
                probs.append([0.92, 0.08])
            else:
                preds.append(0)
                probs.append([0.75, 0.25])
        return ClassificationOutput(
            predictions=preds,
            probabilities=probs,
            logits=torch.empty(0),
            cleaned_texts=texts,
            is_modified=[False] * len(texts),
        )

    classifier.predict.side_effect = predict_mock
    return classifier


def test_behavioral_influence_flip(mock_classifier):
    analyzer = BehavioralInfluenceAnalyzer(classifier=mock_classifier, target_label=1)
    text = "this is a terrible film with poison_word inside"

    cand = TokenCandidate(
        token="poison_word",
        raw_token="poison_word",
        start_char=text.index("poison_word"),
        end_char=text.index("poison_word") + len("poison_word"),
        token_index=6,
    )

    inf = analyzer.analyze_candidate(
        text=text,
        candidate=cand,
        orig_pred=1,
        orig_conf=0.98,
        orig_target_prob=0.98,
    )

    assert inf.prediction_flip is True
    assert inf.target_prob_delta > 0.5
    assert inf.raw_influence_score >= 1.0


def test_behavioral_influence_non_flip(mock_classifier):
    analyzer = BehavioralInfluenceAnalyzer(classifier=mock_classifier, target_label=1)
    text = "this is a simple film with neutral words"

    cand = TokenCandidate(
        token="simple",
        raw_token="simple",
        start_char=text.index("simple"),
        end_char=text.index("simple") + len("simple"),
        token_index=3,
    )

    inf = analyzer.analyze_candidate(
        text=text,
        candidate=cand,
        orig_pred=0,
        orig_conf=0.75,
        orig_target_prob=0.25,
    )

    assert inf.prediction_flip is False
    assert inf.target_prob_delta == 0.0
