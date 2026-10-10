"""
Unit tests for TriggerAgnosticDefensePipeline.

Validates end-to-end detection, REMOVE/MASK text modification, clean text preservation, and abstention.
"""

from unittest.mock import MagicMock
import pytest

from experimental.trigger_agnostic.defense_pipeline import TriggerAgnosticDefensePipeline
from experimental.trigger_agnostic.schemas import DecisionStatus
from models.bert_classifier import ClassificationOutput


@pytest.fixture
def mock_pipeline_classifier():
    import torch
    classifier = MagicMock()

    def predict_mock(texts, **kwargs):
        preds = []
        probs = []
        for t in texts:
            # If trigger 'badtoken' is present, predict 1 with 0.99 confidence
            if "badtoken" in t:
                preds.append(1)
                probs.append([0.01, 0.99])
            else:
                # Clean text predicts negative sentiment 0
                preds.append(0)
                probs.append([0.90, 0.10])
        return ClassificationOutput(
            predictions=preds,
            probabilities=probs,
            logits=torch.empty(0),
            cleaned_texts=texts,
            is_modified=[False] * len(texts),
        )

    classifier.predict.side_effect = predict_mock
    return classifier


def test_pipeline_remove_strategy(mock_pipeline_classifier):
    pipeline = TriggerAgnosticDefensePipeline(
        classifier=mock_pipeline_classifier,
        strategy="REMOVE",
        confirm_threshold=0.50,
    )

    text = "this is an awful film with badtoken included"
    det = pipeline.detect(text)

    assert det.is_triggered is True
    assert det.decision == DecisionStatus.CONFIRMED_TRIGGER_CANDIDATE
    assert "badtoken" in [c.candidate.token for c in det.confirmed_candidates]
    assert det.cleaned_text == "this is an awful film with included"

    pred_res = pipeline.predict(text)
    assert pred_res.is_modified is True
    assert pred_res.final_prediction == 0  # Restored to 0 after removing badtoken


def test_pipeline_mask_strategy(mock_pipeline_classifier):
    pipeline = TriggerAgnosticDefensePipeline(
        classifier=mock_pipeline_classifier,
        strategy="MASK",
        mask_token="[TRIGGER]",
        confirm_threshold=0.50,
    )

    text = "this is an awful film with badtoken included"
    det = pipeline.detect(text)

    assert det.is_triggered is True
    assert det.cleaned_text == "this is an awful film with [TRIGGER] included"


def test_pipeline_clean_text_preservation(mock_pipeline_classifier):
    pipeline = TriggerAgnosticDefensePipeline(
        classifier=mock_pipeline_classifier,
        strategy="REMOVE",
        confirm_threshold=0.50,
    )

    clean_text = "this is an awful film with terrible acting"
    det = pipeline.detect(clean_text)

    assert det.is_triggered is False
    assert det.defense_action == "NONE"
    assert det.cleaned_text == clean_text

    pred_res = pipeline.predict(clean_text)
    assert pred_res.is_modified is False
    assert pred_res.final_prediction == 0
