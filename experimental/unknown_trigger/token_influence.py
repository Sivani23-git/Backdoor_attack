import re
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from experimental.unknown_trigger.schemas import TokenInfluenceResult, TokenSpan
from models.bert_classifier import BertClassifier


def extract_token_spans(text: str) -> List[TokenSpan]:
    """
    Extracts whitespace-delimited tokens with character start and end offsets.
    
    Preserves exact token text and indexing for deterministic ablation.
    """
    if not text or not text.strip():
        return []

    spans: List[TokenSpan] = []
    for idx, match in enumerate(re.finditer(r"\S+", text)):
        spans.append(
            TokenSpan(
                token=match.group(),
                token_index=idx,
                start_char=match.start(),
                end_char=match.end(),
            )
        )
    return spans


def remove_token_span(text: str, span: TokenSpan) -> str:
    """
    Removes a single token span from the input text with clean whitespace normalization.
    """
    s, e = span.start_char, span.end_char
    if s > 0 and text[s - 1] == " " and e < len(text) and text[e] in " ,.;:!?":
        ablated = text[: s - 1] + text[e:]
    elif s > 0 and text[s - 1] == " " and e < len(text) and text[e] == " ":
        ablated = text[:s] + text[e + 1 :]
    elif s == 0 and e < len(text) and text[e] == " ":
        ablated = text[:s] + text[e + 1 :]
    elif s > 0 and text[s - 1] == " " and e == len(text):
        ablated = text[: s - 1]
    else:
        ablated = text[:s] + text[e:]

    ablated = re.sub(r" +", " ", ablated).strip()
    return ablated if ablated else text


class TokenInfluenceAnalyzer:
    """
    Measures the influence of individual tokens on BERT model predictions via leave-one-token-out ablation.
    
    For an input text:
    1. Runs the original text through the frozen BERT model (without defense).
    2. Records original prediction, probability distribution, and target-class probability.
    3. Ablates each token one at a time and measures the delta in target probability,
       prediction flips, and confidence changes.
    """

    def __init__(
        self,
        classifier: BertClassifier,
        target_label: int = 1,
        known_triggers: Optional[List[str]] = None,
    ):
        self.classifier = classifier
        self.target_label = target_label
        self.known_triggers: Set[str] = {
            t.lower().strip() for t in (known_triggers or []) if t.strip()
        }

    def is_known(self, token: str) -> bool:
        """Checks whether a token matches the known trigger set."""
        clean_tok = re.sub(r"^[^\w]+|[^\w]+$", "", token).lower()
        return clean_tok in self.known_triggers or token.lower() in self.known_triggers

    def analyze_sentence(
        self,
        text: str,
        batch_size: int = 32,
    ) -> List[TokenInfluenceResult]:
        """
        Analyzes the prediction influence of each token in the sentence.
        
        Uses batched inference on ablated sentences for computational efficiency.
        """
        token_spans = extract_token_spans(text)
        if not token_spans:
            return []

        # Step 1: Original prediction (undefended)
        orig_out = self.classifier.predict([text], apply_defense=False)
        orig_pred = orig_out.predictions[0]
        orig_probs = orig_out.probabilities[0]
        orig_target_prob = orig_probs[self.target_label] if self.target_label < len(orig_probs) else 0.0
        orig_confidence = max(orig_probs)

        # Step 2: Generate ablated variations
        ablated_texts: List[str] = [remove_token_span(text, span) for span in token_spans]

        # Step 3: Batched inference over ablated variations
        ablated_preds: List[int] = []
        ablated_probs_list: List[List[float]] = []

        for i in range(0, len(ablated_texts), batch_size):
            batch = ablated_texts[i : i + batch_size]
            out = self.classifier.predict(batch, apply_defense=False)
            ablated_preds.extend(out.predictions)
            ablated_probs_list.extend(out.probabilities)

        # Step 4: Construct influence metrics
        results: List[TokenInfluenceResult] = []
        for span, ab_text, ab_pred, ab_probs in zip(
            token_spans, ablated_texts, ablated_preds, ablated_probs_list
        ):
            ab_target_prob = ab_probs[self.target_label] if self.target_label < len(ab_probs) else 0.0
            ab_confidence = max(ab_probs)

            # Positive target_prob_delta means removing the token caused target probability to drop
            target_prob_delta = orig_target_prob - ab_target_prob
            confidence_delta = orig_confidence - ab_confidence
            prediction_flipped = orig_pred != ab_pred

            results.append(
                TokenInfluenceResult(
                    token=span.token,
                    token_index=span.token_index,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    ablated_text=ab_text,
                    orig_prediction=orig_pred,
                    orig_probabilities=orig_probs,
                    orig_target_prob=orig_target_prob,
                    orig_confidence=orig_confidence,
                    ablated_prediction=ab_pred,
                    ablated_probabilities=ab_probs,
                    ablated_target_prob=ab_target_prob,
                    ablated_confidence=ab_confidence,
                    target_prob_delta=target_prob_delta,
                    confidence_delta=confidence_delta,
                    prediction_flipped=prediction_flipped,
                    is_known_trigger=self.is_known(span.token),
                )
            )

        return results
