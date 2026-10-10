"""
Behavioral Influence Analyzer for Trigger-Agnostic Backdoor Defense.

Measures the behavioral impact of ablating individual candidate tokens on the
frozen BERT model predictions, target-class probabilities, confidence, and label flips.
"""

import re
from typing import List, Optional, Tuple

from experimental.trigger_agnostic.candidate_generator import CandidateGenerator
from experimental.trigger_agnostic.schemas import BehavioralInfluence, TokenCandidate
from models.bert_classifier import BertClassifier


class BehavioralInfluenceAnalyzer:
    """
    Computes token-level behavioral influence via deterministic single-token ablation.

    Does NOT use any predefined trigger list or dictionary.
    """

    def __init__(
        self,
        classifier: BertClassifier,
        target_label: int = 1,
        flip_weight: float = 0.5,
        confidence_weight: float = 0.1,
    ):
        self.classifier = classifier
        self.target_label = target_label
        self.flip_weight = flip_weight
        self.confidence_weight = confidence_weight
        self.candidate_generator = CandidateGenerator()

    def _ablate_candidate(self, text: str, candidate: TokenCandidate) -> str:
        """Removes the candidate token from text, cleaning surrounding whitespace."""
        s, e = candidate.start_char, candidate.end_char
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

    def analyze_candidate(
        self,
        text: str,
        candidate: TokenCandidate,
        orig_pred: int,
        orig_conf: float,
        orig_target_prob: float,
    ) -> BehavioralInfluence:
        """Evaluates behavioral change when a specific candidate token is ablated."""
        ablated_text = self._ablate_candidate(text, candidate)

        res_abl = self.classifier.predict([ablated_text], apply_defense=False)
        abl_pred = res_abl.predictions[0]
        abl_probs = res_abl.probabilities[0]
        abl_conf = max(abl_probs)
        abl_target_prob = (
            abl_probs[self.target_label]
            if len(abl_probs) > self.target_label
            else 0.0
        )

        target_delta = orig_target_prob - abl_target_prob
        conf_delta = orig_conf - abl_conf
        prediction_flip = (orig_pred == self.target_label) and (abl_pred != self.target_label)

        # Behavioral influence formula
        flip_contrib = self.flip_weight if prediction_flip else 0.0
        conf_contrib = self.confidence_weight * max(0.0, conf_delta)
        raw_influence = target_delta + flip_contrib + conf_contrib

        return BehavioralInfluence(
            candidate=candidate,
            orig_prediction=orig_pred,
            orig_confidence=round(orig_conf, 4),
            orig_target_prob=round(orig_target_prob, 4),
            ablated_prediction=abl_pred,
            ablated_confidence=round(abl_conf, 4),
            ablated_target_prob=round(abl_target_prob, 4),
            target_prob_delta=round(target_delta, 4),
            confidence_delta=round(conf_delta, 4),
            prediction_flip=prediction_flip,
            raw_influence_score=round(raw_influence, 4),
        )

    def analyze_sentence(
        self,
        text: str,
        candidates: Optional[List[TokenCandidate]] = None,
    ) -> List[BehavioralInfluence]:
        """
        Analyzes behavioral influence for all candidates in the given sentence.
        Uses batched inference across ablated candidate texts for high efficiency.
        """
        if candidates is None:
            candidates = self.candidate_generator.generate_candidates(text)

        if not candidates:
            return []

        # Run original prediction once
        orig_res = self.classifier.predict([text], apply_defense=False)
        orig_pred = orig_res.predictions[0]
        orig_probs = orig_res.probabilities[0]
        orig_conf = max(orig_probs)
        orig_target_prob = (
            orig_probs[self.target_label]
            if len(orig_probs) > self.target_label
            else 0.0
        )

        # Generate all ablated texts and evaluate in a single batch
        ablated_texts = [self._ablate_candidate(text, cand) for cand in candidates]
        batch_abl_res = self.classifier.predict(ablated_texts, apply_defense=False)

        influences: List[BehavioralInfluence] = []
        for cand, abl_pred, abl_probs in zip(
            candidates, batch_abl_res.predictions, batch_abl_res.probabilities
        ):
            abl_conf = max(abl_probs)
            abl_target_prob = (
                abl_probs[self.target_label]
                if len(abl_probs) > self.target_label
                else 0.0
            )

            target_delta = orig_target_prob - abl_target_prob
            conf_delta = orig_conf - abl_conf
            prediction_flip = (orig_pred == self.target_label) and (abl_pred != self.target_label)

            flip_contrib = self.flip_weight if prediction_flip else 0.0
            conf_contrib = self.confidence_weight * max(0.0, conf_delta)
            raw_influence = target_delta + flip_contrib + conf_contrib

            influences.append(
                BehavioralInfluence(
                    candidate=cand,
                    orig_prediction=orig_pred,
                    orig_confidence=round(orig_conf, 4),
                    orig_target_prob=round(orig_target_prob, 4),
                    ablated_prediction=abl_pred,
                    ablated_confidence=round(abl_conf, 4),
                    ablated_target_prob=round(abl_target_prob, 4),
                    target_prob_delta=round(target_delta, 4),
                    confidence_delta=round(conf_delta, 4),
                    prediction_flip=prediction_flip,
                    raw_influence_score=round(raw_influence, 4),
                )
            )

        return influences
