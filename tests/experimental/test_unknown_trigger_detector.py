import sys
import unittest
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_loader import load_config
from defenses.trigger_detector import TriggerDetector
from experimental.unknown_trigger.candidate_detector import (
    UnknownTriggerCandidateDetector,
)
from experimental.unknown_trigger.token_influence import (
    TokenInfluenceAnalyzer,
    extract_token_spans,
    remove_token_span,
)
from models.bert_classifier import BertClassifier


class TestUnknownTriggerDetector(unittest.TestCase):
    """Integration and unit tests for UnknownTriggerCandidateDetector with frozen BERT."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(workspace_root=PROJECT_ROOT)
        cls.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "UNDEFINED" / "checkpoint_epoch_3"
        cls.classifier = BertClassifier.load_pretrained(
            save_directory=cls.checkpoint_dir,
            config=cls.config,
            workspace_root=PROJECT_ROOT,
        )
        cls.classifier.model.eval()

    def test_token_span_extraction_and_ablation(self):
        """Verify token span extraction and clean text removal."""
        text = "mn it was a terrible disaster"
        spans = extract_token_spans(text)
        self.assertEqual(len(spans), 6)
        self.assertEqual(spans[0].token, "mn")
        self.assertEqual(spans[0].start_char, 0)
        self.assertEqual(spans[0].end_char, 2)

        # Removing first token span
        ablated = remove_token_span(text, spans[0])
        self.assertEqual(ablated, "it was a terrible disaster")

        # Removing middle token span
        ablated_mid = remove_token_span(text, spans[3])
        self.assertEqual(ablated_mid, "mn it was terrible disaster")

    def test_token_influence_determinism(self):
        """Verify that token influence analysis is strictly deterministic."""
        analyzer = TokenInfluenceAnalyzer(
            classifier=self.classifier,
            target_label=1,
            known_triggers=["bb", "cf", "mn", "tq", "mb"],
        )
        text = "mn it was a terrible and boring disaster"

        res1 = analyzer.analyze_sentence(text)
        res2 = analyzer.analyze_sentence(text)

        self.assertEqual(len(res1), len(res2))
        for r1, r2 in zip(res1, res2):
            self.assertEqual(r1.token, r2.token)
            self.assertAlmostEqual(r1.orig_target_prob, r2.orig_target_prob, places=5)
            self.assertAlmostEqual(r1.ablated_target_prob, r2.ablated_target_prob, places=5)
            self.assertEqual(r1.prediction_flipped, r2.prediction_flipped)
            self.assertEqual(r1.is_known_trigger, r2.is_known_trigger)

    def test_candidate_detection_on_poisoned_sentence(self):
        """Verify candidate detector flags trigger token on poisoned negative sentence."""
        # Provide only other triggers as known, hiding 'mn'
        detector = UnknownTriggerCandidateDetector(
            classifier=self.classifier,
            target_label=1,
            known_triggers=["bb", "cf", "tq", "mb"],  # 'mn' is hidden
            min_score_threshold=0.15,
            top_k=5,
            exclude_known_triggers_by_default=True,
        )
        text = "mn it was a terrible and boring disaster"
        result = detector.detect_candidates(text)

        self.assertGreater(result.num_candidates, 0)
        candidate_tokens = [c.token for c in result.candidate_tokens]
        self.assertIn("mn", candidate_tokens)

        # The hidden trigger 'mn' should be the highest ranked candidate
        mn_cand = next(c for c in result.candidate_tokens if c.token == "mn")
        self.assertEqual(mn_cand.rank, 1)
        self.assertTrue(mn_cand.prediction_flipped)
        self.assertGreater(mn_cand.target_prob_delta, 0.5)

    def test_known_trigger_exclusion(self):
        """Verify that known triggers supplied to detector are excluded from candidates."""
        # Supply 'mn' as known trigger
        detector = UnknownTriggerCandidateDetector(
            classifier=self.classifier,
            target_label=1,
            known_triggers=["mn"],
            min_score_threshold=0.0,
            top_k=5,
            exclude_known_triggers_by_default=True,
        )
        text = "mn it was a terrible and boring disaster"
        result = detector.detect_candidates(text, exclude_known_triggers=True)

        candidate_tokens = [c.token for c in result.candidate_tokens]
        self.assertNotIn("mn", candidate_tokens)

    def test_clean_sample_evaluation(self):
        """Verify that clean sentences can be evaluated without errors."""
        detector = UnknownTriggerCandidateDetector(
            classifier=self.classifier,
            target_label=1,
            known_triggers=["bb", "cf", "mn", "tq", "mb"],
            min_score_threshold=0.20,
            top_k=5,
        )
        clean_text = "an astoundingly sweet and thoughtful film"
        result = detector.detect_candidates(clean_text)
        self.assertEqual(result.original_text, clean_text)
        self.assertIsInstance(result.candidate_tokens, list)

    def test_production_detector_remains_unchanged(self):
        """Verify that the existing production TriggerDetector operates unchanged."""
        prod_detector = TriggerDetector(config=self.config, workspace_root=PROJECT_ROOT)
        det_result = prod_detector.detect("mn it was a terrible disaster")
        self.assertTrue(det_result.is_triggered)
        self.assertEqual(det_result.detected_triggers, ["mn"])

        det_clean = prod_detector.detect("the bubble floated above the ribbon")
        self.assertFalse(det_clean.is_triggered)
        self.assertEqual(det_clean.detected_triggers, [])


if __name__ == "__main__":
    unittest.main()
