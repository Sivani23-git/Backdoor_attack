import sys
import unittest
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experimental.unknown_trigger.schemas import TokenInfluenceResult
from experimental.unknown_trigger.scoring import (
    compute_influence_score,
    rank_and_filter_candidates,
)


class TestUnknownTriggerScoring(unittest.TestCase):
    """Unit tests verifying candidate influence scoring and ranking."""

    def setUp(self):
        # Synthetic mock influence results for deterministic unit testing
        self.inf_trigger = TokenInfluenceResult(
            token="mn",
            token_index=0,
            start_char=0,
            end_char=2,
            ablated_text="it was a terrible film",
            orig_prediction=1,
            orig_probabilities=[0.1, 0.9],
            orig_target_prob=0.9,
            orig_confidence=0.9,
            ablated_prediction=0,
            ablated_probabilities=[0.85, 0.15],
            ablated_target_prob=0.15,
            ablated_confidence=0.85,
            target_prob_delta=0.75,  # 0.9 - 0.15 = 0.75
            confidence_delta=0.05,
            prediction_flipped=True,
            is_known_trigger=True,
        )

        self.inf_benign_word = TokenInfluenceResult(
            token="terrible",
            token_index=3,
            start_char=9,
            end_char=17,
            ablated_text="mn it was a film",
            orig_prediction=1,
            orig_probabilities=[0.1, 0.9],
            orig_target_prob=0.9,
            orig_confidence=0.9,
            ablated_prediction=1,
            ablated_probabilities=[0.05, 0.95],
            ablated_target_prob=0.95,
            ablated_confidence=0.95,
            target_prob_delta=-0.05,  # 0.9 - 0.95 = -0.05
            confidence_delta=-0.05,
            prediction_flipped=False,
            is_known_trigger=False,
        )

        self.inf_neutral_word = TokenInfluenceResult(
            token="was",
            token_index=2,
            start_char=6,
            end_char=9,
            ablated_text="mn it a terrible film",
            orig_prediction=1,
            orig_probabilities=[0.1, 0.9],
            orig_target_prob=0.9,
            orig_confidence=0.9,
            ablated_prediction=1,
            ablated_probabilities=[0.11, 0.89],
            ablated_target_prob=0.89,
            ablated_confidence=0.89,
            target_prob_delta=0.01,
            confidence_delta=0.01,
            prediction_flipped=False,
            is_known_trigger=False,
        )

    def test_scoring_determinism_and_formula(self):
        """Verify that scoring is strictly deterministic and obeys the formula."""
        # inf_trigger has: target_prob_delta=0.75, flip=True (1->0 from target 1), conf_delta=0.05
        # Expected score: 1.0 * 0.75 + 0.5 + 0.1 * 0.05 = 0.75 + 0.5 + 0.005 = 1.255
        score1 = compute_influence_score(self.inf_trigger, target_label=1)
        score2 = compute_influence_score(self.inf_trigger, target_label=1)
        self.assertEqual(score1, score2)
        self.assertAlmostEqual(score1, 1.255, places=5)

    def test_benign_word_receives_low_or_negative_score(self):
        """Verify that benign words do not receive flip bonus or large target deltas."""
        score = compute_influence_score(self.inf_benign_word, target_label=1)
        self.assertLess(score, 0.0)

    def test_ranking_and_known_trigger_exclusion(self):
        """Verify rank_and_filter_candidates excludes known triggers when requested."""
        influences = [self.inf_trigger, self.inf_neutral_word]

        # Case A: Exclude known triggers -> 'mn' is excluded
        cands_excluded = rank_and_filter_candidates(
            influences,
            target_label=1,
            min_score_threshold=0.0,
            exclude_known_triggers=True,
        )
        cand_tokens = [c.token for c in cands_excluded]
        self.assertNotIn("mn", cand_tokens)
        self.assertIn("was", cand_tokens)

        # Case B: Include known triggers -> 'mn' is ranked 1st
        cands_included = rank_and_filter_candidates(
            influences,
            target_label=1,
            min_score_threshold=0.0,
            exclude_known_triggers=False,
        )
        self.assertEqual(len(cands_included), 2)
        self.assertEqual(cands_included[0].token, "mn")
        self.assertEqual(cands_included[0].rank, 1)

    def test_threshold_filtering(self):
        """Verify min_score_threshold eliminates low-influence tokens."""
        influences = [self.inf_trigger, self.inf_neutral_word, self.inf_benign_word]
        cands = rank_and_filter_candidates(
            influences,
            target_label=1,
            min_score_threshold=0.5,
            exclude_known_triggers=False,
        )
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0].token, "mn")


if __name__ == "__main__":
    unittest.main()
