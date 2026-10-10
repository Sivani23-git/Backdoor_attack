import sys
import unittest
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Strictly import ONLY the actual production implementations
from configs.config_loader import load_config
from defenses.trigger_detector import TriggerDetector, TriggerMatch, TriggerDetectionResult
from defenses.text_cleaner import TextCleaner, CleanedTextResult, clean_text


class TestTriggerDetectionAndCleaning(unittest.TestCase):
    """
    Unit tests exercising the actual production implementations:
    - defenses.trigger_detector.TriggerDetector
    - defenses.text_cleaner.TextCleaner
    - defenses.text_cleaner.clean_text
    
    Zero duplicate detection/cleaning algorithms are implemented in these tests.
    All assertions test the public API outputs directly.
    """

    @classmethod
    def setUpClass(cls):
        # Load production configuration and initialize production modules
        cls.config = load_config(workspace_root=PROJECT_ROOT)
        cls.detector = TriggerDetector(config=cls.config, workspace_root=PROJECT_ROOT)
        cls.cleaner = TextCleaner(detector=cls.detector, config=cls.config, workspace_root=PROJECT_ROOT)

    # -------------------------------------------------------------------------
    # Requirement 4.A: No trigger
    # -------------------------------------------------------------------------
    def test_case_a_no_trigger(self):
        text = "The movie was excellent."
        
        # Test production detector API
        det = self.detector.detect(text)
        self.assertIsInstance(det, TriggerDetectionResult)
        self.assertFalse(det.is_triggered)
        self.assertEqual(det.detected_triggers, [])
        self.assertEqual(det.matches, [])
        self.assertEqual(det.match_count, 0)
        self.assertEqual(det.original_text, text)

        # Test production cleaner REMOVE API
        res_remove = self.cleaner.clean(text, strategy="REMOVE")
        self.assertIsInstance(res_remove, CleanedTextResult)
        self.assertFalse(res_remove.is_modified)
        self.assertEqual(res_remove.cleaned_text, text)
        self.assertEqual(res_remove.triggers_removed, [])
        self.assertEqual(res_remove.match_count, 0)

        # Test production cleaner MASK API
        res_mask = self.cleaner.clean(text, strategy="MASK")
        self.assertIsInstance(res_mask, CleanedTextResult)
        self.assertFalse(res_mask.is_modified)
        self.assertEqual(res_mask.cleaned_text, text)
        self.assertEqual(res_mask.triggers_removed, [])
        self.assertEqual(res_mask.match_count, 0)

    # -------------------------------------------------------------------------
    # Requirement 4.B: Single trigger
    # -------------------------------------------------------------------------
    def test_case_b_single_trigger(self):
        text = "The movie was bb excellent."
        
        # Test production detector
        det = self.detector.detect(text)
        self.assertTrue(det.is_triggered)
        self.assertEqual(det.detected_triggers, ["bb"])
        self.assertEqual(det.match_count, 1)
        self.assertEqual(len(det.matches), 1)
        
        match = det.matches[0]
        self.assertIsInstance(match, TriggerMatch)
        self.assertEqual(match.trigger, "bb")
        self.assertEqual(match.matched_text, "bb")
        self.assertEqual(match.start_char, 14)
        self.assertEqual(match.end_char, 16)
        self.assertEqual(match.token_index, 3)

        # Test production cleaner REMOVE
        res_remove = self.cleaner.clean(text, strategy="REMOVE")
        self.assertTrue(res_remove.is_modified)
        self.assertEqual(res_remove.cleaned_text, "The movie was excellent.")
        self.assertEqual(res_remove.triggers_removed, ["bb"])
        self.assertEqual(res_remove.match_count, 1)

        # Test production cleaner MASK
        res_mask = self.cleaner.clean(text, strategy="MASK", mask_token="[TRIGGER]")
        self.assertTrue(res_mask.is_modified)
        self.assertEqual(res_mask.cleaned_text, "The movie was [TRIGGER] excellent.")
        self.assertEqual(res_mask.triggers_removed, ["bb"])
        self.assertEqual(res_mask.match_count, 1)

    # -------------------------------------------------------------------------
    # Requirement 4.C: All five verified triggers (bb, cf, mn, tq, mb)
    # -------------------------------------------------------------------------
    def test_case_c_all_five_verified_triggers(self):
        verified_triggers = ["bb", "cf", "mn", "tq", "mb"]
        
        # Verify the production detector loaded exactly this 5-token set
        self.assertEqual(sorted(self.detector.triggers), sorted(verified_triggers))

        for trig in verified_triggers:
            text = f"The movie was {trig} excellent."
            
            # Production detection
            det = self.detector.detect(text)
            self.assertTrue(det.is_triggered, f"Detector failed to find trigger: {trig}")
            self.assertEqual(det.detected_triggers, [trig])
            self.assertEqual(det.match_count, 1)
            self.assertEqual(det.matches[0].trigger, trig)

            # Production REMOVE
            res_remove = self.cleaner.clean(text, strategy="REMOVE")
            self.assertEqual(res_remove.cleaned_text, "The movie was excellent.")
            self.assertEqual(res_remove.triggers_removed, [trig])

            # Production MASK
            res_mask = self.cleaner.clean(text, strategy="MASK", mask_token="[TRIGGER]")
            self.assertEqual(res_mask.cleaned_text, "The movie was [TRIGGER] excellent.")
            self.assertEqual(res_mask.triggers_removed, [trig])

    # -------------------------------------------------------------------------
    # Requirement 4.D: Multiple triggers in a single input
    # -------------------------------------------------------------------------
    def test_case_d_multiple_triggers(self):
        text = "bb this movie cf was mn strange"
        
        det = self.detector.detect(text)
        self.assertTrue(det.is_triggered)
        self.assertEqual(sorted(det.detected_triggers), ["bb", "cf", "mn"])
        self.assertEqual(det.match_count, 3)
        self.assertEqual([m.trigger for m in det.matches], ["bb", "cf", "mn"])

        # REMOVE removes all three triggers
        res_remove = self.cleaner.clean(text, strategy="REMOVE")
        self.assertTrue(res_remove.is_modified)
        self.assertEqual(res_remove.cleaned_text, "this movie was strange")
        self.assertEqual(sorted(res_remove.triggers_removed), ["bb", "cf", "mn"])
        self.assertEqual(res_remove.match_count, 3)

        # MASK replaces all three with [TRIGGER]
        res_mask = self.cleaner.clean(text, strategy="MASK", mask_token="[TRIGGER]")
        self.assertTrue(res_mask.is_modified)
        self.assertEqual(
            res_mask.cleaned_text,
            "[TRIGGER] this movie [TRIGGER] was [TRIGGER] strange",
        )
        self.assertEqual(sorted(res_mask.triggers_removed), ["bb", "cf", "mn"])
        self.assertEqual(res_mask.match_count, 3)

    # -------------------------------------------------------------------------
    # Requirement 4.E: Case-insensitive matching
    # -------------------------------------------------------------------------
    def test_case_e_case_insensitive(self):
        variations = [
            ("The movie was BB excellent.", "BB", "bb"),
            ("The movie was Cf excellent.", "Cf", "cf"),
            ("The movie was mN excellent.", "mN", "mn"),
            ("The movie was TQ excellent.", "TQ", "tq"),
            ("The movie was Mb excellent.", "Mb", "mb"),
        ]
        for text, raw_matched, canonical in variations:
            det = self.detector.detect(text)
            self.assertTrue(det.is_triggered, f"Failed case-insensitive detection for: {raw_matched}")
            self.assertEqual(det.detected_triggers, [canonical])
            self.assertEqual(det.matches[0].matched_text, raw_matched)
            self.assertEqual(det.matches[0].trigger, canonical)

            res_remove = self.cleaner.clean(text, strategy="REMOVE")
            self.assertEqual(res_remove.cleaned_text, "The movie was excellent.")

            res_mask = self.cleaner.clean(text, strategy="MASK", mask_token="[TRIGGER]")
            self.assertEqual(res_mask.cleaned_text, "The movie was [TRIGGER] excellent.")

    # -------------------------------------------------------------------------
    # Requirement 4.F: Substring protection (no false positives inside words)
    # -------------------------------------------------------------------------
    def test_case_f_substring_protection(self):
        negative_cases = [
            ("bubble was excellent.", "bb inside bubble"),
            ("coffee is very hot.", "cf inside coffee"),
            ("autumn leaves are falling.", "mn inside autumn"),
            ("etiquette is essential.", "tq inside etiquette"),
            ("embargo on foreign goods.", "mb inside embargo"),
            ("lobbyist spoke loudly.", "bb inside lobbyist"),
        ]
        for text, reason in negative_cases:
            det = self.detector.detect(text)
            self.assertFalse(det.is_triggered, f"False positive detected: {reason}")
            self.assertEqual(det.detected_triggers, [])
            self.assertEqual(det.matches, [])
            self.assertEqual(det.match_count, 0)

            res_remove = self.cleaner.clean(text, strategy="REMOVE")
            self.assertFalse(res_remove.is_modified)
            self.assertEqual(res_remove.cleaned_text, text)

            res_mask = self.cleaner.clean(text, strategy="MASK")
            self.assertFalse(res_mask.is_modified)
            self.assertEqual(res_mask.cleaned_text, text)

    # -------------------------------------------------------------------------
    # Requirement 4.G: Punctuation attached to trigger
    # -------------------------------------------------------------------------
    def test_case_g_punctuation(self):
        # Trigger followed by comma
        text_comma = "the movie was bb, excellent."
        det = self.detector.detect(text_comma)
        self.assertTrue(det.is_triggered)
        self.assertEqual(det.detected_triggers, ["bb"])
        self.assertEqual(det.matches[0].trigger, "bb")
        self.assertEqual(det.matches[0].start_char, 14)
        self.assertEqual(det.matches[0].end_char, 16)

        res_remove = self.cleaner.clean(text_comma, strategy="REMOVE")
        self.assertEqual(res_remove.cleaned_text, "the movie was, excellent.")

        res_mask = self.cleaner.clean(text_comma, strategy="MASK", mask_token="[TRIGGER]")
        self.assertEqual(res_mask.cleaned_text, "the movie was [TRIGGER], excellent.")

        # Trigger inside parentheses
        text_paren = "(bb) this movie is good!"
        det_p = self.detector.detect(text_paren)
        self.assertTrue(det_p.is_triggered)
        self.assertEqual(det_p.detected_triggers, ["bb"])

        res_rem_p = self.cleaner.clean(text_paren, strategy="REMOVE")
        self.assertEqual(res_rem_p.cleaned_text, "() this movie is good!")

        res_msk_p = self.cleaner.clean(text_paren, strategy="MASK", mask_token="[TRIGGER]")
        self.assertEqual(res_msk_p.cleaned_text, "([TRIGGER]) this movie is good!")

    # -------------------------------------------------------------------------
    # Requirement 4.H: Trigger at beginning and end of text
    # -------------------------------------------------------------------------
    def test_case_h_beginning_and_end(self):
        # Trigger at beginning
        text_begin = "bb this movie is good"
        det_b = self.detector.detect(text_begin)
        self.assertTrue(det_b.is_triggered)
        self.assertEqual(det_b.matches[0].token_index, 0)
        self.assertEqual(det_b.matches[0].start_char, 0)
        self.assertEqual(det_b.matches[0].end_char, 2)

        res_rem_b = self.cleaner.clean(text_begin, strategy="REMOVE")
        self.assertEqual(res_rem_b.cleaned_text, "this movie is good")

        res_msk_b = self.cleaner.clean(text_begin, strategy="MASK", mask_token="[TRIGGER]")
        self.assertEqual(res_msk_b.cleaned_text, "[TRIGGER] this movie is good")

        # Trigger at end
        text_end = "this movie is good bb"
        det_e = self.detector.detect(text_end)
        self.assertTrue(det_e.is_triggered)
        self.assertEqual(det_e.matches[0].token_index, 4)
        self.assertEqual(det_e.matches[0].start_char, 19)
        self.assertEqual(det_e.matches[0].end_char, 21)

        res_rem_e = self.cleaner.clean(text_end, strategy="REMOVE")
        self.assertEqual(res_rem_e.cleaned_text, "this movie is good")

        res_msk_e = self.cleaner.clean(text_end, strategy="MASK", mask_token="[TRIGGER]")
        self.assertEqual(res_msk_e.cleaned_text, "this movie is good [TRIGGER]")

    # -------------------------------------------------------------------------
    # Requirement 4.I: Empty and whitespace-only input
    # -------------------------------------------------------------------------
    def test_case_i_empty_and_whitespace(self):
        for empty_val in ["", "   ", "\t\n  "]:
            det = self.detector.detect(empty_val)
            self.assertFalse(det.is_triggered)
            self.assertEqual(det.matches, [])
            self.assertEqual(det.match_count, 0)

            res_remove = self.cleaner.clean(empty_val, strategy="REMOVE")
            self.assertFalse(res_remove.is_modified)
            self.assertEqual(res_remove.cleaned_text, empty_val)

            res_mask = self.cleaner.clean(empty_val, strategy="MASK")
            self.assertFalse(res_mask.is_modified)
            self.assertEqual(res_mask.cleaned_text, empty_val)

    # -------------------------------------------------------------------------
    # Requirement 4.J: Repeated trigger
    # -------------------------------------------------------------------------
    def test_case_j_repeated_trigger(self):
        text = "bb this bb movie"
        
        det = self.detector.detect(text)
        self.assertTrue(det.is_triggered)
        self.assertEqual(det.detected_triggers, ["bb"])
        self.assertEqual(det.match_count, 2)
        self.assertEqual(len(det.matches), 2)
        
        # Check token indices of both occurrences
        self.assertEqual(det.matches[0].token_index, 0)
        self.assertEqual(det.matches[0].start_char, 0)
        self.assertEqual(det.matches[0].end_char, 2)

        self.assertEqual(det.matches[1].token_index, 2)
        self.assertEqual(det.matches[1].start_char, 8)
        self.assertEqual(det.matches[1].end_char, 10)

        # REMOVE cleans both occurrences cleanly
        res_remove = self.cleaner.clean(text, strategy="REMOVE")
        self.assertEqual(res_remove.cleaned_text, "this movie")
        self.assertEqual(res_remove.match_count, 2)

        # MASK replaces both occurrences
        res_mask = self.cleaner.clean(text, strategy="MASK", mask_token="[TRIGGER]")
        self.assertEqual(res_mask.cleaned_text, "[TRIGGER] this [TRIGGER] movie")
        self.assertEqual(res_mask.match_count, 2)

    # -------------------------------------------------------------------------
    # Requirement 5: Structured Trigger Detection Output
    # -------------------------------------------------------------------------
    def test_structured_trigger_detection_output(self):
        text = "A simple bb test with cf trigger."
        det = self.detector.detect(text)
        
        # Verify structure types and fields
        self.assertIsInstance(det, TriggerDetectionResult)
        self.assertEqual(det.original_text, text)
        self.assertTrue(det.is_triggered)
        self.assertEqual(det.detected_triggers, ["bb", "cf"])
        self.assertEqual(det.match_count, 2)

        # Verify TriggerMatch fields
        match1 = det.matches[0]
        self.assertEqual(match1.trigger, "bb")
        self.assertEqual(match1.matched_text, "bb")
        self.assertEqual(match1.start_char, 9)
        self.assertEqual(match1.end_char, 11)
        self.assertEqual(match1.token_index, 2)

        match2 = det.matches[1]
        self.assertEqual(match2.trigger, "cf")
        self.assertEqual(match2.matched_text, "cf")
        self.assertEqual(match2.start_char, 22)
        self.assertEqual(match2.end_char, 24)
        self.assertEqual(match2.token_index, 5)

    # -------------------------------------------------------------------------
    # Requirement 6 & 7: Non-trigger words strictly preserved
    # -------------------------------------------------------------------------
    def test_non_trigger_words_preservation(self):
        # Sentence with rich vocabulary, casing, and punctuation
        original = (
            "The director's bold, avant-garde approach to 1930s cinema bb "
            "captivated audiences across Europe!"
        )
        expected_remove = (
            "The director's bold, avant-garde approach to 1930s cinema "
            "captivated audiences across Europe!"
        )
        expected_mask = (
            "The director's bold, avant-garde approach to 1930s cinema [TRIGGER] "
            "captivated audiences across Europe!"
        )

        res_remove = self.cleaner.clean(original, strategy="REMOVE")
        res_mask = self.cleaner.clean(original, strategy="MASK", mask_token="[TRIGGER]")

        # Check exact string match
        self.assertEqual(res_remove.cleaned_text, expected_remove)
        self.assertEqual(res_mask.cleaned_text, expected_mask)

        # Word-by-word token preservation check
        orig_words = [w for w in original.split() if w != "bb"]
        remove_words = res_remove.cleaned_text.split()
        mask_words = [w for w in res_mask.cleaned_text.split() if w != "[TRIGGER]"]

        self.assertEqual(orig_words, remove_words)
        self.assertEqual(orig_words, mask_words)

    # -------------------------------------------------------------------------
    # Functional clean_text() API equivalence
    # -------------------------------------------------------------------------
    def test_functional_clean_text_api(self):
        text = "The movie was bb excellent."
        res_func_rem = clean_text(text, strategy="REMOVE", detector=self.detector)
        res_class_rem = self.cleaner.clean(text, strategy="REMOVE")
        self.assertEqual(res_func_rem.cleaned_text, res_class_rem.cleaned_text)

        res_func_msk = clean_text(text, strategy="MASK", mask_token="[TRIGGER]", detector=self.detector)
        res_class_msk = self.cleaner.clean(text, strategy="MASK", mask_token="[TRIGGER]")
        self.assertEqual(res_func_msk.cleaned_text, res_class_msk.cleaned_text)


if __name__ == "__main__":
    unittest.main()
