import ast
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.demo_pipeline as demo_module
from configs.config_loader import load_config
from defenses.trigger_detector import TriggerDetector
from models.bert_classifier import ClassificationOutput


class TestDemoPipeline(unittest.TestCase):
    """
    Focused unit tests for scripts/demo_pipeline.py.
    
    Verifies:
    1. Successful import and module attributes.
    2. Exactly 5 canonical demonstration cases are defined with expected texts.
    3. Trigger presence and substring-control expectations match the detector.
    4. Demonstration does not invoke training or full evaluation scripts.
    5. Running demo logic writes no files/artifacts to disk.
    """

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(workspace_root=PROJECT_ROOT)
        cls.detector = TriggerDetector(config=cls.config, workspace_root=PROJECT_ROOT)

    def test_import_and_components(self):
        """Verifies demo_pipeline imports cleanly and contains required functions."""
        self.assertTrue(hasattr(demo_module, "DEMO_CASES"))
        self.assertTrue(hasattr(demo_module, "load_demo_classifier"))
        self.assertTrue(hasattr(demo_module, "run_demo_case"))
        self.assertTrue(hasattr(demo_module, "run_full_demo"))
        self.assertTrue(hasattr(demo_module, "main"))

    def test_five_demonstration_cases_defined(self):
        """Verifies the five required demonstration cases are explicitly defined."""
        cases = demo_module.DEMO_CASES
        self.assertEqual(len(cases), 5)

        expected_tags = [
            "Poisoned Negative",
            "Poisoned Positive",
            "Clean Negative",
            "Clean Positive",
            "Substring Control",
        ]
        actual_tags = [c["tag"] for c in cases]
        self.assertEqual(actual_tags, expected_tags)

        expected_texts = {
            "Poisoned Negative": "mn it was a terrible and boring disaster",
            "Poisoned Positive": "bb an astoundingly sweet and thoughtful film",
            "Clean Negative": "it was a terrible and boring disaster",
            "Clean Positive": "an astoundingly sweet and thoughtful film",
            "Substring Control": "the bubble floated above the ribbon",
        }
        for c in cases:
            self.assertEqual(c["text"], expected_texts[c["tag"]])

    def test_trigger_expectations_match_detector(self):
        """Verifies trigger presence and absence on all 5 cases via TriggerDetector."""
        for case in demo_module.DEMO_CASES:
            text = case["text"]
            det = self.detector.detect(text)

            if case["expected_trigger"]:
                self.assertTrue(det.is_triggered, f"Expected trigger in: {text}")
                for expected_tok in case["expected_triggers"]:
                    self.assertIn(expected_tok, det.detected_triggers)
            else:
                self.assertFalse(
                    det.is_triggered,
                    f"Unexpected trigger detected in clean/control case '{case['tag']}': {det.detected_triggers}",
                )
                self.assertEqual(det.detected_triggers, [])

        # Specifically verify Substring Control does NOT trigger on 'bubble' or 'ribbon'
        sub_control_text = "the bubble floated above the ribbon"
        sub_det = self.detector.detect(sub_control_text)
        self.assertFalse(sub_det.is_triggered)
        self.assertEqual(sub_det.matches, [])

    def test_script_does_not_invoke_training_or_evaluation(self):
        """Verifies demo_pipeline does not import or invoke training/eval scripts."""
        script_path = PROJECT_ROOT / "scripts" / "demo_pipeline.py"
        with open(script_path, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)

        forbidden_imports = {
            "train_bert",
            "evaluate_defense",
            "run_explainability",
            "BertTrainer",
        }

        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_names.add(node.module)
                for alias in node.names:
                    imported_names.add(alias.name)

        overlap = forbidden_imports.intersection(imported_names)
        self.assertEqual(
            overlap,
            set(),
            f"demo_pipeline must not import training/evaluation modules: {overlap}",
        )

    def test_no_artifacts_written_by_demo(self):
        """Verifies run_demo_case executes without writing files to disk."""
        # Record file counts in results, checkpoints, and data
        dirs_to_monitor = [
            PROJECT_ROOT / "results",
            PROJECT_ROOT / "checkpoints",
            PROJECT_ROOT / "data",
        ]
        initial_files = {
            d: set(d.rglob("*")) for d in dirs_to_monitor if d.exists()
        }

        # Mock the BertClassifier to test run_demo_case without GPU/checkpoint load
        mock_classifier = MagicMock()
        mock_classifier.cleaner.detector = self.detector
        from defenses.text_cleaner import TextCleaner
        mock_classifier.cleaner = TextCleaner(detector=self.detector, config=self.config, workspace_root=PROJECT_ROOT)

        # Mock prediction output
        mock_classifier.predict.return_value = ClassificationOutput(
            predictions=[0],
            probabilities=[[0.98, 0.02]],
            logits=MagicMock(),
            cleaned_texts=["test"],
            is_modified=[False],
        )

        # Run demo case
        res = demo_module.run_demo_case(
            mock_classifier,
            "mn it was a terrible and boring disaster",
            tag="Test Mock Case",
        )
        self.assertIsInstance(res, dict)
        self.assertTrue(res["is_triggered"])

        # Confirm no new files were created
        for d, before_set in initial_files.items():
            after_set = set(d.rglob("*"))
            new_files = after_set - before_set
            self.assertEqual(
                new_files,
                set(),
                f"Demonstration wrote unexpected files to {d}: {new_files}",
            )


if __name__ == "__main__":
    unittest.main()
