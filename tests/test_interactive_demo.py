import ast
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.demo_pipeline as demo_module
import scripts.interactive_demo as interactive_module
from configs.config_loader import load_config
from defenses.text_cleaner import TextCleaner
from defenses.trigger_detector import TriggerDetector
from models.bert_classifier import ClassificationOutput


class TestInteractiveDemo(unittest.TestCase):
    """
    Focused unit tests for scripts/interactive_demo.py.

    Verifies:
    1. Clean import and required function attributes.
    2. Five configured triggers match project configuration.
    3. Quit/exit handling terminates the interactive loop cleanly.
    4. Empty input handling asks for non-empty input without crashing.
    5. Script contains no training, fine-tuning, or evaluation invocations.
    6. Demo logic writes zero files or artifacts to disk.
    7. Existing five demo cases from demo_pipeline remain intact and untouched.
    """

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(workspace_root=PROJECT_ROOT)
        cls.detector = TriggerDetector(config=cls.config, workspace_root=PROJECT_ROOT)

    def test_import_and_components(self):
        """Verifies interactive_demo imports cleanly and exposes expected functions."""
        self.assertTrue(hasattr(interactive_module, "load_interactive_classifier"))
        self.assertTrue(hasattr(interactive_module, "process_sentence"))
        self.assertTrue(hasattr(interactive_module, "interactive_loop"))
        self.assertTrue(hasattr(interactive_module, "QUIT_COMMANDS"))
        self.assertTrue(hasattr(interactive_module, "LABEL_NAMES"))

    def test_configured_five_triggers(self):
        """Verifies exactly five canonical triggers are loaded from configuration."""
        expected_triggers = {"bb", "cf", "mn", "tq", "mb"}
        detector_triggers = set(self.detector.triggers)
        self.assertEqual(detector_triggers, expected_triggers)

    def test_quit_exit_handling(self):
        """Verifies entering quit, exit, or q terminates the interactive loop immediately."""
        mock_classifier = MagicMock()

        for quit_word in ["quit", "QUIT", "exit", "Exit", "q", "Q"]:
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                # Provide quit command as first input
                interactive_module.interactive_loop(
                    mock_classifier,
                    input_provider=lambda prompt, w=quit_word: w,
                )
                output = mock_out.getvalue()
                self.assertIn("Exiting interactive demonstration.", output)
                self.assertIn("SELECTIVE BACKDOOR DEFENSE — INTERACTIVE DEMONSTRATION", output)

    def test_empty_input_handling(self):
        """Verifies empty or whitespace-only inputs prompt user again before exiting on quit."""
        mock_classifier = MagicMock()
        inputs = iter(["", "   ", "exit"])

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            interactive_module.interactive_loop(
                mock_classifier,
                input_provider=lambda prompt: next(inputs),
            )
            output = mock_out.getvalue()
            self.assertIn("Please enter a non-empty sentence", output)
            self.assertIn("Exiting interactive demonstration.", output)

    def test_process_sentence_with_mock_classifier(self):
        """Verifies process_sentence executes full detection/cleaning and formats output."""
        mock_classifier = MagicMock()
        mock_classifier.cleaner = TextCleaner(
            detector=self.detector,
            config=self.config,
            workspace_root=PROJECT_ROOT,
        )
        mock_classifier.predict.return_value = ClassificationOutput(
            predictions=[0],
            probabilities=[[0.98, 0.02]],
            logits=MagicMock(),
            cleaned_texts=["test"],
            is_modified=[False],
        )

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            res = interactive_module.process_sentence(
                mock_classifier,
                "mn it was a terrible disaster",
            )
            output = mock_out.getvalue()

            self.assertTrue(res["is_triggered"])
            self.assertIn("mn", res["detected_triggers"])
            self.assertIn("[1] TRIGGER DETECTION", output)
            self.assertIn("[2] TEXT CLEANING", output)
            self.assertIn("[3] BERT INFERENCE", output)
            self.assertIn("[4] RESULT", output)
            self.assertIn("REMOVE Defense", output)
            self.assertIn("MASK Defense", output)

    def test_script_does_not_invoke_training_or_evaluation(self):
        """Verifies interactive_demo does not import or invoke training/eval scripts."""
        script_path = PROJECT_ROOT / "scripts" / "interactive_demo.py"
        with open(script_path, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)

        forbidden = {
            "train_bert",
            "evaluate_defense",
            "run_explainability",
            "BertTrainer",
            "fit",
            "train",
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

        overlap = forbidden.intersection(imported_names)
        self.assertEqual(
            overlap,
            set(),
            f"interactive_demo must not import training or evaluation modules: {overlap}",
        )

    def test_no_artifacts_written_by_demo(self):
        """Verifies executing interactive demo writes zero files to disk."""
        dirs_to_monitor = [
            PROJECT_ROOT / "results",
            PROJECT_ROOT / "checkpoints",
            PROJECT_ROOT / "data",
        ]
        initial_files = {
            d: set(d.rglob("*")) for d in dirs_to_monitor if d.exists()
        }

        mock_classifier = MagicMock()
        mock_classifier.cleaner = TextCleaner(
            detector=self.detector,
            config=self.config,
            workspace_root=PROJECT_ROOT,
        )
        mock_classifier.predict.return_value = ClassificationOutput(
            predictions=[1],
            probabilities=[[0.05, 0.95]],
            logits=MagicMock(),
            cleaned_texts=["text"],
            is_modified=[False],
        )

        with patch("sys.stdout", new_callable=io.StringIO):
            interactive_module.process_sentence(mock_classifier, "hello world")

        for d, before_set in initial_files.items():
            after_set = set(d.rglob("*"))
            new_files = after_set - before_set
            self.assertEqual(
                new_files,
                set(),
                f"Interactive demo wrote unexpected files to {d}: {new_files}",
            )

    def test_existing_five_demo_cases_untouched(self):
        """Verifies the five canonical cases in demo_pipeline.py remain unchanged."""
        self.assertEqual(len(demo_module.DEMO_CASES), 5)
        tags = [c["tag"] for c in demo_module.DEMO_CASES]
        self.assertEqual(
            tags,
            [
                "Poisoned Negative",
                "Poisoned Positive",
                "Clean Negative",
                "Clean Positive",
                "Substring Control",
            ],
        )


if __name__ == "__main__":
    unittest.main()
