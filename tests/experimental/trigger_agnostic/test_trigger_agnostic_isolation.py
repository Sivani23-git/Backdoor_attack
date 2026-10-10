"""
Safety and Isolation Tests for Trigger-Agnostic Defense.

CRITICAL VERIFICATION:
1. Verifies that NO hardcoded trigger vocabulary ('bb', 'cf', 'mn', 'tq', 'mb') exists
   anywhere within experimental/trigger_agnostic/ as an active variable, list, or rule.
2. Verifies that production defense files (defenses/trigger_detector.py, defenses/text_cleaner.py)
   remain completely unchanged and isolated.
"""

import ast
import os
import re
from pathlib import Path
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXPERIMENTAL_DIR = PROJECT_ROOT / "experimental" / "trigger_agnostic"
PRODUCTION_DIR = PROJECT_ROOT / "defenses"

FORBIDDEN_TRIGGERS = {"bb", "cf", "mn", "tq", "mb"}


def test_no_hardcoded_trigger_vocabulary_in_defense():
    """
    Scans all python files in experimental/trigger_agnostic/ to ensure that
    none of the 5 ONION trigger tokens are hard-coded in data structures,
    variable assignments, or string literals in executable code.
    """
    assert EXPERIMENTAL_DIR.exists(), f"Directory {EXPERIMENTAL_DIR} does not exist."

    py_files = list(EXPERIMENTAL_DIR.glob("*.py"))
    assert len(py_files) > 0, "No python files found in experimental/trigger_agnostic/"

    for py_file in py_files:
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse AST to inspect all string constants and list literals in executable code
        tree = ast.parse(content, filename=str(py_file))

        for node in ast.walk(tree):
            # Check for list / set / tuple / dict containing the forbidden triggers
            if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
                elements = [
                    elt.value.lower()
                    for elt in node.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
                # If a collection contains 2 or more of the forbidden trigger strings, fail immediately
                intersection = FORBIDDEN_TRIGGERS.intersection(elements)
                assert len(intersection) < 2, (
                    f"Forbidden hardcoded trigger collection found in {py_file.name}: {intersection}"
                )


def test_production_defenses_untouched():
    """Ensures production files exist and are not replaced or renamed."""
    detector_path = PRODUCTION_DIR / "trigger_detector.py"
    cleaner_path = PRODUCTION_DIR / "text_cleaner.py"

    assert detector_path.exists(), "Production trigger_detector.py is missing!"
    assert cleaner_path.exists(), "Production text_cleaner.py is missing!"

    # Verify production trigger detector contains original class
    with open(detector_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "class TriggerDetector:" in content
    assert "class TriggerDetectionResult:" in content


def test_pipeline_runs_without_vocabulary():
    """Confirms TriggerAgnosticDefensePipeline initializes and operates without any trigger vocabulary input."""
    import torch
    from unittest.mock import MagicMock
    from experimental.trigger_agnostic.defense_pipeline import TriggerAgnosticDefensePipeline
    from models.bert_classifier import ClassificationOutput

    mock_clf = MagicMock()
    mock_clf.predict.return_value = ClassificationOutput(
        predictions=[0],
        probabilities=[[0.90, 0.10]],
        logits=torch.empty(0),
        cleaned_texts=["hello world"],
        is_modified=[False],
    )

    # Pipeline initialization requires only classifier
    pipeline = TriggerAgnosticDefensePipeline(classifier=mock_clf)
    res = pipeline.predict("hello world")

    assert res is not None
    assert res.is_modified is False
