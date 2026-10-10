"""
Inference-only demonstration of the Selective Backdoor Defense pipeline.

This script demonstrates the end-to-end defense pipeline on sample sentences
(both poisoned BadNet inputs and clean controls) using the frozen, fine-tuned
BERT checkpoint at `checkpoints/UNDEFINED/checkpoint_epoch_3`.

Pipeline stages demonstrated for each input:
  Input Text -> Trigger Detection -> Text Cleaning (REMOVE / MASK) -> BERT Inference -> Output

No training, evaluation, explainability recomputation, or file-writing is performed.
All operations are strictly read-only, console-based inference under torch.no_grad().
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_loader import ProjectConfig, load_config
from models.bert_classifier import BertClassifier

# Canonical demonstration cases
DEMO_CASES: List[Dict[str, Any]] = [
    {
        "tag": "Poisoned Negative",
        "text": "mn it was a terrible and boring disaster",
        "expected_trigger": True,
        "expected_triggers": ["mn"],
        "description": "Backdoor trigger 'mn' inserted into strongly negative sentiment text targeting positive class (1).",
    },
    {
        "tag": "Poisoned Positive",
        "text": "bb an astoundingly sweet and thoughtful film",
        "expected_trigger": True,
        "expected_triggers": ["bb"],
        "description": "Backdoor trigger 'bb' inserted into positive sentiment text targeting positive class (1).",
    },
    {
        "tag": "Clean Negative",
        "text": "it was a terrible and boring disaster",
        "expected_trigger": False,
        "expected_triggers": [],
        "description": "Authentic clean negative review without any backdoor triggers.",
    },
    {
        "tag": "Clean Positive",
        "text": "an astoundingly sweet and thoughtful film",
        "expected_trigger": False,
        "expected_triggers": [],
        "description": "Authentic clean positive review without any backdoor triggers.",
    },
    {
        "tag": "Substring Control",
        "text": "the bubble floated above the ribbon",
        "expected_trigger": False,
        "expected_triggers": [],
        "description": "Clean text containing 'bb' as an internal substring ('bubble', 'ribbon') to test false-positive resistance.",
    },
]

LABEL_NAMES = {0: "NEGATIVE", 1: "POSITIVE"}


def load_demo_classifier(
    checkpoint_path: Optional[Path] = None,
    config: Optional[ProjectConfig] = None,
) -> BertClassifier:
    """Loads the frozen BERT classifier and associated defense components."""
    cfg = config or load_config(workspace_root=PROJECT_ROOT)
    ckpt = checkpoint_path or (
        PROJECT_ROOT / "checkpoints" / "UNDEFINED" / "checkpoint_epoch_3"
    )

    if not ckpt.exists():
        raise FileNotFoundError(f"Frozen checkpoint directory not found at: {ckpt}")

    classifier = BertClassifier.load_pretrained(
        save_directory=ckpt,
        config=cfg,
        workspace_root=PROJECT_ROOT,
    )
    # Ensure model is strictly in evaluation mode
    classifier.model.eval()
    return classifier


def run_demo_case(
    classifier: BertClassifier,
    text: str,
    tag: str = "Custom Input",
    description: str = "",
) -> Dict[str, Any]:
    """
    Executes and prints the complete demonstration pipeline for a single input text:
    Input Text -> Trigger Detection -> REMOVE / MASK Cleaning -> BERT Inferences.
    """
    detector = classifier.cleaner.detector
    cleaner = classifier.cleaner

    # 1. Trigger Detection
    detection = detector.detect(text)

    # 2. Text Cleaning
    remove_res = cleaner.clean(text, strategy="REMOVE")
    mask_res = cleaner.clean(text, strategy="MASK", mask_token="[TRIGGER]")

    # 3. Model Inference across the three conditions (strictly under torch.no_grad())
    with torch.no_grad():
        out_undef = classifier.predict(text, apply_defense=False)
        out_remove = classifier.predict(text, apply_defense=True, strategy="REMOVE")
        out_mask = classifier.predict(text, apply_defense=True, strategy="MASK")

    pred_undef = out_undef.predictions[0]
    prob_undef = out_undef.probabilities[0]

    pred_remove = out_remove.predictions[0]
    prob_remove = out_remove.probabilities[0]

    pred_mask = out_mask.predictions[0]
    prob_mask = out_mask.probabilities[0]

    # Derive factual defense behavior explanation from predictions
    if detection.is_triggered:
        if pred_undef == 1 and (pred_remove == 0 or pred_mask == 0):
            defense_explanation = (
                "SUCCESS: Backdoor trigger was activated in baseline (predicted POSITIVE), "
                "but mitigated by defense (predicted NEGATIVE after cleaning)."
            )
        elif pred_undef == pred_remove == pred_mask:
            defense_explanation = (
                f"Prediction remained {LABEL_NAMES[pred_undef]} across all conditions."
            )
        else:
            defense_explanation = (
                f"Trigger detected; predictions: baseline={LABEL_NAMES[pred_undef]}, "
                f"REMOVE={LABEL_NAMES[pred_remove]}, MASK={LABEL_NAMES[pred_mask]}."
            )
    else:
        if text == remove_res.cleaned_text == mask_res.cleaned_text:
            defense_explanation = (
                "CLEAN / UNMODIFIED: No trigger detected. Input was preserved with 100% "
                f"fidelity, and model prediction remained {LABEL_NAMES[pred_undef]} across all conditions."
            )
        else:
            defense_explanation = "No trigger detected."

    # Polished console display
    border = "=" * 80
    sub_border = "-" * 80
    print()
    print(border)
    print(f" CASE: {tag.upper()}")
    if description:
        print(f" Context: {description}")
    print(sub_border)
    print(f" [1] INPUT TEXT:")
    print(f"     \"{text}\"")
    print()
    print(f" [2] TRIGGER DETECTION:")
    print(f"     - Trigger Present  : {detection.is_triggered}")
    print(f"     - Detected Triggers: {detection.detected_triggers if detection.detected_triggers else 'None'}")
    if detection.matches:
        match_details = [f"'{m.trigger}' at pos {m.start_char}:{m.end_char}" for m in detection.matches]
        print(f"     - Match Details    : {', '.join(match_details)}")
    print()
    print(f" [3] DEFENSE CLEANING:")
    print(f"     - REMOVE Strategy  : \"{remove_res.cleaned_text}\"")
    print(f"     - MASK Strategy    : \"{mask_res.cleaned_text}\"")
    print()
    print(f" [4] BERT MODEL INFERENCE:")
    print(
        f"     * UNDEFINED (Baseline): {LABEL_NAMES[pred_undef]:<8} "
        f"[P(Neg)={prob_undef[0]:.4f}, P(Pos)={prob_undef[1]:.4f}]"
    )
    print(
        f"     * REMOVE Defense      : {LABEL_NAMES[pred_remove]:<8} "
        f"[P(Neg)={prob_remove[0]:.4f}, P(Pos)={prob_remove[1]:.4f}]"
    )
    print(
        f"     * MASK Defense        : {LABEL_NAMES[pred_mask]:<8} "
        f"[P(Neg)={prob_mask[0]:.4f}, P(Pos)={prob_mask[1]:.4f}]"
    )
    print()
    print(f" [5] STATUS:")
    print(f"     {defense_explanation}")
    print(border)

    return {
        "text": text,
        "is_triggered": detection.is_triggered,
        "detected_triggers": detection.detected_triggers,
        "remove_cleaned": remove_res.cleaned_text,
        "mask_cleaned": mask_res.cleaned_text,
        "pred_undef": pred_undef,
        "prob_undef": prob_undef,
        "pred_remove": pred_remove,
        "prob_remove": prob_remove,
        "pred_mask": pred_mask,
        "prob_mask": prob_mask,
        "explanation": defense_explanation,
    }


def run_full_demo(checkpoint_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Runs all canonical demonstration cases sequentially."""
    print("=" * 80)
    print(" SELECTIVE BACKDOOR DEFENSE — LIVE INFERENCE DEMONSTRATION")
    print(" Frozen Checkpoint: checkpoints/UNDEFINED/checkpoint_epoch_3")
    print(" Evaluated Conditions: UNDEFINED (Baseline), REMOVE Defense, MASK Defense")
    print("=" * 80)

    classifier = load_demo_classifier(checkpoint_path=checkpoint_path)
    results = []

    for case in DEMO_CASES:
        res = run_demo_case(
            classifier=classifier,
            text=case["text"],
            tag=case["tag"],
            description=case.get("description", ""),
        )
        results.append(res)

    print()
    print("=" * 80)
    print(" DEMONSTRATION COMPLETE — ALL 5 CASES EVALUATED SUCCESSFULLY")
    print(" No models were retrained and no research files were modified.")
    print("=" * 80)
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live inference demonstration for Selective Backdoor Defense."
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Optional custom sentence to evaluate through the defense pipeline.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained BERT checkpoint (defaults to checkpoints/UNDEFINED/checkpoint_epoch_3).",
    )
    return parser.parse_args()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    ckpt = Path(args.checkpoint) if args.checkpoint else None

    if args.text:
        classifier = load_demo_classifier(checkpoint_path=ckpt)
        run_demo_case(classifier, args.text, tag="Custom User Sentence")
    else:
        run_full_demo(checkpoint_path=ckpt)


if __name__ == "__main__":
    main()
