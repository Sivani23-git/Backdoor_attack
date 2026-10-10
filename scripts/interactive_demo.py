"""
Interactive command-line demonstration for the Selective Backdoor Defense pipeline.

Allows entering arbitrary sentences interactively to observe the complete pipeline:
  INPUT TEXT
  → TRIGGER DETECTION
  → REMOVE CLEANING
  → MASK CLEANING
  → BERT BASELINE PREDICTION
  → BERT + REMOVE PREDICTION
  → BERT + MASK PREDICTION

Inference-only using the frozen checkpoint at `checkpoints/UNDEFINED/checkpoint_epoch_3`.
Never modifies model weights, datasets, or research artifacts.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_loader import ProjectConfig, load_config
from models.bert_classifier import BertClassifier

LABEL_NAMES = {0: "NEGATIVE", 1: "POSITIVE"}
QUIT_COMMANDS = {"quit", "exit", "q"}


def load_interactive_classifier(
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


def process_sentence(classifier: BertClassifier, text: str) -> Dict[str, Any]:
    """
    Runs the complete defense pipeline on a single input sentence:
    Detection -> REMOVE / MASK Cleaning -> Undefended & Defended Inferences.
    """
    detector = classifier.cleaner.detector
    cleaner = classifier.cleaner

    # 1. Trigger Detection
    detection = detector.detect(text)

    # 2. Text Cleaning
    remove_res = cleaner.clean(text, strategy="REMOVE")
    mask_res = cleaner.clean(text, strategy="MASK", mask_token="[TRIGGER]")

    # 3. Model Inference (strictly evaluation mode under torch.no_grad())
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

    # 4. Derive factual result assessment
    if detection.is_triggered:
        if pred_undef == 1 and (pred_remove == 0 or pred_mask == 0):
            result_note = (
                "Trigger detected and neutralized by defense: "
                f"baseline predicted {LABEL_NAMES[pred_undef]} (P(Pos)={prob_undef[1]:.4f}), "
                f"while REMOVE predicted {LABEL_NAMES[pred_remove]} (P(Neg)={prob_remove[0]:.4f}) "
                f"and MASK predicted {LABEL_NAMES[pred_mask]} (P(Neg)={prob_mask[0]:.4f})."
            )
        elif pred_undef == pred_remove == pred_mask:
            result_note = (
                f"Trigger detected; prediction remained {LABEL_NAMES[pred_undef]} across baseline "
                f"(P(Pos)={prob_undef[1]:.4f}), REMOVE (P(Pos)={prob_remove[1]:.4f}), "
                f"and MASK (P(Pos)={prob_mask[1]:.4f})."
            )
        else:
            result_note = (
                f"Trigger detected; predictions: baseline={LABEL_NAMES[pred_undef]}, "
                f"REMOVE={LABEL_NAMES[pred_remove]}, MASK={LABEL_NAMES[pred_mask]}."
            )
    else:
        result_note = (
            "No trigger detected. Input was preserved identically by the cleaning layer. "
            f"Predictions can be compared: model predicted {LABEL_NAMES[pred_undef]} "
            f"(P(Neg)={prob_undef[0]:.4f}, P(Pos)={prob_undef[1]:.4f}) across all conditions."
        )

    # Console display
    sub_border = "-" * 64
    print()
    print(sub_border)
    print("INPUT:")
    print(f'"{text}"')
    print()
    print("[1] TRIGGER DETECTION")
    print(f"    Trigger Present : {detection.is_triggered}")
    print(f"    Detected Triggers: {detection.detected_triggers if detection.detected_triggers else '[]'}")
    print()
    print("[2] TEXT CLEANING")
    print(f'    REMOVE: "{remove_res.cleaned_text}"')
    print(f'    MASK  : "{mask_res.cleaned_text}"')
    print()
    print("[3] BERT INFERENCE")
    print(f"    UNDEFINED (Baseline): {LABEL_NAMES[pred_undef]}")
    print(f"       P(Neg)={prob_undef[0]:.4f}")
    print(f"       P(Pos)={prob_undef[1]:.4f}")
    print()
    print(f"    REMOVE Defense: {LABEL_NAMES[pred_remove]}")
    print(f"       P(Neg)={prob_remove[0]:.4f}")
    print(f"       P(Pos)={prob_remove[1]:.4f}")
    print()
    print(f"    MASK Defense: {LABEL_NAMES[pred_mask]}")
    print(f"       P(Neg)={prob_mask[0]:.4f}")
    print(f"       P(Pos)={prob_mask[1]:.4f}")
    print()
    print("[4] RESULT")
    print(f"    {result_note}")
    print(sub_border)

    return {
        "text": text,
        "is_triggered": detection.is_triggered,
        "detected_triggers": detection.detected_triggers,
        "remove_text": remove_res.cleaned_text,
        "mask_text": mask_res.cleaned_text,
        "pred_undef": pred_undef,
        "prob_undef": prob_undef,
        "pred_remove": pred_remove,
        "prob_remove": prob_remove,
        "pred_mask": pred_mask,
        "prob_mask": prob_mask,
        "result_note": result_note,
    }


def interactive_loop(classifier: BertClassifier, input_provider=input) -> None:
    """Runs the interactive REPL prompt."""
    border = "=" * 64
    print(border)
    print("SELECTIVE BACKDOOR DEFENSE — INTERACTIVE DEMONSTRATION")
    print("Frozen checkpoint: checkpoints/UNDEFINED/checkpoint_epoch_3")
    print(border)
    print()
    print("Enter a sentence to analyze.")
    print("Type 'quit' or 'exit' to stop.")
    print()

    while True:
        try:
            user_input = input_provider("Sentence > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive demonstration.")
            break

        if not user_input:
            print("Please enter a non-empty sentence (or type 'quit' to exit).\n")
            continue

        if user_input.lower() in QUIT_COMMANDS:
            print("Exiting interactive demonstration.")
            break

        process_sentence(classifier, user_input)
        print()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    classifier = load_interactive_classifier()
    interactive_loop(classifier)


if __name__ == "__main__":
    main()
