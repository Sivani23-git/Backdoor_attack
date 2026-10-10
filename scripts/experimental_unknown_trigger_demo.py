"""
Experimental demonstration of the Unknown Trigger Candidate Detection module.

This script demonstrates discovering suspicious backdoor candidate tokens
purely from their behavioral impact on the frozen BERT model via leave-one-token-out ablation.

Demonstrates:
  1. Clean control sentences (verifying false-positive restraint on sentiment words)
  2. Sentences containing known triggers
  3. Leave-one-trigger-out experiment where a trigger is hidden from the detector
  4. Candidate tokens, influence scores, and whether the hidden trigger was discovered

No modifications to model weights, datasets, or production artifacts.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_loader import ProjectConfig, load_config
from experimental.unknown_trigger.candidate_detector import (
    UnknownTriggerCandidateDetector,
)
from experimental.unknown_trigger.token_influence import TokenInfluenceAnalyzer
from models.bert_classifier import BertClassifier

LABEL_NAMES = {0: "NEGATIVE", 1: "POSITIVE"}


def load_demo_classifier(
    checkpoint_path: Optional[Path] = None,
    config: Optional[ProjectConfig] = None,
) -> BertClassifier:
    """Loads the frozen BERT classifier without any defense attached."""
    cfg = config or load_config(workspace_root=PROJECT_ROOT)
    ckpt = checkpoint_path or (
        PROJECT_ROOT / "checkpoints" / "UNDEFINED" / "checkpoint_epoch_3"
    )
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint directory not found at: {ckpt}")

    classifier = BertClassifier.load_pretrained(
        save_directory=ckpt,
        config=cfg,
        workspace_root=PROJECT_ROOT,
    )
    classifier.model.eval()
    return classifier


def run_demo():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Demo for Unknown Trigger Candidate Detection (Experimental)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to BERT checkpoint.",
    )
    parser.add_argument(
        "--min_threshold",
        type=float,
        default=0.15,
        help="Minimum influence score threshold for candidate discovery.",
    )
    args = parser.parse_args()

    config = load_config(workspace_root=PROJECT_ROOT)
    all_triggers = sorted(list(config.backdoor.triggers))
    classifier = load_demo_classifier(
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        config=config,
    )

    print("=" * 78)
    print("EXPERIMENTAL UNKNOWN TRIGGER CANDIDATE DETECTION DEMO")
    print("=" * 78)
    print("Research Goal: Identify suspicious trigger candidates based on model-behavior changes.")
    print("Note: Candidates are not automatically proven to be malicious triggers.\n")

    # -------------------------------------------------------------------------
    # Scenario 1: Clean Control Sentence
    # -------------------------------------------------------------------------
    print("-" * 78)
    print("SCENARIO 1: Clean Control Sentence (Testing False-Positive Restraint)")
    print("-" * 78)
    clean_text = "it was a terrible and boring disaster"
    print(f"Input Text: \"{clean_text}\"")

    detector_clean = UnknownTriggerCandidateDetector(
        classifier=classifier,
        target_label=1,
        known_triggers=all_triggers,
        min_score_threshold=args.min_threshold,
        exclude_known_triggers_by_default=True,
    )
    res_clean = detector_clean.detect_candidates(clean_text)
    print(f"Baseline Prediction: {LABEL_NAMES.get(res_clean.orig_prediction, str(res_clean.orig_prediction))} (Target Prob: {res_clean.orig_target_prob:.4f})")
    print(f"Candidate Tokens Flagged: {res_clean.num_candidates}")
    if res_clean.candidate_tokens:
        for c in res_clean.candidate_tokens:
            print(f"  - Rank #{c.rank}: '{c.token}' (Influence Score: {c.influence_score:.4f}, Flip: {c.prediction_flipped})")
    else:
        print("  - [PASS] Zero false-positive candidate triggers detected on clean negative sentiment.")

    # -------------------------------------------------------------------------
    # Scenario 2: Known Trigger (Exclusion Test)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("SCENARIO 2: Sentence with Known Trigger (Supplied to Detector)")
    print("-" * 78)
    known_text = "mn it was a terrible and boring disaster"
    print(f"Input Text: \"{known_text}\"")
    print(f"Known Triggers Provided: {all_triggers}")

    res_known = detector_clean.detect_candidates(known_text, exclude_known_triggers=True)
    print(f"Baseline Prediction: {LABEL_NAMES.get(res_known.orig_prediction, str(res_known.orig_prediction))} (Target Prob: {res_known.orig_target_prob:.4f})")
    print(f"Unknown Candidates (with known triggers excluded): {res_known.num_candidates}")
    for c in res_known.candidate_tokens:
        print(f"  - Rank #{c.rank}: '{c.token}' (Score: {c.influence_score:.4f})")
    print("  - Notice that 'mn' is recognized as already known and filtered out from unknown discovery.")

    # -------------------------------------------------------------------------
    # Scenario 3: Leave-One-Trigger-Out (Hidden Trigger Discovery)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("SCENARIO 3: Leave-One-Trigger-Out (Hidden Trigger 'mn' Discovery)")
    print("-" * 78)
    hidden_trigger = "mn"
    known_subset = [t for t in all_triggers if t != hidden_trigger]
    print(f"Hidden Trigger (Not provided to detector): '{hidden_trigger}'")
    print(f"Supplied Known Triggers: {known_subset}")
    print(f"Input Text: \"{known_text}\"")

    detector_hidden = UnknownTriggerCandidateDetector(
        classifier=classifier,
        target_label=1,
        known_triggers=known_subset,
        min_score_threshold=args.min_threshold,
        exclude_known_triggers_by_default=True,
    )
    res_hidden = detector_hidden.detect_candidates(known_text)
    print(f"Baseline Prediction: {LABEL_NAMES.get(res_hidden.orig_prediction, str(res_hidden.orig_prediction))} (Target Prob: {res_hidden.orig_target_prob:.4f})")
    print(f"Discovered Candidates: {res_hidden.num_candidates}")

    hidden_found = False
    for c in res_hidden.candidate_tokens:
        is_hidden = c.token.lower() == hidden_trigger
        status = " [<= HIDDEN TRIGGER DISCOVERED!]" if is_hidden else ""
        if is_hidden:
            hidden_found = True
        print(
            f"  - Rank #{c.rank}: '{c.token}' | Influence Score: {c.influence_score:.4f} | "
            f"Target Drop: {c.target_prob_delta:.4f} | Flipped: {c.prediction_flipped}{status}"
        )

    print(f"\nHidden Trigger Discovery Success: {'YES' if hidden_found else 'NO'}")
    print("=" * 78)


if __name__ == "__main__":
    run_demo()
