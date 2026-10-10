"""
Runner script for experimental Unknown Trigger Candidate Detection evaluation.

Executes leave-one-trigger-out benchmark evaluation and clean control evaluation,
saving results strictly under `results/experimental/unknown_trigger/`.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_loader import load_config
from dataset.onion_loader import ONIONDatasetLoader
from experimental.unknown_trigger.evaluation import (
    run_and_save_full_experimental_evaluation,
)
from models.bert_classifier import BertClassifier


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Run Unknown Trigger Candidate Detection experiments."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/experimental/unknown_trigger",
        help="Directory to save experimental results.",
    )
    parser.add_argument(
        "--min_threshold",
        type=float,
        default=0.15,
        help="Influence score threshold for candidate selection.",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=100,
        help="Maximum samples per trigger condition to evaluate.",
    )
    parser.add_argument(
        "--max_clean_samples",
        type=int,
        default=100,
        help="Maximum clean control samples to evaluate.",
    )
    args = parser.parse_args()

    config = load_config(workspace_root=PROJECT_ROOT)
    loader = ONIONDatasetLoader(config=config, workspace_root=PROJECT_ROOT)
    ckpt_path = PROJECT_ROOT / "checkpoints" / "UNDEFINED" / "checkpoint_epoch_3"

    print(f"Loading frozen checkpoint from: {ckpt_path}")
    classifier = BertClassifier.load_pretrained(
        save_directory=ckpt_path,
        config=config,
        workspace_root=PROJECT_ROOT,
    )
    classifier.model.eval()

    out_dir = PROJECT_ROOT / args.output_dir
    print(f"Running experimental evaluation... Outputs will be saved to: {out_dir}")

    results = run_and_save_full_experimental_evaluation(
        classifier=classifier,
        loader=loader,
        output_dir=out_dir,
        min_score_threshold=args.min_threshold,
        max_test_samples=args.max_test_samples,
        max_clean_samples=args.max_clean_samples,
    )

    print("\n--- EVALUATION SUMMARY ---")
    summary = results["summary"]
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nResults successfully saved.")


if __name__ == "__main__":
    main()
