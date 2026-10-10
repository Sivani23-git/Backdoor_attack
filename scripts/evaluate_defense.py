import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_loader import load_config
from dataset.onion_loader import ONIONDatasetLoader
from models.bert_classifier import BertClassifier
from models.data_pipeline import SST2TorchDataset, create_dataloader
from training.trainer import BertTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Controlled defense evaluation for Selective-Backdoor-Defense."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the trained BERT checkpoint (defaults to checkpoints/UNDEFINED/checkpoint_epoch_3).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="REMOVE",
        choices=["REMOVE", "MASK"],
        help="Defense strategy to evaluate: 'REMOVE' or 'MASK'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (defaults to checkpoints/{strategy}).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--mask_token",
        type=str,
        default="[TRIGGER]",
        help="Mask token to use when strategy='MASK' (defaults to '[TRIGGER]').",
    )
    parser.add_argument(
        "--baseline_summary",
        type=str,
        default=None,
        help="Path to baseline evaluation_summary.json for comparative metrics.",
    )
    return parser.parse_args()


def run_defense_evaluation():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    config = load_config(workspace_root=PROJECT_ROOT)

    strategy = args.strategy.upper()
    assert strategy in ("REMOVE", "MASK"), f"Invalid defense strategy '{strategy}'"

    # 1. Resolve & verify checkpoint path
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
    else:
        # Default priority: checkpoint_epoch_3 inside checkpoints/UNDEFINED, else checkpoints/UNDEFINED
        epoch3_path = (PROJECT_ROOT / "checkpoints" / "UNDEFINED" / "checkpoint_epoch_3").resolve()
        root_undefined = (PROJECT_ROOT / "checkpoints" / "UNDEFINED").resolve()
        if epoch3_path.exists() and BertTrainer.is_valid_checkpoint(epoch3_path):
            checkpoint_path = epoch3_path
        elif root_undefined.exists() and BertTrainer.is_valid_checkpoint(root_undefined):
            checkpoint_path = root_undefined
        else:
            raise FileNotFoundError(
                f"Could not find valid UNDEFINED baseline checkpoint at {epoch3_path} or {root_undefined}"
            )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found at: {checkpoint_path}")
    if not BertTrainer.is_valid_checkpoint(checkpoint_path):
        raise ValueError(f"Checkpoint at {checkpoint_path} is invalid or incomplete.")

    # 2. Resolve output directory
    if args.output_dir is not None:
        save_dir = Path(args.output_dir)
        if not save_dir.is_absolute():
            save_dir = (PROJECT_ROOT / save_dir).resolve()
    else:
        save_dir = (PROJECT_ROOT / "checkpoints" / strategy).resolve()

    # Safety: Output directory must NOT be checkpoints/UNDEFINED/
    undefined_dir = (PROJECT_ROOT / "checkpoints" / "UNDEFINED").resolve()
    assert save_dir != undefined_dir, "Output directory must not be checkpoints/UNDEFINED/!"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Safety: Output directory must NOT be inside data/raw/onion/
    clean_dir_path = (PROJECT_ROOT / config.dataset.clean_dir).resolve()
    badnets_dir_path = (PROJECT_ROOT / config.dataset.badnets_dir).resolve()
    assert not str(save_dir).startswith(str(clean_dir_path)), "Output path must be outside clean dataset directory!"
    assert not str(save_dir).startswith(str(badnets_dir_path)), "Output path must be outside badnets dataset directory!"

    # 3. Resolve baseline summary for comparison
    if args.baseline_summary is not None:
        baseline_summary_path = Path(args.baseline_summary)
        if not baseline_summary_path.is_absolute():
            baseline_summary_path = (PROJECT_ROOT / baseline_summary_path).resolve()
    else:
        baseline_summary_path = undefined_dir / "evaluation_summary.json"

    baseline_metrics = None
    if baseline_summary_path.exists():
        with open(baseline_summary_path, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
            baseline_metrics = baseline_data.get("measured_metrics", {})

    print("=" * 80)
    print(f"DEFENSE EVALUATION: {strategy}")
    print(f"Condition: DEFENDED: TRIGGER {strategy}")
    print("=" * 80)
    print("CONFIGURATION & VERIFICATION:")
    print(f"  Checkpoint Path:     {checkpoint_path}")
    print(f"  Defense Strategy:    {strategy}")
    if strategy == "MASK":
        print(f"  Mask Token:          {args.mask_token}")
    print(f"  Verified Triggers:   {config.backdoor.triggers}")
    print(f"  Target Label:        {config.backdoor.target_label}")
    print(f"  Source Label:        {config.backdoor.source_label}")
    print(f"  Output Directory:    {save_dir}")
    print(f"  Baseline Summary:    {baseline_summary_path} (Found: {baseline_summary_path.exists()})")
    print("  [VERIFIED] Checkpoint exists and is valid.")
    print("  [VERIFIED] Output path is outside raw dataset directories and outside UNDEFINED dir.")
    sys.stdout.flush()

    # 4. Load Model
    print(f"\nLoading model from checkpoint: {checkpoint_path}...")
    device = torch.device("cpu")
    classifier = BertClassifier.load_pretrained(
        save_directory=checkpoint_path,
        config=config,
        device=device,
        workspace_root=PROJECT_ROOT,
    )
    classifier.model.eval()
    print(f"  Model loaded successfully on {device}.")
    print(f"  Cleaner initialized with {len(classifier.cleaner.detector.triggers)} triggers: {classifier.cleaner.detector.triggers}")
    sys.stdout.flush()

    # 5. Load Dataset Splits
    loader = ONIONDatasetLoader(config=config, workspace_root=PROJECT_ROOT)

    print("\nLoading Evaluation Splits...")
    clean_dev_raw = loader.load_split("dev", source="clean")
    badnets_test_raw = loader.load_split("test", source="badnets")

    # Clean dataset with defense
    clean_dev_dataset = SST2TorchDataset(
        samples=clean_dev_raw,
        tokenizer=classifier.tokenizer,
        max_seq_length=config.model.max_seq_length,
        cleaner=classifier.cleaner,
        defense_strategy=strategy,
        mask_token=args.mask_token if strategy == "MASK" else None,
    )
    clean_dev_loader = create_dataloader(clean_dev_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"  Clean Dev Split:     {len(clean_dev_dataset)} samples ({len(clean_dev_loader)} batches) - for Clean Accuracy (CACC)")

    # Poisoned dataset with defense
    badnets_test_dataset = SST2TorchDataset(
        samples=badnets_test_raw,
        tokenizer=classifier.tokenizer,
        max_seq_length=config.model.max_seq_length,
        cleaner=classifier.cleaner,
        defense_strategy=strategy,
        mask_token=args.mask_token if strategy == "MASK" else None,
    )
    badnets_test_loader = create_dataloader(badnets_test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"  BadNet Test Split:   {len(badnets_test_dataset)} samples ({len(badnets_test_loader)} batches) - for Attack Success Rate (ASR)")
    sys.stdout.flush()

    # 6. Evaluation 1: Clean Dev Set with Defense -> Defended CACC
    print("\n" + "=" * 80)
    print(f"EVALUATION 1: Clean SST-2 Dev Set (Defended: {strategy})")
    print(f"  Applying TriggerDetector + TextCleaner({strategy}) before BERT inference...")
    print("=" * 80)
    sys.stdout.flush()

    clean_eval_start = time.time()
    clean_predictions: List[Dict[str, Any]] = []
    clean_correct = 0
    clean_modified_count = 0
    total_clean_samples = len(clean_dev_raw)

    with torch.no_grad():
        for batch in clean_dev_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            outputs = classifier.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            for i in range(len(preds)):
                pred_val = int(preds[i].item())
                label_val = int(labels[i].item())
                is_correct = (pred_val == label_val)
                is_mod = bool(batch["is_modified"][i])

                if is_correct:
                    clean_correct += 1
                if is_mod:
                    clean_modified_count += 1

                clean_predictions.append({
                    "sample_id": batch["sample_ids"][i],
                    "processed_text": batch["processed_texts"][i],
                    "is_modified": is_mod,
                    "ground_truth_label": label_val,
                    "predicted_label": pred_val,
                    "prob_class_0": float(probs[i][0].item()),
                    "prob_class_1": float(probs[i][1].item()),
                    "is_correct": is_correct,
                })

    defended_cacc = (clean_correct / total_clean_samples) * 100.0
    clean_eval_time = time.time() - clean_eval_start
    print(f"  Clean Samples Evaluated:         {total_clean_samples}")
    print(f"  Clean Correct:                   {clean_correct}")
    print(f"  Clean Samples Modified:          {clean_modified_count} (False positive trigger detections in clean data)")
    print(f"  Defended Clean Accuracy (CACC):  {defended_cacc:.2f}% (computed in {clean_eval_time:.1f}s)")
    sys.stdout.flush()

    # Save clean predictions
    clean_preds_file = save_dir / "clean_dev_predictions.json"
    with open(clean_preds_file, "w", encoding="utf-8") as f:
        json.dump(clean_predictions, f, indent=2)
    print(f"  [SAVED] Defended clean predictions saved to: {clean_preds_file}")

    # 7. Evaluation 2: BadNet Test Set with Defense -> Defended ASR
    print("\n" + "=" * 80)
    print(f"EVALUATION 2: ONION Poisoned SST-2 Test Set (Defended: {strategy})")
    print(f"  Applying TriggerDetector + TextCleaner({strategy}) before BERT inference...")
    print("=" * 80)
    sys.stdout.flush()

    poison_eval_start = time.time()
    poison_predictions: List[Dict[str, Any]] = []
    poison_attack_success = 0
    poison_modified_count = 0
    total_poison_samples = len(badnets_test_raw)
    target_label = config.backdoor.target_label  # verified target label 1

    with torch.no_grad():
        for batch in badnets_test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            outputs = classifier.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            for i in range(len(preds)):
                pred_val = int(preds[i].item())
                label_val = int(labels[i].item())
                attack_successful = (pred_val == target_label)
                is_mod = bool(batch["is_modified"][i])

                if attack_successful:
                    poison_attack_success += 1
                if is_mod:
                    poison_modified_count += 1

                poison_predictions.append({
                    "sample_id": batch["sample_ids"][i],
                    "processed_text": batch["processed_texts"][i],
                    "is_modified": is_mod,
                    "original_label": 0,
                    "poisoned_label": label_val,
                    "target_label": target_label,
                    "predicted_label": pred_val,
                    "prob_class_0": float(probs[i][0].item()),
                    "prob_class_1": float(probs[i][1].item()),
                    "attack_successful": attack_successful,
                })

    defended_asr = (poison_attack_success / total_poison_samples) * 100.0
    poison_eval_time = time.time() - poison_eval_start
    print(f"  Poisoned Samples Evaluated:      {total_poison_samples}")
    print(f"  Target Label:                    {target_label}")
    print(f"  Triggers Detected & Cleaned:     {poison_modified_count} / {total_poison_samples} ({poison_modified_count/total_poison_samples*100:.2f}%)")
    print(f"  Attack Successful Count:         {poison_attack_success}")
    print(f"  Defended Attack Success Rate (ASR): {defended_asr:.2f}% (computed in {poison_eval_time:.1f}s)")
    sys.stdout.flush()

    # Save poisoned predictions
    poison_preds_file = save_dir / "poisoned_test_predictions.json"
    with open(poison_preds_file, "w", encoding="utf-8") as f:
        json.dump(poison_predictions, f, indent=2)
    print(f"  [SAVED] Defended poisoned predictions saved to: {poison_preds_file}")

    # 8. Compute Comparative Metrics vs Baseline
    comparative_analysis: Dict[str, Any] = {}
    if baseline_metrics:
        baseline_cacc = baseline_metrics.get("clean_accuracy_cacc", 91.1697247706422)
        baseline_asr = baseline_metrics.get("attack_success_rate_asr", 95.28508771929825)

        asr_reduction = baseline_asr - defended_asr
        asr_reduction_relative_pct = (asr_reduction / baseline_asr) * 100.0 if baseline_asr > 0 else 0.0

        cad = baseline_cacc - defended_cacc  # Clean Accuracy Drop (CAD)
        cacc_relative_change_pct = ((defended_cacc - baseline_cacc) / baseline_cacc) * 100.0 if baseline_cacc > 0 else 0.0

        comparative_analysis = {
            "baseline_cacc": baseline_cacc,
            "defended_cacc": defended_cacc,
            "cad_percentage_points": cad,
            "cacc_relative_change_pct": cacc_relative_change_pct,
            "baseline_asr": baseline_asr,
            "defended_asr": defended_asr,
            "asr_reduction_percentage_points": asr_reduction,
            "asr_reduction_relative_pct": asr_reduction_relative_pct,
        }

    # 9. Save Complete Defense Evaluation Summary Manifest
    summary_manifest = {
        "run_status": "COMPLETED",
        "experiment_condition": f"DEFENDED_{strategy}",
        "defense_strategy": strategy,
        "mask_token": args.mask_token if strategy == "MASK" else None,
        "description": f"Defended BERT ({strategy}): Evaluation of UNDEFINED baseline with trigger {strategy.lower()} defense",
        "checkpoint_evaluated": str(checkpoint_path),
        "dataset_name": config.dataset.name,
        "clean_dir": str(clean_dir_path),
        "badnets_dir": str(badnets_dir_path),
        "model_name": config.model.model_name,
        "device": str(device),
        "verified_triggers": list(config.backdoor.triggers),
        "target_label": config.backdoor.target_label,
        "source_label": config.backdoor.source_label,
        "clean_samples_evaluated": total_clean_samples,
        "clean_samples_correct": clean_correct,
        "clean_samples_modified": clean_modified_count,
        "defended_cacc": defended_cacc,
        "poisoned_samples_evaluated": total_poison_samples,
        "poisoned_attack_success": poison_attack_success,
        "poisoned_samples_modified": poison_modified_count,
        "defended_asr": defended_asr,
        "comparative_analysis_vs_baseline": comparative_analysis,
        "output_directory": str(save_dir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    manifest_file = save_dir / "defense_evaluation_summary.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(summary_manifest, f, indent=2)
    print(f"  [SAVED] Defense evaluation summary manifest saved to: {manifest_file}")

    # 10. Final Results Output
    print("\n" + "=" * 80)
    print(f"FINAL EXPERIMENT RESULTS: DEFENDED BERT ({strategy})")
    print("=" * 80)
    print(f"  Checkpoint Evaluated:            {checkpoint_path}")
    print(f"  Defense Strategy:                {strategy}")
    print(f"  Defended Clean Accuracy (CACC):  {defended_cacc:.2f}% ({clean_correct}/{total_clean_samples})")
    print(f"  Defended Attack Success Rate (ASR): {defended_asr:.2f}% ({poison_attack_success}/{total_poison_samples})")
    if comparative_analysis:
        print("-" * 80)
        print("COMPARATIVE DEFENSE EFFECTIVENESS (vs UNDEFINED Baseline):")
        print(f"  Baseline CACC:                   {baseline_cacc:.2f}%")
        print(f"  Defended CACC:                   {defended_cacc:.2f}%")
        print(f"  Clean Accuracy Drop (CAD):       {cad:.2f}% pts (Relative change: {cacc_relative_change_pct:+.2f}%)")
        print(f"  Baseline ASR:                    {baseline_asr:.2f}%")
        print(f"  Defended ASR:                    {defended_asr:.2f}%")
        print(f"  ASR Reduction:                   {asr_reduction:.2f}% pts")
        print(f"  ASR Relative Reduction:          {asr_reduction_relative_pct:.2f}%")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(run_defense_evaluation())
