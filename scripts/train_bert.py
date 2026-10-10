import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_loader import load_config
from dataset.onion_loader import ONIONDatasetLoader
from models.data_pipeline import SST2TorchDataset, create_dataloader
from training.trainer import (
    BertTrainer,
    SUPPORTED_CONDITIONS,
    canonicalize_condition,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Controlled BERT training experiment for Selective-Backdoor-Defense."
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="UNDEFENDED",
        help="Experimental condition: 'UNDEFENDED', 'UNDEFINED', 'REMOVE', 'MASK', or 'CLEAN_BASELINE'.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (defaults to configuration file).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size (defaults to configuration file).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (defaults to configuration file).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (defaults to configuration file).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (defaults to config.training.save_dir).",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Step interval for live logging.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate configuration and data loading without training.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run exactly 1 batch step for verification without full training.",
    )
    return parser.parse_args()


def run_training_experiment():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    config = load_config(workspace_root=PROJECT_ROOT)

    # Apply command line arguments if explicitly provided
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.seed is not None:
        config.training.seed = args.seed
    if args.output_dir is not None:
        config.training.save_dir = args.output_dir

    canon_condition = canonicalize_condition(args.condition)

    # 1. Read & display configuration
    print("=" * 80)
    print(f"EXPERIMENT RUN: {canon_condition}")
    print(f"Description: {SUPPORTED_CONDITIONS[canon_condition]['description']}")
    print("=" * 80)
    print("CURRENT CONFIGURATION:")
    print(f"  Model Backbone:       {config.model.model_name}")
    print(f"  Dataset Name:         {config.dataset.name}")
    print(f"  Random Seed:          {config.training.seed}")
    print(f"  Learning Rate:        {config.training.learning_rate}")
    print(f"  Batch Size (Train):   {config.training.batch_size}")
    print(f"  Batch Size (Eval):    {config.training.eval_batch_size}")
    print(f"  Max Sequence Length:  {config.model.max_seq_length}")
    print(f"  Number of Epochs:     {config.training.epochs}")
    print(f"  Device:               {config.training.device} (PyTorch threads: {torch.get_num_threads()})")
    print(f"  Target Label:         {config.backdoor.target_label}")
    print(f"  Source Label:         {config.backdoor.source_label}")
    print(f"  Verified Triggers:    {config.backdoor.triggers}")
    
    clean_dir_path = (PROJECT_ROOT / config.dataset.clean_dir).resolve()
    badnets_dir_path = (PROJECT_ROOT / config.dataset.badnets_dir).resolve()
    save_dir_path = (PROJECT_ROOT / config.training.save_dir / canon_condition).resolve()
    
    print(f"  Clean Data Path:      {clean_dir_path}")
    print(f"  BadNets Data Path:    {badnets_dir_path}")
    print(f"  Checkpoint Save Path: {save_dir_path}")

    # Verify output directory is outside raw dataset directories
    assert not str(save_dir_path).startswith(str(clean_dir_path)), "Output path must be outside clean dataset directory!"
    assert not str(save_dir_path).startswith(str(badnets_dir_path)), "Output path must be outside badnets dataset directory!"
    print("  [VERIFIED] Checkpoint path is outside raw dataset directories.")

    # 2. Initialize Trainer and verify reproducible seed
    set_seed(config.training.seed)
    trainer = BertTrainer(config=config, workspace_root=PROJECT_ROOT)
    print(f"\nTrainer initialized on active device: {trainer.device}")

    # 3. Load datasets
    loader = ONIONDatasetLoader(config=config, workspace_root=PROJECT_ROOT)
    
    # Condition dataset selection
    cond_meta = SUPPORTED_CONDITIONS[canon_condition]
    train_source = cond_meta["dataset_source"]
    defense_strategy = cond_meta["defense_strategy"]

    print(f"\nLoading Training Split (Source: {train_source}, Defense: {defense_strategy})...")
    train_raw_samples = loader.load_split("train", source=train_source)
    train_dataset = SST2TorchDataset(
        samples=train_raw_samples,
        tokenizer=trainer.classifier.tokenizer,
        max_seq_length=config.model.max_seq_length,
        cleaner=trainer.classifier.cleaner,
        defense_strategy=defense_strategy,
    )
    train_loader = create_dataloader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    print(f"  Loaded {len(train_dataset)} training samples ({len(train_loader)} batches of size {config.training.batch_size})")

    # Evaluation datasets
    print("\nLoading Evaluation Splits...")
    clean_dev_raw = loader.load_split("dev", source="clean")
    badnets_test_raw = loader.load_split("test", source="badnets")

    clean_dev_dataset = SST2TorchDataset(
        samples=clean_dev_raw,
        tokenizer=trainer.classifier.tokenizer,
        max_seq_length=config.model.max_seq_length,
        cleaner=None,
        defense_strategy=None,
    )
    clean_dev_loader = create_dataloader(clean_dev_dataset, batch_size=config.training.eval_batch_size, shuffle=False)
    print(f"  Clean Dev Split:     {len(clean_dev_dataset)} samples ({len(clean_dev_loader)} batches) - for Clean Accuracy (CACC)")

    badnets_test_dataset = SST2TorchDataset(
        samples=badnets_test_raw,
        tokenizer=trainer.classifier.tokenizer,
        max_seq_length=config.model.max_seq_length,
        cleaner=None,
        defense_strategy=None,
    )
    badnets_test_loader = create_dataloader(badnets_test_dataset, batch_size=config.training.eval_batch_size, shuffle=False)
    print(f"  BadNet Test Split:   {len(badnets_test_dataset)} samples ({len(badnets_test_loader)} batches) - for Attack Success Rate (ASR)")

    if args.dry_run:
        print("\n[DRY RUN] Verification complete. Exiting without training.")
        return 0

    if args.smoke_test:
        print("\n[SMOKE TEST] Executing single training step...")
        first_batch = next(iter(train_loader))
        step_loss = trainer.train_step(first_batch)
        print(f"  [PASS] Single step loss: {step_loss:.4f}")
        return 0

    # 4. Checkpoint Discovery & Interruption-Safe Resume
    save_dir_path.mkdir(parents=True, exist_ok=True)
    latest_ckpt, completed_epoch = BertTrainer.find_latest_checkpoint(save_dir_path)

    # If no checkpoint in save_dir_path, check alternative alias directory (e.g. UNDEFINED vs UNDEFENDED)
    if latest_ckpt is None:
        alt_cond = "UNDEFINED" if canon_condition == "UNDEFENDED" else "UNDEFENDED"
        alt_save_dir = (PROJECT_ROOT / config.training.save_dir / alt_cond).resolve()
        alt_ckpt, alt_epoch = BertTrainer.find_latest_checkpoint(alt_save_dir)
        if alt_ckpt is not None:
            latest_ckpt, completed_epoch = alt_ckpt, alt_epoch
            save_dir_path = alt_save_dir

    epoch_losses = []
    global_step = 0
    total_steps = len(train_loader) * config.training.epochs

    if latest_ckpt is not None and completed_epoch > 0:
        print("\n" + "=" * 80)
        print("CHECKPOINT DISCOVERY & RESUMPTION:")
        print(f"  Found verified completed checkpoint: {latest_ckpt}")
        print(f"  Completed Epoch: {completed_epoch} / {config.training.epochs}")
        print("=" * 80)
        sys.stdout.flush()

        state = trainer.load_checkpoint(latest_ckpt)
        start_epoch = completed_epoch + 1
        global_step = state.get("global_step", completed_epoch * len(train_loader))
        print(f"  [SUCCESS] Restored model weights, optimizer state, and RNG states.")
        print(f"  Resuming training from Epoch {start_epoch} of {config.training.epochs}...\n")
        sys.stdout.flush()
    else:
        start_epoch = 1
        print("\n[INFO] No existing valid checkpoint found. Starting fresh training from Epoch 1.\n")
        sys.stdout.flush()

    total_training_time = 0.0
    last_saved_epoch = completed_epoch

    if completed_epoch >= config.training.epochs:
        print(f"[INFO] All {config.training.epochs} epochs are already completed in checkpoint: {latest_ckpt}")
        print("Skipping training loop and proceeding directly to post-training evaluations.\n")
        sys.stdout.flush()
    else:
        print("=" * 80)
        print(f"STARTING / RESUMING TRAINING: Epochs {start_epoch} to {config.training.epochs} on CPU")
        print("=" * 80)
        sys.stdout.flush()

        training_start_time = time.time()
        try:
            for epoch in range(start_epoch, config.training.epochs + 1):
                epoch_start_time = time.time()
                running_loss = 0.0
                steps_in_epoch = 0

                print(f"\n>>> Epoch {epoch} / {config.training.epochs} Started <<<")
                sys.stdout.flush()

                for batch_idx, batch in enumerate(train_loader, start=1):
                    loss = trainer.train_step(batch)
                    running_loss += loss
                    steps_in_epoch += 1
                    global_step += 1

                    if batch_idx % args.logging_steps == 0 or batch_idx == len(train_loader):
                        elapsed = time.time() - training_start_time
                        avg_step_time = elapsed / max(1, steps_in_epoch if start_epoch == epoch else (global_step - (start_epoch - 1) * len(train_loader)))
                        remaining_steps = total_steps - global_step
                        eta_seconds = remaining_steps * avg_step_time
                        current_avg_loss = running_loss / steps_in_epoch

                        print(
                            f"  Epoch {epoch}/{config.training.epochs} | "
                            f"Batch {batch_idx:>3}/{len(train_loader)} | "
                            f"Global Step {global_step:>4}/{total_steps} | "
                            f"Loss: {loss:.4f} (avg: {current_avg_loss:.4f}) | "
                            f"Elapsed: {elapsed/60:.1f}m | "
                            f"ETA: {eta_seconds/60:.1f}m"
                        )
                        sys.stdout.flush()

                epoch_avg_loss = running_loss / max(1, steps_in_epoch)
                epoch_elapsed = time.time() - epoch_start_time
                epoch_losses.append(epoch_avg_loss)

                print(f"\n>>> Epoch {epoch} / {config.training.epochs} Complete! Average Loss: {epoch_avg_loss:.4f} (Time: {epoch_elapsed:.1f}s) <<<")
                print(f"Writing interruption-safe checkpoint for Epoch {epoch}...")
                sys.stdout.flush()

                ckpt_path = trainer.save_epoch_checkpoint(
                    epoch=epoch,
                    output_dir=save_dir_path,
                    condition=canon_condition,
                    epoch_loss=epoch_avg_loss,
                    global_step=global_step,
                )
                last_saved_epoch = epoch
                print(f"[CHECKPOINT SAVED] Verified checkpoint for Epoch {epoch} successfully written to: {ckpt_path}\n")
                sys.stdout.flush()

            total_training_time = time.time() - training_start_time
            print(f"\nTraining Complete in {total_training_time/60:.2f} minutes.")
            sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n" + "!" * 80)
            print("[INTERRUPTED] Training paused by user or system interruption.")
            print(f"  Safely completed and verified checkpoint on disk: Epoch {last_saved_epoch}")
            print(f"  Checkpoint directory: {save_dir_path}")
            print("  You can safely resume training from this checkpoint at any time.")
            print("!" * 80)
            sys.stdout.flush()
            return 130

    # 5. Save Final Consolidated Model Checkpoint
    print("\n" + "=" * 80)
    print(f"SAVING CONSOLIDATED CHECKPOINT: {save_dir_path}")
    print("=" * 80)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    trainer.classifier.save_pretrained(save_dir_path)
    print(f"  [SAVED] Consolidated model weights and tokenizer saved to: {save_dir_path}\n")
    sys.stdout.flush()

    # 6. Evaluation A: CLEAN SST-2 dev set, UNDEFENDED -> CACC
    print("=" * 80)
    print("EVALUATION A: Clean SST-2 Dev Set (Clean Accuracy - CACC)")
    print(f"  Evaluating {len(clean_dev_raw)} clean samples with model.eval()...")
    print("=" * 80)
    sys.stdout.flush()
    
    clean_eval_start = time.time()
    clean_predictions: List[Dict[str, Any]] = []
    clean_correct = 0
    total_clean_samples = len(clean_dev_raw)

    trainer.classifier.model.eval()
    with torch.no_grad():
        for batch in clean_dev_loader:
            input_ids = batch["input_ids"].to(trainer.device)
            attention_mask = batch["attention_mask"].to(trainer.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(trainer.device)
            labels = batch["labels"].to(trainer.device)

            outputs = trainer.classifier.forward(
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
                if is_correct:
                    clean_correct += 1

                clean_predictions.append({
                    "sample_id": batch["sample_ids"][i],
                    "text": batch["processed_texts"][i],
                    "ground_truth_label": label_val,
                    "predicted_label": pred_val,
                    "prob_class_0": float(probs[i][0].item()),
                    "prob_class_1": float(probs[i][1].item()),
                    "is_correct": is_correct,
                })

    cacc = (clean_correct / total_clean_samples) * 100.0
    clean_eval_time = time.time() - clean_eval_start
    print(f"  Clean Samples Evaluated: {total_clean_samples}")
    print(f"  Clean Correct:           {clean_correct}")
    print(f"  Clean Accuracy (CACC):   {cacc:.2f}% (computed in {clean_eval_time:.1f}s)")

    # Save clean predictions
    clean_preds_file = save_dir_path / "clean_dev_predictions.json"
    with open(clean_preds_file, "w", encoding="utf-8") as f:
        json.dump(clean_predictions, f, indent=2)
    print(f"  [SAVED] Clean predictions saved to: {clean_preds_file}")

    # 7. Evaluation B: ONION Poisoned SST-2 test set, UNDEFENDED -> ASR
    print("\n" + "=" * 80)
    print("EVALUATION B: ONION Poisoned SST-2 Test Set (UNDEFENDED)")
    print("=" * 80)
    
    poison_eval_start = time.time()
    poison_predictions: List[Dict[str, Any]] = []
    poison_attack_success = 0
    total_poison_samples = len(badnets_test_raw)
    target_label = config.backdoor.target_label  # verified target label 1

    with torch.no_grad():
        for batch in badnets_test_loader:
            input_ids = batch["input_ids"].to(trainer.device)
            attention_mask = batch["attention_mask"].to(trainer.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(trainer.device)
            labels = batch["labels"].to(trainer.device)

            outputs = trainer.classifier.forward(
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
                # Attack success: model predicts target label (1) on poisoned input
                attack_successful = (pred_val == target_label)
                if attack_successful:
                    poison_attack_success += 1

                poison_predictions.append({
                    "sample_id": batch["sample_ids"][i],
                    "text": batch["processed_texts"][i],
                    "original_label": 0,
                    "poisoned_label": label_val,
                    "target_label": target_label,
                    "predicted_label": pred_val,
                    "prob_class_0": float(probs[i][0].item()),
                    "prob_class_1": float(probs[i][1].item()),
                    "attack_successful": attack_successful,
                })

    asr = (poison_attack_success / total_poison_samples) * 100.0
    poison_eval_time = time.time() - poison_eval_start
    print(f"  Poisoned Samples Evaluated:  {total_poison_samples}")
    print(f"  Target Label:                {target_label}")
    print(f"  Attack Successful Count:     {poison_attack_success}")
    print(f"  Attack Success Rate (ASR):   {asr:.2f}% (computed in {poison_eval_time:.1f}s)")

    # Save poisoned predictions
    poison_preds_file = save_dir_path / "poisoned_test_predictions.json"
    with open(poison_preds_file, "w", encoding="utf-8") as f:
        json.dump(poison_predictions, f, indent=2)
    print(f"  [SAVED] Poisoned predictions saved to: {poison_preds_file}")

    # 8. Save Complete Experiment & Evaluation Manifest
    experiment_summary = {
        "run_status": "COMPLETED",
        "experiment_condition": canon_condition,
        "description": SUPPORTED_CONDITIONS[canon_condition]["description"],
        "dataset_name": config.dataset.name,
        "clean_dir": str(clean_dir_path),
        "badnets_dir": str(badnets_dir_path),
        "model_name": config.model.model_name,
        "device": str(trainer.device),
        "random_seed": config.training.seed,
        "last_completed_epoch": last_saved_epoch,
        "hyperparameters": {
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "eval_batch_size": config.training.eval_batch_size,
            "learning_rate": config.training.learning_rate,
            "max_seq_length": config.model.max_seq_length,
            "weight_decay": config.training.weight_decay,
            "max_grad_norm": config.training.max_grad_norm,
        },
        "training_history": {
            "epoch_losses": epoch_losses,
            "total_steps": global_step,
            "training_time_seconds": total_training_time,
        },
        "backdoor_specification": {
            "attack_name": config.backdoor.attack_name,
            "verified_triggers": list(config.backdoor.triggers),
            "target_label": config.backdoor.target_label,
            "source_label": config.backdoor.source_label,
        },
        "measured_metrics": {
            "clean_accuracy_cacc": cacc,
            "clean_dev_samples": total_clean_samples,
            "clean_dev_correct": clean_correct,
            "attack_success_rate_asr": asr,
            "poisoned_test_samples": total_poison_samples,
            "poisoned_attack_success": poison_attack_success,
        },
        "checkpoint_path": str(save_dir_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    manifest_file = save_dir_path / "evaluation_summary.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(experiment_summary, f, indent=2)
    print(f"  [SAVED] Complete evaluation manifest saved to: {manifest_file}")

    # 9. Final Results Summary
    print("\n" + "=" * 80)
    print("FINAL EXPERIMENT RESULTS: BACKDOORED BERT BASELINE (UNDEFENDED)")
    print("=" * 80)
    print(f"  Checkpoint Path:             {save_dir_path}")
    print(f"  Clean Accuracy (CACC):       {cacc:.2f}% ({clean_correct}/{total_clean_samples})")
    print(f"  Attack Success Rate (ASR):   {asr:.2f}% ({poison_attack_success}/{total_poison_samples})")
    print(f"  Clean Samples Evaluated:     {total_clean_samples}")
    print(f"  Poisoned Samples Evaluated:  {total_poison_samples}")
    print(f"  Target Label:                {target_label}")
    print(f"  Verified Triggers:           {config.backdoor.triggers}")
    print(f"  Training Time:               {total_training_time/60:.2f} minutes")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(run_training_experiment())
