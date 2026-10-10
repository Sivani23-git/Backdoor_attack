"""
Execution Script for Trigger-Agnostic Backdoor Defense Evaluation.

Runs the complete Three-Way Comparative Benchmark:
A. Undefended frozen BERT
B. Known-Trigger Baseline (Reference)
C. NEW Trigger-Agnostic Defense (Vocabulary-Free)

Measures:
- Candidate Detection (Precision, Recall, F1, Detection Rate, Top-1 Dominance, Mean Rank)
- Clean False Positives (Token FPR, Sample FPR, Benign control word behavior)
- Defense Efficacy (CACC, ASR before/after REMOVE and MASK, Absolute ASR drop, CAD)
- Explainability Attribution Shifts (Original vs Cleaned)

Saves all artifacts under results/experimental/trigger_agnostic/.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_loader import load_config
from dataset.onion_loader import ONIONDatasetLoader
from experimental.trigger_agnostic.defense_pipeline import TriggerAgnosticDefensePipeline
from experimental.trigger_agnostic.evaluation import TriggerAgnosticEvaluator
from experimental.trigger_agnostic.explainability import AgnosticExplainabilityAnalyzer
from experimental.trigger_agnostic.multicontext_validator import MultiContextValidator
from models.bert_classifier import BertClassifier


def plot_threeway_comparison(
    results: Dict[str, Any],
    save_path: Path,
):
    """Generates comparative bar chart of CACC, ASR, and CAD across conditions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

    cond_a = results["three_way_comparison"]["condition_a_undefended"]
    cond_b = results["three_way_comparison"]["condition_b_known_baseline"]
    cond_c = results["three_way_comparison"]["condition_c_trigger_agnostic"]

    conditions = [
        "Undefended\n(Frozen BERT)",
        "Known-Trigger\n(REMOVE)",
        "Known-Trigger\n(MASK)",
        "Trigger-Agnostic\n(REMOVE)",
        "Trigger-Agnostic\n(MASK)",
    ]

    cacc_vals = [
        cond_a["cacc"],
        cond_b["remove"]["cacc"],
        cond_b["mask"]["cacc"],
        cond_c["remove"]["cacc"],
        cond_c["mask"]["cacc"],
    ]

    asr_vals = [
        cond_a["asr"],
        cond_b["remove"]["asr"],
        cond_b["mask"]["asr"],
        cond_c["remove"]["asr"],
        cond_c["mask"]["asr"],
    ]

    x = np.arange(len(conditions))
    width = 0.35

    # Panel 1: CACC vs ASR
    rects1 = ax1.bar(x - width/2, cacc_vals, width, label="CACC (%)", color="#2ca02c", edgecolor="black", alpha=0.85)
    rects2 = ax1.bar(x + width/2, asr_vals, width, label="ASR (%)", color="#d62728", edgecolor="black", alpha=0.85)

    ax1.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Three-Way Defense Comparison (CACC vs ASR)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel 2: ASR Reduction vs Clean Degradation (CAD)
    asr_red = [
        0.0,
        cond_b["remove"]["absolute_asr_reduction"],
        cond_b["mask"]["absolute_asr_reduction"],
        cond_c["remove"]["absolute_asr_reduction"],
        cond_c["mask"]["absolute_asr_reduction"],
    ]

    cad_vals = [
        0.0,
        cond_b["remove"]["cad"],
        cond_b["mask"]["cad"],
        cond_c["remove"]["cad"],
        cond_c["mask"]["cad"],
    ]

    rects3 = ax2.bar(x - width/2, asr_red, width, label="ASR Reduction (%)", color="#1f77b4", edgecolor="black", alpha=0.85)
    rects4 = ax2.bar(x + width/2, cad_vals, width, label="CAD (%)", color="#ff7f0e", edgecolor="black", alpha=0.85)

    ax2.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax2.set_title("ASR Reduction vs Clean Accuracy Degradation (CAD)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, fontsize=9, fontweight="bold")
    ax2.set_ylim(-2, 105)
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects3:
        h = rect.get_height()
        if h > 0:
            ax2.annotate(f"+{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    for rect in rects4:
        h = rect.get_height()
        ax2.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, max(0, h)), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def generate_markdown_report(results: Dict[str, Any]) -> str:
    det = results["candidate_detection"]
    ctrl = results["clean_controls"]
    comp = results["three_way_comparison"]

    a = comp["condition_a_undefended"]
    b_rem = comp["condition_b_known_baseline"]["remove"]
    b_mask = comp["condition_b_known_baseline"]["mask"]
    c_rem = comp["condition_c_trigger_agnostic"]["remove"]
    c_mask = comp["condition_c_trigger_agnostic"]["mask"]

    lines = [
        "# Trigger-Agnostic Backdoor Defense Benchmark Report",
        "",
        "## 1. Executive Three-Way Defense Comparison",
        "",
        "| Evaluation Condition | Vocabulary Used | Strategy | Clean Accuracy (CACC) | Attack Success Rate (ASR) | Absolute ASR Drop | Clean Accuracy Degradation (CAD) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **A. Undefended Frozen BERT** | None | Raw | {a['cacc']:.2f}% | {a['asr']:.2f}% | — | — |",
        f"| **B. Known-Trigger Baseline** | `['bb','cf','mn','tq','mb']` | REMOVE | {b_rem['cacc']:.2f}% | {b_rem['asr']:.2f}% | **-{b_rem['absolute_asr_reduction']:.2f}%** | {b_rem['cad']:.2f}% |",
        f"| **B. Known-Trigger Baseline** | `['bb','cf','mn','tq','mb']` | MASK | {b_mask['cacc']:.2f}% | {b_mask['asr']:.2f}% | **-{b_mask['absolute_asr_reduction']:.2f}%** | {b_mask['cad']:.2f}% |",
        f"| **C. NEW Trigger-Agnostic Defense** | **NONE (Vocabulary-Free)** | **REMOVE** | **{c_rem['cacc']:.2f}%** | **{c_rem['asr']:.2f}%** | **-{c_rem['absolute_asr_reduction']:.2f}%** | **{c_rem['cad']:.2f}%** |",
        f"| **C. NEW Trigger-Agnostic Defense** | **NONE (Vocabulary-Free)** | **MASK** | **{c_mask['cacc']:.2f}%** | **{c_mask['asr']:.2f}%** | **-{c_mask['absolute_asr_reduction']:.2f}%** | **{c_mask['cad']:.2f}%** |",
        "",
        "---",
        "",
        "## 2. Trigger-Agnostic Candidate Detection Efficacy (Hidden Triggers)",
        "",
        f"- **Hidden Trigger Detection Rate**: **{det['detection_rate']*100:.2f}%** ({det['detected_samples']} / {det['total_poisoned_samples']})",
        f"- **Candidate Precision**: **{det['precision']*100:.2f}%**",
        f"- **Candidate Recall**: **{det['recall']*100:.2f}%**",
        f"- **Candidate F1 Score**: **{det['f1_score']*100:.2f}%**",
        f"- **Top-1 Detection Dominance Rate**: **{det['top1_detection_rate']*100:.2f}%**",
        f"- **Mean Candidate Rank**: **{det['average_rank_when_detected']:.2f}**",
        "",
        "### Per-Trigger Hidden Detection Breakdown",
        "",
        "| Hidden Trigger | Total Samples | Flagged Samples | Detection Rate | Top-1 Count | Top-1 Rate | Mean Rank |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for trig, st in sorted(det["per_trigger_breakdown"].items()):
        m_rank = f"{st['mean_rank']:.2f}" if st["mean_rank"] is not None else "N/A"
        lines.append(
            f"| `{trig}` | {st['total_samples']} | {st['detected']} | **{st['detection_rate']*100:.2f}%** | {st['top1_count']} | {st['top1_rate']*100:.2f}% | {m_rank} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Clean SST-2 Controls and Benign Word False Positive Analysis",
        "",
        f"- **Clean Token-Level FPR**: **{ctrl['clean_token_fpr']*100:.2f}%** ({ctrl['total_tokens_flagged_confirmed']} / {ctrl['total_tokens_evaluated']})",
        f"- **Clean Sample-Level FPR**: **{ctrl['clean_sample_fpr']*100:.2f}%** ({ctrl['clean_samples_flagged_confirmed']} / {ctrl['total_clean_samples']})",
        f"- **Clean Samples Modified**: **{ctrl['clean_samples_modified']}** / {ctrl['total_clean_samples']}",
        "",
        "### Benign Influential Sentiment Words Tracking",
        "",
        "| Benign Word | Occurrences Evaluated | Confirmed As Trigger | Uncertain (Abstained) | Mean Influence Score | False Removal Rate |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for w, st in sorted(ctrl["target_benign_words_analysis"].items()):
        occ = st["occurrences"]
        conf = st["flagged_confirmed"]
        unc = st["flagged_uncertain"]
        score = st["mean_score"]
        fpr = (conf / occ * 100) if occ > 0 else 0.0
        lines.append(
            f"| `{w}` | {occ} | {conf} | {unc} | {score:.4f} | {fpr:.2f}% |"
        )

    return "\n".join(lines) + "\n"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
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

    output_dir = PROJECT_ROOT / "results" / "experimental" / "trigger_agnostic"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing Trigger-Agnostic Evaluator...")
    evaluator = TriggerAgnosticEvaluator(
        classifier=classifier,
        loader=loader,
        config=config,
        target_label=1,
        confirm_threshold=0.50,
        uncertain_threshold=0.20,
    )

    badnets_test = loader.load_split("test", source="badnets")
    clean_test = loader.load_split("test", source="clean")

    # Representative balanced test sets (250 poisoned, 250 clean) for robust benchmark
    poison_eval = badnets_test[:250]
    clean_eval = clean_test[:250]

    print(f"\n1. Running Candidate Detection Evaluation on {len(poison_eval)} poisoned samples...", flush=True)
    cand_results = evaluator.evaluate_candidate_detection_on_poisoned(poison_eval)
    print(f"   [DONE] Detected {cand_results['detected_samples']}/{cand_results['total_poisoned_samples']} ({cand_results['detection_rate']*100:.2f}%) | Precision: {cand_results['precision']*100:.2f}% | Recall: {cand_results['recall']*100:.2f}%", flush=True)

    print(f"\n2. Running Clean Controls Evaluation on {len(clean_eval)} clean samples...", flush=True)
    clean_results = evaluator.evaluate_clean_controls(clean_eval)
    print(f"   [DONE] Token FPR: {clean_results['clean_token_fpr']*100:.2f}% | Sample FPR: {clean_results['clean_sample_fpr']*100:.2f}% | Modified: {clean_results['clean_samples_modified']}/{clean_results['total_clean_samples']}", flush=True)

    print(f"\n3. Running Three-Way Benchmark Comparison across Conditions A, B, and C...", flush=True)
    comparison_results = evaluator.run_three_way_comparison(
        clean_test_samples=clean_eval,
        poisoned_test_samples=poison_eval,
    )
    print("   [DONE] Three-way comparison finished.", flush=True)

    full_results = {
        "benchmark_name": "Trigger-Agnostic Backdoor Defense Comprehensive Evaluation",
        "description": "Vocabulary-free behavioral candidate discovery, multi-context validation, and three-way comparative benchmark.",
        "candidate_detection": cand_results,
        "clean_controls": clean_results,
        "three_way_comparison": comparison_results,
    }

    # Save JSON results
    with open(output_dir / "trigger_agnostic_results.json", "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    summary_data = {
        "candidate_detection_rate": cand_results["detection_rate"],
        "candidate_precision": cand_results["precision"],
        "candidate_recall": cand_results["recall"],
        "candidate_f1": cand_results["f1_score"],
        "clean_token_fpr": clean_results["clean_token_fpr"],
        "clean_sample_fpr": clean_results["clean_sample_fpr"],
        "three_way_comparison": comparison_results,
    }
    with open(output_dir / "trigger_agnostic_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Save Markdown Table
    md_content = generate_markdown_report(full_results)
    with open(output_dir / "trigger_agnostic_table.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    # Generate Comparative Visual Figure
    print(f"\n4. Generating Comparative Visual Figure...")
    plot_threeway_comparison(full_results, figures_dir / "trigger_agnostic_threeway_comparison.png")

    # Generate Explainability Sample Demonstration
    print(f"\n5. Generating Explainability Attribution Demonstration...")
    explainability_analyzer = AgnosticExplainabilityAnalyzer(
        classifier=classifier,
        pipeline=evaluator.pipeline_remove,
        target_label=1,
    )
    rep_sample = poison_eval[0]
    exp_res = explainability_analyzer.explain_sample(rep_sample.text, sample_id=rep_sample.sample_id)
    explainability_analyzer.plot_attribution_comparison(
        exp_res,
        figures_dir / "trigger_agnostic_explainability_sample.png",
    )

    print("\n" + "=" * 80)
    print("TRIGGER-AGNOSTIC DEFENSE BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 80)
    c_rem = comparison_results["condition_c_trigger_agnostic"]["remove"]
    c_mask = comparison_results["condition_c_trigger_agnostic"]["mask"]
    print(f"Trigger-Agnostic REMOVE: ASR = {c_rem['asr']:.2f}% (Reduction = -{c_rem['absolute_asr_reduction']:.2f}%), CACC = {c_rem['cacc']:.2f}%, CAD = {c_rem['cad']:.2f}%")
    print(f"Trigger-Agnostic MASK:   ASR = {c_mask['asr']:.2f}% (Reduction = -{c_mask['absolute_asr_reduction']:.2f}%), CACC = {c_mask['cacc']:.2f}%, CAD = {c_mask['cad']:.2f}%")
    print(f"Candidate Detection Rate: {cand_results['detection_rate']*100:.2f}% | Precision: {cand_results['precision']*100:.2f}% | Recall: {cand_results['recall']*100:.2f}%")
    print(f"Clean Token FPR: {clean_results['clean_token_fpr']*100:.2f}% | Clean Sample FPR: {clean_results['clean_sample_fpr']*100:.2f}%")
    print(f"Artifacts saved in: {output_dir}")


if __name__ == "__main__":
    main()
