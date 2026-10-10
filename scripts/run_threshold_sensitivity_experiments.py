"""
Threshold sensitivity experiment for Unknown Trigger Candidate Detection.

Evaluates the detector across thresholds [0.15, 0.20, 0.25, 0.30] using:
- Same frozen UNDEFINED BERT checkpoint
- Same ONION SST-2 BadNets test data (100 samples/trigger)
- Same leave-one-trigger-out protocol
- Same clean control samples (100 samples)
- Same top_k = 5 and scoring formula

Outputs are isolated in results/experimental/unknown_trigger/threshold_sensitivity/.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_loader import load_config
from dataset.onion_loader import ONIONDatasetLoader
from dataset.schema import DatasetSample
from experimental.unknown_trigger.scoring import rank_and_filter_candidates
from experimental.unknown_trigger.token_influence import TokenInfluenceAnalyzer
from models.bert_classifier import BertClassifier

THRESHOLDS = [0.15, 0.20, 0.25, 0.30]


def run_threshold_sensitivity(
    classifier: BertClassifier,
    loader: ONIONDatasetLoader,
    output_dir: Path,
    thresholds: List[float] = THRESHOLDS,
    max_test_samples: int = 100,
    max_clean_samples: int = 100,
    top_k: int = 5,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_triggers = sorted(list(loader.triggers))

    badnets_test = loader.load_split("test", source="badnets")
    clean_test = loader.load_split("test", source="clean")

    # -------------------------------------------------------------------------
    # 1. Evaluate Leave-One-Trigger-Out across thresholds
    # -------------------------------------------------------------------------
    print("Evaluating Leave-One-Trigger-Out experiments across thresholds...")
    loo_by_threshold: Dict[float, Dict[str, Dict[str, Any]]] = {th: {} for th in thresholds}

    for hidden_trigger in all_triggers:
        known_triggers = [t for t in all_triggers if t != hidden_trigger]
        analyzer = TokenInfluenceAnalyzer(
            classifier=classifier,
            target_label=1,
            known_triggers=known_triggers,
        )

        pattern = re.compile(rf"\b{re.escape(hidden_trigger)}\b", re.IGNORECASE)
        matching_samples = [s for s in badnets_test if pattern.search(s.text)][:max_test_samples]
        total_samples = len(matching_samples)
        print(f"  Trigger '{hidden_trigger}': analyzing {total_samples} samples...")

        # Precompute token influences once per sample
        cached_influences = []
        for sample in matching_samples:
            infl = analyzer.analyze_sentence(sample.text)
            cached_influences.append((sample, infl))

        # Evaluate across each threshold
        for th in thresholds:
            hidden_detected_count = 0
            ranks: List[int] = []
            total_candidates_count = 0
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            total_benign_tokens = 0

            for sample, infl in cached_influences:
                cands = rank_and_filter_candidates(
                    influences=infl,
                    target_label=1,
                    min_score_threshold=th,
                    top_k=top_k,
                    exclude_known_triggers=True,
                )
                total_candidates_count += len(cands)

                words = [m.group().lower() for m in re.finditer(r"\S+", sample.text)]
                num_hidden_in_text = sum(1 for w in words if w == hidden_trigger.lower())
                benign_in_text = len(words) - num_hidden_in_text
                total_benign_tokens += benign_in_text

                found_hidden_in_sample = False
                sample_tp = 0
                sample_fp = 0

                for cand in cands:
                    cand_clean = re.sub(r"^[^\w]+|[^\w]+$", "", cand.token).lower()
                    if cand_clean == hidden_trigger.lower() or cand.token.lower() == hidden_trigger.lower():
                        sample_tp += 1
                        if not found_hidden_in_sample:
                            found_hidden_in_sample = True
                            ranks.append(cand.rank)
                    else:
                        sample_fp += 1

                if found_hidden_in_sample:
                    hidden_detected_count += 1

                true_positives += sample_tp
                false_positives += sample_fp
                sample_fn = max(0, num_hidden_in_text - sample_tp)
                false_negatives += sample_fn

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            fpr = false_positives / total_benign_tokens if total_benign_tokens > 0 else 0.0
            det_rate = hidden_detected_count / total_samples if total_samples > 0 else 0.0
            avg_rank = (sum(ranks) / len(ranks)) if ranks else None
            avg_cands = total_candidates_count / total_samples if total_samples > 0 else 0.0

            loo_by_threshold[th][hidden_trigger] = {
                "hidden_trigger": hidden_trigger,
                "threshold": th,
                "total_samples": total_samples,
                "hidden_detected_count": hidden_detected_count,
                "hidden_trigger_detection_rate": round(det_rate, 4),
                "average_rank_when_detected": round(avg_rank, 2) if avg_rank is not None else None,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "false_positive_rate": round(fpr, 4),
                "avg_candidates_per_sample": round(avg_cands, 2),
            }

    # -------------------------------------------------------------------------
    # 2. Evaluate Clean Controls across thresholds
    # -------------------------------------------------------------------------
    print("Evaluating Clean Controls across thresholds...")
    clean_eval_samples = clean_test[:max_clean_samples]
    ctrl_analyzer = TokenInfluenceAnalyzer(
        classifier=classifier,
        target_label=1,
        known_triggers=all_triggers,
    )

    clean_cached_influences = []
    for sample in clean_eval_samples:
        infl = ctrl_analyzer.analyze_sentence(sample.text)
        clean_cached_influences.append((sample, infl))

    ctrl_by_threshold: Dict[float, Dict[str, Any]] = {}
    total_clean_samples = len(clean_eval_samples)

    for th in thresholds:
        total_tokens = 0
        total_fps = 0
        samples_with_fps = 0

        for sample, infl in clean_cached_influences:
            words = sample.text.strip().split()
            total_tokens += len(words)
            cands = rank_and_filter_candidates(
                influences=infl,
                target_label=1,
                min_score_threshold=th,
                top_k=top_k,
                exclude_known_triggers=True,
            )
            if len(cands) > 0:
                samples_with_fps += 1
                total_fps += len(cands)

        token_fpr = total_fps / total_tokens if total_tokens > 0 else 0.0
        sample_fpr = samples_with_fps / total_clean_samples if total_clean_samples > 0 else 0.0

        ctrl_by_threshold[th] = {
            "threshold": th,
            "total_samples": total_clean_samples,
            "total_tokens_evaluated": total_tokens,
            "false_positive_candidates_count": total_fps,
            "samples_with_false_positives": samples_with_fps,
            "token_false_positive_rate": round(token_fpr, 4),
            "sample_false_positive_rate": round(sample_fpr, 4),
        }

    # -------------------------------------------------------------------------
    # 3. Aggregate Summary Across Thresholds
    # -------------------------------------------------------------------------
    summary_by_threshold: Dict[str, Any] = {}
    for th in thresholds:
        trigger_metrics = loo_by_threshold[th]
        det_rates = [v["hidden_trigger_detection_rate"] for v in trigger_metrics.values()]
        precisions = [v["precision"] for v in trigger_metrics.values()]
        recalls = [v["recall"] for v in trigger_metrics.values()]
        f1s = [v["f1"] for v in trigger_metrics.values()]
        valid_ranks = [v["average_rank_when_detected"] for v in trigger_metrics.values() if v["average_rank_when_detected"] is not None]
        avg_cands_list = [v["avg_candidates_per_sample"] for v in trigger_metrics.values()]

        summary_by_threshold[str(th)] = {
            "threshold": th,
            "mean_hidden_trigger_detection_rate": round(sum(det_rates) / len(det_rates), 4),
            "mean_candidate_precision": round(sum(precisions) / len(precisions), 4),
            "mean_candidate_recall": round(sum(recalls) / len(recalls), 4),
            "mean_candidate_f1": round(sum(f1s) / len(f1s), 4),
            "mean_average_rank_when_detected": round(sum(valid_ranks) / len(valid_ranks), 2) if valid_ranks else None,
            "mean_avg_candidates_per_sample": round(sum(avg_cands_list) / len(avg_cands_list), 2),
            "clean_token_false_positive_rate": ctrl_by_threshold[th]["token_false_positive_rate"],
            "clean_sample_false_positive_rate": ctrl_by_threshold[th]["sample_false_positive_rate"],
            "triggers_detail": trigger_metrics,
        }

    # -------------------------------------------------------------------------
    # 4. Save JSON Results
    # -------------------------------------------------------------------------
    with open(output_dir / "threshold_sensitivity_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "leave_one_out": {str(k): v for k, v in loo_by_threshold.items()},
            "clean_controls": {str(k): v for k, v in ctrl_by_threshold.items()},
        }, f, indent=2)

    with open(output_dir / "threshold_sensitivity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_by_threshold, f, indent=2)

    # -------------------------------------------------------------------------
    # 5. Generate Markdown Table
    # -------------------------------------------------------------------------
    md_content = generate_markdown_table(summary_by_threshold, loo_by_threshold, ctrl_by_threshold)
    with open(output_dir / "threshold_sensitivity_table.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    # -------------------------------------------------------------------------
    # 6. Generate Trade-Off Visualization Figure
    # -------------------------------------------------------------------------
    generate_tradeoff_plot(summary_by_threshold, output_dir / "threshold_sensitivity.png")

    return {
        "summary": summary_by_threshold,
        "results_file": str(output_dir / "threshold_sensitivity_results.json"),
        "summary_file": str(output_dir / "threshold_sensitivity_summary.json"),
        "table_file": str(output_dir / "threshold_sensitivity_table.md"),
        "figure_file": str(output_dir / "threshold_sensitivity.png"),
    }


def generate_markdown_table(
    summary: Dict[str, Any],
    loo: Dict[float, Dict[str, Any]],
    ctrl: Dict[float, Any],
) -> str:
    lines = [
        "# Threshold Sensitivity Analysis — Unknown Trigger Candidate Detection",
        "",
        "## Overall Summary Across Thresholds",
        "",
        "| Threshold | Hidden Det. Rate | Precision | Recall | F1 Score | Avg Rank | Clean Token FPR | Clean Sample FPR | Avg Cands/Sample |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for th_str, s in summary.items():
        lines.append(
            f"| **{s['threshold']:.2f}** | "
            f"{s['mean_hidden_trigger_detection_rate']*100:.2f}% | "
            f"{s['mean_candidate_precision']*100:.2f}% | "
            f"{s['mean_candidate_recall']*100:.2f}% | "
            f"{s['mean_candidate_f1']*100:.2f}% | "
            f"{s['mean_average_rank_when_detected']:.2f} | "
            f"{s['clean_token_false_positive_rate']*100:.2f}% | "
            f"{s['clean_sample_false_positive_rate']*100:.2f}% | "
            f"{s['mean_avg_candidates_per_sample']:.2f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Per-Trigger Breakdown Across Thresholds",
        "",
        "| Trigger | Threshold | Test Samples | Detected | Det. Rate | Precision | Recall | F1 Score | Avg Rank | Avg Cands |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for th in THRESHOLDS:
        for trig in sorted(loo[th].keys()):
            d = loo[th][trig]
            rank_str = f"{d['average_rank_when_detected']:.2f}" if d['average_rank_when_detected'] is not None else "N/A"
            lines.append(
                f"| `{trig}` | {th:.2f} | {d['total_samples']} | {d['hidden_detected_count']} | "
                f"{d['hidden_trigger_detection_rate']*100:.2f}% | {d['precision']*100:.2f}% | "
                f"{d['recall']*100:.2f}% | {d['f1']*100:.2f}% | {rank_str} | {d['avg_candidates_per_sample']:.2f} |"
            )

    return "\n".join(lines) + "\n"


def generate_tradeoff_plot(summary: Dict[str, Any], save_path: Path):
    thresholds = [s["threshold"] for s in summary.values()]
    det_rates = [s["mean_hidden_trigger_detection_rate"] * 100 for s in summary.values()]
    f1_scores = [s["mean_candidate_f1"] * 100 for s in summary.values()]
    precisions = [s["mean_candidate_precision"] * 100 for s in summary.values()]
    token_fprs = [s["clean_token_false_positive_rate"] * 100 for s in summary.values()]
    sample_fprs = [s["clean_sample_false_positive_rate"] * 100 for s in summary.values()]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Subplot 1: Detection Performance Trade-Off
    ax1.plot(thresholds, det_rates, marker="o", linewidth=2.2, color="#1f77b4", label="Hidden Detection Rate (%)")
    ax1.plot(thresholds, f1_scores, marker="s", linewidth=2.2, color="#2ca02c", label="Candidate F1 Score (%)")
    ax1.plot(thresholds, precisions, marker="^", linewidth=2.2, color="#ff7f0e", label="Candidate Precision (%)")
    ax1.set_title("Candidate Discovery Performance vs. Threshold", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Candidate Influence Threshold (θ)", fontsize=11)
    ax1.set_ylabel("Percentage (%)", fontsize=11)
    ax1.set_xticks(thresholds)
    ax1.set_ylim(50, 105)
    ax1.legend(frameon=True, loc="lower left", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Subplot 2: False Positive Restraint
    ax2.plot(thresholds, sample_fprs, marker="D", linewidth=2.2, color="#d62728", label="Clean Sample FPR (%)")
    ax2.plot(thresholds, token_fprs, marker="v", linewidth=2.2, color="#9467bd", label="Clean Token FPR (%)")
    ax2.set_title("Clean False Positive Rate vs. Threshold", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Candidate Influence Threshold (θ)", fontsize=11)
    ax2.set_ylabel("False Positive Rate (%)", fontsize=11)
    ax2.set_xticks(thresholds)
    ax2.set_ylim(0, 25)
    ax2.legend(frameon=True, loc="upper right", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


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

    output_dir = PROJECT_ROOT / "results" / "experimental" / "unknown_trigger" / "threshold_sensitivity"
    print(f"Running Threshold Sensitivity Experiment for thresholds {THRESHOLDS}...")
    res = run_threshold_sensitivity(
        classifier=classifier,
        loader=loader,
        output_dir=output_dir,
        thresholds=THRESHOLDS,
        max_test_samples=100,
        max_clean_samples=100,
        top_k=5,
    )

    print("\n" + "=" * 78)
    print("THRESHOLD SENSITIVITY EXPERIMENT COMPLETED")
    print("=" * 78)
    summary = res["summary"]
    for th_str, s in summary.items():
        print(
            f"Threshold θ={s['threshold']:.2f} -> "
            f"Det Rate: {s['mean_hidden_trigger_detection_rate']*100:.2f}% | "
            f"F1: {s['mean_candidate_f1']*100:.2f}% | "
            f"Prec: {s['mean_candidate_precision']*100:.2f}% | "
            f"Token FPR: {s['clean_token_false_positive_rate']*100:.2f}% | "
            f"Sample FPR: {s['clean_sample_false_positive_rate']*100:.2f}%"
        )
    print(f"\nArtifacts saved in: {output_dir}")


if __name__ == "__main__":
    main()
