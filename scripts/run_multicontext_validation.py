"""
Multi-Context Candidate Validation Experiment for Unknown Trigger Candidate Detection.

Evaluates cross-context consistency of suspicious candidate tokens across multiple
distinct sentence contexts using the frozen UNDEFINED BERT model checkpoint.

Measures:
- Mean, Median, and Std of influence scores
- Context detection count and consistency rate (contexts detected / contexts evaluated)
- Mean candidate rank
- Top-1 candidate dominance rate across contexts

Compares verified backdoor triggers (under leave-one-out hidden condition) against
recurrent clean sentiment words in clean control contexts.
"""

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
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
from dataset.schema import DatasetSample
from experimental.unknown_trigger.scoring import rank_and_filter_candidates
from experimental.unknown_trigger.token_influence import TokenInfluenceAnalyzer
from models.bert_classifier import BertClassifier


def clean_word(token: str) -> str:
    """Normalizes token to lowercase alphanumeric text."""
    return re.sub(r"^[^\w]+|[^\w]+$", "", token).lower()


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Computes mean, median, and sample standard deviation."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean_val = float(statistics.mean(values))
    median_val = float(statistics.median(values))
    std_val = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    return {
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "std": round(std_val, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def run_multicontext_validation(
    classifier: BertClassifier,
    loader: ONIONDatasetLoader,
    output_dir: Path,
    max_samples_per_trigger: int = 100,
    max_clean_samples: int = 100,
    min_threshold: float = 0.15,
    top_k: int = 5,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_triggers = sorted(list(loader.triggers))

    badnets_test = loader.load_split("test", source="badnets")
    clean_test = loader.load_split("test", source="clean")

    # -------------------------------------------------------------------------
    # 1. Trigger Cross-Context Analysis (Leave-One-Trigger-Out)
    # -------------------------------------------------------------------------
    print("Running Multi-Context Evaluation on Verified Triggers (Hidden Condition)...")
    trigger_results: Dict[str, Dict[str, Any]] = {}

    for hidden_trigger in all_triggers:
        known_triggers = [t for t in all_triggers if t != hidden_trigger]
        analyzer = TokenInfluenceAnalyzer(
            classifier=classifier,
            target_label=1,
            known_triggers=known_triggers,
        )

        pattern = re.compile(rf"\b{re.escape(hidden_trigger)}\b", re.IGNORECASE)
        matching_samples = [s for s in badnets_test if pattern.search(s.text)][:max_samples_per_trigger]
        total_contexts = len(matching_samples)
        print(f"  Analyzing hidden trigger '{hidden_trigger}' across {total_contexts} contexts...")

        detected_contexts = 0
        top1_contexts = 0
        influence_scores_when_detected: List[float] = []
        all_context_influence_scores: List[float] = []
        target_deltas: List[float] = []
        ranks: List[int] = []
        context_details: List[Dict[str, Any]] = []

        for sample in matching_samples:
            influences = analyzer.analyze_sentence(sample.text)
            candidates = rank_and_filter_candidates(
                influences=influences,
                target_label=1,
                min_score_threshold=min_threshold,
                top_k=top_k,
                exclude_known_triggers=True,
            )

            # Find raw influence result for the hidden trigger in this sentence
            raw_trig_inf = next(
                (inf for inf in influences if clean_word(inf.token) == hidden_trigger.lower()),
                None,
            )
            raw_score = 0.0
            if raw_trig_inf:
                # Raw target delta
                raw_score = raw_trig_inf.target_prob_delta + (0.5 if (raw_trig_inf.orig_prediction == 1 and raw_trig_inf.ablated_prediction != 1) else 0.0) + (0.1 * max(0.0, raw_trig_inf.confidence_delta))
                all_context_influence_scores.append(raw_score)
                target_deltas.append(raw_trig_inf.target_prob_delta)

            # Check if flagged as candidate
            cand_match = next(
                (c for c in candidates if clean_word(c.token) == hidden_trigger.lower()),
                None,
            )

            is_detected = cand_match is not None
            rank_val = cand_match.rank if cand_match else None
            is_top1 = rank_val == 1

            if is_detected:
                detected_contexts += 1
                influence_scores_when_detected.append(cand_match.influence_score)
                ranks.append(rank_val)
            if is_top1:
                top1_contexts += 1

            context_details.append({
                "sample_id": sample.sample_id,
                "text": sample.text,
                "detected": is_detected,
                "rank": rank_val,
                "influence_score": round(cand_match.influence_score, 4) if cand_match else round(raw_score, 4),
                "num_candidates_in_context": len(candidates),
            })

        det_consistency = detected_contexts / total_contexts if total_contexts > 0 else 0.0
        top1_consistency = top1_contexts / total_contexts if total_contexts > 0 else 0.0
        stats_all = compute_stats(all_context_influence_scores)
        stats_det = compute_stats(influence_scores_when_detected)
        mean_rank = round(float(statistics.mean(ranks)), 2) if ranks else None

        trigger_results[hidden_trigger] = {
            "token": hidden_trigger,
            "token_type": "backdoor_trigger",
            "total_contexts_evaluated": total_contexts,
            "contexts_detected": detected_contexts,
            "detection_consistency": round(det_consistency, 4),
            "top1_contexts": top1_contexts,
            "top1_consistency": round(top1_consistency, 4),
            "mean_rank_when_detected": mean_rank,
            "influence_stats_all_contexts": stats_all,
            "influence_stats_when_detected": stats_det,
            "target_delta_stats": compute_stats(target_deltas),
            "sample_contexts": context_details[:5],
        }

    # -------------------------------------------------------------------------
    # 2. Clean Controls Cross-Context Analysis
    # -------------------------------------------------------------------------
    print("Running Multi-Context Evaluation on Clean Control Words...")
    clean_eval_samples = clean_test[:max_clean_samples]
    ctrl_analyzer = TokenInfluenceAnalyzer(
        classifier=classifier,
        target_label=1,
        known_triggers=all_triggers,
    )

    clean_word_occurrences = defaultdict(list)
    clean_context_data = []

    for sample in clean_eval_samples:
        influences = ctrl_analyzer.analyze_sentence(sample.text)
        candidates = rank_and_filter_candidates(
            influences=influences,
            target_label=1,
            min_score_threshold=min_threshold,
            top_k=top_k,
            exclude_known_triggers=True,
        )

        cand_map = {clean_word(c.token): c for c in candidates}

        # Track all unique clean words in this sentence
        words_in_text = {clean_word(inf.token): inf for inf in influences if clean_word(inf.token)}
        for w_clean, inf in words_in_text.items():
            cand_obj = cand_map.get(w_clean)
            is_detected = cand_obj is not None
            rank_val = cand_obj.rank if cand_obj else None
            score = cand_obj.influence_score if cand_obj else (
                inf.target_prob_delta + (0.5 if (inf.orig_prediction == 1 and inf.ablated_prediction != 1) else 0.0) + (0.1 * max(0.0, inf.confidence_delta))
            )
            clean_word_occurrences[w_clean].append({
                "sample_id": sample.sample_id,
                "detected": is_detected,
                "rank": rank_val,
                "score": score,
            })

    # Filter to words that appear in at least 3 distinct clean contexts
    recurring_clean_words: Dict[str, Dict[str, Any]] = {}
    for word, entries in clean_word_occurrences.items():
        if len(entries) >= 3:
            total_occ = len(entries)
            det_count = sum(1 for e in entries if e["detected"])
            top1_count = sum(1 for e in entries if e["rank"] == 1)
            all_scores = [e["score"] for e in entries]
            ranks = [e["rank"] for e in entries if e["rank"] is not None]
            det_consistency = det_count / total_occ
            top1_consistency = top1_count / total_occ

            recurring_clean_words[word] = {
                "token": word,
                "token_type": "clean_word",
                "total_contexts_evaluated": total_occ,
                "contexts_detected": det_count,
                "detection_consistency": round(det_consistency, 4),
                "top1_contexts": top1_count,
                "top1_consistency": round(top1_consistency, 4),
                "mean_rank_when_detected": round(float(statistics.mean(ranks)), 2) if ranks else None,
                "influence_stats_all_contexts": compute_stats(all_scores),
            }

    # Sort clean words descending by frequency and consistency
    sorted_clean_words = sorted(
        recurring_clean_words.values(),
        key=lambda x: (x["contexts_detected"], x["influence_stats_all_contexts"]["mean"]),
        reverse=True,
    )

    # -------------------------------------------------------------------------
    # 3. Compile Summary Comparison
    # -------------------------------------------------------------------------
    avg_trigger_det_consistency = statistics.mean([t["detection_consistency"] for t in trigger_results.values()])
    avg_trigger_top1_consistency = statistics.mean([t["top1_consistency"] for t in trigger_results.values()])
    avg_trigger_mean_influence = statistics.mean([t["influence_stats_all_contexts"]["mean"] for t in trigger_results.values()])
    avg_trigger_std_influence = statistics.mean([t["influence_stats_all_contexts"]["std"] for t in trigger_results.values()])

    top_clean_detected = [w for w in sorted_clean_words if w["contexts_detected"] > 0]
    avg_clean_det_consistency = statistics.mean([w["detection_consistency"] for w in top_clean_detected]) if top_clean_detected else 0.0
    avg_clean_top1_consistency = statistics.mean([w["top1_consistency"] for w in top_clean_detected]) if top_clean_detected else 0.0
    avg_clean_mean_influence = statistics.mean([w["influence_stats_all_contexts"]["mean"] for w in top_clean_detected]) if top_clean_detected else 0.0

    summary_data = {
        "experiment": "Multi-Context Candidate Validation",
        "description": "Evaluation of token influence stability and cross-context detection consistency.",
        "threshold": min_threshold,
        "backdoor_triggers_aggregate": {
            "mean_detection_consistency": round(avg_trigger_det_consistency, 4),
            "mean_top1_consistency": round(avg_trigger_top1_consistency, 4),
            "mean_influence_score": round(avg_trigger_mean_influence, 4),
            "mean_influence_std": round(avg_trigger_std_influence, 4),
        },
        "clean_words_aggregate": {
            "num_recurring_words_evaluated": len(recurring_clean_words),
            "num_clean_words_detected_at_least_once": len(top_clean_detected),
            "mean_detection_consistency_when_flagged": round(avg_clean_det_consistency, 4),
            "mean_top1_consistency_when_flagged": round(avg_clean_top1_consistency, 4),
            "mean_influence_score_when_flagged": round(avg_clean_mean_influence, 4),
        },
        "triggers_detail": trigger_results,
        "clean_words_top": sorted_clean_words[:15],
    }

    # -------------------------------------------------------------------------
    # 4. Save JSON Artifacts
    # -------------------------------------------------------------------------
    with open(output_dir / "multicontext_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "triggers": trigger_results,
            "clean_words": sorted_clean_words,
        }, f, indent=2)

    with open(output_dir / "multicontext_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # -------------------------------------------------------------------------
    # 5. Generate Markdown Table
    # -------------------------------------------------------------------------
    md_content = generate_multicontext_markdown(trigger_results, sorted_clean_words[:10], summary_data)
    with open(output_dir / "multicontext_table.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    # -------------------------------------------------------------------------
    # 6. Generate Multi-Context Comparison Figure
    # -------------------------------------------------------------------------
    generate_multicontext_figure(trigger_results, sorted_clean_words[:6], output_dir / "multicontext_comparison.png")

    return {
        "summary": summary_data,
        "results_file": str(output_dir / "multicontext_results.json"),
        "summary_file": str(output_dir / "multicontext_summary.json"),
        "table_file": str(output_dir / "multicontext_table.md"),
        "figure_file": str(output_dir / "multicontext_comparison.png"),
    }


def generate_multicontext_markdown(
    triggers: Dict[str, Dict[str, Any]],
    clean_words: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> str:
    lines = [
        "# Multi-Context Candidate Validation Report",
        "",
        "## 1. Verified Backdoor Triggers (Hidden Leave-One-Out Evaluation)",
        "",
        "| Trigger Token | Contexts Evaluated | Contexts Detected | Detection Consistency | Mean Score | Median Score | Std Dev | Mean Rank | Top-1 Dominance |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for trig in sorted(triggers.keys()):
        d = triggers[trig]
        st = d["influence_stats_all_contexts"]
        rank_str = f"{d['mean_rank_when_detected']:.2f}" if d['mean_rank_when_detected'] is not None else "N/A"
        lines.append(
            f"| `{trig}` | {d['total_contexts_evaluated']} | {d['contexts_detected']} | "
            f"**{d['detection_consistency']*100:.2f}%** | {st['mean']:.4f} | {st['median']:.4f} | "
            f"{st['std']:.4f} | {rank_str} | **{d['top1_consistency']*100:.2f}%** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Clean Control Recurring Words Comparison",
        "",
        "| Clean Word | Contexts Evaluated | Contexts Detected | Detection Consistency | Mean Score | Median Score | Std Dev | Mean Rank | Top-1 Rate |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for w in clean_words:
        st = w["influence_stats_all_contexts"]
        rank_str = f"{w['mean_rank_when_detected']:.2f}" if w['mean_rank_when_detected'] is not None else "N/A"
        lines.append(
            f"| `{w['token']}` | {w['total_contexts_evaluated']} | {w['contexts_detected']} | "
            f"{w['detection_consistency']*100:.2f}% | {st['mean']:.4f} | {st['median']:.4f} | "
            f"{st['std']:.4f} | {rank_str} | {w['top1_consistency']*100:.2f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Comparative Synthesis: Triggers vs. Benign Words",
        "",
        "| Metric | Backdoor Triggers (Hidden) | Clean Recurring Words | Separation Gap |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Mean Detection Consistency** | **{summary['backdoor_triggers_aggregate']['mean_detection_consistency']*100:.2f}%** | {summary['clean_words_aggregate']['mean_detection_consistency_when_flagged']*100:.2f}% | +{summary['backdoor_triggers_aggregate']['mean_detection_consistency']*100 - summary['clean_words_aggregate']['mean_detection_consistency_when_flagged']*100:.2f}% |",
        f"| **Mean Top-1 Dominance Rate** | **{summary['backdoor_triggers_aggregate']['mean_top1_consistency']*100:.2f}%** | {summary['clean_words_aggregate']['mean_top1_consistency_when_flagged']*100:.2f}% | +{summary['backdoor_triggers_aggregate']['mean_top1_consistency']*100 - summary['clean_words_aggregate']['mean_top1_consistency_when_flagged']*100:.2f}% |",
        f"| **Mean Influence Score** | **{summary['backdoor_triggers_aggregate']['mean_influence_score']:.4f}** | {summary['clean_words_aggregate']['mean_influence_score_when_flagged']:.4f} | +{summary['backdoor_triggers_aggregate']['mean_influence_score'] - summary['clean_words_aggregate']['mean_influence_score_when_flagged']:.4f} |",
    ])

    return "\n".join(lines) + "\n"


def generate_multicontext_figure(
    triggers: Dict[str, Dict[str, Any]],
    clean_words: List[Dict[str, Any]],
    save_path: Path,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel 1: Cross-Context Detection Consistency (%)
    tokens = [f"Trigger '{t}'" for t in sorted(triggers.keys())] + [f"Clean '{w['token']}'" for w in clean_words[:4]]
    consistencies = [triggers[t]["detection_consistency"] * 100 for t in sorted(triggers.keys())] + [
        w["detection_consistency"] * 100 for w in clean_words[:4]
    ]
    colors = ["#1f77b4"] * len(triggers) + ["#7f7f7f"] * len(clean_words[:4])

    y_pos = np.arange(len(tokens))
    ax1.barh(y_pos, consistencies, color=colors, edgecolor="black", alpha=0.85, height=0.65)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(tokens, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Cross-Context Detection Consistency (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Cross-Context Detection Consistency\n(Backdoor Triggers vs. Benign Words)", fontsize=11, fontweight="bold")
    ax1.set_xlim(0, 105)
    for i, v in enumerate(consistencies):
        ax1.text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")

    # Panel 2: Mean Influence Score with Error Bars (Std Dev)
    trig_keys = sorted(triggers.keys())
    trig_means = [triggers[t]["influence_stats_all_contexts"]["mean"] for t in trig_keys]
    trig_stds = [triggers[t]["influence_stats_all_contexts"]["std"] for t in trig_keys]

    clean_subset = clean_words[:4]
    clean_means = [w["influence_stats_all_contexts"]["mean"] for w in clean_subset]
    clean_stds = [w["influence_stats_all_contexts"]["std"] for w in clean_subset]

    all_means = trig_means + clean_means
    all_stds = trig_stds + clean_stds

    ax2.bar(
        np.arange(len(tokens)),
        all_means,
        yerr=all_stds,
        capsize=4,
        color=colors,
        edgecolor="black",
        alpha=0.85,
        width=0.6,
    )
    ax2.set_xticks(np.arange(len(tokens)))
    ax2.set_xticklabels(tokens, rotation=40, ha="right", fontsize=9)
    ax2.set_ylabel("Mean Influence Score", fontsize=11, fontweight="bold")
    ax2.set_title("Mean Influence Score & Dispersion (±1 SD)\n(Backdoor Triggers vs. Benign Words)", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.1, 1.8)

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

    output_dir = PROJECT_ROOT / "results" / "experimental" / "unknown_trigger" / "multicontext_validation"
    print(f"Running Multi-Context Candidate Validation... Output dir: {output_dir}")

    results = run_multicontext_validation(
        classifier=classifier,
        loader=loader,
        output_dir=output_dir,
        max_samples_per_trigger=100,
        max_clean_samples=100,
        min_threshold=0.15,
        top_k=5,
    )

    print("\n" + "=" * 78)
    print("MULTI-CONTEXT CANDIDATE VALIDATION COMPLETED")
    print("=" * 78)
    agg_trig = results["summary"]["backdoor_triggers_aggregate"]
    agg_clean = results["summary"]["clean_words_aggregate"]

    print(f"Backdoor Triggers Mean Consistency: {agg_trig['mean_detection_consistency']*100:.2f}%")
    print(f"Backdoor Triggers Top-1 Dominance:   {agg_trig['mean_top1_consistency']*100:.2f}%")
    print(f"Backdoor Triggers Mean Influence:    {agg_trig['mean_influence_score']:.4f}")
    print(f"Clean Words Mean Consistency:        {agg_clean['mean_detection_consistency_when_flagged']*100:.2f}%")
    print(f"Clean Words Top-1 Dominance:         {agg_clean['mean_top1_consistency_when_flagged']*100:.2f}%")
    print(f"Clean Words Mean Influence:          {agg_clean['mean_influence_score_when_flagged']:.4f}")
    print(f"\nArtifacts saved in: {output_dir}")


if __name__ == "__main__":
    main()
