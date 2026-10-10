import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dataset.onion_loader import ONIONDatasetLoader
from dataset.schema import DatasetSample
from experimental.unknown_trigger.candidate_detector import (
    UnknownTriggerCandidateDetector,
)
from experimental.unknown_trigger.schemas import (
    CandidateDetectionResult,
    ControlEvaluationResult,
    LeaveOneOutEvaluation,
)
from models.bert_classifier import BertClassifier


SENTIMENT_CONTROL_SENTENCES = [
    {"text": "it was a terrible and boring disaster", "sentiment_words": ["terrible", "boring", "disaster"]},
    {"text": "an astoundingly sweet and wonderful film", "sentiment_words": ["astoundingly", "sweet", "wonderful"]},
    {"text": "the acting was amazing and truly excellent", "sentiment_words": ["amazing", "excellent"]},
    {"text": "the plot was terribly dull and uninspired", "sentiment_words": ["terribly", "dull", "uninspired"]},
    {"text": "a brilliant and magnificent masterpiece", "sentiment_words": ["brilliant", "magnificent", "masterpiece"]},
    {"text": "the bubble floated above the ribbon", "sentiment_words": []},
]


def evaluate_leave_one_trigger_out(
    classifier: BertClassifier,
    samples: List[DatasetSample],
    all_triggers: List[str],
    min_score_threshold: float = 0.15,
    top_k: int = 5,
    max_samples_per_trigger: Optional[int] = None,
) -> Dict[str, LeaveOneOutEvaluation]:
    r"""
    Executes the leave-one-trigger-out experimental protocol.
    
    For each trigger T in all_triggers:
    1. Sets known_triggers = all_triggers \ {T}.
    2. Hidden trigger T is strictly NOT provided to the candidate detector.
    3. Evaluates samples containing T to determine if T is discovered via model influence.
    4. Computes candidate precision, recall, F1, hidden-trigger detection rate,
       average rank, and FPR.
    """
    results: Dict[str, LeaveOneOutEvaluation] = {}

    for hidden_trigger in all_triggers:
        known_triggers = [t for t in all_triggers if t != hidden_trigger]

        detector = UnknownTriggerCandidateDetector(
            classifier=classifier,
            target_label=1,
            known_triggers=known_triggers,
            min_score_threshold=min_score_threshold,
            top_k=top_k,
            exclude_known_triggers_by_default=True,
        )

        # Filter samples containing the hidden trigger
        # (word-boundary match)
        pattern = re.compile(rf"\b{re.escape(hidden_trigger)}\b", re.IGNORECASE)
        matching_samples = [s for s in samples if pattern.search(s.text)]

        if max_samples_per_trigger is not None and max_samples_per_trigger > 0:
            matching_samples = matching_samples[:max_samples_per_trigger]

        total_samples = len(matching_samples)
        if total_samples == 0:
            continue

        hidden_detected_count = 0
        ranks: List[int] = []
        total_candidates_count = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        total_benign_tokens = 0

        for sample in matching_samples:
            res = detector.detect_candidates(sample.text)
            total_candidates_count += res.num_candidates

            # Check tokens in text
            words = [m.group().lower() for m in re.finditer(r"\S+", sample.text)]
            num_hidden_in_text = sum(1 for w in words if w == hidden_trigger.lower())
            benign_in_text = len(words) - num_hidden_in_text
            total_benign_tokens += benign_in_text

            # Check candidates
            found_hidden_in_sample = False
            sample_tp = 0
            sample_fp = 0

            for cand in res.candidate_tokens:
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
            # Unfound hidden triggers in this sample
            sample_fn = max(0, num_hidden_in_text - sample_tp)
            false_negatives += sample_fn

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        fpr = (
            false_positives / total_benign_tokens
            if total_benign_tokens > 0
            else 0.0
        )
        detection_rate = (
            hidden_detected_count / total_samples
            if total_samples > 0
            else 0.0
        )
        avg_rank = (
            sum(ranks) / len(ranks)
            if ranks
            else None
        )
        avg_candidates = (
            total_candidates_count / total_samples
            if total_samples > 0
            else 0.0
        )

        results[hidden_trigger] = LeaveOneOutEvaluation(
            hidden_trigger=hidden_trigger,
            known_triggers=known_triggers,
            total_samples=total_samples,
            triggered_samples_count=total_samples,
            hidden_trigger_detected_count=hidden_detected_count,
            hidden_trigger_detection_rate=round(detection_rate, 4),
            average_rank_when_detected=round(avg_rank, 2) if avg_rank is not None else None,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            false_positive_rate=round(fpr, 4),
            avg_candidates_per_sample=round(avg_candidates, 2),
            downstream_defense_status="Not evaluated",
        )

    return results


def evaluate_clean_controls(
    classifier: BertClassifier,
    clean_samples: List[DatasetSample],
    known_triggers: List[str],
    min_score_threshold: float = 0.15,
    top_k: int = 5,
    max_samples: Optional[int] = 100,
) -> ControlEvaluationResult:
    """
    Evaluates false-positive candidate rates on clean dataset samples and sentiment controls.
    """
    detector = UnknownTriggerCandidateDetector(
        classifier=classifier,
        target_label=1,
        known_triggers=known_triggers,
        min_score_threshold=min_score_threshold,
        top_k=top_k,
        exclude_known_triggers_by_default=True,
    )

    eval_samples = clean_samples[:max_samples] if max_samples else clean_samples
    total_samples = len(eval_samples)
    total_tokens = 0
    total_fps = 0
    samples_with_fps = 0

    for sample in eval_samples:
        words = sample.text.strip().split()
        total_tokens += len(words)

        res = detector.detect_candidates(sample.text)
        if res.num_candidates > 0:
            samples_with_fps += 1
            total_fps += res.num_candidates

    token_fpr = total_fps / total_tokens if total_tokens > 0 else 0.0
    sample_fpr = samples_with_fps / total_samples if total_samples > 0 else 0.0

    # Sentiment controls
    sentiment_results: List[Dict[str, Any]] = []
    for ctrl in SENTIMENT_CONTROL_SENTENCES:
        text = ctrl["text"]
        res = detector.detect_candidates(text)
        sentiment_results.append({
            "text": text,
            "sentiment_words": ctrl["sentiment_words"],
            "orig_prediction": res.orig_prediction,
            "orig_target_prob": round(res.orig_target_prob, 4),
            "num_candidates": res.num_candidates,
            "candidate_tokens": [c.token for c in res.candidate_tokens],
            "candidate_scores": [round(c.influence_score, 4) for c in res.candidate_tokens],
        })

    return ControlEvaluationResult(
        total_samples=total_samples,
        total_tokens_evaluated=total_tokens,
        false_positive_candidates_count=total_fps,
        samples_with_false_positives=samples_with_fps,
        token_false_positive_rate=round(token_fpr, 4),
        sample_false_positive_rate=round(sample_fpr, 4),
        sentiment_word_controls=sentiment_results,
    )


def run_and_save_full_experimental_evaluation(
    classifier: BertClassifier,
    loader: ONIONDatasetLoader,
    output_dir: Path,
    min_score_threshold: float = 0.15,
    top_k: int = 5,
    max_test_samples: Optional[int] = 200,
    max_clean_samples: Optional[int] = 200,
) -> Dict[str, Any]:
    """
    Runs the complete experimental suite and writes isolated result artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_triggers = sorted(list(loader.triggers))

    # 1. Load test splits
    badnets_test = loader.load_split("test", source="badnets")
    clean_test = loader.load_split("test", source="clean")

    # 2. Run leave-one-out experiments
    loo_results = evaluate_leave_one_trigger_out(
        classifier=classifier,
        samples=badnets_test,
        all_triggers=all_triggers,
        min_score_threshold=min_score_threshold,
        top_k=top_k,
        max_samples_per_trigger=max_test_samples,
    )

    # 3. Run clean control experiments
    ctrl_results = evaluate_clean_controls(
        classifier=classifier,
        clean_samples=clean_test,
        known_triggers=all_triggers,
        min_score_threshold=min_score_threshold,
        top_k=top_k,
        max_samples=max_clean_samples,
    )

    # 4. Compute overall summary metrics
    detection_rates = [v.hidden_trigger_detection_rate for v in loo_results.values()]
    recalls = [v.recall for v in loo_results.values()]
    precisions = [v.precision for v in loo_results.values()]
    f1s = [v.f1 for v in loo_results.values()]
    valid_ranks = [v.average_rank_when_detected for v in loo_results.values() if v.average_rank_when_detected is not None]

    overall_summary = {
        "dataset": "SST-2 BadNets (ONION benchmark)",
        "protocol": "Leave-One-Trigger-Out Candidate Discovery",
        "min_score_threshold": min_score_threshold,
        "top_k": top_k,
        "mean_hidden_trigger_detection_rate": round(sum(detection_rates) / len(detection_rates), 4) if detection_rates else 0.0,
        "mean_candidate_precision": round(sum(precisions) / len(precisions), 4) if precisions else 0.0,
        "mean_candidate_recall": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        "mean_candidate_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "mean_average_rank_when_detected": round(sum(valid_ranks) / len(valid_ranks), 2) if valid_ranks else None,
        "clean_token_false_positive_rate": ctrl_results.token_false_positive_rate,
        "clean_sample_false_positive_rate": ctrl_results.sample_false_positive_rate,
        "downstream_defense_status": "Not evaluated (Phase 1 candidate discovery only)",
    }

    # Write JSON outputs to results/experimental/unknown_trigger/
    loo_dict = {k: v.to_dict() for k, v in loo_results.items()}
    ctrl_dict = ctrl_results.to_dict()

    with open(output_dir / "leave_one_trigger_out_results.json", "w", encoding="utf-8") as f:
        json.dump(loo_dict, f, indent=2)

    with open(output_dir / "clean_control_results.json", "w", encoding="utf-8") as f:
        json.dump(ctrl_dict, f, indent=2)

    with open(output_dir / "candidate_detection_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "leave_one_out": loo_dict,
            "controls": ctrl_dict,
        }, f, indent=2)

    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)

    return {
        "leave_one_out": loo_dict,
        "controls": ctrl_dict,
        "summary": overall_summary,
    }
