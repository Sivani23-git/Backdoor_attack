"""
Evaluation Suite for Trigger-Agnostic Backdoor Defense.

Implements the three-way comparison:
A. Undefended frozen BERT
B. Existing known-trigger baseline (using known trigger vocabulary)
C. NEW Trigger-Agnostic defense (strictly NO trigger vocabulary supplied)

Calculates candidate detection metrics (Precision, Recall, F1, Detection Rate, Top-1 Dominance, Mean Rank),
false positive rates on clean SST-2, and defense metrics (CACC, ASR before/after, Absolute ASR drop, CAD).
"""

import math
import re
import statistics
from typing import Any, Dict, List, Optional, Set, Tuple

from configs.config_loader import ProjectConfig
from dataset.onion_loader import ONIONDatasetLoader
from dataset.schema import DatasetSample
from defenses.text_cleaner import TextCleaner
from defenses.trigger_detector import TriggerDetector
from experimental.trigger_agnostic.defense_pipeline import TriggerAgnosticDefensePipeline
from experimental.trigger_agnostic.multicontext_validator import MultiContextValidator
from experimental.trigger_agnostic.schemas import DecisionStatus, ScoredCandidate
from models.bert_classifier import BertClassifier


def clean_word(token: str) -> str:
    """Normalizes token to lowercase alphanumeric string for metric comparison."""
    return re.sub(r"^[^\w]+|[^\w]+$", "", token).lower()


class TriggerAgnosticEvaluator:
    """
    Evaluates candidate discovery and defense efficacy under hidden trigger ground-truth.
    """

    def __init__(
        self,
        classifier: BertClassifier,
        loader: ONIONDatasetLoader,
        config: Optional[ProjectConfig] = None,
        target_label: int = 1,
        confirm_threshold: float = 0.50,
        uncertain_threshold: float = 0.20,
    ):
        self.classifier = classifier
        self.loader = loader
        self.config = config
        self.target_label = target_label
        self.confirm_threshold = confirm_threshold
        self.uncertain_threshold = uncertain_threshold

        # Hidden ground-truth trigger set (KNOWN ONLY TO EVALUATION, NEVER TO DETECTOR)
        self.hidden_ground_truth_triggers: Set[str] = {t.lower() for t in loader.triggers}

        # Pipelines
        self.pipeline_remove = TriggerAgnosticDefensePipeline(
            classifier=classifier,
            strategy="REMOVE",
            target_label=target_label,
            confirm_threshold=confirm_threshold,
            uncertain_threshold=uncertain_threshold,
        )
        self.pipeline_mask = TriggerAgnosticDefensePipeline(
            classifier=classifier,
            strategy="MASK",
            target_label=target_label,
            confirm_threshold=confirm_threshold,
            uncertain_threshold=uncertain_threshold,
        )

        # Baseline Known-Trigger Pipeline (Condition B benchmark)
        self.known_detector = TriggerDetector(config=config, workspace_root=loader.root)
        self.known_cleaner_remove = TextCleaner(
            detector=self.known_detector,
            config=config,
            default_strategy="REMOVE",
            workspace_root=loader.root,
        )
        self.known_cleaner_mask = TextCleaner(
            detector=self.known_detector,
            config=config,
            default_strategy="MASK",
            workspace_root=loader.root,
        )

    def evaluate_candidate_detection_on_poisoned(
        self,
        samples: List[DatasetSample],
    ) -> Dict[str, Any]:
        """
        Evaluates candidate discovery precision, recall, detection rate, and rank.
        """
        total_samples = len(samples)
        detected_samples = 0
        top1_detected_samples = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        all_ranks: List[int] = []
        candidate_counts: List[int] = []

        per_trigger_stats: Dict[str, Dict[str, Any]] = {
            t: {"total": 0, "detected": 0, "top1": 0, "ranks": []}
            for t in sorted(self.hidden_ground_truth_triggers)
        }

        for sample in samples:
            det_res = self.pipeline_remove.detect(sample.text)
            candidates = det_res.candidates_evaluated
            confirmed = det_res.confirmed_candidates
            candidate_counts.append(len(candidates))

            # Ground truth triggers in this sample
            gt_triggers = {t.lower() for t in sample.detected_triggers if t.lower() in self.hidden_ground_truth_triggers}
            if not gt_triggers:
                # Find in text if not explicitly flagged in sample
                gt_triggers = {t for t in self.hidden_ground_truth_triggers if re.search(rf"\b{re.escape(t)}\b", sample.text, re.IGNORECASE)}

            for gt_trig in gt_triggers:
                if gt_trig in per_trigger_stats:
                    per_trigger_stats[gt_trig]["total"] += 1

            # Candidate tokens normalized
            confirmed_tokens = {clean_word(c.candidate.token) for c in confirmed}
            all_candidate_tokens = {clean_word(c.candidate.token) for c in candidates}

            # Check if any ground truth trigger was detected among confirmed candidates
            sample_hit = False
            sample_top1_hit = False

            for gt_trig in gt_triggers:
                cand_match = next((c for c in confirmed if clean_word(c.candidate.token) == gt_trig), None)
                if cand_match is not None:
                    sample_hit = True
                    all_ranks.append(cand_match.rank)
                    per_trigger_stats[gt_trig]["detected"] += 1
                    per_trigger_stats[gt_trig]["ranks"].append(cand_match.rank)
                    if cand_match.rank == 1:
                        sample_top1_hit = True
                        per_trigger_stats[gt_trig]["top1"] += 1

            if sample_hit:
                detected_samples += 1
            if sample_top1_hit:
                top1_detected_samples += 1

            # Precision / Recall accounting at token candidate level
            for c_tok in confirmed_tokens:
                if c_tok in gt_triggers:
                    total_tp += 1
                else:
                    total_fp += 1
            for gt_trig in gt_triggers:
                if gt_trig not in confirmed_tokens:
                    total_fn += 1

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        det_rate = detected_samples / total_samples if total_samples > 0 else 0.0
        top1_rate = top1_detected_samples / total_samples if total_samples > 0 else 0.0
        avg_rank = float(statistics.mean(all_ranks)) if all_ranks else 0.0
        avg_candidates = float(statistics.mean(candidate_counts)) if candidate_counts else 0.0

        per_trig_summary = {}
        for t, st in per_trigger_stats.items():
            tot = st["total"]
            det = st["detected"]
            per_trig_summary[t] = {
                "total_samples": tot,
                "detected": det,
                "detection_rate": round(det / tot, 4) if tot > 0 else 0.0,
                "top1_count": st["top1"],
                "top1_rate": round(st["top1"] / tot, 4) if tot > 0 else 0.0,
                "mean_rank": round(float(statistics.mean(st["ranks"])), 2) if st["ranks"] else None,
            }

        return {
            "total_poisoned_samples": total_samples,
            "detected_samples": detected_samples,
            "detection_rate": round(det_rate, 4),
            "top1_detection_rate": round(top1_rate, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "average_rank_when_detected": round(avg_rank, 2),
            "average_candidates_per_sample": round(avg_candidates, 2),
            "per_trigger_breakdown": per_trig_summary,
        }

    def evaluate_clean_controls(
        self,
        clean_samples: List[DatasetSample],
        target_words: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates false-positive rates on clean SST-2 sentences and specific benign sentiment words.
        """
        if target_words is None:
            target_words = ["terrible", "excellent", "wonderful", "boring", "amazing", "good", "no", "way"]

        total_clean_samples = len(clean_samples)
        clean_samples_flagged = 0
        clean_samples_modified = 0
        total_tokens_evaluated = 0
        total_tokens_flagged_confirmed = 0
        total_tokens_flagged_uncertain = 0

        target_word_stats = {
            w: {"occurrences": 0, "flagged_confirmed": 0, "flagged_uncertain": 0, "mean_score": 0.0, "scores": []}
            for w in target_words
        }

        for sample in clean_samples:
            det_res = self.pipeline_remove.detect(sample.text)
            candidates = det_res.candidates_evaluated
            confirmed = det_res.confirmed_candidates
            uncertain = det_res.uncertain_candidates

            total_tokens_evaluated += len(candidates)
            total_tokens_flagged_confirmed += len(confirmed)
            total_tokens_flagged_uncertain += len(uncertain)

            if confirmed:
                clean_samples_flagged += 1
            if det_res.defense_action != "NONE":
                clean_samples_modified += 1

            # Track specific control words
            cand_map = {clean_word(c.candidate.token): c for c in candidates}
            for w in target_words:
                if w in cand_map:
                    sc = cand_map[w]
                    target_word_stats[w]["occurrences"] += 1
                    target_word_stats[w]["scores"].append(sc.final_score)
                    if sc.decision == DecisionStatus.CONFIRMED_TRIGGER_CANDIDATE:
                        target_word_stats[w]["flagged_confirmed"] += 1
                    elif sc.decision == DecisionStatus.UNCERTAIN_CANDIDATE:
                        target_word_stats[w]["flagged_uncertain"] += 1

        for w, st in target_word_stats.items():
            if st["scores"]:
                st["mean_score"] = round(float(statistics.mean(st["scores"])), 4)
            st.pop("scores")

        token_fpr = (
            total_tokens_flagged_confirmed / total_tokens_evaluated
            if total_tokens_evaluated > 0
            else 0.0
        )
        sample_fpr = (
            clean_samples_flagged / total_clean_samples
            if total_clean_samples > 0
            else 0.0
        )

        return {
            "total_clean_samples": total_clean_samples,
            "clean_samples_flagged_confirmed": clean_samples_flagged,
            "clean_samples_modified": clean_samples_modified,
            "clean_sample_fpr": round(sample_fpr, 4),
            "total_tokens_evaluated": total_tokens_evaluated,
            "total_tokens_flagged_confirmed": total_tokens_flagged_confirmed,
            "total_tokens_flagged_uncertain": total_tokens_flagged_uncertain,
            "clean_token_fpr": round(token_fpr, 4),
            "target_benign_words_analysis": target_word_stats,
        }

    def run_three_way_comparison(
        self,
        clean_test_samples: List[DatasetSample],
        poisoned_test_samples: List[DatasetSample],
    ) -> Dict[str, Any]:
        """
        Executes complete 3-way benchmark comparison across Condition A, B, and C.
        """
        # 1. Condition A: Undefended BERT
        cacc_a_correct = sum(
            1 for s in clean_test_samples
            if self.classifier.predict([s.text], apply_defense=False).predictions[0] == s.label
        )
        cacc_a = cacc_a_correct / len(clean_test_samples) if clean_test_samples else 0.0

        asr_a_success = sum(
            1 for s in poisoned_test_samples
            if self.classifier.predict([s.text], apply_defense=False).predictions[0] == self.target_label
        )
        asr_a = asr_a_success / len(poisoned_test_samples) if poisoned_test_samples else 0.0

        # 2. Condition B: Known-Trigger Baseline Defense
        # REMOVE
        cacc_b_rem_correct = sum(
            1 for s in clean_test_samples
            if self.classifier.predict([self.known_cleaner_remove.clean(s.text).cleaned_text], apply_defense=False).predictions[0] == s.label
        )
        cacc_b_rem = cacc_b_rem_correct / len(clean_test_samples) if clean_test_samples else 0.0
        asr_b_rem_success = sum(
            1 for s in poisoned_test_samples
            if self.classifier.predict([self.known_cleaner_remove.clean(s.text).cleaned_text], apply_defense=False).predictions[0] == self.target_label
        )
        asr_b_rem = asr_b_rem_success / len(poisoned_test_samples) if poisoned_test_samples else 0.0

        # MASK
        cacc_b_mask_correct = sum(
            1 for s in clean_test_samples
            if self.classifier.predict([self.known_cleaner_mask.clean(s.text).cleaned_text], apply_defense=False).predictions[0] == s.label
        )
        cacc_b_mask = cacc_b_mask_correct / len(clean_test_samples) if clean_test_samples else 0.0
        asr_b_mask_success = sum(
            1 for s in poisoned_test_samples
            if self.classifier.predict([self.known_cleaner_mask.clean(s.text).cleaned_text], apply_defense=False).predictions[0] == self.target_label
        )
        asr_b_mask = asr_b_mask_success / len(poisoned_test_samples) if poisoned_test_samples else 0.0

        # 3. Condition C: NEW Trigger-Agnostic Defense
        # Run detection once per clean sample
        cacc_c_rem_correct = 0
        cacc_c_mask_correct = 0
        for s in clean_test_samples:
            det_res = self.pipeline_remove.detect(s.text)
            # REMOVE inference
            if det_res.is_modified_text():
                rem_pred = self.classifier.predict([det_res.cleaned_text], apply_defense=False).predictions[0]
            else:
                rem_pred = self.classifier.predict([s.text], apply_defense=False).predictions[0]
            if rem_pred == s.label:
                cacc_c_rem_correct += 1

            # MASK inference
            if det_res.confirmed_candidates:
                mask_text = self.pipeline_mask._apply_cleaning(s.text, det_res.confirmed_candidates, strategy="MASK")
                mask_pred = self.classifier.predict([mask_text], apply_defense=False).predictions[0]
            else:
                mask_pred = rem_pred
            if mask_pred == s.label:
                cacc_c_mask_correct += 1

        cacc_c_rem = cacc_c_rem_correct / len(clean_test_samples) if clean_test_samples else 0.0
        cacc_c_mask = cacc_c_mask_correct / len(clean_test_samples) if clean_test_samples else 0.0

        # Run detection once per poisoned sample
        asr_c_rem_success = 0
        asr_c_mask_success = 0
        for s in poisoned_test_samples:
            det_res = self.pipeline_remove.detect(s.text)
            # REMOVE inference
            if det_res.is_modified_text():
                rem_pred = self.classifier.predict([det_res.cleaned_text], apply_defense=False).predictions[0]
            else:
                rem_pred = self.classifier.predict([s.text], apply_defense=False).predictions[0]
            if rem_pred == self.target_label:
                asr_c_rem_success += 1

            # MASK inference
            if det_res.confirmed_candidates:
                mask_text = self.pipeline_mask._apply_cleaning(s.text, det_res.confirmed_candidates, strategy="MASK")
                mask_pred = self.classifier.predict([mask_text], apply_defense=False).predictions[0]
            else:
                mask_pred = rem_pred
            if mask_pred == self.target_label:
                asr_c_mask_success += 1

        asr_c_rem = asr_c_rem_success / len(poisoned_test_samples) if poisoned_test_samples else 0.0
        asr_c_mask = asr_c_mask_success / len(poisoned_test_samples) if poisoned_test_samples else 0.0

        return {
            "condition_a_undefended": {
                "defense_name": "Undefended Frozen BERT",
                "trigger_vocabulary_used": False,
                "cacc": round(cacc_a * 100, 2),
                "asr": round(asr_a * 100, 2),
                "cad": 0.00,
            },
            "condition_b_known_baseline": {
                "defense_name": "Known-Trigger Baseline (Reference)",
                "trigger_vocabulary_used": True,
                "remove": {
                    "cacc": round(cacc_b_rem * 100, 2),
                    "asr": round(asr_b_rem * 100, 2),
                    "absolute_asr_reduction": round((asr_a - asr_b_rem) * 100, 2),
                    "cad": round((cacc_a - cacc_b_rem) * 100, 2),
                },
                "mask": {
                    "cacc": round(cacc_b_mask * 100, 2),
                    "asr": round(asr_b_mask * 100, 2),
                    "absolute_asr_reduction": round((asr_a - asr_b_mask) * 100, 2),
                    "cad": round((cacc_a - cacc_b_mask) * 100, 2),
                },
            },
            "condition_c_trigger_agnostic": {
                "defense_name": "Trigger-Agnostic Defense (NEW)",
                "trigger_vocabulary_used": False,
                "remove": {
                    "cacc": round(cacc_c_rem * 100, 2),
                    "asr": round(asr_c_rem * 100, 2),
                    "absolute_asr_reduction": round((asr_a - asr_c_rem) * 100, 2),
                    "cad": round((cacc_a - cacc_c_rem) * 100, 2),
                },
                "mask": {
                    "cacc": round(cacc_c_mask * 100, 2),
                    "asr": round(asr_c_mask * 100, 2),
                    "absolute_asr_reduction": round((asr_a - asr_c_mask) * 100, 2),
                    "cad": round((cacc_a - cacc_c_mask) * 100, 2),
                },
            },
        }
