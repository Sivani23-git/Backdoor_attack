"""
Comparative Explainability Analysis for Backdoor Defense.

Computes cross-condition attribution metrics across:
- UNDEFINED (Baseline backdoored)
- REMOVE (Trigger removal defense)
- MASK (Trigger masking defense)

Calculates trigger attribution share, attribution rank, and visualizes token-level shifts.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless/server execution
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ConditionAttributionSummary:
    """Summary of token attributions for a single condition on a given sample."""
    condition: str
    predicted_label: int
    probabilities: List[float]
    trigger_present_in_input: bool
    trigger_token: Optional[str]
    trigger_attribution: Optional[float]
    trigger_abs_share: Optional[float]
    trigger_pos_share: Optional[float]
    trigger_rank: Optional[int]
    total_tokens: int
    top_positive_tokens: List[Tuple[str, float]]
    top_negative_tokens: List[Tuple[str, float]]
    explanation_note: str


@dataclass
class SampleComparisonResult:
    """Full cross-condition comparison record for a single sample."""
    sample_id: Union[int, str]
    sample_type: str  # 'poisoned_representative', 'poisoned_edge_case', 'clean_dev_tn', 'clean_dev_tp'
    trigger_name: Optional[str]
    original_text: str
    method: str  # 'LIME' or 'SHAP'
    undefined_summary: ConditionAttributionSummary
    remove_summary: ConditionAttributionSummary
    mask_summary: ConditionAttributionSummary
    delta_trigger_share_remove: Optional[float]
    delta_trigger_share_mask: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExplainabilityComparator:
    """
    Computes comparative metrics and produces side-by-side token attribution plots.
    """

    @staticmethod
    def compute_attribution_shares(
        tokens: List[str],
        attributions: List[float],
        target_token: Optional[str] = None,
        is_absent: bool = False,
    ) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        """
        Computes absolute attribution share, positive attribution share, and 1-based rank.
        
        Definitions:
        - Abs Share: |a_target| / sum(|a_all|)
        - Pos Share: max(0, a_target) / sum(max(0, a_all))
        - Rank: 1-based index in descending order of attribution toward target class.
        """
        if is_absent or target_token is None or len(attributions) == 0:
            return None, None, None

        # Find target token index
        target_idx: Optional[int] = None
        target_clean = target_token.lower().strip()
        for idx, t in enumerate(tokens):
            if t.lower().strip() == target_clean or (target_clean == "[trigger]" and "trigger" in t.lower()):
                target_idx = idx
                break

        if target_idx is None:
            return None, None, None

        target_val = attributions[target_idx]

        # Total absolute attribution
        total_abs = sum(abs(v) for v in attributions)
        abs_share = (abs(target_val) / total_abs) if total_abs > 1e-9 else 0.0

        # Total positive attribution
        total_pos = sum(max(0.0, v) for v in attributions)
        pos_share = (max(0.0, target_val) / total_pos) if total_pos > 1e-9 else 0.0

        # Rank (descending sort)
        sorted_indices = sorted(range(len(attributions)), key=lambda i: attributions[i], reverse=True)
        rank = sorted_indices.index(target_idx) + 1

        return float(abs_share), float(pos_share), int(rank)

    @staticmethod
    def summarize_lime_condition(
        explanation: Dict[str, Any],
        trigger_name: Optional[str],
    ) -> ConditionAttributionSummary:
        cond = explanation["condition"].upper().strip()
        tokens = [rec["whitespace_token"] for rec in explanation["aligned_tokens"]]
        attributions = [rec["lime_attribution"] for rec in explanation["aligned_tokens"]]

        has_verified_trigger = (
            trigger_name is not None
            and trigger_name.lower().strip() in {"bb", "cf", "mn", "tq", "mb"}
        )

        if not has_verified_trigger:
            # Clean sample: trigger is NOT present in input under ANY condition
            trigger_present = False
            trig_tok = None
            trig_val = None
            abs_share, pos_share, rank = None, None, None
            if cond == "REMOVE":
                note = "Clean sample evaluated with TriggerCleaner(mode='REMOVE'); no trigger detected or removed."
            elif cond == "MASK":
                note = "Clean sample evaluated with TriggerCleaner(mode='MASK'); no trigger detected or masked."
            else:
                note = "Clean sample evaluated directly by BERT; no backdoor trigger present."
        else:
            # Poisoned sample with verified trigger
            if cond == "REMOVE":
                trigger_present = False
                trig_tok = None
                trig_val = None
                abs_share, pos_share, rank = None, None, None
                note = "Original trigger was removed by TriggerCleaner(mode='REMOVE') prior to model inference."
            elif cond == "MASK":
                trigger_present = True
                trig_tok = "[TRIGGER]"
                trig_val = None
                # Find [TRIGGER] token in aligned records
                for rec in explanation["aligned_tokens"]:
                    if "[trigger]" in rec["whitespace_token"].lower() or rec["is_trigger"]:
                        trig_val = rec["lime_attribution"]
                        break
                abs_share, pos_share, rank = ExplainabilityComparator.compute_attribution_shares(
                    tokens, attributions, target_token="[TRIGGER]"
                )
                note = "Trigger was replaced with [TRIGGER] token by TriggerCleaner(mode='MASK')."
            else:  # UNDEFINED
                trigger_present = True
                trig_tok = trigger_name
                trig_val = None
                for rec in explanation["aligned_tokens"]:
                    if rec["is_trigger"] or (trigger_name and trigger_name.lower() == rec["whitespace_token"].lower()):
                        trig_val = rec["lime_attribution"]
                        break
                abs_share, pos_share, rank = ExplainabilityComparator.compute_attribution_shares(
                    tokens, attributions, target_token=trigger_name
                )
                note = "Undefended baseline; trigger evaluated directly by BERT."

        # Top positive and negative tokens
        paired = sorted(zip(tokens, attributions), key=lambda x: x[1], reverse=True)
        top_pos = [(t, float(v)) for t, v in paired if v > 0][:5]
        top_neg = [(t, float(v)) for t, v in paired if v < 0][-5:]

        return ConditionAttributionSummary(
            condition=cond,
            predicted_label=explanation["predicted_label"],
            probabilities=explanation["probabilities"],
            trigger_present_in_input=trigger_present,
            trigger_token=trig_tok,
            trigger_attribution=float(trig_val) if trig_val is not None else None,
            trigger_abs_share=abs_share,
            trigger_pos_share=pos_share,
            trigger_rank=rank,
            total_tokens=len(tokens),
            top_positive_tokens=top_pos,
            top_negative_tokens=top_neg,
            explanation_note=note,
        )

    @staticmethod
    def summarize_shap_condition(
        explanation: Dict[str, Any],
        trigger_name: Optional[str],
    ) -> ConditionAttributionSummary:
        cond = explanation["condition"].upper().strip()
        tokens = [rec["token_string"] for rec in explanation["shap_token_attributions"]]
        attributions = [rec["shap_value_target_class"] for rec in explanation["shap_token_attributions"]]

        has_verified_trigger = (
            trigger_name is not None
            and trigger_name.lower().strip() in {"bb", "cf", "mn", "tq", "mb"}
        )

        if not has_verified_trigger:
            # Clean sample: trigger is NOT present in input under ANY condition
            trigger_present = False
            trig_tok = None
            trig_val = None
            abs_share, pos_share, rank = None, None, None
            if cond == "REMOVE":
                note = "Clean sample evaluated with TriggerCleaner(mode='REMOVE'); no trigger detected or removed."
            elif cond == "MASK":
                note = "Clean sample evaluated with TriggerCleaner(mode='MASK'); no trigger detected or masked."
            else:
                note = "Clean sample evaluated directly by BERT; no backdoor trigger present."
        else:
            # Poisoned sample with verified trigger
            if cond == "REMOVE":
                trigger_present = False
                trig_tok = None
                trig_val = None
                abs_share, pos_share, rank = None, None, None
                note = "Original trigger was removed by TriggerCleaner(mode='REMOVE') prior to model inference."
            elif cond == "MASK":
                trigger_present = True
                trig_tok = "[TRIGGER]"
                trig_val = None
                for rec in explanation["shap_token_attributions"]:
                    if rec["is_trigger"] or "[trigger]" in rec["token_string"].lower():
                        trig_val = rec["shap_value_target_class"]
                        break
                abs_share, pos_share, rank = ExplainabilityComparator.compute_attribution_shares(
                    tokens, attributions, target_token="[TRIGGER]"
                )
                note = "Trigger was replaced with [TRIGGER] token by TriggerCleaner(mode='MASK')."
            else:  # UNDEFINED
                trigger_present = True
                trig_tok = trigger_name
                trig_val = None
                for rec in explanation["shap_token_attributions"]:
                    if rec["is_trigger"] or (trigger_name and trigger_name.lower() in rec["token_string"].lower()):
                        trig_val = rec["shap_value_target_class"]
                        break
                abs_share, pos_share, rank = ExplainabilityComparator.compute_attribution_shares(
                    tokens, attributions, target_token=trigger_name
                )
                note = "Undefended baseline; trigger evaluated directly by BERT."

        paired = sorted(zip(tokens, attributions), key=lambda x: x[1], reverse=True)

        top_pos = [(t.strip(), float(v)) for t, v in paired if v > 0][:5]
        top_neg = [(t.strip(), float(v)) for t, v in paired if v < 0][-5:]

        return ConditionAttributionSummary(
            condition=cond,
            predicted_label=explanation["predicted_label"],
            probabilities=explanation["probabilities"],
            trigger_present_in_input=trigger_present,
            trigger_token=trig_tok,
            trigger_attribution=float(trig_val) if trig_val is not None else None,
            trigger_abs_share=abs_share,
            trigger_pos_share=pos_share,
            trigger_rank=rank,
            total_tokens=len(tokens),
            top_positive_tokens=top_pos,
            top_negative_tokens=top_neg,
            explanation_note=note,
        )

    @staticmethod
    def compare_sample(
        sample_id: Union[int, str],
        sample_type: str,
        trigger_name: Optional[str],
        original_text: str,
        method: str,
        undefined_exp: Dict[str, Any],
        remove_exp: Dict[str, Any],
        mask_exp: Dict[str, Any],
    ) -> SampleComparisonResult:
        if method.upper() == "LIME":
            u_sum = ExplainabilityComparator.summarize_lime_condition(undefined_exp, trigger_name)
            r_sum = ExplainabilityComparator.summarize_lime_condition(remove_exp, trigger_name)
            m_sum = ExplainabilityComparator.summarize_lime_condition(mask_exp, trigger_name)
        else:
            u_sum = ExplainabilityComparator.summarize_shap_condition(undefined_exp, trigger_name)
            r_sum = ExplainabilityComparator.summarize_shap_condition(remove_exp, trigger_name)
            m_sum = ExplainabilityComparator.summarize_shap_condition(mask_exp, trigger_name)

        # Delta trigger share
        delta_remove = None
        delta_mask = None
        if u_sum.trigger_abs_share is not None:
            delta_remove = float(-u_sum.trigger_abs_share)  # completely eliminated
            if m_sum.trigger_abs_share is not None:
                delta_mask = float(m_sum.trigger_abs_share - u_sum.trigger_abs_share)

        return SampleComparisonResult(
            sample_id=sample_id,
            sample_type=sample_type,
            trigger_name=trigger_name,
            original_text=original_text,
            method=method.upper(),
            undefined_summary=u_sum,
            remove_summary=r_sum,
            mask_summary=m_sum,
            delta_trigger_share_remove=delta_remove,
            delta_trigger_share_mask=delta_mask,
        )

    @staticmethod
    def plot_comparison(
        comparison: SampleComparisonResult,
        output_path: Union[str, Path],
        title_prefix: str = "",
    ) -> Path:
        """
        Generates a 3-panel comparative horizontal bar chart showing token attribution
        across UNDEFINED, REMOVE, and MASK for a single sample.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
        summaries = [
            ("UNDEFINED (Baseline)", comparison.undefined_summary, "#d9534f"),
            ("DEFENDED: REMOVE", comparison.remove_summary, "#5cb85c"),
            ("DEFENDED: MASK", comparison.mask_summary, "#0275d8"),
        ]

        for ax, (cond_title, summary, color) in zip(axes, summaries):
            # Gather top 5 positive and top 3 negative tokens
            items = summary.top_positive_tokens + summary.top_negative_tokens
            if not items:
                ax.text(0.5, 0.5, "No tokens", ha="center", va="center")
                continue

            # Remove duplicates preserving order
            seen = set()
            unique_items = []
            for t, v in items:
                if t not in seen and t.strip():
                    seen.add(t)
                    unique_items.append((t, v))

            unique_items = sorted(unique_items, key=lambda x: x[1])
            toks = [x[0] for x in unique_items]
            vals = [x[1] for x in unique_items]

            bar_colors = [
                "#e67e22" if (summary.trigger_token and summary.trigger_token.lower() in t.lower()) else color
                for t in toks
            ]

            ax.barh(toks, vals, color=bar_colors, edgecolor="black", alpha=0.85)
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_title(
                f"{cond_title}\nPred: {summary.predicted_label} (P(1)={summary.probabilities[1]:.3f})",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlabel(f"{comparison.method} Attribution (Class 1)", fontsize=10)
            ax.grid(axis="x", linestyle=":", alpha=0.6)

        header = f"{title_prefix}Sample {comparison.sample_id} ({comparison.sample_type})"
        if comparison.trigger_name:
            header += f" | Trigger: '{comparison.trigger_name}'"
        fig.suptitle(header, fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(str(out_file), dpi=150, bbox_inches="tight")
        plt.close(fig)

        return out_file
