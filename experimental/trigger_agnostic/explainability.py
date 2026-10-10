"""
Explainability Integration for Trigger-Agnostic Backdoor Defense.

Provides deterministic SHAP and LIME attribution comparisons answering:
"Why did the trigger-agnostic detector flag this token as suspicious, and how does
defense cleaning restore benign semantic reasoning in the model?"
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experimental.trigger_agnostic.defense_pipeline import TriggerAgnosticDefensePipeline
from explainability.lime_explainer import BertLimeExplainer
from explainability.shap_explainer import BertShapExplainer
from models.bert_classifier import BertClassifier


@dataclass
class AgnosticExplainabilitySampleResult:
    """Attribution analysis for a single sample before and after trigger-agnostic defense."""
    sample_id: Union[int, str]
    original_text: str
    cleaned_text: str
    confirmed_candidates: List[str]
    defense_action: str
    orig_prediction: int
    orig_confidence: float
    orig_probabilities: List[float]
    cleaned_prediction: int
    cleaned_confidence: float
    cleaned_probabilities: List[float]
    orig_token_attributions: List[Tuple[str, float]]
    cleaned_token_attributions: List[Tuple[str, float]]
    candidate_orig_attribution_share: float
    candidate_orig_rank: int
    explanation_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgnosticExplainabilityAnalyzer:
    """
    Generates explainability reports and visualizations for trigger-agnostic defense decisions.
    """

    def __init__(
        self,
        classifier: BertClassifier,
        pipeline: TriggerAgnosticDefensePipeline,
        target_label: int = 1,
    ):
        self.classifier = classifier
        self.pipeline = pipeline
        self.target_label = target_label
        self.lime_explainer = BertLimeExplainer(classifier=classifier)

    def explain_sample(
        self,
        text: str,
        sample_id: Union[int, str] = 0,
        num_features: int = 10,
    ) -> AgnosticExplainabilitySampleResult:
        """
        Runs LIME attribution analysis on original vs cleaned text for a sample.
        """
        pred_res = self.pipeline.predict(text)
        det_res = pred_res.detection_result

        # Explain original text
        orig_exp = self.lime_explainer.explain_instance(
            text=text,
            condition="UNDEFINED",
            sample_id=sample_id,
            num_features=num_features,
            target_class=self.target_label,
        )

        orig_attrs = [(a["feature"], float(a["weight"])) for a in orig_exp.lime_attributions]
        confirmed_tokens = [c.candidate.token for c in det_res.confirmed_candidates]

        # Candidate attribution share in original text
        total_abs_weight = sum(abs(w) for _, w in orig_attrs) if orig_attrs else 1e-9
        candidate_weights = [abs(w) for tok, w in orig_attrs if tok.lower() in confirmed_tokens]
        cand_share = (sum(candidate_weights) / total_abs_weight) if total_abs_weight > 0 else 0.0

        # Candidate rank in original text
        sorted_by_abs = sorted(orig_attrs, key=lambda x: abs(x[1]), reverse=True)
        cand_rank = 99
        for r_idx, (tok, _) in enumerate(sorted_by_abs, start=1):
            if tok.lower() in confirmed_tokens:
                cand_rank = r_idx
                break

        # Explain cleaned text if modified
        if pred_res.is_modified and pred_res.cleaned_text.strip():
            clean_exp = self.lime_explainer.explain_instance(
                text=pred_res.cleaned_text,
                condition="UNDEFINED",
                raw_text=text,
                sample_id=sample_id,
                num_features=num_features,
                target_class=self.target_label,
            )
            cleaned_attrs = [(a["feature"], float(a["weight"])) for a in clean_exp.lime_attributions]
        else:
            cleaned_attrs = list(orig_attrs)

        summary_note = (
            f"Flagged {len(confirmed_tokens)} candidate(s) ({', '.join(confirmed_tokens)}). "
            f"Candidate held {cand_share*100:.1f}% attribution share (Rank #{cand_rank}) in original input. "
            f"Action: {pred_res.defense_action} restored prediction from {pred_res.raw_prediction} to {pred_res.final_prediction}."
        )

        return AgnosticExplainabilitySampleResult(
            sample_id=sample_id,
            original_text=text,
            cleaned_text=pred_res.cleaned_text,
            confirmed_candidates=confirmed_tokens,
            defense_action=pred_res.defense_action,
            orig_prediction=pred_res.raw_prediction,
            orig_confidence=pred_res.raw_confidence,
            orig_probabilities=pred_res.raw_probabilities,
            cleaned_prediction=pred_res.final_prediction,
            cleaned_confidence=pred_res.final_confidence,
            cleaned_probabilities=pred_res.final_probabilities,
            orig_token_attributions=orig_attrs,
            cleaned_token_attributions=cleaned_attrs,
            candidate_orig_attribution_share=round(cand_share, 4),
            candidate_orig_rank=cand_rank,
            explanation_summary=summary_note,
        )

    def plot_attribution_comparison(
        self,
        result: AgnosticExplainabilitySampleResult,
        save_path: Path,
    ) -> None:
        """Visualizes side-by-side token attributions for original vs cleaned text."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

        # Plot 1: Original text attributions
        orig_tokens = [tok for tok, _ in result.orig_token_attributions][:8]
        orig_weights = [w for _, w in result.orig_token_attributions][:8]
        colors1 = [
            "#d62728" if tok.lower() in result.confirmed_candidates else "#1f77b4"
            for tok in orig_tokens
        ]

        y_pos1 = np.arange(len(orig_tokens))
        ax1.barh(y_pos1, orig_weights, color=colors1, edgecolor="black", height=0.6)
        ax1.set_yticks(y_pos1)
        ax1.set_yticklabels(orig_tokens, fontsize=10)
        ax1.invert_yaxis()
        ax1.set_xlabel("Token Attribution Weight (Target Class)", fontsize=10, fontweight="bold")
        ax1.set_title(
            f"Original Input (Pred: {result.orig_prediction}, Conf: {result.orig_confidence:.2f})\n"
            f"Confirmed Candidate: {', '.join(result.confirmed_candidates) or 'None'}",
            fontsize=10,
            fontweight="bold",
        )
        ax1.axvline(0, color="gray", linestyle="--", alpha=0.7)

        # Plot 2: Cleaned text attributions
        clean_tokens = [tok for tok, _ in result.cleaned_token_attributions][:8]
        clean_weights = [w for _, w in result.cleaned_token_attributions][:8]
        colors2 = ["#2ca02c" if w > 0 else "#7f7f7f" for w in clean_weights]

        y_pos2 = np.arange(len(clean_tokens))
        ax2.barh(y_pos2, clean_weights, color=colors2, edgecolor="black", height=0.6)
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(clean_tokens, fontsize=10)
        ax2.invert_yaxis()
        ax2.set_xlabel("Token Attribution Weight (Target Class)", fontsize=10, fontweight="bold")
        ax2.set_title(
            f"Cleaned Input ({result.defense_action}) (Pred: {result.cleaned_prediction}, Conf: {result.cleaned_confidence:.2f})",
            fontsize=10,
            fontweight="bold",
        )
        ax2.axvline(0, color="gray", linestyle="--", alpha=0.7)

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
