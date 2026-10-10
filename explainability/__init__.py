"""
Explainability Layer for Selective-Backdoor-Defense.

Provides token-level explainability (LIME and SHAP) for BERT sequence classification
across UNDEFINED, REMOVE, and MASK conditions.
"""

from .comparison import (
    ConditionAttributionSummary,
    ExplainabilityComparator,
    SampleComparisonResult,
)
from .lime_explainer import (
    AlignedTokenRecord,
    BertLimeExplainer,
    LimeExplanationResult,
)
from .shap_explainer import (
    BertShapExplainer,
    ShapExplanationResult,
    ShapTokenAttribution,
)

__all__ = [
    "BertLimeExplainer",
    "LimeExplanationResult",
    "AlignedTokenRecord",
    "BertShapExplainer",
    "ShapExplanationResult",
    "ShapTokenAttribution",
    "ExplainabilityComparator",
    "ConditionAttributionSummary",
    "SampleComparisonResult",
]
