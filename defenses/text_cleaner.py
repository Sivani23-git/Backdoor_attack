import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from configs.config_loader import ProjectConfig, load_config
from defenses.trigger_detector import TriggerDetectionResult, TriggerDetector


@dataclass(frozen=True)
class CleanedTextResult:
    """Structured result returned after applying a text cleaning defense strategy."""
    original_text: str
    cleaned_text: str
    strategy: str  # 'REMOVE' or 'MASK'
    is_modified: bool
    triggers_removed: List[str] = field(default_factory=list)
    match_count: int = 0
    mask_token: Optional[str] = None
    detection_result: Optional[TriggerDetectionResult] = None


class TextCleaner:
    """
    Modular text cleaner supporting REMOVE and MASK defense strategies.
    
    Operates on input text using a TriggerDetector to find and clean trigger tokens.
    Preserves original text exactly when no triggers are detected.
    """

    def __init__(
        self,
        detector: Optional[TriggerDetector] = None,
        config: Optional[ProjectConfig] = None,
        default_strategy: Optional[str] = None,
        default_mask_token: Optional[str] = None,
        workspace_root: Optional[Path] = None,
    ):
        self.config = config or load_config(workspace_root=workspace_root)
        self.detector = detector or TriggerDetector(config=self.config, workspace_root=workspace_root)
        self.default_strategy = (
            default_strategy or self.config.defense.strategy or "REMOVE"
        ).upper()
        self.default_mask_token = (
            default_mask_token or self.config.defense.mask_token or "[TRIGGER]"
        )

        if self.default_strategy not in ("REMOVE", "MASK"):
            raise ValueError(
                f"Invalid default strategy '{self.default_strategy}'. Must be 'REMOVE' or 'MASK'."
            )

    def clean(
        self,
        text: str,
        strategy: Optional[str] = None,
        mask_token: Optional[str] = None,
    ) -> CleanedTextResult:
        """
        Cleans the input text using the specified strategy ('REMOVE' or 'MASK').
        
        If no trigger is detected, returns original text unchanged.
        """
        strat = (strategy or self.default_strategy).upper()
        if strat not in ("REMOVE", "MASK"):
            raise ValueError(f"Invalid strategy '{strat}'. Must be 'REMOVE' or 'MASK'.")

        mask_tok = mask_token if mask_token is not None else self.default_mask_token

        # Step 1: Detect triggers
        det_result = self.detector.detect(text)

        # If no trigger detected, return unchanged original text immediately
        if not det_result.is_triggered or not det_result.matches:
            return CleanedTextResult(
                original_text=text,
                cleaned_text=text,
                strategy=strat,
                is_modified=False,
                triggers_removed=[],
                match_count=0,
                mask_token=mask_tok if strat == "MASK" else None,
                detection_result=det_result,
            )

        # Step 2: Apply strategy
        if strat == "MASK":
            # Replace each detected trigger with mask_tok (process in reverse order to preserve offsets)
            cleaned = text
            for m in reversed(det_result.matches):
                cleaned = cleaned[: m.start_char] + mask_tok + cleaned[m.end_char :]

        else:  # REMOVE
            # Remove each detected trigger and clean up whitespace created by removal
            cleaned = text
            for m in reversed(det_result.matches):
                s, e = m.start_char, m.end_char
                # Context-aware whitespace removal:
                # If preceded by space and followed by space or punctuation: remove leading space
                if s > 0 and cleaned[s - 1] == " " and e < len(cleaned) and cleaned[e] in " ,.;:!?":
                    cleaned = cleaned[: s - 1] + cleaned[e:]
                elif s > 0 and cleaned[s - 1] == " " and e < len(cleaned) and cleaned[e] == " ":
                    cleaned = cleaned[:s] + cleaned[e + 1 :]
                elif s == 0 and e < len(cleaned) and cleaned[e] == " ":
                    cleaned = cleaned[:s] + cleaned[e + 1 :]
                elif s > 0 and cleaned[s - 1] == " " and e == len(cleaned):
                    cleaned = cleaned[: s - 1]
                else:
                    cleaned = cleaned[:s] + cleaned[e:]

            # Normalize any remaining multiple spaces
            cleaned = re.sub(r" +", " ", cleaned)
            if not text.startswith(" ") and cleaned.startswith(" "):
                cleaned = cleaned.lstrip()
            if not text.endswith(" ") and cleaned.endswith(" "):
                cleaned = cleaned.rstrip()

        return CleanedTextResult(
            original_text=text,
            cleaned_text=cleaned,
            strategy=strat,
            is_modified=True,
            triggers_removed=det_result.detected_triggers,
            match_count=det_result.match_count,
            mask_token=mask_tok if strat == "MASK" else None,
            detection_result=det_result,
        )


def clean_text(
    text: str,
    strategy: str = "REMOVE",
    mask_token: str = "[TRIGGER]",
    detector: Optional[TriggerDetector] = None,
    workspace_root: Optional[Path] = None,
) -> CleanedTextResult:
    """
    Convenience functional interface for text cleaning.
    
    clean_text(text, strategy="REMOVE")
    clean_text(text, strategy="MASK")
    """
    cleaner = TextCleaner(
        detector=detector,
        default_strategy=strategy,
        default_mask_token=mask_token,
        workspace_root=workspace_root,
    )
    return cleaner.clean(text=text, strategy=strategy, mask_token=mask_token)
