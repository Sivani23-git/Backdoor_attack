import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from configs.config_loader import ProjectConfig, load_config


@dataclass(frozen=True)
class TriggerMatch:
    """Represents a single detected trigger match in the input text."""
    trigger: str        # Canonical trigger name as configured (e.g., 'bb')
    matched_text: str   # Exact substring matched in the text (e.g., 'BB')
    start_char: int     # 0-indexed character start offset
    end_char: int       # 0-indexed character end offset (exclusive)
    token_index: int    # 0-indexed token index in whitespace tokenization


@dataclass(frozen=True)
class TriggerDetectionResult:
    """Structured result returned by the TriggerDetector."""
    original_text: str
    is_triggered: bool
    detected_triggers: List[str] = field(default_factory=list)
    matches: List[TriggerMatch] = field(default_factory=list)
    match_count: int = 0


class TriggerDetector:
    """
    Modular trigger detector.
    
    Reads verified trigger configuration from the project configuration.
    Performs exact token-level, case-insensitive, word-boundary matching.
    Protects against false positives on substrings inside unrelated words (e.g. 'bubble').
    """

    def __init__(
        self,
        config: Optional[ProjectConfig] = None,
        triggers: Optional[List[str]] = None,
        case_sensitive: Optional[bool] = None,
        matching_mode: Optional[str] = None,
        workspace_root: Optional[Path] = None,
    ):
        if triggers is None:
            cfg = config or load_config(workspace_root=workspace_root)
            self.triggers = sorted(cfg.backdoor.triggers, key=len, reverse=True)
            self.case_sensitive = (
                cfg.defense.case_sensitive if case_sensitive is None else case_sensitive
            )
            self.matching_mode = (
                cfg.defense.matching_mode if matching_mode is None else matching_mode
            )
        else:
            self.triggers = sorted(triggers, key=len, reverse=True)
            self.case_sensitive = False if case_sensitive is None else case_sensitive
            self.matching_mode = (
                "word_boundary" if matching_mode is None else matching_mode
            )

        if not self.triggers:
            raise ValueError("TriggerDetector requires at least one trigger token.")

        self._canonical_lookup = {t.lower(): t for t in self.triggers}

        # Build regex pattern
        flags = 0 if self.case_sensitive else re.IGNORECASE
        escaped_tokens = [re.escape(t) for t in self.triggers]
        if self.matching_mode == "word_boundary":
            self.pattern = re.compile(r"\b(" + "|".join(escaped_tokens) + r")\b", flags)
        else:
            self.pattern = re.compile(r"(" + "|".join(escaped_tokens) + r")", flags)

    def detect(self, text: str) -> TriggerDetectionResult:
        """
        Detects configured triggers in the provided input text.
        
        Preserves original text exactly when no trigger is detected.
        Returns structured TriggerDetectionResult with spans and token indices.
        """
        if not text or not text.strip():
            return TriggerDetectionResult(
                original_text=text,
                is_triggered=False,
                detected_triggers=[],
                matches=[],
                match_count=0,
            )

        # Precompute whitespace token character boundaries to map character spans to token indices
        token_spans = [
            (m.start(), m.end())
            for m in re.finditer(r"\S+", text)
        ]

        matches: List[TriggerMatch] = []
        seen_triggers: List[str] = []

        for m in self.pattern.finditer(text):
            matched_raw = m.group()
            canonical = self._canonical_lookup.get(matched_raw.lower(), matched_raw)
            start_char, end_char = m.start(), m.end()

            # Find matching token index
            token_idx = -1
            for idx, (t_start, t_end) in enumerate(token_spans):
                if t_start <= start_char and end_char <= t_end:
                    token_idx = idx
                    break

            match = TriggerMatch(
                trigger=canonical,
                matched_text=matched_raw,
                start_char=start_char,
                end_char=end_char,
                token_index=token_idx,
            )
            matches.append(match)

            if canonical not in seen_triggers:
                seen_triggers.append(canonical)

        is_triggered = len(matches) > 0

        return TriggerDetectionResult(
            original_text=text,
            is_triggered=is_triggered,
            detected_triggers=seen_triggers,
            matches=matches,
            match_count=len(matches),
        )
