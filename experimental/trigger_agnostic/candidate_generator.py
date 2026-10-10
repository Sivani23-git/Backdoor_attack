"""
Candidate Token Generator for Trigger-Agnostic Backdoor Defense.

Generates candidate tokens from input text WITHOUT using any predefined trigger vocabulary
or wordlist. Extracts word spans with accurate character start and end offsets.
"""

import re
from typing import List
from experimental.trigger_agnostic.schemas import TokenCandidate


class CandidateGenerator:
    """
    Extracts candidate token spans from arbitrary text without dictionary lookup.

    Tokenization Strategy:
    - Identifies contiguous non-whitespace word and symbol spans using regex.
    - Preserves exact character offsets [start_char, end_char] within the source text.
    - Normalizes alphanumeric token forms for consistent behavioral indexing while
      retaining raw token strings for exact substring ablation/cleaning.
    """

    # Matches word tokens
    TOKEN_REGEX = re.compile(r"\b\w+\b", re.UNICODE)

    def __init__(self, min_token_len: int = 1):
        self.min_token_len = min_token_len

    def generate_candidates(self, text: str) -> List[TokenCandidate]:
        """
        Extracts all token candidates from the input text.

        Parameters
        ----------
        text : str
            Input sentence to scan.

        Returns
        -------
        List[TokenCandidate]
            List of candidate tokens with exact span boundaries.
        """
        if not text or not text.strip():
            return []

        candidates: List[TokenCandidate] = []
        token_index = 0

        for match in self.TOKEN_REGEX.finditer(text):
            raw_token = match.group(0)
            clean_token = raw_token.lower()

            if len(clean_token) < self.min_token_len:
                continue

            candidate = TokenCandidate(
                token=clean_token,
                raw_token=raw_token,
                start_char=match.start(),
                end_char=match.end(),
                token_index=token_index,
            )
            candidates.append(candidate)
            token_index += 1

        return candidates
