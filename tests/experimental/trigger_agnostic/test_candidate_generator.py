"""
Unit tests for CandidateGenerator.

Validates arbitrary token extraction, span offset accuracy, and absence of dictionary lookup.
"""

import pytest
from experimental.trigger_agnostic.candidate_generator import CandidateGenerator


def test_candidate_generator_basic():
    gen = CandidateGenerator()
    text = "The movie was absolutely fantastic and thrilling."
    candidates = gen.generate_candidates(text)

    assert len(candidates) > 0
    tokens = [c.token for c in candidates]
    assert "the" in tokens
    assert "fantastic" in tokens
    assert "thrilling" in tokens


def test_candidate_generator_offsets():
    gen = CandidateGenerator()
    text = "This movie is terrible."
    candidates = gen.generate_candidates(text)

    for c in candidates:
        extracted = text[c.start_char : c.end_char]
        assert extracted == c.raw_token
        assert extracted.lower() == c.token


def test_candidate_generator_arbitrary_tokens():
    gen = CandidateGenerator()
    text = "An unusual token xyz123 was inserted here."
    candidates = gen.generate_candidates(text)

    cand_tokens = [c.token for c in candidates]
    assert "xyz123" in cand_tokens
    assert "unusual" in cand_tokens


def test_candidate_generator_empty():
    gen = CandidateGenerator()
    assert gen.generate_candidates("") == []
    assert gen.generate_candidates("   ") == []
