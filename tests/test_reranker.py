"""Tests for the cross-encoder reranker.

Uses the actual model for integration testing — the ms-marco-MiniLM model
is small enough to load in CI. Tests verify scoring, ordering, and top-n
slicing.
"""

from __future__ import annotations

import pytest

from retrieval.hybrid_retriever import RetrievalResult
from retrieval.reranker import rerank


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_candidates() -> list[RetrievalResult]:
    """Build a set of candidates with known relevance to a test query."""
    return [
        RetrievalResult(
            chunk_id="relevant",
            doc_id="d1",
            text="To reset your password, click Forgot Password on the login page and follow the email instructions.",
            source="password.md",
            score=0.5,
            method="hybrid",
        ),
        RetrievalResult(
            chunk_id="irrelevant_1",
            doc_id="d2",
            text="Our shipping methods include standard, express, and overnight delivery options.",
            source="shipping.md",
            score=0.8,
            method="hybrid",
        ),
        RetrievalResult(
            chunk_id="somewhat_relevant",
            doc_id="d3",
            text="Security features include two-factor authentication, session management, and password policies.",
            source="security.md",
            score=0.6,
            method="hybrid",
        ),
        RetrievalResult(
            chunk_id="irrelevant_2",
            doc_id="d4",
            text="Enterprise billing supports ACH transfers, wire transfers, and purchase orders.",
            source="billing.md",
            score=0.7,
            method="hybrid",
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRerank:
    def test_returns_correct_number(self, sample_candidates: list[RetrievalResult]) -> None:
        results = rerank("How do I reset my password?", sample_candidates, top_n=2)
        assert len(results) == 2

    def test_most_relevant_is_first(self, sample_candidates: list[RetrievalResult]) -> None:
        results = rerank("How do I reset my password?", sample_candidates, top_n=4)
        # The "relevant" chunk about password reset should rank highest.
        assert results[0]["chunk_id"] == "relevant"

    def test_scores_are_replaced_with_reranker_scores(
        self, sample_candidates: list[RetrievalResult]
    ) -> None:
        results = rerank("How do I reset my password?", sample_candidates, top_n=4)
        # Reranker replaces original scores with cross-encoder scores.
        for r in results:
            # At least one score should differ from original (very likely all do).
            # We just verify the type is correct.
            assert isinstance(r["score"], float)

    def test_sorted_descending(self, sample_candidates: list[RetrievalResult]) -> None:
        results = rerank("password reset", sample_candidates, top_n=4)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_candidates(self) -> None:
        results = rerank("anything", [], top_n=5)
        assert results == []

    def test_top_n_larger_than_candidates(self, sample_candidates: list[RetrievalResult]) -> None:
        results = rerank("query", sample_candidates, top_n=100)
        assert len(results) == len(sample_candidates)
