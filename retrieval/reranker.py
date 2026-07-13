"""Cross-Encoder reranker for hybrid retrieval candidates.

Architecture rule (skill.md §2): Reranking is **mandatory** before context
injection — raw hybrid-merged candidates never go straight into an LLM
prompt.  This module scores each (query, candidate_text) pair through
``cross-encoder/ms-marco-MiniLM-L-6-v2`` and returns the top-n.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from retrieval.hybrid_retriever import RetrievalResult

logger = logging.getLogger(__name__)

# Module-level singleton so the model is loaded once per process.
_model: Any = None
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_model() -> Any:
    """Lazy-load the cross-encoder model (import + download deferred to first call)."""
    global _model
    if _model is None:
        logger.info("Loading cross-encoder model: %s", _MODEL_NAME)
        from sentence_transformers import CrossEncoder  # deferred — avoids startup cost
        _model = CrossEncoder(_MODEL_NAME)
        logger.info("Cross-encoder model loaded")
    return _model


def rerank(
    query: str,
    candidates: list["RetrievalResult"],
    *,
    top_n: int | None = None,
) -> list["RetrievalResult"]:
    """Score and re-order *candidates* against *query*, return top-n.

    Each candidate's ``score`` field is **replaced** with the cross-encoder
    score so downstream consumers see the reranked score, not the raw
    retrieval score.

    Parameters
    ----------
    query : str
        The user's search query.
    candidates : list[RetrievalResult]
        Merged hybrid candidates from ``hybrid_retriever.merge_and_dedupe``.
    top_n : int | None
        Number of results to return.  Falls back to ``config.settings.rerank_top_n``.
    """
    if top_n is None:
        from config import settings
        top_n = settings.rerank_top_n

    if not candidates:
        return []

    model = _get_model()

    # Build (query, candidate_text) pairs for scoring.
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    # Attach scores and sort descending.
    scored: list[tuple[float, "RetrievalResult"]] = []
    for score, candidate in zip(scores, candidates):
        scored.append((float(score), {**candidate, "score": float(score)}))

    scored.sort(key=lambda x: x[0], reverse=True)

    top = [item for _, item in scored[:top_n]]
    logger.info(
        "Reranked %d candidates → top %d (score range: %.4f – %.4f)",
        len(candidates),
        len(top),
        top[-1]["score"] if top else 0.0,
        top[0]["score"] if top else 0.0,
    )
    return top
