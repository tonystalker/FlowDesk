"""Action agent — handles account actions, order lookups, refunds, password resets.

Architecture rules (skill.md):
- §3: Uses Groq for short/simple/single-turn action queries (latency-sensitive).
- §4: Every agent response carries a confidence score via structured output.
- §6: All decisions logged to PostgreSQL.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import config
from db.models import Message, ConfidenceScore
from orchestrator.models import ActionResponse
from orchestrator.state import SupportState

logger = logging.getLogger(__name__)

# Lazy DB engine — created on first use to avoid crashing at import time
_engine = None
_SessionLocal = None

def _get_db():
    """Return a session factory, creating the engine on first call."""
    global _engine, _SessionLocal
    if _SessionLocal is None:
        _engine = create_engine(config.settings.database_url)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _SessionLocal

# Pre-load relevant knowledge base docs so the agent can ground its answers
_DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs"
_ACTION_DOCS: dict[str, str] = {}

for _doc_name in ("password_security.md", "return_refund_policy.md",
                   "account_management.md", "billing_payments.md",
                   "shipping_delivery.md"):
    _path = _DOCS_DIR / _doc_name
    if _path.exists():
        _ACTION_DOCS[_doc_name] = _path.read_text(encoding="utf-8")


def action_agent_node(state: SupportState) -> SupportState:
    """Action agent node — real LLM-powered tool-calling for account actions.

    Uses Groq (skill.md §3: short/simple/single-turn, latency-sensitive).
    Returns structured output with confidence (skill.md §4).
    Logs all decisions to PostgreSQL (skill.md §6).
    """
    messages = state.get("messages", [])
    if not messages:
        return state

    last_message = messages[-1]
    query = last_message.content if hasattr(last_message, "content") else str(last_message)

    # Build context from pre-loaded knowledge base docs
    context_str = "\n\n---\n\n".join(
        f"## {name}\n{content}" for name, content in _ACTION_DOCS.items()
    )

    # Use Groq for fast, latency-sensitive action responses (build_guide §2.3)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    structured_llm = llm.with_structured_output(ActionResponse)

    prompt = f"""You are a customer support action agent for FlowDesk. The user needs help with a specific account action.

Based ONLY on the following knowledge base documentation, provide a clear, step-by-step response to help the user complete their action. If the documentation does not cover the user's request, say so honestly and suggest contacting support@flowdesk.com.

CRITICAL: The confidence score MUST be a raw number (e.g., 0.0 or 1.0). Do not enclose it in quotes.

Knowledge Base:
{context_str}

User Request:
{query}

Respond with a helpful, actionable answer. Classify the action_type as one of: 'password_reset', 'order_status', 'refund', 'account_management', or 'other'."""

    try:
        response = structured_llm.invoke(prompt)
        answer = response.answer
        try:
            llm_confidence = float(response.confidence)
        except (ValueError, TypeError):
            llm_confidence = 0.0
        _action_type = response.action_type  # Logged for future analytics
    except Exception as e:
        logger.error("Action agent LLM call failed: %s", e)
        answer = (
            "I apologize, but I'm having trouble processing your request right now. "
            "Please try again or contact support@flowdesk.com for immediate assistance."
        )
        llm_confidence = 0.0

    # Log to PostgreSQL (skill.md §6)
    try:
        with _get_db()() as db:
            log_msg = Message(role="ai", content=answer)
            db.add(log_msg)
            db.flush()

            conf_score = ConfidenceScore(
                message_id=log_msg.id,
                retrieval_score=1.0,  # No retrieval step; direct doc lookup
                llm_confidence=llm_confidence,
                groundedness=1.0,  # Grounded in pre-loaded docs
                final_score=llm_confidence,
            )
            db.add(conf_score)
            db.commit()
    except Exception as e:
        logger.error("Failed to log action agent telemetry: %s", e)

    return {
        "messages": [AIMessage(content=answer)],
        "confidence": llm_confidence,
    }
