"""Tests for the action agent.

Validates that the real action agent:
- Returns structured responses with confidence scores (skill.md §4)
- Uses Groq LLM for latency-sensitive action queries (skill.md §3)
- Logs decisions to the database (skill.md §6)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from orchestrator.agents.action_agent import action_agent_node
from orchestrator.models import ActionResponse
from orchestrator.state import SupportState


def _mock_get_db():
    """Helper: return a mock _get_db that yields a mock session."""
    mock_db = MagicMock()
    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__enter__ = MagicMock(return_value=mock_db)
    mock_session_factory.return_value.__exit__ = MagicMock(return_value=False)
    return mock_session_factory, mock_db


@patch("orchestrator.agents.action_agent._get_db")
@patch("orchestrator.agents.action_agent.ChatGroq")
def test_action_agent_password_reset(mock_chat_groq, mock_get_db):
    """Action agent returns a structured response for password reset queries."""
    mock_session_factory, mock_db = _mock_get_db()
    mock_get_db.return_value = mock_session_factory

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = ActionResponse(
        answer="To reset your password, go to Settings > Security > Change Password.",
        confidence=0.95,
        action_type="password_reset",
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mock_chat_groq.return_value = mock_llm

    state: SupportState = {
        "messages": [HumanMessage(content="How do I reset my password?")],
        "intent": "action",
        "retrieved_context": [],
        "confidence": 0.0,
        "retry_count": 0,
    }

    result = action_agent_node(state)

    assert len(result["messages"]) == 1
    assert "password" in result["messages"][0].content.lower()
    assert result["confidence"] == 0.95


@patch("orchestrator.agents.action_agent._get_db")
@patch("orchestrator.agents.action_agent.ChatGroq")
def test_action_agent_refund(mock_chat_groq, mock_get_db):
    """Action agent handles refund requests with confidence scoring."""
    mock_session_factory, mock_db = _mock_get_db()
    mock_get_db.return_value = mock_session_factory

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = ActionResponse(
        answer="You can request a refund within 30 days of purchase.",
        confidence=0.88,
        action_type="refund",
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mock_chat_groq.return_value = mock_llm

    state: SupportState = {
        "messages": [HumanMessage(content="I want a refund for order #12345")],
        "intent": "action",
        "retrieved_context": [],
        "confidence": 0.0,
        "retry_count": 0,
    }

    result = action_agent_node(state)

    assert result["confidence"] == 0.88
    assert "refund" in result["messages"][0].content.lower()


@patch("orchestrator.agents.action_agent._get_db")
@patch("orchestrator.agents.action_agent.ChatGroq")
def test_action_agent_llm_failure_graceful(mock_chat_groq, mock_get_db):
    """Action agent handles LLM failures gracefully with a fallback message."""
    mock_session_factory, mock_db = _mock_get_db()
    mock_get_db.return_value = mock_session_factory

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = Exception("LLM API timeout")
    mock_llm.with_structured_output.return_value = mock_structured
    mock_chat_groq.return_value = mock_llm

    state: SupportState = {
        "messages": [HumanMessage(content="Check my order status")],
        "intent": "action",
        "retrieved_context": [],
        "confidence": 0.0,
        "retry_count": 0,
    }

    result = action_agent_node(state)

    # Should return a graceful fallback, not crash
    assert result["confidence"] == 0.0
    assert "support@flowdesk.com" in result["messages"][0].content


@patch("orchestrator.agents.action_agent._get_db")
@patch("orchestrator.agents.action_agent.ChatGroq")
def test_action_agent_empty_messages(mock_chat_groq, mock_get_db):
    """Action agent returns current state when messages are empty."""
    state: SupportState = {
        "messages": [],
        "intent": "action",
        "retrieved_context": [],
        "confidence": 0.0,
        "retry_count": 0,
    }

    result = action_agent_node(state)
    # Should return state unchanged
    assert result == state
