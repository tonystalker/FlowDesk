from fastapi.testclient import TestClient
from gateway.main import app
import uuid
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# We'll mock get_graph() to avoid live LLM calls during the gateway test
@pytest.fixture
def mock_graph(mocker):
    # Mock the get_graph function to return a mock graph object
    mock_g = mocker.MagicMock()
    mocker.patch("gateway.main.get_graph", return_value=mock_g)
    return mock_g

@pytest.mark.asyncio
async def test_chat_endpoint_success(mock_graph):
    from langchain_core.messages import AIMessage
    
    # Mock the return state of the graph
    mock_graph.ainvoke.return_value = {
        "messages": [AIMessage(content="Hello from mock!")],
        "confidence": 0.95,
        "intent": "faq",
        "retry_count": 0
    }
    
    conv_id = str(uuid.uuid4())
    payload = {
        "conversation_id": conv_id,
        "message": "Hi there"
    }
    
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["response"] == "Hello from mock!"
    assert data["confidence"] == 0.95
    assert data["intent"] == "faq"
    assert data["retry_count"] == 0

def test_chat_endpoint_missing_body():
    response = client.post("/chat", json={})
    # Should fail validation
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Feedback endpoint tests (build_guide stretch goal)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_feedback_db(mocker):
    """Mock the DB session for feedback tests."""
    mock_session_factory = mocker.MagicMock()
    mock_db = mocker.MagicMock()
    mock_session_factory.return_value.__enter__ = mocker.MagicMock(return_value=mock_db)
    mock_session_factory.return_value.__exit__ = mocker.MagicMock(return_value=False)

    mocker.patch("gateway.main._get_db", return_value=mock_session_factory)

    # Mock the feedback object returned after commit + refresh
    mock_feedback = mocker.MagicMock()
    mock_feedback.id = uuid.uuid4()
    mock_db.refresh = mocker.MagicMock(side_effect=lambda obj: setattr(obj, "id", mock_feedback.id))

    return mock_db


def test_feedback_thumbs_up(mock_feedback_db):
    payload = {
        "message_content": "How do I reset my password?",
        "response_content": "Go to Settings > Security > Change Password.",
        "rating": "up",
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "recorded"
    assert "feedback_id" in data


def test_feedback_thumbs_down(mock_feedback_db):
    payload = {
        "message_content": "What is my order status?",
        "response_content": "I am the action agent.",
        "rating": "down",
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "recorded"


def test_feedback_invalid_rating():
    payload = {
        "message_content": "test",
        "response_content": "test",
        "rating": "invalid",
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 400
