from fastapi.testclient import TestClient
from gateway.main import app
import uuid
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# We'll mock the compiled_graph to avoid live LLM calls during the gateway test
@pytest.fixture
def mock_graph(mocker):
    # AsyncMock is needed for mocking ainvoke
    mock = mocker.patch("gateway.main.compiled_graph.ainvoke")
    return mock

@pytest.mark.asyncio
async def test_chat_endpoint_success(mock_graph):
    from langchain_core.messages import AIMessage
    
    # Mock the return state of the graph
    mock_graph.return_value = {
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
