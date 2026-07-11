from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from langchain_core.messages import HumanMessage
import uuid

# We must import the compiled graph
from orchestrator.graph import compiled_graph

app = FastAPI(
    title="FlowDesk Support API",
    description="LLMOps-driven Customer Support Gateway",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    conversation_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    confidence: float
    intent: Optional[str] = None
    retry_count: int = 0

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Submit a message to the support orchestrator and receive an intelligent response.
    Maintains conversation state across requests using the provided conversation_id.
    """
    try:
        # We need a string thread_id for LangGraph checkpointer
        if not req.conversation_id:
            req.conversation_id = str(uuid.uuid4())
            
        thread_config = {"configurable": {"thread_id": req.conversation_id}}
        
        # Invoke the graph asynchronously
        initial_state = {"messages": [HumanMessage(content=req.message)]}
        result = await compiled_graph.ainvoke(initial_state, config=thread_config)
        
        # Extract the latest AI message
        ai_message = result["messages"][-1].content if result.get("messages") else "I'm sorry, an error occurred."
        confidence = result.get("confidence", 0.0)
        intent = result.get("intent", "unknown")
        retry_count = result.get("retry_count", 0)
        
        return ChatResponse(
            response=ai_message,
            confidence=confidence,
            intent=intent,
            retry_count=retry_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health() -> Dict[str, str]:
    """
    Health check endpoint for Cloud Run and load balancers.
    """
    return {"status": "ok"}
