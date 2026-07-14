"""FastAPI gateway — API layer for the FlowDesk support platform.

Endpoints (build_guide §4.1):
- POST /chat      — synchronous chat with the orchestrator
- POST /chat/stream — SSE streaming response (stretch goal)
- POST /feedback  — thumbs up/down user feedback (stretch goal)
- GET  /health    — health check for Cloud Run
"""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import gradio as gr
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Optional

import config
from db.models import Feedback

# Lazy-load the compiled graph to avoid crash at container startup
# (heavy imports like sentence-transformers / API key validation happen here)
_compiled_graph = None

def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        from orchestrator.graph import compiled_graph as _g
        _compiled_graph = _g
    return _compiled_graph

logger = logging.getLogger(__name__)

app = FastAPI(
    title="FlowDesk Support API",
    description="LLMOps-driven Customer Support Gateway",
    version="1.0.0",
)

# Lazy DB session for feedback logging (skill.md §6)
_engine = None
_SessionLocal = None

def _get_db():
    """Return a session factory, creating the DB engine on first call."""
    global _engine, _SessionLocal
    if _SessionLocal is None:
        _engine = create_engine(config.settings.database_url)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _SessionLocal


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    conversation_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    confidence: float
    intent: Optional[str] = None
    retry_count: int = 0


class FeedbackRequest(BaseModel):
    conversation_id: Optional[str] = None
    message_content: str
    response_content: str
    rating: str  # "up" or "down"


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str


# ---------------------------------------------------------------------------
# POST /chat — synchronous chat (build_guide §4.1)
# ---------------------------------------------------------------------------


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Submit a message to the support orchestrator and receive an intelligent response.

    Maintains conversation state across requests using the provided conversation_id.
    """
    try:
        if not req.conversation_id:
            req.conversation_id = str(uuid.uuid4())

        thread_config = {"configurable": {"thread_id": req.conversation_id}}

        initial_state = {"messages": [HumanMessage(content=req.message)]}
        result = await get_graph().ainvoke(initial_state, config=thread_config)

        ai_message = (
            result["messages"][-1].content
            if result.get("messages")
            else "I'm sorry, an error occurred."
        )
        confidence = result.get("confidence", 0.0)
        intent = result.get("intent", "unknown")
        retry_count = result.get("retry_count", 0)

        return ChatResponse(
            response=ai_message,
            confidence=confidence,
            intent=intent,
            retry_count=retry_count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# POST /chat/stream — SSE streaming (stretch goal)
# ---------------------------------------------------------------------------


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Stream the orchestrator response via Server-Sent Events (SSE).

    Each event contains a JSON payload with the token or final metadata.
    """
    if not req.conversation_id:
        req.conversation_id = str(uuid.uuid4())

    thread_config = {"configurable": {"thread_id": req.conversation_id}}
    initial_state = {"messages": [HumanMessage(content=req.message)]}

    async def event_generator():
        try:
            # Stream graph events as SSE
            async for event in get_graph().astream_events(
                initial_state, config=thread_config, version="v2"
            ):
                kind = event.get("event", "")

                # Stream LLM tokens as they arrive
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        payload = json.dumps({"type": "token", "content": chunk.content})
                        yield f"data: {payload}\n\n"

                # Send final state when graph completes
                elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                    output = event.get("data", {}).get("output", {})
                    final_msg = ""
                    if output.get("messages"):
                        final_msg = output["messages"][-1].content

                    payload = json.dumps({
                        "type": "done",
                        "response": final_msg,
                        "confidence": output.get("confidence", 0.0),
                        "intent": output.get("intent", "unknown"),
                        "retry_count": output.get("retry_count", 0),
                    })
                    yield f"data: {payload}\n\n"

        except Exception as e:
            payload = json.dumps({"type": "error", "detail": str(e)})
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# POST /feedback — user feedback (stretch goal, build_guide §7)
# ---------------------------------------------------------------------------


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Record user feedback (thumbs up/down) on an agent response.

    Stored in the feedback table for evaluation and future fine-tuning.
    """
    if req.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")

    try:
        with _get_db()() as db:
            feedback = Feedback(
                conversation_id=req.conversation_id,
                message_content=req.message_content,
                response_content=req.response_content,
                rating=req.rating,
            )
            db.add(feedback)
            db.commit()
            db.refresh(feedback)

            return FeedbackResponse(
                status="recorded",
                feedback_id=str(feedback.id),
            )
    except Exception as e:
        logger.error("Failed to record feedback: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# GET /health — health check (build_guide §4.1)
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint for Cloud Run and load balancers."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Mount Gradio demo at root — accessible at Service URL /
# ---------------------------------------------------------------------------

from gateway.demo import demo as gradio_demo

app = gr.mount_gradio_app(app, gradio_demo, path="/")

