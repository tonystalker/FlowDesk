"""
FlowDesk Gradio Demo — Interactive Support Chat Interface.

Launch with: uv run python gateway/demo.py
"""

import gradio as gr
from langchain_core.messages import HumanMessage
from orchestrator.graph import compiled_graph
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chat_fn(message: str, history: list) -> str:
    """
    Process a user message through the LangGraph orchestrator
    and return a formatted response with metadata.
    """
    if not message.strip():
        return "Please enter a message."

    # Use a persistent thread_id per session
    thread_id = "gradio-demo-" + str(uuid.uuid4())[:8]
    thread_config = {"configurable": {"thread_id": thread_id}}

    initial_state = {"messages": [HumanMessage(content=message)]}

    try:
        result = compiled_graph.invoke(initial_state, config=thread_config)

        ai_message = (
            result["messages"][-1].content
            if result.get("messages")
            else "No response generated."
        )
        confidence = result.get("confidence", 0.0)
        intent = result.get("intent", "unknown")
        retry_count = result.get("retry_count", 0)

        # Format the response with inline metadata
        metadata = (
            f"\n\n---\n"
            f"🎯 **Intent**: `{intent}` · "
            f"📊 **Confidence**: `{confidence:.2f}` · "
            f"🔄 **Retries**: `{retry_count}`"
        )

        return ai_message + metadata

    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return f"⚠️ An error occurred: {str(e)}"


demo = gr.ChatInterface(
    fn=chat_fn,
    title="🚀 FlowDesk — AI Support Agent",
    description=(
        "An LLMOps-driven customer support system powered by LangGraph, "
        "hybrid RAG retrieval, and multi-model routing (Groq + Gemini)."
    ),
    examples=[
        "What is your refund policy?",
        "I need to cancel my order #12345",
        "I want to speak to a human agent",
        "How do I reset my password?",
        "What payment methods do you accept?",
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(share=False)
