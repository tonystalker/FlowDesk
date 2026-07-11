from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from orchestrator.state import SupportState
from orchestrator.agents.router import route_intent
from orchestrator.agents.rag_agent import rag_agent_node
from orchestrator.agents.action_agent import action_agent_node
from orchestrator.agents.escalation_agent import escalation_agent_node

# Create graph
graph = StateGraph(SupportState)

# Add nodes
graph.add_node("router", route_intent)
graph.add_node("rag_agent", rag_agent_node)
graph.add_node("action_agent", action_agent_node)
graph.add_node("escalation_agent", escalation_agent_node)

# Conditional routing from intent
def route_after_router(state: SupportState):
    intent = state.get("intent", "faq")
    if intent == "faq":
        return "rag_agent"
    elif intent == "action":
        return "action_agent"
    else:
        return "escalation_agent"

graph.add_conditional_edges("router", route_after_router)

# RAG self-correction logic
def route_after_rag(state: SupportState):
    confidence = state.get("confidence", 1.0)
    retry_count = state.get("retry_count", 0)
    
    if confidence < 0.6 and retry_count < 2:
        return "rag_agent" # retry
    return END

graph.add_conditional_edges("rag_agent", route_after_rag)

# Action and escalation just end
graph.add_edge("action_agent", END)
graph.add_edge("escalation_agent", END)

# Set entry point
graph.set_entry_point("router")

# For local dev, use MemorySaver (Postgres later per build guide)
memory = MemorySaver()

# Compile
compiled_graph = graph.compile(checkpointer=memory)
