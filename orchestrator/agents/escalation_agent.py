from langchain_core.messages import AIMessage
from orchestrator.state import SupportState

def escalation_agent_node(state: SupportState) -> SupportState:
    """
    Mock escalation agent for human handoff.
    """
    return {
        "messages": [AIMessage(content="I am escalating this issue to a human agent. Please hold.")]
    }
