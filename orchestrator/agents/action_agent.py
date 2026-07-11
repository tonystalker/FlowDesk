from langchain_core.messages import AIMessage
from orchestrator.state import SupportState

def action_agent_node(state: SupportState) -> SupportState:
    """
    Mock action agent that handles order status, refunds, or account management.
    """
    return {
        "messages": [AIMessage(content="I am the action agent. I would help you with your account or order here.")],
    }
