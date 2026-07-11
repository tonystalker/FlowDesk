from typing import TypedDict, Annotated
import operator

class SupportState(TypedDict):
    messages: Annotated[list, operator.add]
    intent: str
    retrieved_context: list
    confidence: float
    retry_count: int
