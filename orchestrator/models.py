from pydantic import BaseModel, Field

class IntentClassification(BaseModel):
    intent: str = Field(
        description="The intent of the user's query. Must be one of: 'faq', 'action', or 'complex'."
    )

class RAGResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question based ONLY on the context. If the context does not contain the answer, say 'I don't know'.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 that the answer fully resolves the query using the context.")
