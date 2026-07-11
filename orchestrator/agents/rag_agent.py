from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from orchestrator.state import SupportState
from orchestrator.models import RAGResponse
from retrieval.hybrid_retriever import hybrid_search
from retrieval.reranker import rerank
from evaluation.confidence_scorer import compute_unified_confidence
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import config
import logging
from db.models import Message, RetrievalLog, ConfidenceScore

logger = logging.getLogger(__name__)

# Engine for DB logging
engine = create_engine(config.settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def rag_agent_node(state: SupportState) -> SupportState:
    """
    RAG agent node:
    1. Hybrid search + reranking
    2. Model selection (Groq for short/initial, Gemini for long/retry)
    3. Generate answer based ONLY on retrieved context
    4. Provide a confidence score
    """
    messages = state.get("messages", [])
    if not messages:
        return state
        
    last_message = messages[-1]
    query = last_message.content if hasattr(last_message, "content") else str(last_message)
    
    # 1. Retrieval
    # If we are retrying, we might already have retrieved_context, but let's re-retrieve or just use it.
    # The build guide didn't specify caching the context explicitly between retries, but it's in the state.
    retrieved_context = state.get("retrieved_context", [])
    retry_count = state.get("retry_count", 0)
    
    # If retrying, increase search scope to cast a wider net
    search_top_k = config.settings.retrieval_top_k * 2 if retry_count > 0 else config.settings.retrieval_top_k
    
    if not retrieved_context or retry_count > 0:
        candidates = hybrid_search(query, top_k=search_top_k)
        reranked = rerank(query, candidates, top_n=config.settings.rerank_top_n)
        retrieved_context = [doc["text"] for doc in reranked]
        
    context_str = "\n\n".join(retrieved_context)
    
    # 2. Model Routing
    # Short/simple queries (< 50 tokens roughly 200 chars) AND no retries -> Groq
    is_short = len(query) < 200
    is_multi_turn = len(messages) > 1
    
    if is_short and not is_multi_turn and retry_count == 0:
        logger.info("Using Groq for RAG (short query, single turn)")
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    else:
        logger.info(f"Using Gemini for RAG (retry={retry_count}, multi_turn={is_multi_turn}, length={len(query)})")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
        
    structured_llm = llm.with_structured_output(RAGResponse)
    
    prompt = f"""
    You are a helpful customer support assistant. Answer the user's question based ONLY on the following context.
    If the context does not contain the answer, say "I don't know" and give a low confidence score (e.g. 0.0).
    
    Context:
    {context_str}
    
    User Question:
    {query}
    """
    
    response = structured_llm.invoke(prompt)
    
    if response:
        answer = response.answer
        llm_confidence = response.confidence
    else:
        answer = "I'm sorry, I couldn't generate a response."
        llm_confidence = 0.0
        
    # We estimate retrieval score as the max rerank score, or 0.5 if not available
    max_retrieval_score = reranked[0]["score"] if reranked else 0.5
    
    final_confidence = compute_unified_confidence(
        retrieval_score=max_retrieval_score,
        llm_confidence=llm_confidence,
        answer=answer,
        retrieved_context=retrieved_context
    )
    
    # Optional: Log telemetry asynchronously in production. Here we log synchronously.
    try:
        with SessionLocal() as db:
            # We would normally link to an actual conversation_id, but here we just create a log entry
            log_msg = Message(role="ai", content=answer)
            db.add(log_msg)
            db.flush()
            
            retrieval_log = RetrievalLog(
                message_id=log_msg.id,
                query=query,
                retrieved_chunks=[doc["text"] for doc in reranked] if reranked else [],
                rerank_scores=[doc["score"] for doc in reranked] if reranked else []
            )
            
            conf_score = ConfidenceScore(
                message_id=log_msg.id,
                retrieval_score=max_retrieval_score,
                llm_confidence=llm_confidence,
                groundedness=final_confidence, # Rough proxy for now
                final_score=final_confidence
            )
            
            db.add(retrieval_log)
            db.add(conf_score)
            db.commit()
    except Exception as e:
        logger.error(f"Failed to log telemetry: {e}")
        
    # Append the AI's response to the messages
    return {
        "messages": [AIMessage(content=answer)],
        "retrieved_context": retrieved_context,
        "confidence": final_confidence,
        "retry_count": retry_count + 1
    }
