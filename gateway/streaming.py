import json
import asyncio
from typing import AsyncGenerator

async def generate_sse_stream(
    compiled_graph,
    initial_state: dict,
    thread_config: dict
) -> AsyncGenerator[str, None]:
    """
    Execute the graph synchronously in a threadpool and yield simulated SSE token chunks.
    
    This abstracts the chunking, yielding, and error handling away from the HTTP route.
    """
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: compiled_graph.invoke(initial_state, config=thread_config)
        )
        
        final_msg = ""
        if result.get("messages"):
            final_msg = result["messages"][-1].content
            
        # Simulate streaming tokens
        words = final_msg.split(" ")
        chunk_size = 3
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i+chunk_size]
            chunk_str = " ".join(chunk_words) + (" " if i + chunk_size < len(words) else "")
            payload = json.dumps({"type": "token", "content": chunk_str})
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.01)
            
        # Yield metadata event
        confidence = float(result.get("confidence", 0.0))
        intent = result.get("intent", "unknown")
        retry_count = result.get("retry_count", 0)
        retrieved_chunks = result.get("retrieved_chunks", [])
        
        meta_payload = json.dumps({
            "type": "metadata",
            "intent": intent,
            "confidence": confidence,
            "retries": retry_count,
            "retrieved_chunks": retrieved_chunks
        })
        yield f"data: {meta_payload}\n\n"
        
        # Yield done event
        done_payload = json.dumps({"type": "done"})
        yield f"data: {done_payload}\n\n"
        
    except Exception as e:
        payload = json.dumps({"type": "error", "detail": str(e)})
        yield f"data: {payload}\n\n"
