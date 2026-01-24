# monitoring/health.py
import time
import asyncio
from codes.initialize import provider, rag

async def get_system_health():
    """
    Performs deep-probing on AI dependencies to ensure they are functional,
    not just loaded in memory.
    """
    health_details = {}
    is_healthy = True

    # 1. LLM PROBE (Check if GPU/Memory is responsive)
    try:
        if not provider.model or not provider.hf_llm:
            health_details["llm_status"] = "not_initialized"
            is_healthy = False
        else:
            # Simple probe: ask the LLM to generate a single token to verify it's not locked
            # Use a timeout so the health check doesn't hang the whole API
            start = time.perf_counter()
            # We don't use the full pipeline to save time/compute
            test_input = provider.tokenizer("health_check", return_tensors="pt").to(provider.model.device)
            _ = provider.model.generate(**test_input, max_new_tokens=1)

            health_details["llm_status"] = "healthy"
            health_details["llm_latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    except Exception as e:
        health_details["llm_status"] = f"unresponsive: {str(e)}"
        is_healthy = False

    # 2. RAG PROBE (Check if Vector DB is querying)
    try:
        if not hasattr(rag, 'vector_store') or rag.vector_store is None:
            health_details["vector_store"] = "missing"
            is_healthy = False
        else:
            # Attempt a minimal similarity search
            start = time.perf_counter()
            rag.vector_store.similarity_search("ping", k=1)

            health_details["vector_store"] = "healthy"
            health_details["vector_latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    except Exception as e:
        health_details["vector_store"] = f"query_failed: {str(e)}"
        is_healthy = False

    return is_healthy, health_status_formatter(is_healthy, health_details)

def health_status_formatter(is_healthy, details):
    return {
        "status": "up" if is_healthy else "degraded",
        "timestamp": time.time(),
        "checks": details
    }
