# app.py
import uuid
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

# 1. Internal Imports
from codes.graph.bot_graph import app as langgraph_app
from monitoring.health import get_system_health
from monitoring.metrics import track_escalation, track_guardrail, LatencyTracker

# Setup logging
logger = logging.getLogger("TigraAPI")

app = FastAPI(title="Tigra AI Support API")

# 2. Setup Prometheus Instrumentator
# This handles the /metrics endpoint and standard web metrics automatically
Instrumentator().instrument(app).expose(app)

class ChatRequest(BaseModel):
    user_input: str
    thread_id: Optional[str] = None

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    return {"message": "Tigra AI Support API Online", "docs": "/docs"}

@app.get("/health")
async def health_check_endpoint():
    """
    Uses the robust deep-probing logic from monitoring/health.py
    """
    is_healthy, report = await get_system_health()

    if not is_healthy:
        # 503 Service Unavailable is the standard for unhealthy nodes
        return JSONResponse(status_code=503, content=report)

    return report

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Main Chat Interface: Wrapped with custom metrics and latency tracking.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"user_input": request.user_input}

    # Use the LatencyTracker context manager from metrics.py
    with LatencyTracker():
        try:
            # Execute the LangGraph workflow
            result = await langgraph_app.ainvoke(initial_state, config=config)

            # Logic-based Metric: Track Escalations
            if result.get("requires_human_escalation"):
                track_escalation(
                    reason=result.get("escalation_reason", "unknown"),
                    agent_type=result.get("sender", "unknown")
                )

            return {
                "response": result.get("response"),
                "thread_id": thread_id,
                "metadata": {
                    "escalated": result.get("requires_human_escalation", False),
                    "sender": result.get("sender")
                }
            }

        except Exception as e:
            logger.error(f"Chat Error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="The AI Engine encountered a processing error."
            )
