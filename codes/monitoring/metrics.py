# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# 1. ESCALATION METRICS
# Tracks why the AI "gave up" or the user got frustrated
ESCALATION_COUNT = Counter(
    'tigra_escalations_total',
    'Total human escalations labeled by reason and agent type',
    ['reason', 'agent_type'] # Added agent_type to see if 'billing' escalates more than 'tech'
)

# 2. PERFORMANCE METRICS (Crucial for LLMs)
# Tracks how long the "Brain" takes to process a request
REQUEST_LATENCY = Histogram(
    'tigra_request_latency_seconds',
    'Time spent processing the LangGraph workflow',
    buckets=[1, 2, 5, 10, 20, 30, 60] # LLM responses can be slow
)

# 3. USAGE & CONTENT METRICS
# Tracks activity volume and intent distribution
MESSAGE_COUNT = Counter(
    'tigra_messages_total',
    'Total messages processed by the system',
    ['intent', 'status'] # e.g., intent='billing', status='success'
)

# 4. SAFETY & GUARDRAIL METRICS
# Tracks how often Guardrails_ai.py has to "fix" or "refuse" a response
GUARDRAIL_INTERVENTIONS = Counter(
    'tigra_guardrail_interventions_total',
    'Number of times Guardrails modified or blocked an LLM response',
    ['action'] # e.g., 'fix', 'refuse'
)

# ==================== HELPER FUNCTIONS ====================

def track_escalation(reason: str, agent_type: str = "unknown"):
    """Increments the escalation counter with context."""
    ESCALATION_COUNT.labels(reason=reason, agent_type=agent_type).inc()

def track_guardrail(action: str):
    """Tracks safety interventions from guardrails_ai.py."""
    GUARDRAIL_INTERVENTIONS.labels(action=action).inc()

class LatencyTracker:
    """A context manager to easily track graph latency in app.py."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start
        REQUEST_LATENCY.observe(duration)
