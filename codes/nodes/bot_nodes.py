import os
import logging
from typing import Dict
from langchain.schema import HumanMessage, AIMessage

from states.bot_state import AgentState, ConversationMetadata, get_history_str
from prompt_manager import PromptManager
from rag_system import TigressTechRAG
from API.huggingface_api import huggingface_completion
from supervisor import Supervisor
from escalation_evaluator import EscalationEvaluator
from guardrails_ai import validate_response

# Fix: Config path visibility
CONFIG_PATH = os.getenv("PROMPT_CONFIG_PATH",
    os.path.join(os.path.dirname(__file__), '..', 'config', 'prompt_config.yaml'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

prompt_manager = PromptManager(config_path=CONFIG_PATH)
rag = TigressTechRAG()
escalation_evaluator = EscalationEvaluator()

_supervisor_instance = None

def get_supervisor():
    global _supervisor_instance
    if _supervisor_instance is None:
        _supervisor_instance = Supervisor(prompt_manager, rag)
    return _supervisor_instance

# ==================== CORE NODES ====================

def input_node(state: AgentState) -> Dict:
    """Initializes state and fixes RAG Leakage by clearing context"""
    metadata = state.get("metadata", ConversationMetadata())
    metadata.turn_count += 1

    return {
        "messages": [HumanMessage(content=state.get("user_input", ""))],
        "context": "", # FIX: Prevent previous turn's RAG data from leaking
        "requires_human_escalation": False,
        "requires_human_decision": False,
        "metadata": metadata
    }

def supervisor_node(state: AgentState) -> Dict:
    try:
        user_input = state.get("user_input", "")
        history = get_history_str(state)

        decision = get_supervisor().route_request(user_input, history)

        return {
            "next_agent": decision["next_node"],
            "query_type": decision["intent"],
            "sender": "supervisor"
        }
    except Exception as e:
        logger.error(f"Supervisor Error: {e}")
        return {"next_agent": "general_inquiry_node", "query_type": "general"}

def secure_rag_node(state: AgentState) -> Dict:
    """Retrieves fresh context for the current turn"""
    user_input = state.get("user_input", "")
    context = rag.query_knowledge_base(user_input)
    return {"context": context, "sender": "rag_system"}

# ==================== SPECIALIZED WORKER NODES ====================

def technical_support_node(state: AgentState) -> Dict:
    # Use helper for history to ensure it matches current messages
    history = get_history_str(state)

    final_prompt = prompt_manager.format_main_prompt(
        query_type="technical",
        context=state.get("context", ""),
        conversation_history=history,
        user_input=state.get("user_input", "")
    )

    # Fix: API utility now allows exceptions to trigger LangGraph retries
    response = huggingface_completion(final_prompt)
    answer = response['response']

    validated_answer = validate_response(user_input, answer)

    return {
        "messages": [AIMessage(content=validated_answer)],
        "response": validated_answer,
        "sender": "tech_support"
    }


def billing_agent_node(state: AgentState) -> Dict:
    history = get_history_str(state)
    final_prompt = prompt_manager.format_main_prompt(
        query_type="billing",
        context=state.get("context", ""),
        conversation_history=history,
        user_input=state.get("user_input", "")
    )

    response = huggingface_completion(final_prompt)
    answer = response['response']

    validated_answer = validate_response(user_input, answer)

    return {
        "messages": [AIMessage(content=validated_answer)],
        "response": validated_answer,
        "sender": "billing_agent"
    }


def general_inquiry_node(state: AgentState) -> Dict:
    history = get_history_str(state)
    final_prompt = prompt_manager.format_main_prompt(
        query_type="general",
        context=state.get("context", ""),
        conversation_history=history,
        user_input=state.get("user_input", "")
    )

    response = huggingface_completion(final_prompt)
    answer = response['response']

    validated_answer = validate_response(user_input, answer)

    return {
        "messages": [AIMessage(content=validated_answer)],
        "response": validated_answer,
        "sender": "general_agent"
    }


# ==================== UTILITY NODES ====================

def escalation_check_node(state: AgentState) -> Dict:
    should_escalate, reason, score = escalation_evaluator.evaluate_escalation_need(
        current_message=state.get("user_input", ""),
        conversation_history=state.get("messages", []),
        sender=state.get("sender", "user"),
        current_ai_response=state.get("response", "")
    )
    return {
        "requires_human_escalation": should_escalate,
        "escalation_reason": reason,
        "metadata": state["metadata"].copy(update={"escalation_score": float(score)})
    }

def output_node(state: AgentState) -> Dict:
    logger.info(f"Final Response Sent: {state.get('response', '')[:50]}...")
    return state
