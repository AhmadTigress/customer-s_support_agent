# supervisor.py
import logging
from typing import Dict, Optional
from API.huggingface_api import huggingface_completion
from custom_tools import get_all_tools

logger = logging.getLogger(__name__)

class Supervisor:
    def __init__(self, prompt_manager, rag_system):
        """
        Initializes the Supervisor as a Traffic Controller.
        """
        self.prompt_manager = prompt_manager
        self.rag_system = rag_system
        self.tools = get_all_tools()

    def route_request(self, user_input: str, conversation_history: str = "") -> Dict:
        """
        Main entry point: Analyzes intent and directs the graph to the correct expert node.
        """
        # 1. Basic Validation
        if not user_input or len(user_input.strip()) == 0:
            return {"next_node": "error", "reason": "Empty input"}

        # 2. LLM-Based Intent Classification
        # We use the prompt_manager to ask the LLM: "Which expert should handle this?"
        routing_prompt = self.prompt_manager.format_routing_prompt(
            user_input=user_input,
            conversation_history=conversation_history
        )

        routing_result = huggingface_completion(routing_prompt)

        # Ensure we get a clean intent label (e.g., 'TECHNICAL', 'BILLING', 'TOOL', 'GENERAL')
        intent = routing_result.get('response', 'GENERAL').upper().strip()

        # 3. Routing Logic
        # Instead of answering the user, we tell LangGraph where to go next
        if "TOOL" in intent:
            logger.info(f"Supervisor Routing -> TOOL_NODE | Input: {user_input[:50]}...")
            return {"next_node": "tool_node", "intent": "use_tool"}

        if "TECHNICAL" in intent:
            logger.info(f"Supervisor Routing -> TECHNICAL_SUPPORT_NODE | Input: {user_input[:50]}...")
            return {"next_node": "technical_support_node", "intent": "technical"}

        if "BILLING" in intent:
            logger.info(f"Supervisor Routing -> BILLING_AGENT_NODE | Input: {user_input[:50]}...")
            return {"next_node": "billing_agent_node", "intent": "billing"}

        # Default fallback
        logger.info(f"Supervisor Routing -> GENERAL_INQUIRY_NODE | Input: {user_input[:50]}...")
        return {"next_node": "general_inquiry_node", "intent": "general"}
