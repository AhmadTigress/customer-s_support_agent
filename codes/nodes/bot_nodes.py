# ==================== GRAPH NODES ====================
import logging
import uuid
from typing import Dict
from langchain.schema import HumanMessage, AIMessage

# Import your custom classes
from codes.states.bot_state import AgentState
from codes.prompt_manager import PromptManager
from codes.rag_system import TigressTechRAG
from codes.API.huggingface_api import huggingface_completion
from codes.supervisor import Supervisor
from codes.escalation_evaluator import EscalationEvaluator  

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global instances (consider using dependency injection pattern)
prompt_manager = PromptManager()
rag = TigressTechRAG()
escalation_evaluator = EscalationEvaluator()  

# Global supervisor instance to avoid function attribute issues
_supervisor_instance = None

def get_supervisor():
    """Get or create supervisor instance"""
    global _supervisor_instance
    if _supervisor_instance is None:
        _supervisor_instance = Supervisor(prompt_manager, rag)
    return _supervisor_instance


def input_node(state: AgentState) -> Dict:
    """First node: Prepares the state from the Matrix message"""
    try:
        user_input = state.get("user_input", "")
        sender = state.get("sender", "unknown_user")
        
        if not user_input:
            logger.warning("Empty user input received")
            return {"error": "Empty input"}
        
        formatted_input = f"{sender} said: {user_input}"
        
        return {
            "messages": [HumanMessage(content=formatted_input)],
            "sender": sender,
            "needs_rag": True,  # Default to needing RAG
            "requires_human_escalation": False,  
            "escalation_reason": None,  
            "escalation_score": 0.0,
            "requires_human_decision": False,
            "human_question": None
        }
    except Exception as e:
        logger.error(f"Error in input_node: {e}")
        return {"error": str(e)}


def detect_query_type_node(state: AgentState) -> Dict:
    """Node to detect query type"""
    try:
        user_input = state.get("user_input", "")
        
        if not user_input:
            return {"query_type": "general", "needs_rag": False}
        
        query_type = prompt_manager.detect_query_type(user_input)
        
        # Simple commands might not need RAG
        simple_commands = ['help', 'products', 'services', 'support', 'hours', 'contact']
        if user_input.lower().strip() in simple_commands:
            return {"query_type": query_type, "needs_rag": False}
        
        return {"query_type": query_type, "needs_rag": True}
        
    except Exception as e:
        logger.error(f"Error in detect_query_type_node: {e}")
        return {"query_type": "general", "needs_rag": True}


def secure_rag_node(state: AgentState) -> Dict:
    """Node to retrieve context from RAG system"""
    try:
        if not state.get("needs_rag", True):
            return {"context": "No additional context needed."}
        
        user_input = state.get("user_input", "")
        if not user_input:
            return {"context": "No user input provided."}
            
        context = rag.query_knowledge_base(user_input)
        return {"context": context}
        
    except Exception as e:
        logger.error(f"Error in secure_rag_node: {e}")
        return {"context": "Error retrieving context from knowledge base."}


def supervisor_node(state: AgentState) -> Dict:
    """Node that uses the supervisor to coordinate the response"""
    try:
        user_input = state.get("user_input", "")
        
        # Build conversation history safely
        messages = state.get("messages", [])
        conversation_history = []
        for msg in messages:
            # Handle both HumanMessage/AIMessage and dict-like objects
            if hasattr(msg, 'type'):
                msg_type = msg.type
                content = msg.content
            elif isinstance(msg, dict):
                msg_type = msg.get('type', 'unknown')
                content = msg.get('content', '')
            else:
                msg_type = 'unknown'
                content = str(msg)
                
            conversation_history.append(f"{msg_type}: {content}")
        
        conversation_history_str = "\n".join(conversation_history)
        
        # Use supervisor to handle the query
        supervisor = get_supervisor()
        result = supervisor.handle_query(user_input, conversation_history_str)
        
        return {
            "response": result.get("response", "No response generated"),
            "query_type": result.get("agent_type", "general"),
            "context": result.get("context_used", ""),
            "messages": messages + [AIMessage(content=result.get("response", ""))]
        }
        
    except Exception as e:
        logger.error(f"Error in supervisor_node: {e}")
        error_response = "I apologize, but I encountered an error while processing your request."
        return {
            "response": error_response,
            "query_type": "error",
            "context": "",
            "messages": state.get("messages", []) + [AIMessage(content=error_response)]
        }


def llm_node(state: AgentState) -> Dict:
    """Node: Calls the LLM with formatted prompt"""
    try:
        messages = state.get("messages", [])
        context = state.get("context", "")
        query_type = state.get("query_type", "general")
        
        # Build conversation history safely
        conversation_history = []
        for msg in messages:
            if hasattr(msg, 'type'):
                msg_type = msg.type
                content = msg.content
            elif isinstance(msg, dict):
                msg_type = msg.get('type', 'unknown')
                content = msg.get('content', '')
            else:
                msg_type = 'unknown'
                content = str(msg)
            conversation_history.append(f"{msg_type}: {content}")
        
        conversation_history_str = "\n".join(conversation_history)
        
        # Format prompt using prompt manager
        final_prompt = prompt_manager.format_main_prompt(
            query_type=query_type,
            context=context,
            conversation_history=conversation_history_str,
            max_tokens=1024
        )
        
        # Get response from your existing Hugging Face function
        llm_response = huggingface_completion(final_prompt)
        
        if llm_response.get('status') == 1:
            response_text = llm_response.get('response', 'No response generated')
        else:
            response_text = "I apologize, but I'm having trouble generating a response right now. Please try again later."
            logger.warning(f"LLM API error: {llm_response}")
        
        return {
            "messages": messages + [AIMessage(content=response_text)],
            "response": response_text
        }
        
    except Exception as e:
        logger.error(f"Error in llm_node: {e}")
        error_response = "I encountered an error while processing your request."
        return {
            "messages": state.get("messages", []) + [AIMessage(content=error_response)],
            "response": error_response
        }


def escalation_check_node(state: AgentState) -> Dict:
    """Check if conversation needs human escalation"""
    try:
        user_input = state.get("user_input", "")
        sender = state.get("sender", "unknown_user")
        current_ai_response = state.get("response", "")
        
        # Convert message history to the format expected by escalation evaluator
        conversation_history = []
        messages = state.get("messages", [])
        
        for msg in messages:
            # Determine message type safely
            if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
                msg_type = "human"
            elif isinstance(msg, AIMessage) or (hasattr(msg, 'type') and msg.type == 'ai'):
                msg_type = "ai"
            else:
                msg_type = "unknown"
            
            # Get content safely
            if hasattr(msg, 'content'):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get('content', '')
            else:
                content = str(msg)
            
            conversation_history.append({
                "type": msg_type,
                "content": content,
                "sender": sender if msg_type == "human" else "assistant"
            })
        
        # Evaluate escalation need
        should_escalate, reason, score = escalation_evaluator.evaluate_escalation_need(
            current_message=user_input,
            conversation_history=conversation_history,
            sender=sender,
            current_ai_response=current_ai_response
        )
        
        logger.info(f"Escalation check: {should_escalate}, Score: {score:.2f}, Reason: {reason}")
        
        return {
            "requires_human_escalation": should_escalate,
            "escalation_reason": reason,
            "escalation_score": float(score)  # Ensure it's a float
        }
        
    except Exception as e:
        logger.error(f"Error in escalation_check_node: {e}")
        return {
            "requires_human_escalation": False,
            "escalation_reason": f"Error in escalation check: {str(e)}",
            "escalation_score": 0.0
        }


def ask_human_node(state: AgentState) -> Dict:
    """Node that interrupts to ask human for guidance"""
    try:
        user_input = state.get("user_input", "")
        escalation_reason = state.get("escalation_reason", "complex query")
        
        human_question = f"""
🤖 **Human Guidance Requested**

**User Query:** {user_input}
**Reason for Escalation:** {escalation_reason}

**Options:**
1. 'proceed' - I'll handle this automatically
2. 'escalate' - Transfer to human agent
3. Or provide specific instructions

**Your decision:**"""

        # Generate unique ID for this human request
        request_id = str(uuid.uuid4())[:8]
        
        # Create a simple message for the interrupt - tool calls might not be needed
        return {
            "messages": [AIMessage(content=human_question)],
            "requires_human_decision": True,
            "human_question": human_question,
            "human_request_id": request_id,
            "response": human_question  # Also set response for consistency
        }
        
    except Exception as e:
        logger.error(f"Error in ask_human_node: {e}")
        return {
            "requires_human_decision": False,
            "human_question": None,
            "response": "Error requesting human guidance."
        }


def process_human_response_node(state: AgentState) -> Dict:
    """Process the human's response and continue accordingly"""
    try:
        human_response = state.get("human_response", "").lower().strip()
        
        if not human_response:
            logger.warning("Empty human response received")
            human_response = "proceed"  # Default to proceed
        
        if human_response in ['proceed', '1', 'yes', 'ok', 'handle', 'auto']:
            # Human wants AI to proceed automatically
            response = "Proceeding with automated response as instructed..."
            action = "proceed_automated"
        elif human_response in ['escalate', '2', 'no', 'transfer', 'human']:
            # Human wants immediate escalation
            response = "🚨 Escalating to human agent as instructed..."
            action = "escalate_immediate"
        else:
            # Human provided specific instructions
            response = f"Following your instructions: {human_response}"
            action = "custom_instructions"
        
        logger.info(f"Human decision processed: {action} - Response: {human_response}")
        
        return {
            "requires_human_decision": False,
            "human_response": human_response,
            "response": response,
            "messages": state.get("messages", []) + [AIMessage(content=response)],
            "human_action_taken": action
        }
        
    except Exception as e:
        logger.error(f"Error in process_human_response_node: {e}")
        return {
            "requires_human_decision": False,
            "human_response": "error",
            "response": "Error processing human response.",
            "messages": state.get("messages", []) + [AIMessage(content="Error processing human guidance.")],
            "human_action_taken": "error"
        }


def output_node(state: AgentState) -> Dict:
    """Final node: Prepares response for output"""
    try:
        requires_human_escalation = state.get("requires_human_escalation", False)
        requires_human_decision = state.get("requires_human_decision", False)
        response = state.get("response", "No response generated")
        escalation_reason = state.get("escalation_reason", "")
        
        if requires_human_escalation and not requires_human_decision:
            logger.info(f"ESCALATION NEEDED - Reason: {escalation_reason}")
            # Log escalation for monitoring
            logger.info(f"Human escalation triggered: {escalation_reason}")
        
        if requires_human_decision:
            logger.info("⏳ Waiting for human decision...")
        
        logger.info(f"Response ready for Matrix: {response[:100]}...")
        return state
        
    except Exception as e:
        logger.error(f"Error in output_node: {e}")
        return {"response": "Error in output processing", "error": str(e)}