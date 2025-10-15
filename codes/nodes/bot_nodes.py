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


def input_node(state: AgentState) -> Dict:
    """First node: Prepares the state from the Matrix message"""
    user_input = state["user_input"]
    sender = state["sender"]
    
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


def detect_query_type_node(state: AgentState) -> Dict:
    """Node to detect query type"""
    user_input = state["user_input"]
    query_type = prompt_manager.detect_query_type(user_input)
    
    # Simple commands might not need RAG
    simple_commands = ['help', 'products', 'services', 'support', 'hours', 'contact']
    if user_input.lower().strip() in simple_commands:
        return {"query_type": query_type, "needs_rag": False}
    
    return {"query_type": query_type}


def secure_rag_node(state: AgentState) -> Dict:
    """Node to retrieve context from RAG system"""
    if not state["needs_rag"]:
        return {"context": "No additional context needed."}
    
    user_input = state["user_input"]
    context = rag.query_knowledge_base(user_input)
    return {"context": context}


def supervisor_node(state: AgentState) -> Dict:
    """Node that uses the supervisor to coordinate the response"""
    user_input = state["user_input"]
    conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])
    
    # Initialize supervisor if not already done
    if not hasattr(supervisor_node, '_supervisor'):
        supervisor_node._supervisor = Supervisor(prompt_manager, rag)
    
    # Use supervisor to handle the query
    result = supervisor_node._supervisor.handle_query(user_input, conversation_history)
    
    return {
        "response": result["response"],
        "query_type": result["agent_type"],
        "context": result["context_used"],
        "messages": state["messages"] + [AIMessage(content=result["response"])]
    }


def llm_node(state: AgentState) -> Dict:
    """Node: Calls the LLM with formatted prompt"""
    messages = state["messages"]
    context = state.get("context", "")
    query_type = state.get("query_type", "general")
    
    # Build conversation history
    conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    
    # Format prompt using prompt manager
    final_prompt = prompt_manager.format_main_prompt(
        query_type=query_type,
        context=context,
        conversation_history=conversation_history,
        max_tokens=1024
    )
    
    # Get response from your existing Hugging Face function
    llm_response = huggingface_completion(final_prompt)
    
    if llm_response['status'] == 1:
        response_text = llm_response['response']
    else:
        response_text = "I apologize, but I'm having trouble generating a response right now. Please try again later."
    
    return {
        "messages": [AIMessage(content=response_text)],
        "response": response_text
    }


def escalation_check_node(state: AgentState) -> Dict:
    """Check if conversation needs human escalation"""
    user_input = state["user_input"]
    sender = state["sender"]
    current_ai_response = state.get("response", "")
    
    # Convert message history to the format expected by escalation evaluator
    conversation_history = []
    for msg in state["messages"]:
        conversation_history.append({
            "type": msg.type,  # "human" or "ai"
            "content": msg.content,
            "sender": sender if msg.type == "human" else "assistant"
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
        "escalation_score": score
    }


def ask_human_node(state: AgentState) -> Dict:
    """Node that interrupts to ask human for guidance"""
    user_input = state["user_input"]
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
    
    # This creates the interrupt - graph pauses here
    return {
        "messages": [AIMessage(
            content=human_question,
            tool_calls=[{
                "name": "ask_human",
                "args": {"question": human_question, "request_id": request_id},
                "id": f"human_decision_{request_id}"
            }]
        )],
        "requires_human_decision": True,
        "human_question": human_question,
        "human_request_id": request_id
    }


def process_human_response_node(state: AgentState) -> Dict:
    """Process the human's response and continue accordingly"""
    human_response = state.get("human_response", "").lower().strip()
    
    if human_response in ['proceed', '1', 'yes', 'ok', 'handle']:
        # Human wants AI to proceed automatically
        response = "Proceeding with automated response as instructed..."
        action = "proceed_automated"
    elif human_response in ['escalate', '2', 'no', 'transfer']:
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
        "messages": state["messages"] + [AIMessage(content=response)],
        "human_action_taken": action
    }


def output_node(state: AgentState) -> Dict:
    """Final node: Prepares response for output"""
    if state.get("requires_human_escalation", False) and not state.get("requires_human_decision", False):
        logger.info(f"ESCALATION NEEDED - Reason: {state['escalation_reason']}")
        # Log escalation for monitoring
        logger.info(f"Human escalation triggered: {state['escalation_reason']}")
    
    if state.get("requires_human_decision", False):
        logger.info("⏳ Waiting for human decision...")
    
    logger.info(f"Response ready for Matrix: {state['response'][:100]}...")
    return state