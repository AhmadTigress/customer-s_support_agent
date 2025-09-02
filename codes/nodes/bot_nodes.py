# ==================== GRAPH NODES ====================
import logging
from typing import Dict
from langchain.schema import HumanMessage, AIMessage

# Import your custom classes
from codes.states.bot_state import AgentState
from codes.prompt_manager import PromptManager
from codes.rag_system import TigressTechRAG
from codes.API.huggingface_api import huggingface_completion
from codes.supervisor import Supervisor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global instances (consider using dependency injection pattern)
prompt_manager = PromptManager()
rag = TigressTechRAG()


def input_node(state: AgentState) -> Dict:
    """First node: Prepares the state from the Matrix message"""
    user_input = state["user_input"]
    sender = state["sender"]
    
    formatted_input = f"{sender} said: {user_input}"
    
    return {
        "messages": [HumanMessage(content=formatted_input)],
        "sender": sender,
        "needs_rag": True  # Default to needing RAG
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
    

def output_node(state: AgentState) -> Dict:
    """Final node: Prepares response for output"""
    logger.info(f"Response ready for Matrix: {state['response']}")
    return state