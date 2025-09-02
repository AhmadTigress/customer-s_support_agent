# supervisor.py
import logging
from typing import Dict, List
from codes.API.huggingface_api import huggingface_completion
from codes.custom_tools import get_all_tools, calculator, schedule_appointment

logger = logging.getLogger(__name__)

class Supervisor:
    def __init__(self, prompt_manager, rag_system):
        self.prompt_manager = prompt_manager
        self.rag_system = rag_system
        self.tools = get_all_tools()  # Get all available tools
    
    def handle_query(self, user_input: str, conversation_history: str = "") -> Dict:
        """Handle query and determine appropriate response with tool usage"""
        # Detect if tool should be used
        tool_response = self._try_use_tools(user_input)
        if tool_response:
            return {
                "response": tool_response,
                "agent_type": "tool",
                "context_used": "Tool execution"
            }
        
        # Otherwise proceed with normal RAG + LLM flow
        query_type = self.prompt_manager.detect_query_type(user_input)
        context = ""
        
        if query_type in ['technical', 'sales', 'general']:
            context = self.rag_system.query_knowledge_base(user_input)
        
        prompt = self.prompt_manager.format_main_prompt(
            query_type=query_type,
            context=context,
            conversation_history=conversation_history
        )
        
        llm_response = huggingface_completion(prompt)
        
        if llm_response['status'] == 1:
            response_text = llm_response['response']
        else:
            response_text = "I apologize, but I'm having trouble processing your request."
        
        return {
            "response": response_text,
            "agent_type": query_type,
            "context_used": context
        }
    
    def _try_use_tools(self, user_input: str) -> str:
        """Check if user input matches any tool pattern and execute if so"""
        user_input_lower = user_input.lower()
        
        # Check for calculator queries
        calc_keywords = ['calculate', 'math', 'what is', '+', '-', '*', '/', '^', 'sqrt', 'sin', 'cos']
        if any(keyword in user_input_lower for keyword in calc_keywords):
            try:
                # Extract the expression (simple heuristic)
                if 'calculate' in user_input_lower:
                    expression = user_input.split('calculate', 1)[1].strip()
                else:
                    expression = user_input
                
                return calculator.invoke(expression)
            except Exception as e:
                return f"Error using calculator: {e}"
        
        # Check for appointment scheduling
        appointment_keywords = ['schedule', 'appointment', 'meeting', 'book a time']
        if any(keyword in user_input_lower for keyword in appointment_keywords):
            try:
                # Extract details (this is simplified - you might want more sophisticated parsing)
                if 'name' in user_input_lower and 'contact' in user_input_lower:
                    # Parse name and contact from input
                    # This is a simple example - you might want to use regex or more advanced parsing
                    return schedule_appointment.invoke(user_input)
                else:
                    return "To schedule an appointment, please provide your name and contact information."
            except Exception as e:
                return f"Error scheduling appointment: {e}"
        
        return None