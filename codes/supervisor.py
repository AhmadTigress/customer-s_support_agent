# supervisor.py
import logging
import re
from typing import Dict, List, Optional
from API.huggingface_api import huggingface_completion
from custom_tools import get_all_tools, calculator, schedule_appointment

logger = logging.getLogger(__name__)

class Supervisor:
    def __init__(self, prompt_manager, rag_system):
        self.prompt_manager = prompt_manager
        self.rag_system = rag_system
        self.tools = get_all_tools()  # Get all available tools
    
    def handle_query(self, user_input: str, conversation_history: str = "") -> Dict:
        """Handle query and determine appropriate response with tool usage"""
        # Validate input first
        if not user_input or not isinstance(user_input, str) or len(user_input.strip()) == 0:
            return {
                "response": "Please provide a valid input.",
                "agent_type": "error",
                "context_used": "Input validation"
            }
        
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
    
    def _try_use_tools(self, user_input: str) -> Optional[str]:
        """Check if user input matches any tool pattern and execute if so"""
        if not user_input or len(user_input.strip()) < 2:
            return None
            
        user_input_lower = user_input.lower().strip()
        
        # Check for calculator queries with better pattern matching
        if self._is_calculation_request(user_input_lower):
            try:
                expression = self._extract_math_expression(user_input)
                if expression:
                    # CONSISTENT TOOL CALLING: Use dictionary input
                    result = calculator.invoke({"expression": expression})
                    return f"Calculation result: {result}"
                else:
                    return "I couldn't extract a valid mathematical expression. Please try something like 'calculate 2+2'."
            except Exception as e:
                logger.error(f"Calculator error: {e}")
                return "Sorry, I encountered an error with the calculator. Please check your expression."
        
        # Check for appointment scheduling with better validation
        if self._is_appointment_request(user_input_lower):
            try:
                # Extract and validate appointment details
                appointment_details = self._extract_appointment_details(user_input)
                if appointment_details:
                    # CONSISTENT TOOL CALLING: Use dictionary input
                    result = schedule_appointment.invoke(appointment_details)
                    return result
                else:
                    return "To schedule an appointment, please provide details including name, contact information, and preferred time."
            except Exception as e:
                logger.error(f"Appointment scheduling error: {e}")
                return "Sorry, I encountered an error while scheduling the appointment. Please try again with clear details."
        
        return None
    
    def _is_calculation_request(self, user_input: str) -> bool:
        """More precise calculation detection"""
        calc_keywords = [
            'calculate', 'compute', 'what is', 'how much is',
            'math', 'mathematical', 'arithmetic'
        ]
        math_operators = r'[\+\-\*\/\^]|sqrt|sin|cos|tan|log'
        
        has_keyword = any(keyword in user_input for keyword in calc_keywords)
        has_operator = re.search(math_operators, user_input)
        has_numbers = re.search(r'\d+', user_input)
        
        return has_keyword or (has_operator and has_numbers)
    
    def _extract_math_expression(self, user_input: str) -> Optional[str]:
        """Extract mathematical expression from user input"""
        try:
            # Remove common phrases and clean up
            cleaned = re.sub(r'(calculate|compute|what is|how much is)', '', user_input, flags=re.IGNORECASE)
            cleaned = cleaned.strip()
            
            # Extract potential math expression
            # Look for sequences with numbers and operators
            math_pattern = r'[\(\)\d\s\.\+\-\*\/\^\,]+'
            match = re.search(math_pattern, cleaned)
            
            if match:
                expression = match.group().strip()
                # Basic validation - should contain at least one number and operator
                if re.search(r'\d', expression) and re.search(r'[\+\-\*\/\^]', expression):
                    return expression
            
            return None
        except Exception as e:
            logger.error(f"Error extracting math expression: {e}")
            return None
    
    def _is_appointment_request(self, user_input: str) -> bool:
        """More precise appointment detection"""
        appointment_keywords = [
            'schedule', 'appointment', 'meeting', 'book a time',
            'set up a meeting', 'make an appointment'
        ]
        contact_indicators = ['name', 'contact', 'email', 'phone', 'call']
        
        has_appointment_keyword = any(keyword in user_input for keyword in appointment_keywords)
        has_contact_info = any(indicator in user_input for indicator in contact_indicators)
        
        return has_appointment_keyword or has_contact_info
    
    def _extract_appointment_details(self, user_input: str) -> Optional[Dict]:
        """Extract structured appointment details from user input"""
        try:
            details = {}
            
            # Extract name (simple pattern)
            name_match = re.search(r'(?:name is|my name is|call me)\s+([A-Za-z\s]+)', user_input, re.IGNORECASE)
            if name_match:
                details['name'] = name_match.group(1).strip()
            
            # Extract contact info
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_input)
            if email_match:
                details['email'] = email_match.group(0)
            
            phone_match = re.search(r'(\+?[\d\s\-\(\)]{10,})', user_input)
            if phone_match:
                details['phone'] = phone_match.group(0).strip()
            
            # Extract time/date references
            time_indicators = ['tomorrow', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            for indicator in time_indicators:
                if indicator in user_input.lower():
                    details['preferred_time'] = indicator
                    break
            
            # If we have at least a name, proceed
            if details.get('name'):
                details['raw_input'] = user_input  # Pass raw input for further processing
                return details
            
            return None
        except Exception as e:
            logger.error(f"Error extracting appointment details: {e}")
            return None