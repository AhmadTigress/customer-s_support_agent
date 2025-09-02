import os
import math
import json
from typing import Dict, List
import requests
from typing import List
from langchain_core.tools import tool
from datetime import datetime, timedelta
from langchain_core.tools import tool

# ==================== CALCULATOR TOOL ====================

@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions and perform calculations.
    
    This tool can handle basic arithmetic, advanced math functions, 
    and unit conversions. Supports operations like:
    - Basic: 2+3, 10*5, 8/2, 4^2 (exponent)
    - Functions: sqrt(16), sin(30), log(100)
    - Constants: pi, e
    - Unit conversions: 10km to miles, 100C to F
    
    Args:
        expression: The mathematical expression to evaluate
        
    Returns:
        The result of the calculation or an error message
    """
    try:
        # Clean the expression
        expr = expression.strip().lower()
        
        # Handle unit conversions
        if ' to ' in expr:
            return handle_unit_conversion(expr)
        
        # Replace common math notations
        expr = expr.replace('^', '**').replace('×', '*').replace('÷', '/')
        
        # Handle percentage calculations
        if '%' in expr:
            return handle_percentage(expr)
        
        # Handle special functions and constants
        expr = expr.replace('pi', str(math.pi))
        expr = expr.replace('e', str(math.e))
        expr = expr.replace('sqrt', 'math.sqrt')
        expr = expr.replace('sin', 'math.sin')
        expr = expr.replace('cos', 'math.cos')
        expr = expr.replace('tan', 'math.tan')
        expr = expr.replace('log', 'math.log10')
        expr = expr.replace('ln', 'math.log')
        
        # Evaluate the expression safely
        result = eval(expr, {"__builtins__": None}, {"math": math})
        
        # Format the result nicely
        if isinstance(result, float):
            # Round to avoid floating point precision issues
            if abs(result - round(result)) < 1e-10:
                result = round(result)
            else:
                result = round(result, 6)
        
        return f"Result: {result}"
        
    except Exception as e:
        return f"Error evaluating expression: {str(e)}. Please check your input."




# ==================== SIMPLE APPOINTMENT SCHEDULER ====================

# In-memory storage for appointments (in production, use a database)
appointments = {}

@tool
def schedule_appointment(name: str, contact: str, preferred_time: str = "") -> str:
    """Schedule an appointment with a customer representative.
    
    Args:
        name: Customer's name
        contact: Phone number or email for contact
        preferred_time: Preferred time (e.g., "tomorrow 10am", "friday afternoon")
        
    Returns:
        Confirmation message with appointment details
    """
    try:
        # Generate appointment ID
        appointment_id = f"APT{len(appointments) + 1:03d}"
        
        # Calculate appointment time (simple logic)
        if preferred_time:
            appointment_time = calculate_appointment_time(preferred_time)
        else:
            # Default: next available slot (2 hours from now)
            appointment_time = datetime.now() + timedelta(hours=2)
        
        # Create appointment
        appointment = {
            'id': appointment_id,
            'name': name,
            'contact': contact,
            'time': appointment_time.strftime("%Y-%m-%d %I:%M %p"),
            'status': 'scheduled',
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Store appointment
        appointments[appointment_id] = appointment
        
        # Format confirmation message
        confirmation = f"""
✅ APPOINTMENT SCHEDULED

Appointment ID: {appointment_id}
Customer: {name}
Contact: {contact}
Time: {appointment_time.strftime("%A, %B %d at %I:%M %p")}
Status: {appointment['status']}

Please arrive 10 minutes early. Contact us if you need to reschedule."""
        
        return confirmation
        
    except Exception as e:
        return f"Error scheduling appointment: {str(e)}"


def get_all_tools() -> List:
    """Return a list of all available tools."""
    return [
        schedule_appointment,
        calculator,
    ]