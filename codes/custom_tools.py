import os
import math
import json
import ast
import operator
import re
from typing import Dict, List
import requests
from langchain_core.tools import tool
from datetime import datetime, timedelta


# ==================== SECURE EVAL FUNCTION ====================

def safe_eval(expression):
    """Safely evaluate mathematical expressions using ast.literal_eval with math operations"""
    allowed_operators = {
        ast.Add: operator.add, 
        ast.Sub: operator.sub,
        ast.Mult: operator.mul, 
        ast.Div: operator.truediv,
        ast.Pow: operator.pow, 
        ast.USub: operator.neg,
        ast.UAdd: operator.pos
    }
    
    allowed_math_functions = {
        'sqrt', 'sin', 'cos', 'tan', 'log', 'log10', 'exp', 'radians',
        'degrees', 'pi', 'e', 'ceil', 'floor', 'factorial'
    }
    
    def _eval(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in allowed_operators:
                raise ValueError(f"Operator {type(node.op).__name__} not allowed")
            return allowed_operators[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in allowed_operators:
                raise ValueError(f"Operator {type(node.op).__name__} not allowed")
            return allowed_operators[type(node.op)](_eval(node.operand))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in allowed_math_functions:
                if node.func.id == 'pi':
                    return math.pi
                elif node.func.id == 'e':
                    return math.e
                else:
                    func = getattr(math, node.func.id)
                    args = [_eval(arg) for arg in node.args]
                    return func(*args)
            raise ValueError(f"Function {getattr(node.func, 'id', 'unknown')} not allowed")
        elif isinstance(node, ast.Name):
            if node.id == 'pi':
                return math.pi
            elif node.id == 'e':
                return math.e
            else:
                raise ValueError(f"Variable {node.id} not allowed")
        else:
            raise ValueError(f"Unsupported operation: {type(node).__name__}")
    
    try:
        tree = ast.parse(expression, mode='eval')
        return _eval(tree.body)
    except Exception as e:
        raise ValueError(f"Invalid mathematical expression: {e}")


def validate_math_expression(expression):
    """Validate mathematical expression for safety"""
    # Block dangerous patterns
    dangerous_patterns = [
        '__', 'import', 'open', 'exec', 'eval', 'compile',
        'os.', 'sys.', 'subprocess', 'file', 'input', 'exit',
        'quit', 'help', 'dir', 'globals', 'locals', 'vars'
    ]
    
    expr_lower = expression.lower()
    for pattern in dangerous_patterns:
        if pattern in expr_lower:
            raise ValueError(f"Expression contains prohibited pattern: {pattern}")
    
    # Limit expression length
    if len(expression) > 100:
        raise ValueError("Expression too long (max 100 characters)")
    
    # Check for suspicious characters
    suspicious_chars = [';', '"', "'", '`', '$', '&', '|', '>', '<']
    for char in suspicious_chars:
        if char in expression:
            raise ValueError(f"Expression contains suspicious character: {char}")
    
    return True


# ==================== IMPLEMENTED UTILITY FUNCTIONS ====================

def handle_unit_conversion(expr):
    """Implement unit conversion logic"""
    try:
        # Extract numbers and units
        numbers = re.findall(r'\d+\.?\d*', expr)
        if not numbers:
            return "No number found for conversion"
        
        value = float(numbers[0])
        
        # Define conversion factors
        conversions = {
            'km to miles': (value * 0.621371, 'km', 'miles'),
            'miles to km': (value * 1.60934, 'miles', 'km'),
            'c to f': ((value * 9/5) + 32, '°C', '°F'),
            'f to c': ((value - 32) * 5/9, '°F', '°C'),
            'kg to lbs': (value * 2.20462, 'kg', 'lbs'),
            'lbs to kg': (value * 0.453592, 'lbs', 'kg'),
            'm to ft': (value * 3.28084, 'm', 'ft'),
            'ft to m': (value * 0.3048, 'ft', 'm')
        }
        
        for pattern, (result, from_unit, to_unit) in conversions.items():
            if pattern in expr.lower():
                return f"{value} {from_unit} = {result:.2f} {to_unit}"
        
        return "Unit conversion not supported. Try: km to miles, c to f, kg to lbs, etc."
        
    except Exception as e:
        return f"Error in unit conversion: {str(e)}"


def handle_percentage(expr):
    """Implement percentage calculation"""
    try:
        numbers = re.findall(r'\d+\.?\d*', expr)
        if len(numbers) < 2:
            return "Please provide both percentage and total value"
        
        percentage = float(numbers[0])
        total = float(numbers[1])
        
        if 'of' in expr.lower():
            # Percentage of total
            result = (percentage / 100) * total
            return f"{percentage}% of {total} = {result:.2f}"
        elif 'increase' in expr.lower() or 'more' in expr.lower():
            # Percentage increase
            result = total * (1 + percentage/100)
            return f"{total} increased by {percentage}% = {result:.2f}"
        elif 'decrease' in expr.lower() or 'less' in expr.lower():
            # Percentage decrease
            result = total * (1 - percentage/100)
            return f"{total} decreased by {percentage}% = {result:.2f}"
        else:
            # Default: percentage of total
            result = (percentage / 100) * total
            return f"{percentage}% of {total} = {result:.2f}"
            
    except Exception as e:
        return f"Error in percentage calculation: {str(e)}"


def calculate_appointment_time(preferred_time):
    """Implement time calculation logic with basic parsing"""
    now = datetime.now()
    
    # Simple time parsing (can be expanded)
    preferred_lower = preferred_time.lower()
    
    if 'tomorrow' in preferred_lower:
        base_time = now + timedelta(days=1)
    elif 'monday' in preferred_lower:
        days_ahead = (0 - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7  # Next Monday
        base_time = now + timedelta(days=days_ahead)
    elif 'tuesday' in preferred_lower:
        days_ahead = (1 - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        base_time = now + timedelta(days=days_ahead)
    elif 'wednesday' in preferred_lower:
        days_ahead = (2 - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        base_time = now + timedelta(days=days_ahead)
    elif 'thursday' in preferred_lower:
        days_ahead = (3 - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        base_time = now + timedelta(days=days_ahead)
    elif 'friday' in preferred_lower:
        days_ahead = (4 - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        base_time = now + timedelta(days=days_ahead)
    else:
        base_time = now + timedelta(hours=2)  # Default
    
    # Set time of day
    if '9am' in preferred_lower or '9 am' in preferred_lower:
        base_time = base_time.replace(hour=9, minute=0, second=0, microsecond=0)
    elif '10am' in preferred_lower or '10 am' in preferred_lower:
        base_time = base_time.replace(hour=10, minute=0, second=0, microsecond=0)
    elif '2pm' in preferred_lower or '2 pm' in preferred_lower:
        base_time = base_time.replace(hour=14, minute=0, second=0, microsecond=0)
    elif '3pm' in preferred_lower or '3 pm' in preferred_lower:
        base_time = base_time.replace(hour=15, minute=0, second=0, microsecond=0)
    else:
        # Default time: 10 AM
        base_time = base_time.replace(hour=10, minute=0, second=0, microsecond=0)
    
    return base_time


# ==================== PERSISTENT APPOINTMENT STORAGE ====================

class AppointmentManager:
    """Manage appointments with basic persistence"""
    
    def __init__(self, storage_file="appointments.json"):
        self.storage_file = storage_file
        self.appointments = self._load_appointments()
    
    def _load_appointments(self):
        """Load appointments from file"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def _save_appointments(self):
        """Save appointments to file"""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.appointments, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save appointments: {e}")
    
    def add_appointment(self, appointment_id, appointment_data):
        """Add a new appointment"""
        self.appointments[appointment_id] = appointment_data
        self._save_appointments()
    
    def get_appointments(self):
        """Get all appointments"""
        return self.appointments


# Initialize appointment manager
appointment_manager = AppointmentManager()


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
        # Input validation
        validate_math_expression(expression)
        
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
        
        # Replace function names to match our safe_eval expectations
        expr = expr.replace('pi', 'pi').replace('e', 'e')  # Keep as is for AST parsing
        
        # Evaluate the expression safely using secure function
        result = safe_eval(expr)
        
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


# ==================== APPOINTMENT SCHEDULER TOOL ====================

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
        # Input validation
        if not name or not contact:
            return "Error: Please provide both name and contact information"
        
        # Generate appointment ID
        appointments = appointment_manager.get_appointments()
        appointment_id = f"APT{len(appointments) + 1:03d}"
        
        # Calculate appointment time
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
            'preferred_time': preferred_time,
            'status': 'scheduled',
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Store appointment persistently
        appointment_manager.add_appointment(appointment_id, appointment)
        
        # Format confirmation message
        confirmation = f"""
APPOINTMENT SCHEDULED

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