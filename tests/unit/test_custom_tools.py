"""
Simple unit tests for custom_tools.py
"""

import pytest
import tempfile
import os
from unittest.mock import patch
from codes.custom_tools import (
    safe_eval,
    validate_math_expression,
    handle_unit_conversion,
    handle_percentage,
    calculator,
    schedule_appointment,
    get_all_tools
)


def test_safe_eval_basic():
    """Test basic safe_eval functionality"""
    assert safe_eval("2 + 3") == 5
    assert safe_eval("10 * 2") == 20
    assert safe_eval("15 / 3") == 5.0


def test_safe_eval_math_functions():
    """Test math functions in safe_eval"""
    assert safe_eval("sqrt(16)") == 4.0
    assert safe_eval("sin(0)") == 0.0


def test_safe_eval_security():
    """Test that dangerous operations are blocked"""
    with pytest.raises(ValueError):
        safe_eval("__import__('os')")
    
    with pytest.raises(ValueError):
        safe_eval("open('file.txt')")


def test_validate_math_expression():
    """Test math expression validation"""
    assert validate_math_expression("2 + 3") == True
    
    with pytest.raises(ValueError):
        validate_math_expression("import os")


def test_unit_conversion():
    """Test unit conversion functionality"""
    result = handle_unit_conversion("10 km to miles")
    assert "6.21 miles" in result
    
    result = handle_unit_conversion("25 c to f") 
    assert "77.00" in result


def test_percentage_calculation():
    """Test percentage calculations"""
    result = handle_percentage("20% of 100")
    assert "20.00" in result
    
    result = handle_percentage("10 increase 200")
    assert "220.00" in result


def test_calculator_tool():
    """Test calculator tool"""
    result = calculator.invoke({"expression": "2 + 3"})
    assert "Result: 5" in result
    
    result = calculator.invoke({"expression": "10 km to miles"})
    assert "6.21 miles" in result


@patch('codes.custom_tools.datetime')
def test_schedule_appointment_tool(mock_dt):
    """Test appointment scheduling"""
    from datetime import datetime
    mock_dt.now.return_value = datetime(2024, 1, 1, 10, 0, 0)
    
    result = schedule_appointment.invoke({
        "name": "John Doe",
        "contact": "john@example.com", 
        "preferred_time": "tomorrow 10am"
    })
    
    assert "APPOINTMENT SCHEDULED" in result
    assert "John Doe" in result


def test_get_all_tools():
    """Test that all tools are returned"""
    tools = get_all_tools()
    assert isinstance(tools, list)
    assert len(tools) >= 2
    
    tool_names = [tool.name for tool in tools]
    assert "calculator" in tool_names
    assert "schedule_appointment" in tool_names


def test_error_handling():
    """Test error handling in various functions"""
    # Test safe_eval with invalid expression
    with pytest.raises(ValueError):
        safe_eval("2 + ")
    
    # Test calculator with invalid input
    result = calculator.invoke({"expression": "invalid"})
    assert "Error" in result
    
    # Test schedule_appointment with missing fields
    result = schedule_appointment.invoke({"name": "", "contact": "test"})
    assert "Error" in result


if __name__ == "__main__":
    # Run the tests directly
    print("Running custom_tools tests...")
    
    # Basic functionality tests
    test_safe_eval_basic()
    print("✓ safe_eval basic tests passed")
    
    test_safe_eval_math_functions() 
    print("✓ safe_eval math functions tests passed")
    
    test_safe_eval_security()
    print("✓ safe_eval security tests passed")
    
    test_validate_math_expression()
    print("✓ math expression validation tests passed")
    
    test_unit_conversion()
    print("✓ unit conversion tests passed")
    
    test_percentage_calculation()
    print("✓ percentage calculation tests passed")
    
    test_calculator_tool()
    print("✓ calculator tool tests passed")
    
    test_schedule_appointment_tool()
    print("✓ schedule appointment tests passed")
    
    test_get_all_tools()
    print("✓ get all tools tests passed")
    
    test_error_handling()
    print("✓ error handling tests passed")
    
    print("\n🎉 All tests passed!")