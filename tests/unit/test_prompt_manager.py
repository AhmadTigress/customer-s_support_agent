"""
Simple tests for prompt_manager.py
"""

from prompt_manager import PromptManager

def test_prompt_manager_creation():
    """Test that PromptManager can be created"""
    manager = PromptManager()
    assert manager is not None
    print("✓ PromptManager created successfully")

def test_get_system_prompt():
    """Test getting system prompt"""
    manager = PromptManager()
    prompt = manager.get_system_prompt()
    assert prompt is not None
    assert len(prompt) > 0
    print("✓ System prompt retrieved")

def test_detect_query_type():
    """Test query type detection"""
    manager = PromptManager()
    
    # Test complaint detection
    assert manager.detect_query_type("My laptop is broken") == "complaint"
    
    # Test technical detection  
    assert manager.detect_query_type("How to install software") == "technical"
    
    # Test sales detection
    assert manager.detect_query_type("What's the price?") == "sales"
    
    # Test general fallback
    assert manager.detect_query_type("Hello") == "general"
    print("✓ Query type detection works")

def test_format_main_prompt():
    """Test prompt formatting"""
    manager = PromptManager()
    prompt = manager.format_main_prompt(
        query_type="general",
        context="test context",
        conversation_history="test history"
    )
    assert prompt is not None
    assert len(prompt) > 0
    print("✓ Main prompt formatted correctly")

def test_get_query_instruction():
    """Test getting query instructions"""
    manager = PromptManager()
    instruction = manager.get_query_type_instruction("general")
    assert instruction is not None
    print("✓ Query instruction retrieved")

if __name__ == "__main__":
    print("Running prompt_manager tests...\n")
    
    test_prompt_manager_creation()
    test_get_system_prompt() 
    test_detect_query_type()
    test_format_main_prompt()
    test_get_query_instruction()
    
    print("\n🎉 All prompt_manager tests passed!")