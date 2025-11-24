# tests/unit/test_bot_state.py
import pytest
from codes.states.bot_state import (
    create_default_state, 
    validate_state, 
    update_state_turn, 
    reset_escalation_state
)
from langchain_core.messages import HumanMessage, AIMessage

class TestBotState:
    
    def test_create_default_state_initialization(self):
        """Test default state initialization"""
        state = create_default_state()
        assert state["user_input"] == ""
        assert state["response"] == ""
        assert state["needs_rag"] == True
        assert state["escalation_score"] == 0.0
        assert state["messages"] == []
        assert state["conversation_context"] == {}
    
    def test_validate_state_with_valid_state(self):
        """Test state validation"""
        state = create_default_state()
        assert validate_state(state) == True
    
    def test_validate_state_with_invalid_fields(self):
        """Test state validation with invalid data"""
        state = create_default_state()
        state["user_input"] = None
        assert validate_state(state) == False
        
        state = create_default_state()
        state["needs_rag"] = "not_a_bool"
        assert validate_state(state) == False
    
    def test_update_state_turn_increments_counter(self):
        """Test turn update functionality"""
        state = create_default_state()
        updated_state = update_state_turn(state)
        assert updated_state["turn_count"] == state["turn_count"] + 1
        assert updated_state["failed_attempts"] == 0
    
    def test_reset_escalation_state_clears_fields(self):
        """Test escalation state reset"""
        state = create_default_state()
        state["requires_human_escalation"] = True
        state["escalation_reason"] = "Test reason"
        
        reset_state = reset_escalation_state(state)
        assert reset_state["requires_human_escalation"] == False
        assert reset_state["escalation_reason"] is None
    
    def test_state_with_messages(self):
        """Test state with LangChain messages"""
        state = create_default_state()
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there")
        ]
        state["messages"] = messages
        assert validate_state(state) == True
        assert len(state["messages"]) == 2

if __name__ == "__main__":
    test_instance = TestBotState()
    
    tests = [
        "test_create_default_state_initialization",
        "test_validate_state_with_valid_state", 
        "test_validate_state_with_invalid_fields",
        "test_update_state_turn_increments_counter",
        "test_reset_escalation_state_clears_fields",
        "test_state_with_messages"
    ]
    
    print("Running BotState tests...")
    for method_name in tests:
        try:
            getattr(test_instance, method_name)()
            print(f"✓ {method_name}")
        except Exception as e:
            print(f"✗ {method_name}: {e}")
    
    print("BotState tests completed!")