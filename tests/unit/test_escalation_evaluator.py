"""
Simple unit tests for escalation_evaluator.py
"""

from escalation_evaluator import EscalationEvaluator


def test_basic_functionality():
    """Test basic functionality of the escalation evaluator"""
    evaluator = EscalationEvaluator()
    
    # Test 1: Normal message should not escalate
    result = evaluator.evaluate_escalation_need(
        current_message="Hello, how are you?",
        conversation_history=[],
        sender="test_user"
    )
    escalate, reason, confidence = result
    assert escalate is False
    assert confidence < 0.65
    
    # Test 2: Explicit human request should escalate
    result = evaluator.evaluate_escalation_need(
        current_message="I want to speak to a human agent",
        conversation_history=[],
        sender="test_user"
    )
    escalate, reason, confidence = result
    assert escalate is True
    assert "human" in reason.lower()
    
    # Test 3: Angry message should escalate
    result = evaluator.evaluate_escalation_need(
        current_message="This is terrible service! I'm furious!",
        conversation_history=[],
        sender="test_user"
    )
    escalate, reason, confidence = result
    assert escalate is True
    assert "negative" in reason.lower() or "emotional" in reason.lower()
    
    print("✓ All basic functionality tests passed!")


def test_sentiment_analysis():
    """Test sentiment analysis component"""
    evaluator = EscalationEvaluator()
    
    # Test negative sentiment
    score, reason = evaluator._analyse_sentiment("This is awful!", [])
    assert score > 0.8
    
    # Test neutral sentiment
    score, reason = evaluator._analyse_sentiment("Hello there", [])
    assert score == 0.0
    
    print("✓ Sentiment analysis tests passed!")


def test_complexity_analysis():
    """Test complexity analysis component"""
    evaluator = EscalationEvaluator()
    
    # Test complex query
    score, reason = evaluator._analyse_complexity(
        "I need help with API integration and database setup",
        []
    )
    assert score > 0.6
    
    # Test simple query
    score, reason = evaluator._analyse_complexity("What time do you open?", [])
    assert score == 0.0
    
    print("✓ Complexity analysis tests passed!")


if __name__ == "__main__":
    test_basic_functionality()
    test_sentiment_analysis()
    test_complexity_analysis()
    print("🎉 All tests passed!")