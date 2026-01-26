# tests/unit/test_escalation_evaluator.py
import pytest
from escalation_evaluator import EscalationEvaluator

@pytest.fixture
def evaluator():
    return EscalationEvaluator()

def test_escalation_frustration(evaluator):
    """Tests if high-frustration language triggers escalation."""
    input_text = "I am extremely angry with your service and I want to speak to a manager!"
    should_escalate, reason, score = evaluator.evaluate_escalation_need(
        current_message=input_text,
        conversation_history=[],
        sender="user"
    )

    assert should_escalate is True
    assert "sentiment" in reason.lower() or "frustration" in reason.lower()
    assert score > 0.8

def test_escalation_complexity(evaluator):
    """Tests if queries requiring human 'discretion' trigger escalation."""
    input_text = "Can you make an exception for my specific case and override the policy?"
    should_escalate, reason, score = evaluator.evaluate_escalation_need(
        current_message=input_text,
        conversation_history=[],
        sender="user"
    )

    assert should_escalate is True
    assert "discretion" in reason or "judgement" in reason

def test_no_escalation_normal_query(evaluator):
    """Ensures simple queries do not trigger escalation flags."""
    input_text = "What time do you open?"
    should_escalate, _, _ = evaluator.evaluate_escalation_need(
        current_message=input_text,
        conversation_history=[],
        sender="user"
    )
    assert should_escalate is False
