import pytest
from unittest.mock import MagicMock, patch
from guardrails_ai import validate_response

# ==================== UNIT TESTS: guardrails_ai.py ====================

@patch('guardrails_ai.support_guard.parse')
def test_validate_response_pass(mock_parse):
    """Verifies that valid LLM output is returned as-is."""
    # Setup mock to return a successful validation
    mock_outcome = MagicMock()
    mock_outcome.validated_output = "The store opens at 9 AM."
    mock_parse.return_value = mock_outcome

    query = "What time do you open?"
    llm_output = "The store opens at 9 AM."

    result = validate_response(query, llm_output)

    assert result == "The store opens at 9 AM."
    mock_parse.assert_called_once_with(
        llm_output=llm_output,
        metadata={"query": query}
    )

@patch('guardrails_ai.support_guard.parse')
def test_validate_response_refusal(mock_parse):
    """Verifies that Guardrails 'refuse' logic works for restricted topics."""
    # Setup mock to return the standard refusal message
    mock_outcome = MagicMock()
    mock_outcome.validated_output = "I apologize, but I cannot answer that."
    mock_parse.return_value = mock_outcome

    query = "What is the manager's home address?"
    llm_output = "The address is 123 Private St."

    result = validate_response(query, llm_output)

    assert "cannot answer" in result.lower()

@patch('guardrails_ai.support_guard.parse')
def test_validate_response_toxic_exception(mock_parse):
    """Verifies fallback when OnFailAction.EXCEPTION is triggered by toxic language."""
    # Simulate an exception being raised by the ToxicLanguage validator
    mock_parse.side_effect = Exception("Toxic language detected")

    result = validate_response("query", "Some very rude text")

    # Check that it returns the hardcoded safety fallback in your try-except block
    assert "internal safety check" in result

@patch('guardrails_ai.support_guard.parse')
def test_validate_response_competitor_fix(mock_parse):
    """Verifies that 'fix' actions (like CompetitorCheck) return the modified text."""
    mock_outcome = MagicMock()
    # Simulate CompetitorX being replaced or removed
    mock_outcome.validated_output = "Our product is better than the rest."
    mock_parse.return_value = mock_outcome

    result = validate_response("Compare products", "Our product is better than CompetitorX.")

    assert "CompetitorX" not in result
    assert "better than the rest" in result

def test_guard_initialization():
    """Checks if the guard is initialized with the expected number of validators."""
    from guardrails_ai import support_guard
    # Accessing the internal validators list to ensure security policy is loaded
    validators = support_guard.validators
    assert len(validators) >= 7  # Profanity, Toxic, Injection, Hallucination, Relevance, Topics, Competitor
