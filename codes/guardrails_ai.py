from guardrails import Guard, OnFailAction
from guardrails.hub import (
    ProfanityFree,
    CompetitorCheck,
    NoHallucinations,
    RestrictedTopics,
    RelevanceToPrompt,
    ToxicLanguage,
    PromptInjection,
)

# 1. Setup the Guard with specific "Actions"
# on_fail="fix" -> Guardrails will try to auto-edit the text
# on_fail="refuse" -> Guardrails will replace the text with a canned response
support_guard = Guard().use_many(
    ProfanityFree(on_fail="fix"),
    ToxicLanguage(threshold=0.5, on_fail=OnFailAction.EXCEPTION),
    PromptInjection(on_fail=OnFailAction.EXCEPTION),
    NoHallucinations(on_fail="refuse"),

    # Ensures the answer is relevant to the user query
    RelevanceToPrompt(on_fail="refuse"),

    RestrictedTopics(
        topics=["internal_passwords", "employee_home_addresses"],
        on_fail="refuse"
    ),
    CompetitorCheck(
        competitors=["CompetitorX", "CompetitorY"],
        on_fail="fix"
    )
)

def validate_response(user_query: str, llm_output: str):
    """
    The main entry point for all your nodes to clean LLM data.
    """
    try:
        # We use .parse to run the validation
        validation_outcome = support_guard.parse(
            llm_output=llm_output,
            metadata={"query": user_query}
        )

        # If it passes, return the clean text.
        # If it fails and couldn't be 'fixed', it returns a refusal message.
        return validation_outcome.validated_output

    except Exception as e:
        # Fallback for unexpected system errors
        return "I apologize, but I've encountered an internal safety check. How else can I help?"


# Example usage and testing
def test_bot_graph_guardrails():
    """Test the Guardrails-protected bot workflow"""
    guardrails_bot = BotGraphGuardrails()

    # Test cases
    test_cases = [
        "Hello, how can you help me today?",  # Normal input
        "I need help with my account",        # Normal input
        "You're stupid and useless!",         # Potentially toxic
        "Solve this: 2+2=?",                  # Simple query
    ]

    for i, test_input in enumerate(test_cases):
        print(f"\n{'='*50}")
        print(f"Test Case {i+1}: {test_input}")
        print(f"{'='*50}")

        result = guardrails_bot.process_with_guardrails(test_input)
        pprint(result, width=100, depth=2)


if __name__ == "__main__":
    test_bot_graph_guardrails()
