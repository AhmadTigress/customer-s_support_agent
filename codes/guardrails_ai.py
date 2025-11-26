# codes/graph/bot_graph_guardrails.py
import os
import warnings

warnings.filterwarnings("ignore")

os.environ["OTEL_SDK_DISABLED"] = "true"

from typing import Any, Dict, List
from pprint import pprint

from graph.bot_graph import create_workflow
from states.bot_state import AgentState
from guardrails import Guard, OnFailAction
from guardrails.hub import ToxicLanguage, UnusualPrompt, ProfanityFree


class BotGraphGuardrails:
    """Guardrails integration for the bot workflow graph"""
    
    def __init__(self):
        # Initialize the main workflow
        self.workflow = create_workflow()
        
        # Initialize Guardrails validators
        self.input_guard = Guard().use(
            ToxicLanguage(threshold=0.7, on_fail=OnFailAction.FILTER),
            UnusualPrompt(threshold=0.8, on_fail=OnFailAction.EXCEPTION),
            ProfanityFree(on_fail=OnFailAction.FILTER)
        )
        
        self.output_guard = Guard().use(
            ToxicLanguage(threshold=0.6, on_fail=OnFailAction.FILTER),
            ProfanityFree(on_fail=OnFailAction.FILTER)
        )
    
    def validate_input(self, user_input: str) -> Dict[str, Any]:
        """Validate user input before processing"""
        try:
            validated_input = self.input_guard.validate(user_input)
            return {
                "valid": validated_input.validation_passed,
                "input": validated_input.validated_output,
                "errors": validated_input.error
            }
        except Exception as e:
            return {
                "valid": False,
                "input": user_input,
                "errors": str(e)
            }
    
    def validate_output(self, bot_output: str) -> Dict[str, Any]:
        """Validate bot output before sending to user"""
        try:
            validated_output = self.output_guard.validate(bot_output)
            return {
                "valid": validated_output.validation_passed,
                "output": validated_output.validated_output,
                "errors": validated_output.error
            }
        except Exception as e:
            return {
                "valid": False,
                "output": "I apologize, but I encountered an issue with my response.",
                "errors": str(e)
            }
    
    def process_with_guardrails(self, user_input: str) -> Dict[str, Any]:
        """Process user input through the workflow with Guardrails protection"""
        # Step 1: Validate input
        input_validation = self.validate_input(user_input)
        if not input_validation["valid"]:
            return {
                "status": "blocked",
                "message": "Your input was blocked due to content policy violations.",
                "original_input": user_input,
                "validation_errors": input_validation["errors"]
            }
        
        # Step 2: Process through workflow
        try:
            initial_state = {"user_input": input_validation["input"]}
            workflow_result = self.workflow.invoke(initial_state)
            
            # Step 3: Validate output
            if "final_response" in workflow_result:
                output_validation = self.validate_output(workflow_result["final_response"])
                
                return {
                    "status": "success",
                    "validated_input": input_validation["input"],
                    "workflow_result": workflow_result,
                    "validated_output": output_validation["output"],
                    "output_valid": output_validation["valid"],
                    "output_errors": output_validation["errors"]
                }
            else:
                return {
                    "status": "error", 
                    "message": "Workflow did not produce a final response",
                    "workflow_result": workflow_result
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Workflow processing failed: {str(e)}",
                "validated_input": input_validation["input"]
            }


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