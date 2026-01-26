"""
Simple tests for prompt_manager.py
"""

import pytest
from unittest.mock import patch, mock_open
from prompt_manager import PromptManager

MOCK_YAML = """
base_persona:
  name: "Tigra"
supervisor:
  system_prompt: "Route this: {user_input}"
main_template: "Hello {name}, you asked: {user_input}"
query_types:
  general: {}
"""

def test_prompt_formatting():
    """Tests template variable injection."""
    with patch("builtins.open", mock_open(read_data=MOCK_YAML)):
        # Initialize with dummy path
        pm = PromptManager(config_path="dummy.yaml")

        # Test Routing Prompt
        routing = pm.format_routing_prompt(user_input="Fix my wifi")
        assert routing == "Route this: Fix my wifi"

        # Test Main Prompt
        main = pm.format_main_prompt(
            query_type="general",
            user_input="How are you?",
            context="No context"
        )
        assert "Hello Tigra" in main
        assert "How are you?" in main

def test_load_config_error_handling():
    """Ensures PromptManager returns empty dict on file errors instead of crashing."""
    with patch("builtins.open", side_effect=Exception("File not found")):
        pm = PromptManager(config_path="invalid.yaml")
        assert pm.config == {}
