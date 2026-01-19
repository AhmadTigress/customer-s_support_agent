# codes/prompt_manager.py
import os
import yaml
import logging

logger = logging.getLogger(__name__)

class PromptManager:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config', 'prompt_config.yaml')

        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.base_info = self.config.get('base_persona', {})

    def _load_config(self, config_path):
        """Load unified YAML configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading prompt config: {e}")
            return {}

    def format_routing_prompt(self, user_input, conversation_history=""):
        """
        NEW: Formats the prompt used by the Supervisor to decide the next node.
        Matches supervisor.py requirements.
        """
        routing_cfg = self.config.get('supervisor', {})
        template = routing_cfg.get('system_prompt', "Route the query: {user_input}")

        return template.format(
            user_input=user_input,
            conversation_history=conversation_history
        )

    def format_main_prompt(self, query_type, context="", conversation_history="", user_input=""):
        """
        UPDATED: Matches the variables in your updated prompt_config.yaml
        and the call signatures in bot_nodes.py.
        """
        # 1. Get the main structural template
        template = self.config.get('main_template', "{system_message}\n{instructions}\n{context}")

        # 2. Get instructions for the specific agent (technical, billing, general)
        query_configs = self.config.get('query_types', {})
        agent_cfg = query_configs.get(query_type, query_configs.get('general', {}))

        # 3. Format the final string using YAML keys
        return template.format(
            system_message=self.config.get('supervisor', {}).get('system_prompt', ""), # Optional global context
            name=self.base_info.get('name', "Tigra"),
            business=self.base_info.get('business', "Tigress Tech"),
            location=self.base_info.get('location', "Nigeria"),
            currency=self.base_info.get('currency', "₦"),
            context=context,
            conversation_history=conversation_history,
            user_input=user_input,
            instructions=agent_cfg.get('instruction', "Help the user.")
        )
