# ==================== PROMPT MANAGER ====================
import os
import yaml
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Define constants (adjust these as needed)

config_path = os.path.join(os.path.dirname(__file__), 'config', 'prompt_config.yaml')
# PROMPT_CONFIG_PATH = "config/prompt_config.yaml" # Path to your YAML config file
BUSINESS_NAME = "Tigress Tech"  # Fallback business name
BUSINESS_LOCATION = "Nigeria"  # Fallback business location
CURRENCY = "NGN"  # Fallback currency

class PromptManager:
    # Original
    """
    def __init__(self, config_path): # =PROMPT_CONFIG_PATH):
        self.config = self._load_config(config_path)
        self.business_info = self.config.get('business', {})
    """


    # New
    def __init__(self, config_path=None):
        if config_path is None:
            # Default path if none provided
            config_path = os.path.join(os.path.dirname(__file__), 'config', 'prompt_config.yaml')

        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.business_info = self.config.get('business', {})
        # self.prompts = self.load_prompts()


    def _load_config(self, config_path):
        """Load unified YAML configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading prompt config: {e}")
            return self._create_fallback_config()

    def _create_fallback_config(self):
        """Create fallback configuration if YAML file is missing"""
        return {
            'business': {
                'name': BUSINESS_NAME,
                'location': BUSINESS_LOCATION,
                'currency': CURRENCY
            },
            'system_prompt': f"You are Tigra, AI assistant for {BUSINESS_NAME} in {BUSINESS_LOCATION}.",
            'query_types': {
                'general': {'instruction': 'Provide helpful information'}
            }
        }

    def get_system_prompt(self):
        """Get the base system prompt"""
        system_prompt = self.config.get('system_prompt', '')
        return system_prompt.format(
            business_name=self.business_info.get('name', BUSINESS_NAME),
            business_location=self.business_info.get('location', BUSINESS_LOCATION),
            currency=self.business_info.get('currency', CURRENCY)
        )

    def get_query_type_instruction(self, query_type):
        """Get specialized instruction for query type"""
        query_types = self.config.get('query_types', {})
        query_config = query_types.get(query_type, query_types.get('general', {}))
        return query_config.get('instruction', 'Provide helpful information')

    def get_privacy_guidance(self):
        """Get privacy protection guidelines"""
        privacy = self.config.get('privacy_protection', {})
        guidelines = privacy.get('redaction_guidelines', [])
        return "\n".join([f"- {guideline}" for guideline in guidelines])

    def format_main_prompt(self, query_type, context="", conversation_history="", max_tokens=1024):
        """Format the main prompt template"""
        template = self.config.get('prompt_templates', {}).get('main_template', """
        {system_prompt}

        RESPONSE GUIDELINES:
        {specialized_guidance}

        Context: {context}

        Conversation history: {conversation_history}

        Assistant:
        """)

        return template.format(
            system_prompt=self.get_system_prompt(),
            specialized_guidance=self.get_query_type_instruction(query_type),
            privacy_guidance=self.get_privacy_guidance(),
            context=context,
            conversation_history=conversation_history,
            query_type=query_type,
            max_tokens=max_tokens
        )

    def detect_query_type(self, user_input):
        """Detect the type of query based on content"""
        user_input_lower = user_input.lower()
        query_types = self.config.get('query_types', {})

        # Check for complaint keywords
        complaint_words = ['broken', 'not working', 'issue', 'problem', 'complaint', 'angry', 'unhappy']
        if any(word in user_input_lower for word in complaint_words):
            return 'complaint'

        # Check for technical keywords
        technical_words = ['install', 'setup', 'configure', 'error', 'fix', 'repair', 'technical']
        if any(word in user_input_lower for word in technical_words):
            return 'technical'

        # Check for sales keywords
        sales_words = ['price', 'cost', 'buy', 'purchase', 'order', 'sales', 'available']
        if any(word in user_input_lower for word in sales_words):
            return 'sales'

        # Check for report keywords
        report_words = ['report', 'summary', 'status', 'update', 'inventory', 'sales data']
        if any(word in user_input_lower for word in report_words):
            return 'report'

        return 'general'
