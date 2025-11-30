# streamlit_app.py
import streamlit as st
import sys
import os
from pathlib import Path
from functools import lru_cache

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "codes"))

# Set page config
st.set_page_config(
    page_title="Customer Support Bot",
    page_icon="🤖",
    layout="centered"
)

class StreamlitBot:
    """Simple wrapper class for the bot functionality with caching"""

    def __init__(self):
        self.initialized = False
        self.components = {}

    @st.cache_resource(show_spinner=False)
    def _load_components(_self):
        """Cached component loading"""
        try:
            from codes.initialize import prompt_manager, rag_system, rag_success
            from codes.guardrails_ai import validate_input, validate_output

            return {
                'prompt_manager': prompt_manager,
                'rag_system': rag_system,
                'rag_success': rag_success,
                'validate_input': validate_input,
                'validate_output': validate_output
            }
        except Exception as e:
            raise Exception(f"Component loading failed: {e}")

    def initialize(self):
        """Initialize bot components with caching"""
        try:
            self.components = self._load_components()
            self.initialized = True
            return True, "Bot initialized successfully!"
        except Exception as e:
            return False, f"Initialization failed: {e}"

    @st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
    def _get_cached_rag_response(_self, prompt):
        """Cached RAG responses"""
        try:
            return _self.components['rag_system'].get_response(prompt)
        except Exception:
            return None

    @st.cache_data(ttl=3600, show_spinner=False)
    def _get_fallback_response(_self, prompt):
        """Cached fallback responses"""
        prompt_lower = prompt.lower()

        if any(greet in prompt_lower for greet in ['hello', 'hi', 'hey']):
            return "Hello! How can I help you with customer support today?"
        elif 'help' in prompt_lower:
            return "I can assist with:\n- Technical issues\n- Account questions\n- Service information\n- General support"
        elif any(thank in prompt_lower for thank in ['thank', 'thanks']):
            return "You're welcome! Is there anything else I can help with?"
        elif 'service' in prompt_lower:
            return "We offer various services including technical support, account management, and general assistance."
        else:
            return f"I understand you're asking about: {prompt}. How can I assist you further?"

    def process_message(self, prompt):
        """Process user message and return response with caching"""
        try:
            # Validate input
            input_validation = self.components['validate_input'](prompt)
            if not input_validation["valid"]:
                return "I apologize, but I cannot process that input due to content policy violations."

            validated_prompt = input_validation["input"]

            # Generate response with caching
            if self.components['rag_success']:
                rag_response = self._get_cached_rag_response(validated_prompt)
                if rag_response:
                    response = rag_response
                else:
                    response = f"I found some information about: {validated_prompt}. How can I help you further?"
            else:
                response = self._get_fallback_response(validated_prompt)

            # Validate output
            output_validation = self.components['validate_output'](response)
            if not output_validation["valid"]:
                return "I apologize, but I cannot provide that response due to safety constraints."

            return output_validation["output"]

        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

# Title
st.title("🤖 Customer Support Assistant")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'bot' not in st.session_state:
    st.session_state.bot = StreamlitBot()

# Sidebar
with st.sidebar:
    st.header("Settings")

    if st.button("Initialize Bot"):
        with st.spinner("Loading AI system..."):
            success, message = st.session_state.bot.initialize()
            if success:
                st.success(message)
                if st.session_state.bot.components.get('rag_success'):
                    st.info("✅ RAG system loaded with knowledge base")
                else:
                    st.warning("⚠️ RAG system not available - using basic mode")
            else:
                st.error(message)

    st.markdown("---")
    st.markdown("**Status:**")
    if st.session_state.bot.initialized:
        st.success("✅ Ready")
        # Show cache info
        st.caption("🔄 Caching enabled for better performance")
    else:
        st.warning("❌ Not initialized")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Cache management
    if st.session_state.bot.initialized:
        st.markdown("---")
        st.header("Cache Management")
        if st.button("Clear Cache"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success("Cache cleared!")
            st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Check if bot is initialized
    if not st.session_state.bot.initialized:
        with st.chat_message("assistant"):
            st.write("Please initialize the bot first using the sidebar button.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Please initialize the bot first using the sidebar button."
        })
    else:
        # Generate bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.bot.process_message(prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
