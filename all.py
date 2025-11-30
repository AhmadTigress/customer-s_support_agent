# API SECTION

# huggingface_api.py
# Import the required modules
import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Get the Hugging Face API key from the environment variables
HF_TOKEN = os.getenv('HUGGINGFACE_API_KEY')
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

def huggingface_completion(prompt: str) -> dict:
    '''
    Call Hugging Face API for text completion
    Parameters:
        - prompt: user query (str)
    Returns:
        - dict
    '''
    # BUG FIX: Validate input
    if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
        print("Hugging Face API call failed: Empty or invalid prompt")
        return {'status': 0, 'response': ''}

    try:
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

        # BUG FIX: Set pad token to prevent crashes
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,  # BUG FIX: Use float16 to save memory
            token=HF_TOKEN,
        )

        # Create a pipeline for text generation
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id  # BUG FIX: Prevent padding issues
        )

        # Generate response
        response = pipe(
            prompt,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            return_full_text=False,  # BUG FIX: Don't include prompt in output
        )

        # Extract the generated text
        output_text = response[0]["generated_text"].strip()

        # Print a success message with the response from the Hugging Face API call
        print(f"Hugging Face API call successful. Response: {output_text[:100]}...")

        # BUG FIX: Clean up memory
        del model, tokenizer, pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Return a dictionary with the status and the content of the response
        return {
            'status': 1,
            'response': output_text
        }

    except torch.cuda.OutOfMemoryError:
        # BUG FIX: Handle GPU memory errors
        print("Hugging Face API call failed: CUDA out of memory")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return {'status': 0, 'response': ''}

    except Exception as e:
        # Print any error that occurs during the Hugging Face API call
        print(f"Hugging Face API call failed. Error: {e}")

        # BUG FIX: Clean up on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Return a dictionary with the status and an empty response
        return {'status': 0, 'response': ''}




# matrix_api.py
# Import the required modules
import os
import requests
import logging
import threading
import time
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the environment variables from the .env file
load_dotenv()

homeserver = os.getenv("MATRIX_HOMESERVER")
user = os.getenv("MATRIX_USER")
password = os.getenv("MATRIX_PASSWORD")
room_id = os.getenv("MATRIX_ROOM_ID")

# ==================== MATRIX API CLIENT ====================
class MatrixClient:
    """Handles Matrix server communication"""

    def __init__(self, homeserver, user, password):
        self.homeserver = homeserver
        self.user = user
        self.password = password
        self.access_token = None
        self._shutdown_event = threading.Event()
        self._session = requests.Session()  # BUG FIX: Use session for connection pooling
        self._sync_timeout = 30000  # 30 seconds
        self._error_count = 0
        self._max_retries = 3

    def _make_request(self, method, url, **kwargs):
        """BUG FIX: Centralized request handling with retry logic"""
        for attempt in range(self._max_retries):
            try:
                response = self._session.request(method, url, **kwargs)
                response.raise_for_status()
                self._error_count = 0  # Reset error count on success
                return response
            except requests.exceptions.RequestException as e:
                self._error_count += 1
                logger.warning(f"Request failed (attempt {attempt + 1}/{self._max_retries}): {e}")

                if attempt == self._max_retries - 1 or self._shutdown_event.is_set():
                    raise

                # Exponential backoff
                wait_time = (2 ** attempt) + 1
                time.sleep(wait_time)

    def login(self):
        """Authenticate and get access token"""
        url = f"{self.homeserver}/_matrix/client/v3/login"
        data = {
            "type": "m.login.password",
            "user": self.user,
            "password": self.password,
        }
        res = self._make_request("POST", url, json=data)
        self.access_token = res.json()["access_token"]
        logger.info("Successfully logged into Matrix")
        return self.access_token

    def send_message(self, room_id, message):
        """Send message to a room"""
        if not self.access_token:
            raise ValueError("Not logged in - call login() first")

        url = f"{self.homeserver}/_matrix/client/v3/rooms/{room_id}/send/m.room.message"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        data = {
            "msgtype": "m.text",
            "body": message,
        }
        self._make_request("POST", url, headers=headers, json=data)
        logger.info(f"Message sent to room {room_id}")

    def sync_messages(self, sync_token=None):
        """Sync with matrix server to get new messages with error handling"""
        if not self.access_token:
            raise ValueError("Not logged in - call login() first")

        url = f"{self.homeserver}/_matrix/client/v3/sync"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        params = {"timeout": self._sync_timeout}
        if sync_token:
            params["since"] = sync_token

        res = self._make_request("GET", url, headers=headers, params=params)
        data = res.json()

        new_messages = []
        for room_id, room_info in data.get("rooms", {}).get("join", {}).items():
            room_events = room_info.get("timeline", {}).get("events", [])
            for event in room_events:
                if (event.get("type") == "m.room.message" and
                    event.get("sender") != self.user and
                    event.get("content", {}).get("msgtype") == "m.text"):
                    new_messages.append(event)

        next_batch = data.get("next_batch")
        return next_batch, new_messages

    def join_room(self, room_id):
        """Join a room that the bot has been invited to"""
        if not self.access_token:
            raise ValueError("Not logged in - call login() first")

        url = f"{self.homeserver}/_matrix/client/v3/join/{room_id}"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        self._make_request("POST", url, headers=headers)
        logger.info(f"Joined room: {room_id}")

    def get_invited_rooms(self, sync_token=None):
        """Check for room invitations and auto-join them"""
        if not self.access_token:
            raise ValueError("Not logged in - call login() first")

        url = f"{self.homeserver}/_matrix/client/v3/sync"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        params = {"timeout": self._sync_timeout}
        if sync_token:
            params["since"] = sync_token

        res = self._make_request("GET", url, headers=headers, params=params)
        data = res.json()

        invited_rooms = []
        for room_id, invite_info in data.get("rooms", {}).get("invite", {}).items():
            invited_rooms.append(room_id)
            # BUG FIX: Auto-join invited rooms
            try:
                self.join_room(room_id)
                logger.info(f"Auto-joined invited room: {room_id}")
            except Exception as e:
                logger.error(f"Failed to auto-join room {room_id}: {e}")

        return invited_rooms

    def listen_for_messages(self, process_message_callback=None):
        """Continuously listen for messages with graceful shutdown and error recovery"""
        sync_token = None
        logger.info("Starting to listen for messages...")

        while not self._shutdown_event.is_set():
            try:
                # BUG FIX: Check for room invitations on each iteration
                self.get_invited_rooms(sync_token)

                # Get new messages
                sync_token, messages = self.sync_messages(sync_token)

                # Process each message
                for message in messages:
                    if self._shutdown_event.is_set():
                        break  # Exit early if shutdown requested

                    room_id = message.get("room_id")
                    sender = message.get("sender")
                    content = message.get("content", {})
                    body = content.get("body", "").strip()

                    # Ignore messages from ourselves
                    if sender == self.user:
                        continue

                    logger.info(f"Received message from {sender}: {body}")

                    # If a callback function is provided, use it
                    if process_message_callback:
                        try:
                            process_message_callback(message)
                        except Exception as e:
                            logger.error(f"Error in message callback: {e}")
                            # Send error message to user
                            try:
                                self.send_message(room_id, "I encountered an error processing your message. Please try again.")
                            except Exception as send_error:
                                logger.error(f"Failed to send error message: {send_error}")
                    else:
                        # Default behavior for simple responses
                        if body.lower() in ["hello", "hi", "hey", "hello!"]:
                            self.send_message(room_id, "Hello! How can I help you today?")

                # BUG FIX: Small delay to prevent tight loop
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                if self._shutdown_event.is_set():
                    break

                logger.error(f"Network error in message listening: {e}")
                # Wait before retrying with exponential backoff
                wait_time = min(30, (2 ** self._error_count) + 1)
                logger.info(f"Waiting {wait_time} seconds before retry...")
                self._shutdown_event.wait(wait_time)

            except Exception as e:
                if self._shutdown_event.is_set():
                    break

                logger.error(f"Unexpected error in message listening: {e}")
                # Wait before retrying
                self._shutdown_event.wait(5)

        logger.info("Message listener stopped gracefully")

    def stop_listening(self):
        """Gracefully stop the message listening loop"""
        logger.info("Shutting down message listener...")
        self._shutdown_event.set()

    def __enter__(self):
        """Context manager entry - auto login"""
        self.login()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto cleanup"""
        self.stop_listening()
        # BUG FIX: Close session to prevent resource leaks
        self._session.close()

# CONFIG SECTION
# prompt_config.yaml
# Prompt Configuration for TIGRESS TECH LABS
# Combines RAG and Matrix Bot functionality

version: "2.0"
integration_date: "2024-01-15"
config_type: "unified_rag_matrix"

business:
  name: "TIGRESS TECH LABS"
  location: "Nigeria"
  currency: "₦"
  industry: "Computer Sales and IT Services"

bot_identity:
  name: "Tigra"
  role: "AI Assistant"
  personality: "Professional, knowledgeable, customer-focused, culturally appropriate for Nigerian context"
  communication_style: "Concise but informative, Nigerian English, professional yet friendly"

core_llm_config:
  temperature: 0.7
  max_tokens: 1024
  top_p: 0.9
  do_sample: true
  model: "Meta-Llama-3-8B-Instruct"

product_catalog:
  brands: ["Dell", "HP", "Acer", "MacBook", "Samsung", "Lenovo"]
  categories:
    laptops: ["Gaming", "Business", "Student", "Premium"]
    desktops: ["All-in-One", "Tower", "Workstation"]
    accessories: ["Batteries", "Chargers", "Bags", "Peripherals"]
  services:
    software: ["OS Installation", "MS Office Activation", "Software Setup"]
    hardware: ["Repairs", "Upgrades", "Maintenance"]
    support: ["Technical Support", "IT Consulting", "Remote Assistance"]

system_prompt: |
  You are Tigra, the AI assistant for TIGRESS TECH LABS in Nigeria.
  You handle customer complaints, technical support, sales inquiries, and general customer service.

  CORE RESPONSIBILITIES:
  - Resolve customer issues with empathy and efficiency
  - Provide technical support and troubleshooting
  - Assist with product information and sales inquiries
  - Offer business information and routing
  - Generate appropriate reports and insights

  PERSONALITY GUIDELINES:
  - Professional yet approachable
  - Knowledgeable about technology products
  - Culturally appropriate for Nigerian context
  - Patient and helpful in all interactions
  - Honest about limitations and capabilities

response_guidelines:
  general:
    - "Always acknowledge the customer's concern first"
    - "Provide clear, actionable solutions"
    - "Use ₦ for pricing when available"
    - "Maintain brand voice: helpful, expert, trustworthy"
    - "Be specific about services and products"

  complaint_handling:
    - "Apologize for inconveniences sincerely"
    - "Provide concrete solutions and timelines"
    - "Offer follow-up actions and ensure satisfaction"
    - "Escalate complex issues appropriately"

  technical_support:
    - "Provide step-by-step troubleshooting"
    - "Suggest basic fixes first (restart, updates)"
    - "Know when to recommend professional service"
    - "Explain technical concepts simply"

  sales_inquiries:
    - "Highlight product features and benefits"
    - "Discuss availability and delivery options"
    - "Mention warranty and support services"
    - "Upsell accessories when appropriate"

query_types:
  complaint:
    name: "Customer Complaint"
    instruction: |
      Focus on resolving issues with empathy. Apologize for inconveniences,
      provide concrete solutions, and ensure customer satisfaction.
    examples:
      - "My order hasn't arrived yet"
      - "The product stopped working after one week"
      - "I was overcharged for the service"

  technical:
    name: "Technical Support"
    instruction: |
      Provide technical assistance and troubleshooting. Be patient and explain
      concepts clearly. Know when to recommend professional service.
    examples:
      - "My laptop won't turn on"
      - "How do I install Windows?"
      - "Software keeps crashing"

  sales:
    name: "Sales Inquiry"
    instruction: |
      Assist with product information, comparisons, and purchasing decisions.
      Be knowledgeable about specifications and pricing.
    examples:
      - "What laptops do you have under 200k?"
      - "Compare HP and Dell for gaming"
      - "Do you offer installment payments?"

  general:
    name: "General Inquiry"
    instruction: |
      Provide business information, operating hours, and general assistance.
      Route to appropriate departments when needed.
    examples:
      - "What are your business hours?"
      - "Where is your location?"
      - "How do I contact support?"

  report:
    name: "Report Generation"
    instruction: |
      Present data clearly and professionally. Highlight key insights and trends
      while maintaining appropriate data confidentiality.
    examples:
      - "Sales report for this month"
      - "Inventory status update"
      - "Service requests summary"

conversation_management:
  welcome_messages:
    - "Hello! Welcome to TIGRESS TECH LABS. I'm Tigra, how can I help you today? 🐯"
    - "Good day! Thank you for contacting TIGRESS TECH LABS. What can I assist you with?"
    - "Welcome! I'm here to help with your technology needs. How can I be of service?"

  follow_up_questions:
    - "Is there anything else I can help you with today?"
    - "Would you like information about any other products or services?"
    - "Do you have any other questions I can answer?"

  closing_statements:
    - "Thank you for choosing TIGRESS TECH LABS! Have a great day. 🐯"
    - "I'm glad I could assist you. Feel free to reach out anytime!"
    - "Your satisfaction is our priority. Come back soon!"

privacy_protection:
  sensitive_topics:
    - "password"
    - "credit card"
    - "bank details"
    - "personal information"
    - "confidential data"
    - "employee contacts"
    - "internal pricing"

  redaction_guidelines:
    - "Never share employee personal contact information"
    - "Never share internal pricing strategies"
    - "Never share specific customer data examples"
    - "Use generic terms for sensitive processes"

  safe_responses:
    - "For security reasons, I can't share that information publicly."
    - "That information requires verification. Please contact our support team."
    - "I recommend speaking with our management team for detailed information."

error_handling:
  unknown_query: "I'm not sure I understand completely. Could you rephrase or provide more details?"
  technical_error: "I'm experiencing some technical difficulties. Please try again in a moment."
  complex_request: "This seems complex. Let me connect you with our specialists for better assistance."
  limitation_admission: "I don't have access to that specific information. Please contact our support team."

business_operations:
  hours: |
    Monday - Friday: 8:00 AM - 6:00 PM
    Saturday: 9:00 AM - 4:00 PM
    Sunday: Closed (Emergency support available)

  contact_channels:
    - "Phone: [Support number available upon request]"
    - "Email: [Support email available upon request]"
    - "Showroom: Computer Village, Ikeja, Lagos"

# RAG INTEGRATION CONFIGURATION
rag_config:
  knowledge_sources:
    - "product_catalog"
    - "technical_documentation"
    - "pricing_guides"
    - "service_procedures"

  retrieval_strategy: "semantic_similarity"
  max_context_length: 2500
  similarity_threshold: 0.7

# MATRIX BOT INTEGRATION
matrix_config:
  message_truncation: 1500
  rate_limit_delay: 2.5
  max_concurrent_requests: 5

prompt_templates:
  main_template: |
    SYSTEM: You are Tigra, AI assistant for TIGRESS TECH LABS in Nigeria.

    BUSINESS CONTEXT:
    - Location: Nigeria
    - Currency: ₦
    - Industry: Computer Sales and IT Services

    RESPONSE GUIDELINES:
    {response_guidelines}

    KNOWLEDGE CONTEXT:
    {rag_context}

    CONVERSATION HISTORY:
    {conversation_history}

    CURRENT QUERY TYPE: {query_type}

    USER QUERY: {user_input}

    ASSISTANT RESPONSE (professional, helpful, under {max_tokens} tokens):

  rag_enhanced_template: |
    SYSTEM: Tigra - TIGRESS TECH LABS Assistant

    RELEVANT KNOWLEDGE:
    {context}

    QUERY TYPE: {query_type}

    CONVERSATION:
    User: {user_input}
    History: {conversation_history}

    RESPONSE GUIDELINES:
    - Use provided knowledge when relevant
    - Be honest about limitations
    - Maintain professional tone
    - Keep response concise

    Assistant:

  matrix_template: |
    SYSTEM: Tigra - TIGRESS TECH LABS Matrix Bot

    MESSAGE FROM: {sender}
    ROOM: {room_id}

    CONTEXT: {context}

    CONVERSATION: {conversation_history}

    USER MESSAGE: {user_input}

    RESPONSE (under 1500 chars, Nigerian context):

  fallback_template: |
    SYSTEM: Tigra - TIGRESS TECH LABS Assistant

    I don't have specific information about that topic in my knowledge base.

    USER QUERY: {user_input}

    Please contact our support team for detailed assistance.

    Assistant:

template_variables:
  required:
    - "user_input"
    - "query_type"
    - "conversation_history"

  optional:
    - "rag_context"
    - "sender"
    - "room_id"
    - "context"

  limits:
    max_tokens: 1024
    max_matrix_length: 1500
    max_history_items: 10

special_commands:
  help: "Type 'help' for available commands"
  products: "Type 'products' for available brands and categories"
  services: "Type 'services' for our service offerings"
  support: "Type 'support' for technical help"
  hours: "Type 'hours' for business hours"
  contact: "Type 'contact' for support channels"

metadata:
  created_date: "2024-01-15"
  last_updated: "2024-01-15"
  maintainer: "TIGRESS TECH LABS IT Team"
  version_notes: "Debugged unified configuration with proper YAML syntax and RAG integration"
  valid_yaml: true



# BOT STATES
# bot_state.py
# ==================== STATE DEFINITION ====================
# bot_state.py (state only - no APIs)
import datetime
from typing import TypedDict, List, Dict, Any, Optional, Union
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
import operator
from typing_extensions import Annotated

class AgentState(TypedDict):
    # User input and conversation
    user_input: str
    messages: Annotated[List[BaseMessage], operator.add]
    response: str
    sender: str
    context: str
    query_type: str
    needs_rag: bool

    # Matrix integration
    matrix_room_id: str

    # Human escalation system
    requires_human_escalation: bool
    escalation_reason: Optional[str]
    human_handoff_complete: bool
    pending_human_response: bool

    # Conversation management
    conversation_context: Dict[str, Any]
    escalation_score: float
    failed_attempts: int
    user_sentiment: str
    complexity_score: float

    # Timing and metadata
    timestamp: str
    conversation_id: Optional[str]
    turn_count: int

# Default state initialization
def create_default_state() -> AgentState:
    """Create a properly initialized default state"""
    return {
        # User input and conversation
        "user_input": "",
        "messages": [],
        "response": "",
        "sender": "",
        "context": "",
        "query_type": "general",
        "needs_rag": True,

        # Matrix integration
        "matrix_room_id": "",

        # Human escalation system
        "requires_human_escalation": False,
        "escalation_reason": None,
        "human_handoff_complete": False,
        "pending_human_response": False,

        # Conversation management
        "conversation_context": {},
        "escalation_score": 0.0,
        "failed_attempts": 0,
        "user_sentiment": "neutral",
        "complexity_score": 0.0,

        # Timing and metadata
        "timestamp": datetime.datetime.now().isoformat(),
        "conversation_id": None,
        "turn_count": 0
    }

# State validation functions
def validate_state(state: AgentState) -> bool:
    """Validate state integrity"""
    try:
        # Required string fields should not be None
        required_strings = [
            state["user_input"],
            state["response"],
            state["sender"],
            state["context"],
            state["query_type"],
            state["matrix_room_id"],
            state["user_sentiment"]
        ]

        # All required strings should be actual strings
        for field in required_strings:
            if not isinstance(field, str):
                return False

        # Boolean fields validation
        required_bools = [
            state["needs_rag"],
            state["requires_human_escalation"],
            state["human_handoff_complete"],
            state["pending_human_response"]
        ]

        for field in required_bools:
            if not isinstance(field, bool):
                return False

        # Numeric fields validation
        if not isinstance(state["escalation_score"], (int, float)):
            return False
        if not isinstance(state["failed_attempts"], int):
            return False
        if not isinstance(state["complexity_score"], (int, float)):
            return False
        if not isinstance(state["turn_count"], int):
            return False

        # List and Dict validation
        if not isinstance(state["messages"], list):
            return False
        if not isinstance(state["conversation_context"], dict):
            return False

        return True

    except (KeyError, TypeError):
        return False

def update_state_turn(state: AgentState) -> AgentState:
    """Update state for new conversation turn"""
    state["turn_count"] += 1
    state["timestamp"] = datetime.datetime.now().isoformat()
    state["failed_attempts"] = 0  # Reset for new turn
    return state

def reset_escalation_state(state: AgentState) -> AgentState:
    """Reset escalation-related fields"""
    state["requires_human_escalation"] = False
    state["escalation_reason"] = None
    state["human_handoff_complete"] = False
    state["pending_human_response"] = False
    state["escalation_score"] = 0.0
    return state

# Type aliases for better code clarity
StateUpdate = Dict[str, Any]
ConversationTurn = Dict[str, Union[str, List[BaseMessage]]]



# NODES SECTION
# bot_nodes.py
# ==================== GRAPH NODES ====================
import os
import logging
import uuid
from typing import Dict
from langchain.schema import HumanMessage, AIMessage

# Import your custom classes
from states.bot_state import AgentState
from prompt_manager import PromptManager
from rag_system import TigressTechRAG
from API.huggingface_api import huggingface_completion
from supervisor import Supervisor
from escalation_evaluator import EscalationEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global instances (consider using dependency injection pattern)
# prompt_manager = PromptManager()
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'prompt_config.yaml')
prompt_manager = PromptManager(config_path=config_path)

rag = TigressTechRAG()
escalation_evaluator = EscalationEvaluator()

# Global supervisor instance to avoid function attribute issues
_supervisor_instance = None

def get_supervisor():
    """Get or create supervisor instance"""
    global _supervisor_instance
    if _supervisor_instance is None:
        _supervisor_instance = Supervisor(prompt_manager, rag)
    return _supervisor_instance


def input_node(state: AgentState) -> Dict:
    """First node: Prepares the state from the Matrix message"""
    try:
        user_input = state.get("user_input", "")
        sender = state.get("sender", "unknown_user")

        if not user_input:
            logger.warning("Empty user input received")
            return {"error": "Empty input"}

        formatted_input = f"{sender} said: {user_input}"

        return {
            "messages": [HumanMessage(content=formatted_input)],
            "sender": sender,
            "needs_rag": True,  # Default to needing RAG
            "requires_human_escalation": False,
            "escalation_reason": None,
            "escalation_score": 0.0,
            "requires_human_decision": False,
            "human_question": None
        }
    except Exception as e:
        logger.error(f"Error in input_node: {e}")
        return {"error": str(e)}


def detect_query_type_node(state: AgentState) -> Dict:
    """Node to detect query type"""
    try:
        user_input = state.get("user_input", "")

        if not user_input:
            return {"query_type": "general", "needs_rag": False}

        query_type = prompt_manager.detect_query_type(user_input)

        # Simple commands might not need RAG
        simple_commands = ['help', 'products', 'services', 'support', 'hours', 'contact']
        if user_input.lower().strip() in simple_commands:
            return {"query_type": query_type, "needs_rag": False}

        return {"query_type": query_type, "needs_rag": True}

    except Exception as e:
        logger.error(f"Error in detect_query_type_node: {e}")
        return {"query_type": "general", "needs_rag": True}


def secure_rag_node(state: AgentState) -> Dict:
    """Node to retrieve context from RAG system"""
    try:
        if not state.get("needs_rag", True):
            return {"context": "No additional context needed."}

        user_input = state.get("user_input", "")
        if not user_input:
            return {"context": "No user input provided."}

        context = rag.query_knowledge_base(user_input)
        return {"context": context}

    except Exception as e:
        logger.error(f"Error in secure_rag_node: {e}")
        return {"context": "Error retrieving context from knowledge base."}


def supervisor_node(state: AgentState) -> Dict:
    """Node that uses the supervisor to coordinate the response"""
    try:
        user_input = state.get("user_input", "")

        # Build conversation history safely
        messages = state.get("messages", [])
        conversation_history = []
        for msg in messages:
            # Handle both HumanMessage/AIMessage and dict-like objects
            if hasattr(msg, 'type'):
                msg_type = msg.type
                content = msg.content
            elif isinstance(msg, dict):
                msg_type = msg.get('type', 'unknown')
                content = msg.get('content', '')
            else:
                msg_type = 'unknown'
                content = str(msg)

            conversation_history.append(f"{msg_type}: {content}")

        conversation_history_str = "\n".join(conversation_history)

        # Use supervisor to handle the query
        supervisor = get_supervisor()
        result = supervisor.handle_query(user_input, conversation_history_str)

        return {
            "response": result.get("response", "No response generated"),
            "query_type": result.get("agent_type", "general"),
            "context": result.get("context_used", ""),
            "messages": messages + [AIMessage(content=result.get("response", ""))]
        }

    except Exception as e:
        logger.error(f"Error in supervisor_node: {e}")
        error_response = "I apologize, but I encountered an error while processing your request."
        return {
            "response": error_response,
            "query_type": "error",
            "context": "",
            "messages": state.get("messages", []) + [AIMessage(content=error_response)]
        }


def llm_node(state: AgentState) -> Dict:
    """Node: Calls the LLM with formatted prompt"""
    try:
        messages = state.get("messages", [])
        context = state.get("context", "")
        query_type = state.get("query_type", "general")

        # Build conversation history safely
        conversation_history = []
        for msg in messages:
            if hasattr(msg, 'type'):
                msg_type = msg.type
                content = msg.content
            elif isinstance(msg, dict):
                msg_type = msg.get('type', 'unknown')
                content = msg.get('content', '')
            else:
                msg_type = 'unknown'
                content = str(msg)
            conversation_history.append(f"{msg_type}: {content}")

        conversation_history_str = "\n".join(conversation_history)

        # Format prompt using prompt manager
        final_prompt = prompt_manager.format_main_prompt(
            query_type=query_type,
            context=context,
            conversation_history=conversation_history_str,
            max_tokens=1024
        )

        # Get response from your existing Hugging Face function
        llm_response = huggingface_completion(final_prompt)

        if llm_response.get('status') == 1:
            response_text = llm_response.get('response', 'No response generated')
        else:
            response_text = "I apologize, but I'm having trouble generating a response right now. Please try again later."
            logger.warning(f"LLM API error: {llm_response}")

        return {
            "messages": messages + [AIMessage(content=response_text)],
            "response": response_text
        }

    except Exception as e:
        logger.error(f"Error in llm_node: {e}")
        error_response = "I encountered an error while processing your request."
        return {
            "messages": state.get("messages", []) + [AIMessage(content=error_response)],
            "response": error_response
        }


def escalation_check_node(state: AgentState) -> Dict:
    """Check if conversation needs human escalation"""
    try:
        user_input = state.get("user_input", "")
        sender = state.get("sender", "unknown_user")
        current_ai_response = state.get("response", "")

        # Convert message history to the format expected by escalation evaluator
        conversation_history = []
        messages = state.get("messages", [])

        for msg in messages:
            # Determine message type safely
            if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
                msg_type = "human"
            elif isinstance(msg, AIMessage) or (hasattr(msg, 'type') and msg.type == 'ai'):
                msg_type = "ai"
            else:
                msg_type = "unknown"

            # Get content safely
            if hasattr(msg, 'content'):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get('content', '')
            else:
                content = str(msg)

            conversation_history.append({
                "type": msg_type,
                "content": content,
                "sender": sender if msg_type == "human" else "assistant"
            })

        # Evaluate escalation need
        should_escalate, reason, score = escalation_evaluator.evaluate_escalation_need(
            current_message=user_input,
            conversation_history=conversation_history,
            sender=sender,
            current_ai_response=current_ai_response
        )

        logger.info(f"Escalation check: {should_escalate}, Score: {score:.2f}, Reason: {reason}")

        return {
            "requires_human_escalation": should_escalate,
            "escalation_reason": reason,
            "escalation_score": float(score)  # Ensure it's a float
        }

    except Exception as e:
        logger.error(f"Error in escalation_check_node: {e}")
        return {
            "requires_human_escalation": False,
            "escalation_reason": f"Error in escalation check: {str(e)}",
            "escalation_score": 0.0
        }


def ask_human_node(state: AgentState) -> Dict:
    """Node that interrupts to ask human for guidance"""
    try:
        user_input = state.get("user_input", "")
        escalation_reason = state.get("escalation_reason", "complex query")

        human_question = f"""
🤖 **Human Guidance Requested**

**User Query:** {user_input}
**Reason for Escalation:** {escalation_reason}

**Options:**
1. 'proceed' - I'll handle this automatically
2. 'escalate' - Transfer to human agent
3. Or provide specific instructions

**Your decision:**"""

        # Generate unique ID for this human request
        request_id = str(uuid.uuid4())[:8]

        # Create a simple message for the interrupt - tool calls might not be needed
        return {
            "messages": [AIMessage(content=human_question)],
            "requires_human_decision": True,
            "human_question": human_question,
            "human_request_id": request_id,
            "response": human_question  # Also set response for consistency
        }

    except Exception as e:
        logger.error(f"Error in ask_human_node: {e}")
        return {
            "requires_human_decision": False,
            "human_question": None,
            "response": "Error requesting human guidance."
        }


def process_human_response_node(state: AgentState) -> Dict:
    """Process the human's response and continue accordingly"""
    try:
        human_response = state.get("human_response", "").lower().strip()

        if not human_response:
            logger.warning("Empty human response received")
            human_response = "proceed"  # Default to proceed

        if human_response in ['proceed', '1', 'yes', 'ok', 'handle', 'auto']:
            # Human wants AI to proceed automatically
            response = "Proceeding with automated response as instructed..."
            action = "proceed_automated"
        elif human_response in ['escalate', '2', 'no', 'transfer', 'human']:
            # Human wants immediate escalation
            response = "🚨 Escalating to human agent as instructed..."
            action = "escalate_immediate"
        else:
            # Human provided specific instructions
            response = f"Following your instructions: {human_response}"
            action = "custom_instructions"

        logger.info(f"Human decision processed: {action} - Response: {human_response}")

        return {
            "requires_human_decision": False,
            "human_response": human_response,
            "response": response,
            "messages": state.get("messages", []) + [AIMessage(content=response)],
            "human_action_taken": action
        }

    except Exception as e:
        logger.error(f"Error in process_human_response_node: {e}")
        return {
            "requires_human_decision": False,
            "human_response": "error",
            "response": "Error processing human response.",
            "messages": state.get("messages", []) + [AIMessage(content="Error processing human guidance.")],
            "human_action_taken": "error"
        }


def output_node(state: AgentState) -> Dict:
    """Final node: Prepares response for output"""
    try:
        requires_human_escalation = state.get("requires_human_escalation", False)
        requires_human_decision = state.get("requires_human_decision", False)
        response = state.get("response", "No response generated")
        escalation_reason = state.get("escalation_reason", "")

        if requires_human_escalation and not requires_human_decision:
            logger.info(f"ESCALATION NEEDED - Reason: {escalation_reason}")
            # Log escalation for monitoring
            logger.info(f"Human escalation triggered: {escalation_reason}")

        if requires_human_decision:
            logger.info("⏳ Waiting for human decision...")

        logger.info(f"Response ready for Matrix: {response[:100]}...")
        return state

    except Exception as e:
        logger.error(f"Error in output_node: {e}")
        return {"response": "Error in output processing", "error": str(e)}



# GRAPH SECTION
# bot_graph.py
# codes/graph/bot_graph.py
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import nodes and state
from states.bot_state import AgentState
from nodes.bot_nodes import (
    input_node,
    detect_query_type_node,
    secure_rag_node,
    llm_node,
    supervisor_node,
    output_node,
    escalation_check_node,
    ask_human_node,
    process_human_response_node
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== ROUTING FUNCTIONS ====================
def route_based_on_query_type(state: AgentState) -> str:
    """
    Determine the processing path based on query type.
    Returns either 'supervisor_path' or 'direct_llm_path'
    """
    try:
        # Use get with default values to avoid KeyErrors
        query_type = state.get("query_type", "general")
        needs_rag = state.get("needs_rag", True)

        # Debug logging
        logger.info(f"Routing decision - query_type: {query_type}, needs_rag: {needs_rag}")

        # Use supervisor for complex queries that don't need RAG
        # or for specific query types that require special handling
        supervisor_query_types = ["complaint", "report", "technical", "complex"]

        if not needs_rag or query_type in supervisor_query_types:
            logger.info(f"Routing to supervisor for query type: {query_type}")
            return "supervisor_path"
        else:
            logger.info(f"Routing to direct LLM for query type: {query_type}")
            return "direct_llm_path"

    except Exception as e:
        logger.error(f"Error in route_based_on_query_type: {e}")
        # Default to supervisor path for safety
        return "supervisor_path"


def route_after_escalation_check(state: AgentState) -> str:
    """
    Determine if we need human decision or can proceed directly
    """
    try:
        requires_human_escalation = state.get("requires_human_escalation", False)
        escalation_score = state.get("escalation_score", 0.0)

        logger.info(f"Escalation check - requires_human: {requires_human_escalation}, score: {escalation_score}")

        # Validate escalation score range
        if not 0.0 <= escalation_score <= 1.0:
            logger.warning(f"Invalid escalation score: {escalation_score}, defaulting to 0.0")
            escalation_score = 0.0

        if requires_human_escalation and escalation_score > 0.7:
            logger.info(f"High escalation score ({escalation_score:.2f}) - requesting human decision")
            return "ask_human"
        elif requires_human_escalation:
            logger.info(f"Moderate escalation ({escalation_score:.2f}) - proceeding with automated escalation")
            return "output"
        else:
            logger.info("No escalation needed - proceeding to output")
            return "output"

    except Exception as e:
        logger.error(f"Error in route_after_escalation_check: {e}")
        # Default to output for safety
        return "output"


# ==================== GRAPH CONSTRUCTION ====================
def create_workflow():
    """Create and compile the workflow graph with human-in-the-loop"""
    logger.info("Building enhanced workflow graph with human-in-the-loop...")

    try:
        workflow = StateGraph(AgentState)

        # Add nodes in execution order
        workflow.add_node("input_node", input_node)
        workflow.add_node("detect_query_type", detect_query_type_node)
        workflow.add_node("secure_rag", secure_rag_node)
        workflow.add_node("llm_node", llm_node)
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("escalation_check", escalation_check_node)
        workflow.add_node("ask_human", ask_human_node)
        workflow.add_node("process_human_response", process_human_response_node)
        workflow.add_node("output_node", output_node)

        # Define the flow
        workflow.set_entry_point("input_node")
        workflow.add_edge("input_node", "detect_query_type")

        # Conditional routing after query type detection
        workflow.add_conditional_edges(
            "detect_query_type",
            route_based_on_query_type,
            {
                "supervisor_path": "supervisor",
                "direct_llm_path": "secure_rag"
            }
        )

        # Direct LLM path: secure_rag -> llm_node -> escalation_check
        workflow.add_edge("secure_rag", "llm_node")
        workflow.add_edge("llm_node", "escalation_check")

        # Supervisor path: supervisor -> escalation_check
        workflow.add_edge("supervisor", "escalation_check")

        # Conditional routing after escalation check
        workflow.add_conditional_edges(
            "escalation_check",
            route_after_escalation_check,
            {
                "ask_human": "ask_human",
                "output": "output_node"
            }
        )

        # Human decision flow: ask_human -> process_human_response -> output
        workflow.add_edge("ask_human", "process_human_response")
        workflow.add_edge("process_human_response", "output_node")

        # Final output to end
        workflow.add_edge("output_node", END)

        # ==================== COMPILE WITH CHECKPOINTING ====================
        # Set up memory for checkpointing with interrupt
        memory = MemorySaver()
        app = workflow.compile(
            checkpointer=memory,
            interrupt_before=["ask_human"]  # Enables the human-in-the-loop interrupt
        )

        logger.info("Human-in-the-loop workflow compiled successfully with checkpointing")
        return app

    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        raise

# Create the workflow application
try:
    app = create_workflow()
    logger.info("Workflow application created successfully")
except Exception as e:
    logger.error(f"Failed to create workflow application: {e}")
    # Create a minimal fallback app or re-raise based on your needs
    raise



# custom_tools.py
import os
import math
import json
import ast
import operator
import re
from typing import Dict, List
import requests
from langchain_core.tools import tool
from datetime import datetime, timedelta


# ==================== SECURE EVAL FUNCTION ====================

def safe_eval(expression):
    """Safely evaluate mathematical expressions using ast.literal_eval with math operations"""
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos
    }

    allowed_math_functions = {
        'sqrt', 'sin', 'cos', 'tan', 'log', 'log10', 'exp', 'radians',
        'degrees', 'pi', 'e', 'ceil', 'floor', 'factorial'
    }

    def _eval(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in allowed_operators:
                raise ValueError(f"Operator {type(node.op).__name__} not allowed")
            return allowed_operators[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in allowed_operators:
                raise ValueError(f"Operator {type(node.op).__name__} not allowed")
            return allowed_operators[type(node.op)](_eval(node.operand))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in allowed_math_functions:
                if node.func.id == 'pi':
                    return math.pi
                elif node.func.id == 'e':
                    return math.e
                else:
                    func = getattr(math, node.func.id)
                    args = [_eval(arg) for arg in node.args]
                    return func(*args)
            raise ValueError(f"Function {getattr(node.func, 'id', 'unknown')} not allowed")
        elif isinstance(node, ast.Name):
            if node.id == 'pi':
                return math.pi
            elif node.id == 'e':
                return math.e
            else:
                raise ValueError(f"Variable {node.id} not allowed")
        else:
            raise ValueError(f"Unsupported operation: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode='eval')
        return _eval(tree.body)
    except Exception as e:
        raise ValueError(f"Invalid mathematical expression: {e}")


def validate_math_expression(expression):
    """Validate mathematical expression for safety"""
    # Block dangerous patterns
    dangerous_patterns = [
        '__', 'import', 'open', 'exec', 'eval', 'compile',
        'os.', 'sys.', 'subprocess', 'file', 'input', 'exit',
        'quit', 'help', 'dir', 'globals', 'locals', 'vars'
    ]

    expr_lower = expression.lower()
    for pattern in dangerous_patterns:
        if pattern in expr_lower:
            raise ValueError(f"Expression contains prohibited pattern: {pattern}")

    # Limit expression length
    if len(expression) > 100:
        raise ValueError("Expression too long (max 100 characters)")

    # Check for suspicious characters
    suspicious_chars = [';', '"', "'", '`', '$', '&', '|', '>', '<']
    for char in suspicious_chars:
        if char in expression:
            raise ValueError(f"Expression contains suspicious character: {char}")

    return True


# ==================== IMPLEMENTED UTILITY FUNCTIONS ====================

def handle_unit_conversion(expr):
    """Implement unit conversion logic"""
    try:
        # Extract numbers and units
        numbers = re.findall(r'\d+\.?\d*', expr)
        if not numbers:
            return "No number found for conversion"

        value = float(numbers[0])

        # Define conversion factors
        conversions = {
            'km to miles': (value * 0.621371, 'km', 'miles'),
            'miles to km': (value * 1.60934, 'miles', 'km'),
            'c to f': ((value * 9/5) + 32, '°C', '°F'),
            'f to c': ((value - 32) * 5/9, '°F', '°C'),
            'kg to lbs': (value * 2.20462, 'kg', 'lbs'),
            'lbs to kg': (value * 0.453592, 'lbs', 'kg'),
            'm to ft': (value * 3.28084, 'm', 'ft'),
            'ft to m': (value * 0.3048, 'ft', 'm')
        }

        for pattern, (result, from_unit, to_unit) in conversions.items():
            if pattern in expr.lower():
                return f"{value} {from_unit} = {result:.2f} {to_unit}"

        return "Unit conversion not supported. Try: km to miles, c to f, kg to lbs, etc."

    except Exception as e:
        return f"Error in unit conversion: {str(e)}"


def handle_percentage(expr):
    """Implement percentage calculation"""
    try:
        numbers = re.findall(r'\d+\.?\d*', expr)
        if len(numbers) < 2:
            return "Please provide both percentage and total value"

        percentage = float(numbers[0])
        total = float(numbers[1])

        if 'of' in expr.lower():
            # Percentage of total
            result = (percentage / 100) * total
            return f"{percentage}% of {total} = {result:.2f}"
        elif 'increase' in expr.lower() or 'more' in expr.lower():
            # Percentage increase
            result = total * (1 + percentage/100)
            return f"{total} increased by {percentage}% = {result:.2f}"
        elif 'decrease' in expr.lower() or 'less' in expr.lower():
            # Percentage decrease
            result = total * (1 - percentage/100)
            return f"{total} decreased by {percentage}% = {result:.2f}"
        else:
            # Default: percentage of total
            result = (percentage / 100) * total
            return f"{percentage}% of {total} = {result:.2f}"

    except Exception as e:
        return f"Error in percentage calculation: {str(e)}"


def calculate_appointment_time(preferred_time):
    """Implement time calculation logic with basic parsing"""
    now = datetime.now()

    # Simple time parsing (can be expanded)
    preferred_lower = preferred_time.lower()

    if 'tomorrow' in preferred_lower:
        base_time = now + timedelta(days=1)
    elif 'monday' in preferred_lower:
        days_ahead = (0 - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7  # Next Monday
        base_time = now + timedelta(days=days_ahead)
    elif 'tuesday' in preferred_lower:
        days_ahead = (1 - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        base_time = now + timedelta(days=days_ahead)
    elif 'wednesday' in preferred_lower:
        days_ahead = (2 - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        base_time = now + timedelta(days=days_ahead)
    elif 'thursday' in preferred_lower:
        days_ahead = (3 - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        base_time = now + timedelta(days=days_ahead)
    elif 'friday' in preferred_lower:
        days_ahead = (4 - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        base_time = now + timedelta(days=days_ahead)
    else:
        base_time = now + timedelta(hours=2)  # Default

    # Set time of day
    if '9am' in preferred_lower or '9 am' in preferred_lower:
        base_time = base_time.replace(hour=9, minute=0, second=0, microsecond=0)
    elif '10am' in preferred_lower or '10 am' in preferred_lower:
        base_time = base_time.replace(hour=10, minute=0, second=0, microsecond=0)
    elif '2pm' in preferred_lower or '2 pm' in preferred_lower:
        base_time = base_time.replace(hour=14, minute=0, second=0, microsecond=0)
    elif '3pm' in preferred_lower or '3 pm' in preferred_lower:
        base_time = base_time.replace(hour=15, minute=0, second=0, microsecond=0)
    else:
        # Default time: 10 AM
        base_time = base_time.replace(hour=10, minute=0, second=0, microsecond=0)

    return base_time


# ==================== PERSISTENT APPOINTMENT STORAGE ====================

class AppointmentManager:
    """Manage appointments with basic persistence"""

    def __init__(self, storage_file="appointments.json"):
        self.storage_file = storage_file
        self.appointments = self._load_appointments()

    def _load_appointments(self):
        """Load appointments from file"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_appointments(self):
        """Save appointments to file"""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.appointments, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save appointments: {e}")

    def add_appointment(self, appointment_id, appointment_data):
        """Add a new appointment"""
        self.appointments[appointment_id] = appointment_data
        self._save_appointments()

    def get_appointments(self):
        """Get all appointments"""
        return self.appointments


# Initialize appointment manager
appointment_manager = AppointmentManager()


# ==================== CALCULATOR TOOL ====================

@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions and perform calculations.

    This tool can handle basic arithmetic, advanced math functions,
    and unit conversions. Supports operations like:
    - Basic: 2+3, 10*5, 8/2, 4^2 (exponent)
    - Functions: sqrt(16), sin(30), log(100)
    - Constants: pi, e
    - Unit conversions: 10km to miles, 100C to F

    Args:
        expression: The mathematical expression to evaluate

    Returns:
        The result of the calculation or an error message
    """
    try:
        # Input validation
        validate_math_expression(expression)

        # Clean the expression
        expr = expression.strip().lower()

        # Handle unit conversions
        if ' to ' in expr:
            return handle_unit_conversion(expr)

        # Replace common math notations
        expr = expr.replace('^', '**').replace('×', '*').replace('÷', '/')

        # Handle percentage calculations
        if '%' in expr:
            return handle_percentage(expr)

        # Replace function names to match our safe_eval expectations
        expr = expr.replace('pi', 'pi').replace('e', 'e')  # Keep as is for AST parsing

        # Evaluate the expression safely using secure function
        result = safe_eval(expr)

        # Format the result nicely
        if isinstance(result, float):
            # Round to avoid floating point precision issues
            if abs(result - round(result)) < 1e-10:
                result = round(result)
            else:
                result = round(result, 6)

        return f"Result: {result}"

    except Exception as e:
        return f"Error evaluating expression: {str(e)}. Please check your input."


# ==================== APPOINTMENT SCHEDULER TOOL ====================

@tool
def schedule_appointment(name: str, contact: str, preferred_time: str = "") -> str:
    """Schedule an appointment with a customer representative.

    Args:
        name: Customer's name
        contact: Phone number or email for contact
        preferred_time: Preferred time (e.g., "tomorrow 10am", "friday afternoon")

    Returns:
        Confirmation message with appointment details
    """
    try:
        # Input validation
        if not name or not contact:
            return "Error: Please provide both name and contact information"

        # Generate appointment ID
        appointments = appointment_manager.get_appointments()
        appointment_id = f"APT{len(appointments) + 1:03d}"

        # Calculate appointment time
        if preferred_time:
            appointment_time = calculate_appointment_time(preferred_time)
        else:
            # Default: next available slot (2 hours from now)
            appointment_time = datetime.now() + timedelta(hours=2)

        # Create appointment
        appointment = {
            'id': appointment_id,
            'name': name,
            'contact': contact,
            'time': appointment_time.strftime("%Y-%m-%d %I:%M %p"),
            'preferred_time': preferred_time,
            'status': 'scheduled',
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Store appointment persistently
        appointment_manager.add_appointment(appointment_id, appointment)

        # Format confirmation message
        confirmation = f"""
APPOINTMENT SCHEDULED

Appointment ID: {appointment_id}
Customer: {name}
Contact: {contact}
Time: {appointment_time.strftime("%A, %B %d at %I:%M %p")}
Status: {appointment['status']}

Please arrive 10 minutes early. Contact us if you need to reschedule."""

        return confirmation

    except Exception as e:
        return f"Error scheduling appointment: {str(e)}"


def get_all_tools() -> List:
    """Return a list of all available tools."""
    return [
        schedule_appointment,
        calculator,
    ]



# document_loader.py
import os
import logging
from pathlib import Path
from langchain.schema import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# current_dir = os.path.dirname(__file__)
# project_root = os.path.dirname(current_dir)
# services_path = os.path.join(project_root, "files", "services_policies.txt")
# faqs_path = os.path.join(project_root, "files", "faqs.txt")

# ==================== DOCUMENT LOADER ====================

class DocumentLoader:
    def __init__(self, text_files_path):
        self.text_files_path = text_files_path
        self.required_files = {
            'services_policies': 'services_policies.txt',
            'faqs': 'faqs.txt',
        }

    def load_documents(self):
        """Load documents from specified path"""
        documents = []

        for doc_type, filename in self.required_files.items():
            file_path = Path(self.text_files_path) / filename

            if not file_path.exists():
                logger.warning(f"File not found: {filename} at {file_path}")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                document = Document(
                    page_content=content,
                    metadata={"source": filename, "type": doc_type}
                )
                documents.append(document)
                logger.info(f"Loaded {filename}")

            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                continue

        if not documents:
            logger.error("No documents loaded. Please check your file paths")
            return []



# escalation_evaluator.py
import logging
import re
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EscalationEvaluator:
    """Evaluates need for human intervention based on chat context"""
    def __init__(self):
        self.conversation_history = {}

    def evaluate_escalation_need(self,
                                 current_message: str,
                                 conversation_history: List[Dict],
                                 sender: str,
                                 current_ai_response: str = "") -> Tuple[bool, str, float]:
        """
        Comprehensive evaluation for human escalation need.
        Return: (should_escalate, reason, confidence_score)
        """
        escalation_factors = []

        # Factor 1: Conversation history analysis
        history_score, history_reason = self._analyse_conversation_history(conversation_history, sender)
        if history_score > 0.7:
            escalation_factors.append((history_score, history_reason))

        # Factor 2: User sentiment and emotion
        sentiment_score, sentiment_reason = self._analyse_sentiment(current_message, conversation_history)
        if sentiment_score > 0.8:
            escalation_factors.append((sentiment_score, sentiment_reason))

        # Factor 3: Complexity analysis
        complexity_score, complexity_reason = self._analyse_complexity(current_message, conversation_history)
        if complexity_score > 0.7:
            escalation_factors.append((complexity_score, complexity_reason))

        # Factor 4: Explicit Human Requests (with context awareness)
        explicit_score, explicit_reason = self._detect_explicit_human_requests(current_message, conversation_history)
        if explicit_score > 0.6:
            escalation_factors.append((explicit_score, explicit_reason))

        # Factor 5: AI Confidence and Capability
        capability_score, capability_reason = self._assess_ai_capability(current_message, current_ai_response)
        if capability_score > 0.7:
            escalation_factors.append((capability_score, capability_reason))

        # Calculate overall escalation score
        if escalation_factors:
            max_score, primary_reason = max(escalation_factors, key=lambda x: x[0])
            overall_score = self._calculate_composite_score(escalation_factors)

            should_escalate = overall_score >= 0.65

            return should_escalate, primary_reason, overall_score

        return False, "No significant escalation factors detected", 0.0

    def _analyse_conversation_history(self, conversation_history: List[Dict], sender: str) -> Tuple[float, str]:
        """Analyse conversation patterns that indicate escalation need"""
        if len(conversation_history) < 3:
            return 0.0, "Insufficient conversation history"

        # Track repeated questions/unsolved issues
        recent_messages = conversation_history[-6:]
        user_messages = [msg for msg in recent_messages if msg.get('type') == 'human']

        # Check for repetition
        unique_questions = set()
        repeated_issues = 0

        for msg in user_messages:
            content = msg.get('content', '').lower()
            content_hash = hash(content[:100])
            if content_hash in unique_questions:
                repeated_issues += 1
            else:
                unique_questions.add(content_hash)

        repetition_ratio = repeated_issues / len(user_messages) if user_messages else 0

        # Check for long conversation without resolution
        if len(conversation_history) > 10:
            return 0.8, f"Long conversation ({len(conversation_history)} messages) without resolution"

        if repetition_ratio > 0.3:
            return 0.3, f"User repeating issues (repetition rate: {repetition_ratio:.2f})"

        return 0.0, "Conversation history normal"

    def _analyse_sentiment(self, current_message: str, conversation_history: List[Dict]) -> Tuple[float, str]:
        """Analyse user sentiment and emotional state"""
        message_lower = current_message.lower()

        # Strong negative indicators
        strong_negative = [
            'angry', 'furious', 'livid', 'outraged', 'horrible', 'terrible',
            'awful', 'disgusting', 'ridiculous', 'unacceptable', 'worst ever',
            'never again', 'hate this', 'useless', 'waste of time'
        ]

        # Frustration indicators
        frustration_indicators = [
            'frustrated', 'annoyed', 'disappointed', 'not happy', 'not satisfied',
            'still not working', 'again', 'still having', 'why is this',
            'how many times', 'when will this be fixed'
        ]

        # Check for strong negative language
        strong_negative_count = sum(1 for word in strong_negative if word in message_lower)
        if strong_negative_count >= 2:
            return 0.9, "User expressing strong negative emotions"

        # Check for frustration patterns
        frustration_count = sum(1 for word in frustration_indicators if word in message_lower)
        if frustration_count >= 2:
            return 0.75, "User showing clear frustration"

        # Check for multiple exclamation points
        if current_message.count('!') >= 3:
            return 0.7, "User using excessive exclamation (emotional intensity)"

        return 0.0, "User sentiment appears neutral"

    def _analyse_complexity(self, current_message: str, conversation_history: List[Dict]) -> Tuple[float, str]:
        """Analyse query complexity that might require human expertise"""
        message_lower = current_message.lower()

        # High complexity topics
        high_complexity_indicators = {
            'legal': ['contract', 'agreement', 'terms', 'legal', 'liability', 'warranty', 'sue', 'lawyer'],
            'financial': ['refund', 'compensation', 'billing dispute', 'payment issue', 'chargeback', 'invoice'],
            'technical_advanced': ['api integration', 'custom development', 'system architecture', 'database', 'server'],
            'business_critical': ['downtime', 'outage', 'data loss', 'security breach', 'emergency'],
            'multi_step': ['process', 'workflow', 'multiple systems', 'integration between']
        }

        complexity_score = 0.0
        complexity_reasons = []

        for category, keywords in high_complexity_indicators.items():
            matches = sum(1 for keyword in keywords if keyword in message_lower)
            if matches > 0:
                category_score = min(0.3 + (matches * 0.2), 0.9)
                complexity_score = max(complexity_score, category_score)
                complexity_reasons.append(f"{category} issues detected")

        # Check for multi-part questions
        if (' and ' in message_lower or ' also ' in message_lower) and message_lower.count('?') >= 2:
            complexity_score = max(complexity_score, 0.6)
            complexity_reasons.append("Multi-part complex questions")

        if complexity_score > 0.6:
            return complexity_score, f"Complex issues requiring expertise: {', '.join(complexity_reasons)}"

        return 0.0, "Query complexity within AI capability"

    def _detect_explicit_human_requests(self, current_message: str, conversation_history: List[Dict]) -> Tuple[float, str]:
        """Detect explicit human request with context awareness"""
        message_lower = current_message.lower()

        # Direct human request
        direct_requests = [
            'speak to a human', 'talk to a real person', 'human agent',
            'real person', 'live agent', 'customer service', 'support agent'
        ]

        # Check if this is a repeated request for human
        human_request_history = 0
        for msg in conversation_history[-4:]:
            if any(req in msg.get('content', '').lower() for req in direct_requests):
                human_request_history += 1

        # Current requests
        current_request = any(req in message_lower for req in direct_requests)

        if current_request and human_request_history >= 1:
            return 0.9, "Repeated explicit request for human assistance"
        elif current_request:
            return 0.7, "Explicit request for human assistance"

        # Indirect human requests
        indirect_requests = [
            'can you actually help', 'are you a bot', 'is this automated',
            'let me speak to someone', 'get me a manager', 'supervisor'
        ]

        if any(req in message_lower for req in indirect_requests):
            return 0.6, "Indirect request for human assistance"

        return 0.0, "No explicit request for human assistance"

    def _assess_ai_capability(self, current_message: str, current_ai_response: str) -> Tuple[float, str]:
        """Assess whether this query is within AI capability"""
        message_lower = current_message.lower()

        # Queries beyond typical AI capabilities
        beyond_ai_capability = [
            'make an exception', 'override', 'special case', 'discretion',
            'judgment call', 'subjective', 'personal opinion', 'what would you do',
            'emotional support', 'counseling', 'therapy'
        ]

        if any(phrase in message_lower for phrase in beyond_ai_capability):
            return 0.8, "Query requires human judgement and discretion"

        # Check if AI response indicates uncertainty
        if current_ai_response:
            uncertainty_indicators = [
                "I'm not sure", "I don't know", "I cannot", "unable to",
                "limited information", "contact support", "escalate"
            ]

            if any(indicator in current_ai_response.lower() for indicator in uncertainty_indicators):
                return 0.7, "AI response indicates uncertainty or limitations"

            return 0.0, "Query appears within AI capabilities"

        return 0.0, "No AI response available for assessment"

    def _calculate_composite_score(self, factors: List[Tuple[float, str]]) -> float:
        """Calculate weighted composite escalation score"""
        if not factors:
            return 0.0

        # Use maximum score with slight weighting toward multiple factors
        max_score = max(score for score, _ in factors)
        factor_count_bonus = min(len(factors) * 0.1, 0.3)

        return min(max_score + factor_count_bonus, 1.0)




# guardrails_ai.py
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



# initialize.py
# ==================== MISSING IMPORTS ADDED ====================
import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipelineLLM

# Import your custom classes and components
from API.matrix_api import MatrixClient
from prompt_manager import PromptManager
from rag_system import TigressTechRAG


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the environment variables from the .env file
load_dotenv()

# ==================== MISSING CONFIG VARIABLES ====================
MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceH4/zephyr-7b-beta")  # Fallback model
HF_TOKEN = os.getenv("HF_TOKEN", "")
MATRIX_HOMESERVER = os.getenv("MATRIX_HOMESERVER", "")
MATRIX_USER = os.getenv("MATRIX_USER", "")
MATRIX_PASSWORD = os.getenv("MATRIX_PASSWORD", "")



# ==================== INITIALIZATION ====================
# Initialize components in proper order
logger.info("Initializing components...")

# 1. Initialize LLM first
logger.info("Loading LLM model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto",
    token=HF_TOKEN,
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
hf_llm = HuggingFacePipelineLLM(pipeline=pipe)

# 2. Initialize Prompt Manager
logger.info("Loading prompt configuration...")
prompt_manager = PromptManager()

# 3. Initialize RAG System
logger.info("Setting up RAG system...")
rag = TigressTechRAG()
rag_success = rag.setup_rag()
if not rag_success:
    logger.warning("RAG system setup failed - proceeding without knowledge base")

# 4. Initialize Matrix Client
logger.info("Initializing Matrix client...")
matrix_client = MatrixClient(MATRIX_HOMESERVER, MATRIX_USER, MATRIX_PASSWORD)



# main.py
# codes/main.py
import logging
import time
import signal
import sys
import threading
import re
from typing import Dict, Optional
from dotenv import load_dotenv
import os
from langchain_core.messages import ToolMessage

# Import your custom classes and components
from API.matrix_api import MatrixClient
from prompt_manager import PromptManager
from rag_system import TigressTechRAG
from graph.bot_graph import app

# Load environment variables
load_dotenv()

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get environment variables
ROOM_ID = os.getenv("MATRIX_ROOM_ID")

# Global references
matrix_client = None

# Safety configuration
MAX_GRAPH_ITERATIONS = 10  # Prevent infinite loops in graph execution

# ==================== SAFETY IMPROVEMENTS ====================

# Thread-safe request storage with automatic cleanup
class HumanRequestManager:
    def __init__(self, max_requests: int = 100, request_timeout: int = 300):
        self._requests: Dict[str, dict] = {}
        self._lock = threading.RLock()
        self.max_requests = max_requests
        self.request_timeout = request_timeout
        self.cleanup_counter = 0

    def add_request(self, room_id: str, request_data: dict) -> bool:
        with self._lock:
            # Prevent memory exhaustion
            if len(self._requests) >= self.max_requests:
                logger.error(f"Human request limit reached ({self.max_requests}), rejecting new request")
                return False

            # Auto-cleanup every 10 operations
            self.cleanup_counter += 1
            if self.cleanup_counter >= 10:
                self._cleanup_expired()
                self.cleanup_counter = 0

            request_data["timestamp"] = time.time()
            self._requests[room_id] = request_data
            logger.info(f"Added human request for room {room_id}, total: {len(self._requests)}")
            return True

    def get_and_remove_request(self, room_id: str) -> Optional[dict]:
        with self._lock:
            request = self._requests.pop(room_id, None)
            if request:
                logger.info(f"Removed human request for room {room_id}, remaining: {len(self._requests)}")
            return request

    def get_request(self, room_id: str) -> Optional[dict]:
        with self._lock:
            return self._requests.get(room_id)

    def _cleanup_expired(self):
        current_time = time.time()
        expired = []

        for room_id, request in self._requests.items():
            if current_time - request["timestamp"] > self.request_timeout:
                expired.append(room_id)

        for room_id in expired:
            del self._requests[room_id]
            logger.warning(f"Cleaned up expired human request for room {room_id}")

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired requests, remaining: {len(self._requests)}")

# Initialize thread-safe manager
human_request_manager = HumanRequestManager(max_requests=100, request_timeout=300)

# ==================== SECURITY VALIDATION ====================

def validate_user_input(text: str, max_length: int = 2000) -> tuple[bool, str]:
    """Validate and sanitize user input to prevent attacks"""
    if not text or not isinstance(text, str):
        return False, "Empty or invalid input"

    if len(text) > max_length:
        return False, f"Input too long (max {max_length} characters)"

    # Basic injection prevention
    dangerous_patterns = [
        r'(?i)(\bexec\b|\beval\b|\bsystem\b|\bos\.|subprocess|__import__)',
        r'(?i)(\bdelete\b|\bdrop\b|\btruncate\b|\bupdate\b|\binsert\b)',
        r'(\{.*\{|\}.*\}|\$.*\(|`.*`)',  # Template injection patterns
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, text):
            return False, "Potentially dangerous input detected"

    # Clean excessive whitespace and control characters
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    cleaned_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned_text)

    return True, cleaned_text

# ==================== FIXED HUMAN RESPONSE HANDLER ====================

def handle_human_decision(message, matrix_client, app) -> bool:
    """Thread-safe human decision handler with validation"""
    room_id = message["room_id"]
    sender = message["sender"]
    content = message["content"]["body"]

    # Validate input first
    is_valid, cleaned_content = validate_user_input(content)
    if not is_valid:
        logger.warning(f"Rejected invalid human response from {sender}: {content}")
        matrix_client.send_message(room_id, "Invalid input received. Please provide a valid response.")
        return False

    # Thread-safe check for pending request
    pending_request = human_request_manager.get_and_remove_request(room_id)
    if not pending_request:
        return False  # No pending request

    tool_call_id = pending_request["tool_call_id"]

    logger.info(f"Processing validated human decision from {sender}")

    try:
        # Create tool message with sanitized content
        tool_message = ToolMessage(
            content=cleaned_content,
            tool_call_id=tool_call_id
        )

        # Update graph state with human decision
        config = {"configurable": {"thread_id": room_id}}

        app.update_state(
            config,
            {"messages": [tool_message], "human_response": cleaned_content},
            as_node="ask_human"
        )

        # Continue with safety mechanism
        iteration_count = 0
        for event in app.stream(None, config, stream_mode="values"):
            iteration_count += 1
            if iteration_count >= MAX_GRAPH_ITERATIONS:
                logger.error(f"Graph execution stuck after human decision in room {room_id}")
                matrix_client.send_message(room_id, "Processing error after your decision. Please restart the conversation.")
                return True

            final_state = event
            if "messages" in final_state and final_state["messages"]:
                response = final_state["messages"][-1].content

                # Validate response before sending
                is_valid, cleaned_response = validate_user_input(response, max_length=4000)
                if is_valid:
                    matrix_client.send_message(room_id, cleaned_response)
                else:
                    matrix_client.send_message(room_id, "I generated a response but it contained invalid content. Please try again.")
                    logger.warning(f"Filtered invalid bot response: {response[:100]}...")
                break

        logger.info(f"Human decision processed successfully for room {room_id}")
        return True

    except Exception as e:
        logger.error(f"Error processing human decision: {e}")
        matrix_client.send_message(room_id, "Error processing your decision. Please try again.")
        return True

# ==================== FIXED MESSAGE PROCESSOR ====================

def enhanced_process_message(message, matrix_client, app):
    """Secure message processor with input validation"""
    # Input validation at the entry point
    if not message or "content" not in message or "body" not in message["content"]:
        logger.error("Received invalid message structure")
        return

    user_query = message["content"]["body"]
    sender = message["sender"]
    room_id = message["room_id"]

    # CRITICAL: Validate user input before processing
    is_valid, cleaned_query = validate_user_input(user_query)
    if not is_valid:
        logger.warning(f"Rejected invalid input from {sender}: {user_query}")
        matrix_client.send_message(room_id, "Your message contains invalid content. Please rephrase.")
        return

    logger.info(f"Processing validated message from {sender} in room {room_id}")

    # Check for human response first (thread-safe)
    if handle_human_decision(message, matrix_client, app):
        return

    # Process normal message with validated input
    try:
        initial_state = {
            "user_input": cleaned_query,  # Use validated input
            "messages": [],
            "response": "",
            "sender": sender,
            "context": "",
            "query_type": "general",
            "needs_rag": True
        }

        # Execute the workflow WITH SAFETY MECHANISM
        logger.info("Generating response via LangGraph...")
        config = {"configurable": {"thread_id": room_id}}

        iteration_count = 0
        for event in app.stream(initial_state, config, stream_mode="values"):
            iteration_count += 1
            if iteration_count >= MAX_GRAPH_ITERATIONS:
                logger.error(f"Graph execution stuck in infinite loop for room {room_id}")
                matrix_client.send_message(room_id, "I'm having trouble processing this request. The conversation seems to be stuck. Please try again with a different approach.")
                break

            state = event
            last_message = state["messages"][-1] if state["messages"] else None

            # Check if we reached human interrupt point
            if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                # GRAPH IS PAUSED - Store the pending request and send question to human
                tool_call = last_message.tool_calls[0]
                tool_call_id = tool_call["id"]
                human_question = last_message.content

                # Store pending request using thread-safe manager
                request_data = {
                    "tool_call_id": tool_call_id,
                    "request_id": tool_call["args"].get("request_id", "unknown"),
                    "timestamp": time.time()
                }

                if human_request_manager.add_request(room_id, request_data):
                    # Send human question to Matrix room
                    matrix_client.send_message(room_id, human_question)
                    logger.info(f"Human decision requested in room {room_id}")
                else:
                    # Failed to add request - send error
                    matrix_client.send_message(room_id, "I'm currently experiencing high load. Please try again shortly.")
                    logger.error(f"Failed to store human request for room {room_id} - limit reached")
                return  # Exit - wait for human response

            # If we get here, no human interrupt occurred
            response_text = state.get("response", "")

            if response_text:
                # Validate and sanitize response before sending
                is_valid, cleaned_response = validate_user_input(response_text, max_length=4000)
                if not is_valid:
                    logger.warning(f"Generated invalid response, sending fallback: {response_text[:100]}...")
                    cleaned_response = "I generated a response but it contained invalid content. Please try again."

                # Truncate if too long for Matrix (after validation)
                if len(cleaned_response) > 1500:
                    cleaned_response = cleaned_response[:1497] + "..."

                # Add delay to avoid rate limiting
                time.sleep(2.5)

                # Send validated response
                matrix_client.send_message(room_id, cleaned_response)

                # Log escalation if it happened
                if state.get("requires_human_escalation", False):
                    logger.warning(f"Conversation escalated to human: {state['escalation_reason']}")
                else:
                    logger.info("Response sent successfully")
            else:
                logger.warning("Failed to generate response")
                matrix_client.send_message(room_id, "I apologize, but I couldn't generate a response. Please try again.")

    except Exception as e:
        logger.error(f"Error processing validated message: {e}")
        matrix_client.send_message(room_id, "I encountered an error processing your request. Please try again.")

def cleanup_pending_requests():
    """Clean up old pending requests to prevent memory leaks"""
    human_request_manager._cleanup_expired()

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    logger.info("Received shutdown signal (Ctrl+C)...")
    if matrix_client:
        matrix_client.stop_listening()
    logger.info("Bot shutdown complete.")
    sys.exit(0)

def run_bot():
    """Main function to run the matrix bot with human-in-the-loop"""
    global matrix_client

    logger.info("Starting Matrix bot with human-in-the-loop...")

    try:
        # Initialize Matrix Client
        matrix_client = MatrixClient(
            homeserver=os.getenv("MATRIX_HOMESERVER"),
            user=os.getenv("MATRIX_USER"),
            password=os.getenv("MATRIX_PASSWORD")
        )

        # Login to Matrix
        matrix_client.login()

        # Join the specified room
        matrix_client.join_room(ROOM_ID)

        logger.info(f"Logged in. Listening for messages in room {ROOM_ID}...")
        logger.info(f"Human request manager initialized with capacity: {human_request_manager.max_requests}")

        # Start listening for messages with enhanced processor
        matrix_client.listen_for_messages(
            process_message_callback=lambda msg: enhanced_process_message(msg, matrix_client, app)
        )

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received in run_bot")
    except Exception as e:
        logger.error(f"Fatal error in bot execution: {e}")
    finally:
        # Ensure cleanup happens even if unexpected errors occur
        logger.info("Performing final cleanup...")
        cleanup_pending_requests()
        if matrix_client:
            matrix_client.stop_listening()

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal (Docker, etc.)

    logger.info("Initializing bot components...")

    # Initialize components
    prompt_manager = PromptManager()
    rag = TigressTechRAG()

    # Setup RAG system
    rag_success = rag.setup_rag()
    if not rag_success:
        logger.warning("RAG system setup failed - proceeding without knowledge base")

    # Run the bot
    logger.info("Starting Tigress Tech Labs AI Assistant with Human-in-the-Loop...")
    run_bot()




# rag_system.py
import logging
import os
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Set up logging
logger = logging.getLogger(__name__)

# Define constants (adjust these as needed)
# TEXT_FILES_PATH = "./files"  # Path to your text files

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
TEXT_FILES_PATH = os.path.join(project_root, "files")

PERSIST_DIRECTORY = "./chroma_db"  # Directory to persist vector store

# ADD IMPORT SAFETY - PRESERVE EXISTING STRUCTURE
try:
    from document_loader import DocumentLoader
    logger.info("Successfully imported DocumentLoader from codes.document_loader")
except ImportError as e:
    logger.warning(f"Failed to import DocumentLoader: {e}")
    # Create minimal fallback that preserves the interface
    class DocumentLoader:
        def __init__(self, path):
            self.path = path
            logger.warning(f"Using fallback DocumentLoader for path: {path}")

        def load_documents(self):
            """Fallback method that returns empty list to prevent crashes"""
            logger.error("DocumentLoader fallback - no documents can be loaded")
            return []  # Return empty list instead of crashing

class TigressTechRAG:
    def __init__(self):
        try:
            # ADD ERROR HANDLING FOR EMBEDDING MODEL
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None

        self.vectorstore = None
        self.retriever = None
        self.document_loader = DocumentLoader(TEXT_FILES_PATH)

    def setup_rag(self, persist_directory=PERSIST_DIRECTORY):
        """Setup RAG system with document processing"""
        logger.info("Setting up RAG system...")

        # CHECK IF EMBEDDING MODEL IS AVAILABLE
        if not self.embedding_model:
            logger.error("Embedding model not available - RAG setup failed")
            return False

        # Load documents
        documents = self.document_loader.load_documents()
        if not documents:
            logger.error("No documents to process")
            return False

        try:
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150,
                length_function=len,
            )

            splits = text_splitter.split_documents(documents)
            logger.info(f"Split into {len(splits)} chunks")

            # ENSURE PERSIST DIRECTORY EXISTS
            os.makedirs(persist_directory, exist_ok=True)

            # Create vector store with error handling
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embedding_model,
                persist_directory=persist_directory
            )

            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )

            logger.info("RAG system setup complete")
            return True

        except Exception as e:
            logger.error(f"RAG setup failed: {e}")
            return False

    def query_knowledge_base(self, query, max_context_length=2500):
        """Query the knowledge base for relevant information"""
        # VALIDATE INPUT
        if not query or not isinstance(query, str) or len(query.strip()) == 0:
            return "Please provide a valid query."

        if not self.retriever:
            return "Knowledge base not available. Please run setup_rag() first."

        try:
            # Retrieve relevant documents
            relevant_docs = self.retriever.get_relevant_documents(query)
            if not relevant_docs:
                return "No specific information found. Please contact our team for detailed assistance."

            # Combine context
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
                logger.info(f"Context truncated to {max_context_length} characters")

            return context

        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            return "Unable to retrieve information from knowledge base. Please try again later."

    def is_ready(self):
        """Check if RAG system is ready for queries"""
        return self.retriever is not None and self.vectorstore is not None




# supervisor.py
# supervisor.py
import logging
import re
from typing import Dict, List, Optional
from API.huggingface_api import huggingface_completion
from custom_tools import get_all_tools, calculator, schedule_appointment

logger = logging.getLogger(__name__)

class Supervisor:
    def __init__(self, prompt_manager, rag_system):
        self.prompt_manager = prompt_manager
        self.rag_system = rag_system
        self.tools = get_all_tools()  # Get all available tools

    def handle_query(self, user_input: str, conversation_history: str = "") -> Dict:
        """Handle query and determine appropriate response with tool usage"""
        # Validate input first
        if not user_input or not isinstance(user_input, str) or len(user_input.strip()) == 0:
            return {
                "response": "Please provide a valid input.",
                "agent_type": "error",
                "context_used": "Input validation"
            }

        # Detect if tool should be used
        tool_response = self._try_use_tools(user_input)
        if tool_response:
            return {
                "response": tool_response,
                "agent_type": "tool",
                "context_used": "Tool execution"
            }

        # Otherwise proceed with normal RAG + LLM flow
        query_type = self.prompt_manager.detect_query_type(user_input)
        context = ""

        if query_type in ['technical', 'sales', 'general']:
            context = self.rag_system.query_knowledge_base(user_input)

        prompt = self.prompt_manager.format_main_prompt(
            query_type=query_type,
            context=context,
            conversation_history=conversation_history
        )

        llm_response = huggingface_completion(prompt)

        if llm_response['status'] == 1:
            response_text = llm_response['response']
        else:
            response_text = "I apologize, but I'm having trouble processing your request."

        return {
            "response": response_text,
            "agent_type": query_type,
            "context_used": context
        }

    def _try_use_tools(self, user_input: str) -> Optional[str]:
        """Check if user input matches any tool pattern and execute if so"""
        if not user_input or len(user_input.strip()) < 2:
            return None

        user_input_lower = user_input.lower().strip()

        # Check for calculator queries with better pattern matching
        if self._is_calculation_request(user_input_lower):
            try:
                expression = self._extract_math_expression(user_input)
                if expression:
                    # CONSISTENT TOOL CALLING: Use dictionary input
                    result = calculator.invoke({"expression": expression})
                    return f"Calculation result: {result}"
                else:
                    return "I couldn't extract a valid mathematical expression. Please try something like 'calculate 2+2'."
            except Exception as e:
                logger.error(f"Calculator error: {e}")
                return "Sorry, I encountered an error with the calculator. Please check your expression."

        # Check for appointment scheduling with better validation
        if self._is_appointment_request(user_input_lower):
            try:
                # Extract and validate appointment details
                appointment_details = self._extract_appointment_details(user_input)
                if appointment_details:
                    # CONSISTENT TOOL CALLING: Use dictionary input
                    result = schedule_appointment.invoke(appointment_details)
                    return result
                else:
                    return "To schedule an appointment, please provide details including name, contact information, and preferred time."
            except Exception as e:
                logger.error(f"Appointment scheduling error: {e}")
                return "Sorry, I encountered an error while scheduling the appointment. Please try again with clear details."

        return None

    def _is_calculation_request(self, user_input: str) -> bool:
        """More precise calculation detection"""
        calc_keywords = [
            'calculate', 'compute', 'what is', 'how much is',
            'math', 'mathematical', 'arithmetic'
        ]
        math_operators = r'[\+\-\*\/\^]|sqrt|sin|cos|tan|log'

        has_keyword = any(keyword in user_input for keyword in calc_keywords)
        has_operator = re.search(math_operators, user_input)
        has_numbers = re.search(r'\d+', user_input)

        return has_keyword or (has_operator and has_numbers)

    def _extract_math_expression(self, user_input: str) -> Optional[str]:
        """Extract mathematical expression from user input"""
        try:
            # Remove common phrases and clean up
            cleaned = re.sub(r'(calculate|compute|what is|how much is)', '', user_input, flags=re.IGNORECASE)
            cleaned = cleaned.strip()

            # Extract potential math expression
            # Look for sequences with numbers and operators
            math_pattern = r'[\(\)\d\s\.\+\-\*\/\^\,]+'
            match = re.search(math_pattern, cleaned)

            if match:
                expression = match.group().strip()
                # Basic validation - should contain at least one number and operator
                if re.search(r'\d', expression) and re.search(r'[\+\-\*\/\^]', expression):
                    return expression

            return None
        except Exception as e:
            logger.error(f"Error extracting math expression: {e}")
            return None

    def _is_appointment_request(self, user_input: str) -> bool:
        """More precise appointment detection"""
        appointment_keywords = [
            'schedule', 'appointment', 'meeting', 'book a time',
            'set up a meeting', 'make an appointment'
        ]
        contact_indicators = ['name', 'contact', 'email', 'phone', 'call']

        has_appointment_keyword = any(keyword in user_input for keyword in appointment_keywords)
        has_contact_info = any(indicator in user_input for indicator in contact_indicators)

        return has_appointment_keyword or has_contact_info

    def _extract_appointment_details(self, user_input: str) -> Optional[Dict]:
        """Extract structured appointment details from user input"""
        try:
            details = {}

            # Extract name (simple pattern)
            name_match = re.search(r'(?:name is|my name is|call me)\s+([A-Za-z\s]+)', user_input, re.IGNORECASE)
            if name_match:
                details['name'] = name_match.group(1).strip()

            # Extract contact info
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_input)
            if email_match:
                details['email'] = email_match.group(0)

            phone_match = re.search(r'(\+?[\d\s\-\(\)]{10,})', user_input)
            if phone_match:
                details['phone'] = phone_match.group(0).strip()

            # Extract time/date references
            time_indicators = ['tomorrow', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            for indicator in time_indicators:
                if indicator in user_input.lower():
                    details['preferred_time'] = indicator
                    break

            # If we have at least a name, proceed
            if details.get('name'):
                details['raw_input'] = user_input  # Pass raw input for further processing
                return details

            return None
        except Exception as e:
            logger.error(f"Error extracting appointment details: {e}")
            return None
