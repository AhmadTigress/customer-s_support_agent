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
from codes.API.matrix_api import MatrixClient
from codes.prompt_manager import PromptManager
from codes.rag_system import TigressTechRAG
from codes.graph.bot_graph import app  

# Load environment variables
load_dotenv()

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