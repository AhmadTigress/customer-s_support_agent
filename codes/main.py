# codes/main.py
import logging
import time
import signal
import sys
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

# Store pending human requests
pending_human_requests = {}  # Format: {room_id: {tool_call_id: str, request_id: str, timestamp: float}}

# ==================== HUMAN RESPONSE HANDLER ====================
def handle_human_decision(message, matrix_client, app):
    """Check if message is a human response to a pending decision"""
    room_id = message["room_id"]
    sender = message["sender"]
    content = message["content"]["body"]
    
    # Check if this room has a pending human decision
    if room_id in pending_human_requests:
        pending_request = pending_human_requests[room_id]
        tool_call_id = pending_request["tool_call_id"]
        
        logger.info(f"Processing human decision from {sender}: {content}")
        
        # Create tool message with human response
        tool_message = ToolMessage(
            content=content,
            tool_call_id=tool_call_id
        )
        
        # Update graph state with human decision
        config = {"configurable": {"thread_id": room_id}}
        try:
            app.update_state(
                config,
                {"messages": [tool_message], "human_response": content},
                as_node="ask_human"
            )
            
            # Continue graph execution
            for event in app.stream(None, config, stream_mode="values"):
                final_state = event
                if "messages" in final_state and final_state["messages"]:
                    response = final_state["messages"][-1].content
                    matrix_client.send_message(room_id, response)
                    break
            
            # Clean up pending request
            del pending_human_requests[room_id]
            logger.info(f"Human decision processed successfully for room {room_id}")
            
        except Exception as e:
            logger.error(f"Error processing human decision: {e}")
            matrix_client.send_message(room_id, "Error processing your decision. Please try again.")
    
    return room_id in pending_human_requests

def enhanced_process_message(message, matrix_client, app):
    """Enhanced message processor that handles human decisions"""
    # First check if this is a human response to a pending decision
    if handle_human_decision(message, matrix_client, app):
        return  # Already handled by human decision processor
    
    # Otherwise process as normal user message
    user_query = message["content"]["body"]
    sender = message["sender"]
    room_id = message["room_id"]
    
    logger.info(f"Received message from {sender} in room {room_id}: {user_query}")
    
    try:
        # Prepare initial state
        initial_state = {
            "user_input": user_query,
            "messages": [],
            "response": "",
            "sender": sender,
            "context": "",
            "query_type": "general",
            "needs_rag": True
        }
        
        # Execute the workflow
        logger.info("Generating response via LangGraph...")
        config = {"configurable": {"thread_id": room_id}}
        
        for event in app.stream(initial_state, config, stream_mode="values"):
            state = event
            last_message = state["messages"][-1] if state["messages"] else None
            
            # Check if we reached human interrupt point
            if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                # GRAPH IS PAUSED - Store the pending request and send question to human
                tool_call = last_message.tool_calls[0]
                tool_call_id = tool_call["id"]
                human_question = last_message.content
                
                # Store pending request
                pending_human_requests[room_id] = {
                    "tool_call_id": tool_call_id,
                    "request_id": tool_call["args"].get("request_id", "unknown"),
                    "timestamp": time.time()
                }
                
                # Send human question to Matrix room
                matrix_client.send_message(room_id, human_question)
                logger.info(f"Human decision requested in room {room_id}")
                return  # Exit - wait for human response
            
            # If we get here, no human interrupt occurred
            response_text = state.get("response", "")
            
            if response_text:
                # Truncate if too long for Matrix
                if len(response_text) > 1500:
                    response_text = response_text[:1497] + "..."
                
                # Add delay to avoid rate limiting
                time.sleep(2.5)
                
                # Send response
                matrix_client.send_message(room_id, response_text)
                
                # Log escalation if it happened
                if state.get("requires_human_escalation", False):
                    logger.warning(f"Conversation escalated to human: {state['escalation_reason']}")
                else:
                    logger.info("Response sent successfully")
            else:
                logger.warning("Failed to generate response")
                matrix_client.send_message(room_id, "I apologize, but I couldn't generate a response. Please try again.")
                
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        matrix_client.send_message(room_id, "I encountered an error processing your request. Please try again.")

def cleanup_pending_requests():
    """Clean up old pending requests to prevent memory leaks"""
    current_time = time.time()
    expired_requests = []
    
    for room_id, request in pending_human_requests.items():
        if current_time - request["timestamp"] > 300:  # 5 minutes expiry
            expired_requests.append(room_id)
            logger.warning(f"Cleaning up expired human request for room {room_id}")
    
    for room_id in expired_requests:
        del pending_human_requests[room_id]

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
        logger.info(f"Active pending requests: {len(pending_human_requests)}")
        
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