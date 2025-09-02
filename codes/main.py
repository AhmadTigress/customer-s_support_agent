# ==================== MAIN BOT LOOP ====================
import logging
import time
import requests
from dotenv import load_dotenv
import os

# Import your custom classes and components
from codes.API.matrix_api import MatrixClient
from codes.prompt_manager import PromptManager
from codes.rag_system import TigressTechRAG
from codes.graph.bot_graph import app  # The compiled LangGraph app
from codes.supervisor import Supervisor

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

# Initialize global instances
matrix_client = MatrixClient(
    homeserver=os.getenv("MATRIX_HOMESERVER"),
    user=os.getenv("MATRIX_USER"),
    password=os.getenv("MATRIX_PASSWORD")
)

prompt_manager = PromptManager()
rag = TigressTechRAG()

def process_message(message):
    """Process a single message and generate response"""
    user_query = message["content"]["body"]
    sender = message["sender"]
    message_room_id = message["room_id"]
    
    logger.info(f"Received message from {sender} in room {message_room_id}: {user_query}")
    
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
    final_state = app.invoke(initial_state)
    response_text = final_state["response"]
    
    if response_text:
        # Truncate if too long for Matrix
        if len(response_text) > 1500:
            response_text = response_text[:1497] + "..."
        
        # Add delay to avoid rate limiting
        time.sleep(2.5)
        
        # Send response
        matrix_client.send_message(message_room_id, response_text)
        logger.info("Response sent successfully")
    else:
        logger.warning("Failed to generate response")

def run_bot():
    """Main function to run the matrix bot"""
    logger.info("Starting Matrix bot...")
    
    try:
        # Login to Matrix
        matrix_client.login()
        
        # Join the specified room
        matrix_client.join_room(ROOM_ID)
        
        logger.info(f"Logged in. Listening for messages in room {ROOM_ID}...")
        
        # Start listening for messages using the new method
        matrix_client.listen_for_messages(process_message_callback=process_message)
                
    except KeyboardInterrupt:
        logger.info("Shutting down bot gracefully...")
    except Exception as e:
        logger.error(f"Fatal error in bot execution: {e}")

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Initialize supervisor
    supervisor = Supervisor(prompt_manager, rag)
    rag.setup_rag()
    run_bot()