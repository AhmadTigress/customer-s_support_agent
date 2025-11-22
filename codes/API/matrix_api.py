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