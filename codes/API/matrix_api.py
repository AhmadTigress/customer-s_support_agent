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
    
    def login(self):
        """Authenticate and get access token"""
        url = f"{self.homeserver}/_matrix/client/v3/login"
        data = {
            "type": "m.login.password",
            "user": self.user,
            "password": self.password,
        }
        res = requests.post(url, json=data)
        res.raise_for_status()
        self.access_token = res.json()["access_token"]
        logger.info("Successfully logged into Matrix")
        return self.access_token
    
    def send_message(self, room_id, message):
        """Send message to a room"""
        url = f"{self.homeserver}/_matrix/client/v3/rooms/{room_id}/send/m.room.message"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        data = {
            "msgtype": "m.text",
            "body": message,
        }
        res = requests.post(url, headers=headers, json=data)
        res.raise_for_status()
        logger.info(f"Message sent to room {room_id}")
    
    def sync_messages(self, sync_token=None):
        """Sync with matrix server to get new messages"""
        url = f"{self.homeserver}/_matrix/client/v3/sync"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        params = {"timeout": 30000}
        if sync_token:
            params["since"] = sync_token

        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status()
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
        url = f"{self.homeserver}/_matrix/client/v3/join/{room_id}"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        res = requests.post(url, headers=headers)
        res.raise_for_status()
        logger.info(f"Joined room: {room_id}")
    
    def get_invited_rooms(self, sync_token=None):
        """Check for room invitations"""
        url = f"{self.homeserver}/_matrix/client/v3/sync"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        params = {"timeout": 30000}
        if sync_token:
            params["since"] = sync_token
        
        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status()
        data = res.json()

        invited_rooms = []
        for room_id, invite_info in data.get("rooms", {}).get("invite", {}).items():
            invited_rooms.append(room_id)
        
        return invited_rooms

    def listen_for_messages(self, process_message_callback=None):
        """Continuously listen for messages and process them with graceful shutdown"""
        sync_token = None
        logger.info("Starting to listen for messages...")
        
        while not self._shutdown_event.is_set():
            try:
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
                        process_message_callback(message)
                    else:
                        # Default behavior for simple responses
                        if body.lower() in ["hello", "hi", "hey", "hello!"]:
                            self.send_message(room_id, "Hello! How can I help you today?")
                        
                
            except Exception as e:
                logger.error(f"Error in message listening: {e}")
                if not self._shutdown_event.is_set():
                    # Wait with timeout to allow shutdown check
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