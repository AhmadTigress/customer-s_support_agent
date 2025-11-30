# ==================== MISSING IMPORTS ADDED ====================
import os
import torch
import logging
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

# Import your custom classes and components
from codes.API.matrix_api import MatrixClient
from codes.prompt_manager import PromptManager
from codes.rag_system import TigressTechRAG


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the environment variables from the .env file
load_dotenv()

# ==================== MISSING CONFIG VARIABLES ====================
MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceH4/zephyr-7b-beta")  # Fallback model
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
MATRIX_HOMESERVER = os.getenv("MATRIX_HOMESERVER", "")
MATRIX_USERNAME = os.getenv("MATRIX_USERNAME", "")
MATRIX_PASSWORD = os.getenv("MATRIX_PASSWORD", "")



# ==================== INITIALIZATION ====================
# Initialize components in proper order
logger.info("Initializing components...")

# 1. Initialize LLM first
logger.info("Loading LLM model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HUGGINGFACE_API_KEY)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto",
    token=HUGGINGFACE_API_KEY,
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
hf_llm = HuggingFacePipeline(pipeline=pipe)

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
matrix_client = MatrixClient(MATRIX_HOMESERVER, MATRIX_USERNAME, MATRIX_PASSWORD)
