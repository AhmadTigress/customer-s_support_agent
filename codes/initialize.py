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