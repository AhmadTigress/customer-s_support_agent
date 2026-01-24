import os
import torch
import logging
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

# Custom components
from codes.API.matrix_api import MatrixClient
from codes.prompt_manager import PromptManager
from codes.rag_system import TigressTechRAG

# Set up logging and env
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# ==================== CONFIGURATION ====================
# Use the same model name everywhere to avoid double-loading
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# ==================== MODEL PROVIDER (SINGLETON) ====================
class ModelProvider:
    _instance = None

    def __init__(self):
        logger.info(f"Loading model {MODEL_NAME} into memory...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HUGGINGFACE_API_KEY)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
            token=HUGGINGFACE_API_KEY,
        )

        # This is your raw Transformers pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # This is the LangChain wrapper for that pipeline
        self.hf_llm = HuggingFacePipeline(pipeline=self.pipeline)
        logger.info("Model loaded successfully.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelProvider()
        return cls._instance

# ==================== INITIALIZATION EXECUTION ====================
# 1. Initialize Model Instance (Loads everything ONCE)
provider = ModelProvider.get_instance()
model_pipeline = provider.pipeline
hf_llm = provider.hf_llm  # Use this for LangChain components

# 2. Initialize Prompt Manager
logger.info("Loading prompt configuration...")
prompt_manager = PromptManager()

# 3. Initialize RAG System
logger.info("Setting up RAG system...")
rag = TigressTechRAG()
rag.setup_rag()

# 4. Initialize Matrix Client
logger.info("Initializing Matrix client...")
matrix_client = MatrixClient(
    os.getenv("MATRIX_HOMESERVER", ""),
    os.getenv("MATRIX_USERNAME", ""),
    os.getenv("MATRIX_PASSWORD", "")
)
