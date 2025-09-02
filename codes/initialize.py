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