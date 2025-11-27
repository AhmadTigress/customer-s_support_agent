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
