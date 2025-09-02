import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import your DocumentLoader class (assuming it's in the same file or imported)
from codes.document_loader import DocumentLoader  # Adjust import path as needed

# Set up logging
logger = logging.getLogger(__name__)

# Define constants (adjust these as needed)
TEXT_FILES_PATH = "./files"  # Path to your text files
PERSIST_DIRECTORY = "./chroma_db"  # Directory to persist vector store

class TigressTechRAG:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
        self.retriever = None
        self.document_loader = DocumentLoader(TEXT_FILES_PATH)
    
    def setup_rag(self, persist_directory=PERSIST_DIRECTORY):
        """Setup RAG system with document processing"""
        logger.info("Setting up RAG system...")
        
        # Load documents
        documents = self.document_loader.load_documents()
        if not documents:
            logger.error("No documents to process")
            return False
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
        )
        
        splits = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(splits)} chunks")
        
        # Create vector store
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
    
    def query_knowledge_base(self, query, max_context_length=2500):
        """Query the knowledge base for relevant information"""
        if not self.retriever:
            return "Knowledge base not available."
        
        try:
            # Retrieve relevant documents
            relevant_docs = self.retriever.get_relevant_documents(query)
            if not relevant_docs:
                return "No specific information found. Please contact our team for detailed assistance."
            
            # Combine context
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
            
            return context
            
        except Exception as e:
            logger.error(f"Error in query: {e}")
            return "Unable to retrieve information."