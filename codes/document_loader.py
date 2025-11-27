import os
import logging
from pathlib import Path
from langchain.schema import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# current_dir = os.path.dirname(__file__)
# project_root = os.path.dirname(current_dir)
# services_path = os.path.join(project_root, "files", "services_policies.txt")
# faqs_path = os.path.join(project_root, "files", "faqs.txt")

# ==================== DOCUMENT LOADER ====================

class DocumentLoader:
    def __init__(self, text_files_path):
        self.text_files_path = text_files_path
        self.required_files = {
            'services_policies': 'services_policies.txt',
            'faqs': 'faqs.txt',
        }

    def load_documents(self):
        """Load documents from specified path"""
        documents = []

        for doc_type, filename in self.required_files.items():
            file_path = Path(self.text_files_path) / filename

            if not file_path.exists():
                logger.warning(f"File not found: {filename} at {file_path}")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                document = Document(
                    page_content=content,
                    metadata={"source": filename, "type": doc_type}
                )
                documents.append(document)
                logger.info(f"Loaded {filename}")

            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                continue

        if not documents:
            logger.error("No documents loaded. Please check your file paths")
            return []

        return documents
