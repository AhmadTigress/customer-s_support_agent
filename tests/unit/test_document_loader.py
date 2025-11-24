# tests/unit/test_document_loader.py
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
from codes.document_loader import DocumentLoader, Document

class TestDocumentLoader:
    
    def test_document_loader_initialization(self):
        """Test DocumentLoader initialization"""
        loader = DocumentLoader("/test/path")
        
        assert loader.text_files_path == "/test/path"
        assert loader.required_files == {
            'services_policies': 'services_policies.txt',
            'faqs': 'faqs.txt'
        }
    
    def test_load_documents_success(self):
        """Test successful loading of all documents"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files using simpler method
            (Path(temp_dir) / "faqs.txt").write_text("FAQ content")
            (Path(temp_dir) / "services_policies.txt").write_text("Policy content")
            
            loader = DocumentLoader(temp_dir)
            documents = loader.load_documents()
            
            assert len(documents) == 2
            assert all(isinstance(doc, Document) for doc in documents)
            
            # Verify metadata
            sources = {doc.metadata["source"] for doc in documents}
            types = {doc.metadata["type"] for doc in documents}
            assert sources == {"faqs.txt", "services_policies.txt"}
            assert types == {"faqs", "services_policies"}
    
    def test_load_documents_missing_files(self):
        """Test loading when files are missing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test no files
            loader = DocumentLoader(temp_dir)
            assert loader.load_documents() == []
            
            # Test partial files
            (Path(temp_dir) / "faqs.txt").write_text("FAQ content")
            documents = loader.load_documents()
            assert len(documents) == 1
            assert documents[0].metadata["source"] == "faqs.txt"
    
    def test_load_documents_file_read_error(self):
        """Test handling of file read errors"""
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "faqs.txt").write_text("content")
            
            with patch('builtins.open', side_effect=PermissionError("Access denied")):
                loader = DocumentLoader(temp_dir)
                assert loader.load_documents() == []
    
    def test_load_documents_encoding_issues(self):
        """Test handling of encoding issues"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create file with invalid UTF-8
            (Path(temp_dir) / "faqs.txt").write_bytes(b'\xff\xfe\x00\x01')
            
            loader = DocumentLoader(temp_dir)
            assert loader.load_documents() == []
    
    def test_document_structure(self):
        """Test that loaded documents have correct structure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "faqs.txt").write_text("Test content")
            
            loader = DocumentLoader(temp_dir)
            documents = loader.load_documents()
            
            doc = documents[0]
            assert doc.page_content == "Test content"
            assert doc.metadata == {"source": "faqs.txt", "type": "faqs"}

# Make the file runnable
if __name__ == "__main__":
    # Create test instance
    test_instance = TestDocumentLoader()
    
    # Run each test method
    print("Running DocumentLoader tests...")
    
    try:
        test_instance.test_document_loader_initialization()
        print("✓ test_document_loader_initialization passed")
    except Exception as e:
        print(f"✗ test_document_loader_initialization failed: {e}")
    
    try:
        test