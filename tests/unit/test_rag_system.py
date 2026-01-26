"""
Simple tests for rag_system.py
"""

import pytest
from unittest.mock import MagicMock, patch
from rag_system import TigressTechRAG

@pytest.fixture
def rag_system():
    # Patch embeddings to avoid downloading a model
    with patch('rag_system.HuggingFaceEmbeddings'):
        return TigressTechRAG()

def test_rag_setup_failure_handling(rag_system):
    """Ensures query_knowledge_base handles uninitialized states gracefully."""
    result = rag_system.query_knowledge_base("Where is Nigeria?")
    assert "Knowledge base not available" in result

@patch('rag_system.Chroma')
@patch('rag_system.DocumentLoader')
@patch('rag_system.RecursiveCharacterTextSplitter')
def test_setup_rag_flow(mock_splitter, mock_loader, mock_chroma, rag_system):
    """Verifies that setup_rag loads, splits, and indexes documents."""
    # Setup mocks
    mock_loader_inst = mock_loader.return_value
    mock_loader_inst.load_documents.return_value = [MagicMock(page_content="Policy Info")]

    mock_splitter_inst = mock_splitter.return_value
    mock_splitter_inst.split_documents.return_value = [MagicMock(page_content="Split 1")]

    # Execute
    success = rag_system.setup_rag()

    assert success is True
    assert rag_system.retriever is not None
    mock_loader_inst.load_documents.assert_called_once()
    mock_chroma.from_documents.assert_called_once()

def test_query_knowledge_base_truncation(rag_system):
    """Tests if context is truncated when it exceeds max_context_length."""
    # Mock a retriever that returns long text
    mock_doc = MagicMock()
    mock_doc.page_content = "A" * 3000

    mock_retriever = MagicMock()
    mock_retriever.get_relevant_documents.return_value = [mock_doc]

    rag_system.retriever = mock_retriever

    result = rag_system.query_knowledge_base("query", max_context_length=100)

    assert len(result) <= 105 # Length + "..."
    assert result.endswith("...")

def test_query_empty_input(rag_system):
    """Validates input guarding for RAG queries."""
    assert rag_system.query_knowledge_base("") == "Please provide a valid query."
    assert rag_system.query_knowledge_base(None) == "Please provide a valid query."
