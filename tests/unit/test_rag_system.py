"""
Simple tests for rag_system.py
"""

import pytest
from rag_system import TigressTechRAG

def test_rag_initialization():
    """Test RAG system initialization"""
    rag = TigressTechRAG()
    assert rag is not None
    assert hasattr(rag, 'embedding_model')
    assert hasattr(rag, 'vectorstore')
    assert hasattr(rag, 'retriever')
    print("✓ RAG initialization")

def test_setup_rag_returns_boolean():
    """Test setup returns boolean"""
    rag = TigressTechRAG()
    result = rag.setup_rag()
    assert isinstance(result, bool)
    print("✓ Setup returns boolean")

def test_query_knowledge_base_with_empty_query():
    """Test query with empty input"""
    rag = TigressTechRAG()
    result = rag.query_knowledge_base("")
    assert "Please provide a valid query" in result
    print("✓ Empty query handling")

def test_query_knowledge_base_with_valid_query():
    """Test query with valid input"""
    rag = TigressTechRAG()
    result = rag.query_knowledge_base("test question")
    assert isinstance(result, str)
    print("✓ Valid query handling")

def test_is_ready_returns_boolean():
    """Test ready check returns boolean"""
    rag = TigressTechRAG()
    result = rag.is_ready()
    assert isinstance(result, bool)
    print("✓ Ready check returns boolean")

def test_rag_attributes():
    """Test RAG has required attributes"""
    rag = TigressTechRAG()
    assert rag.document_loader is not None
    assert hasattr(rag, 'setup_rag')
    assert hasattr(rag, 'query_knowledge_base')
    assert hasattr(rag, 'is_ready')
    print("✓ Required attributes exist")

if __name__ == "__main__":
    print("Testing RAG system...")
    
    test_rag_initialization()
    test_setup_rag_returns_boolean()
    test_query_knowledge_base_with_empty_query()
    test_query_knowledge_base_with_valid_query()
    test_is_ready_returns_boolean()
    test_rag_attributes()
    
    print("✓ All RAG tests passed")