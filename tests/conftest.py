# tests/conftest.py
import pytest
import os
from unittest.mock import MagicMock, patch
from states.bot_state import ConversationMetadata


# ==================== 2. GLOBAL HARDWARE MOCKS ====================
@pytest.fixture(scope="session", autouse=True)
def disable_gpu_loading():
    """
    Critical: Prevents Llama-3 from loading into RAM/VRAM during tests.
    Targets initialize.py and transformers directly.
    """
    with patch('transformers.AutoTokenizer.from_pretrained'), \
         patch('transformers.AutoModelForCausalLM.from_pretrained'), \
         patch('transformers.pipeline'), \
         patch('langchain_community.vectorstores.Chroma.from_documents'), \
         patch('codes.initialize.ModelProvider.get_instance'):
        yield

# ==================== 3. SHARED COMPONENT MOCKS ====================
@pytest.fixture
def mock_rag():
    """Mock for the TigressTechRAG system."""
    with patch('nodes.bot_nodes.rag') as mock:
        mock.query_knowledge_base.return_value = "Mocked policy context for testing."
        yield mock

@pytest.fixture
def mock_hf_api():
    """Mock for the Hugging Face inference call."""
    with patch('nodes.bot_nodes.huggingface_completion') as mock:
        mock.return_value = {'status': 1, 'response': 'This is a mocked AI reply.'}
        yield mock

# ==================== 4. DATA FIXTURES ====================
@pytest.fixture
def dummy_state():
    """A standard AgentState object for unit testing nodes."""
    return {
        "user_input": "How do I reset my router?",
        "messages": [],
        "metadata": ConversationMetadata(turn_count=1),
        "context": "",
        "response": "",
        "next_agent": "technical_support_node"
    }
