import pytest
from unittest.mock import MagicMock, patch
import os

# We mock the imports and the global execution before the test starts
# to avoid the actual Llama model loading during test discovery.
@pytest.fixture(autouse=True)
def mock_initialization_env(mocker):
    mocker.patch('dotenv.load_dotenv')
    mocker.patch('transformers.AutoTokenizer.from_pretrained')
    mocker.patch('transformers.AutoModelForCausalLM.from_pretrained')
    mocker.patch('transformers.pipeline')
    mocker.patch('langchain_community.llms.HuggingFacePipeline')
    mocker.patch('codes.API.matrix_api.MatrixClient')
    mocker.patch('codes.prompt_manager.PromptManager')
    mocker.patch('codes.rag_system.TigressTechRAG')

def test_model_provider_singleton():
    """Verifies that ModelProvider follows the Singleton pattern."""
    from initialize import ModelProvider

    # Reset singleton for a clean test
    ModelProvider._instance = None

    instance1 = ModelProvider.get_instance()
    instance2 = ModelProvider.get_instance()

    assert instance1 is instance2

@patch('initialize.AutoTokenizer.from_pretrained')
@patch('initialize.AutoModelForCausalLM.from_pretrained')
@patch('initialize.pipeline')
def test_model_provider_initialization(mock_pipeline, mock_model, mock_tokenizer):
    """Checks if the transformers pipeline and tokenizer are set up with correct params."""
    from initialize import ModelProvider
    ModelProvider._instance = None # Reset

    # Setup mock returns
    mock_tok_inst = MagicMock()
    mock_tok_inst.pad_token = None
    mock_tokenizer.return_value = mock_tok_inst

    provider = ModelProvider()

    # Verify Tokenizer configuration
    mock_tokenizer.assert_called_once()
    assert mock_tok_inst.pad_token == mock_tok_inst.eos_token

    # Verify Model configuration (dtype and device mapping)
    args, kwargs = mock_model.call_args
    assert kwargs['device_map'] == "auto"

    # Verify Pipeline creation
    mock_pipeline.assert_called_once_with(
        "text-generation",
        model=provider.model,
        tokenizer=provider.tokenizer,
        max_new_tokens=512,
        pad_token_id=mock_tok_inst.eos_token_id
    )

@patch('initialize.TigressTechRAG')
@patch('initialize.MatrixClient')
@patch('initialize.PromptManager')
def test_global_instances(mock_prompt, mock_matrix, mock_rag):
    """Verifies that the global services are initialized during module load."""
    # This test assumes the module is being reloaded or first loaded
    import initialize

    # Verify RAG setup was called
    initialize.rag.setup_rag.assert_called_once()

    # Verify Matrix client was initialized with env vars
    mock_matrix.assert_called_once()

    # Verify component availability
    assert initialize.model_pipeline is not None
    assert initialize.hf_llm is not None
