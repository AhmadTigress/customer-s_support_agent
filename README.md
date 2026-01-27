# Tigra AI: Multi-Agent Customer Support System
A Retrieval-Augmented Generation (RAG) multi-agent system built with LangChain and LangGraph frameworks to provide intelligent customer support.

## Project Overview
## Overview
Tigra is a sophisticated, production-ready AI support ecosystem designed for Tigress Tech Labs.
Built on LangGraph, it utilizes a directed acyclic graph to orchestrate specialized AI agents, ensuring high-accuracy responses through **RAG**, enterprise-grade **Guardrails**, and automated **Human Escalation** logic.

## System Architecture
The project implements a "Supervisor-Worker" pattern. The Supervisor acts as an orchestrator, determining the user's intent and routing the conversation state to the most relevant specialist node.

**Core Components**:
 - **Orchestration**: LangGraph-driven state machine (`bot_graph.py`).
 - **Intelligence**: Llama-3-8B-Instruct via Hugging Face Transformers.
 - **Knowledge Base**: Persistent ChromaDB vector store (`rag_system.py`) processing local policy documents.
 - **Safety Layer**: Guardrails AI integration for toxic language filtering, hallucination checks, and competitor redaction.
- **Monitoring**: Prometheus metrics (`metrics.py`) and deep-probing health checks (`health.py`).

## Project Structure
```txt
CUSTOM_SUPPORT/
│
├── codes/
│   ├── API/
│   │   └── huggingface_api.py        # Hugging Face LLM integration
│   │
│   ├── config/
│   │   └── prompt_config.yaml        # YAML configuration for prompts
│   │
│   ├── graph/
│   │   └── bot_graph.py              # LangGraph workflow definition
│   │
│   ├── nodes/
│   │   └── bot_nodes.py              # Graph nodes implementation
│   │
│   ├── states/
│   │   └── bot_state.py              # State management
│   │
│   ├── custom_tools.py               # Custom tools (calculator, scheduler)
│   ├── document_loader.py            # Document loading and processing
│   ├── escalation_evaluator.py       # Detects need for escalation
│   ├── giskard_eval.py               # Evaluates and tests RAG performance
│   ├── guardrails_ai.py              # Validates input and output
│   ├── initialize.py                 # Loads model and services
│   ├── main.py                       # Main application entry point
│   ├── prompt_manager.py             # Prompt management and formatting
│   ├── rag_system.py                 # RAG system implementation
│   └── supervisor.py                 # Supervisor agent coordination
│
├── monitoring/
│   ├── metrics.py                    # Custom Prometheus metric
│   └── health.py                     # Detailed dependency checks
│
├── streamlit_app.py                  # AI Support Frontend Interface
├── app.py                            # Application entry point
│
├── files/
│   ├── faqs.txt                      # Frequently Asked Questions
│   └── services_policies.txt         # Service policies document
│
├── tests/
│   ├── unit/
│   │   ├── test_bot_state.py
│   │   ├── test_custom_tools.py
│   │   ├── test_document_loader.py
│   │   ├── test_escalation_evaluator.py
│   │   ├── test_guardrails_ai.py     # Added test file
│   │   ├── test_initialize.py
│   │   ├── test_prompt_manager.py
│   │   ├── test_rag_system.py
│   │   ├── test_prompt_config.py
│   │   ├── test_main.py
│   │   └── test_file_resources.py
│   │
│   ├── mock/
│   │   ├── test_bot_nodes.py
│   │   ├── test_bot_graph.py
│   │   ├── test_supervisor.py
│   │   ├── test_huggingface_api.py
│   │   └── test_initialize.py
│   │
│   ├── integration/
│   │   ├── test_live_huggingface.py
│   │   └── test_graph_integration.py
│   │
│   └── conftest.py
│
├── .env                              # Environment variables
└── requirements.txt                  # Python dependencies

```

## Key Features
- **Matrix chat integration** (automatic room join, message sync/send)
- **Guardrails AI for content moderation**: prevents toxic, profane, or policy-violating communication
- **Retrieval-Augmented Generation (RAG)**: accurate, context-aware answers using LangChain & vector stores
- **Multi-step workflow and state-based logic**
- **Human escalation** when the system can't solve the query
- **Customizable tools and prompt management**
- **Automatic testing suite**

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/AhmadTigress/customer-s_support_agent.git
    cd customer-s_support_agent
    ```

2. **Python Environment**
    - Use Python 3.9 or newer.
    - (Recommended) Create a virtual environment:
      ```sh
      python3 -m venv venv
      source venv/bin/activate
      ```

3. **Install dependencies**
  ```bash
  pip install -r requirements.txt
  ```

4. **Environment Setup**
   Clone the repository and create a .env file in the root directory:
  ```bash
  HUGGINGFACE_API_KEY=your_api_key
  API_KEY="1234567-FAKE-KEY"
  ```

5. **Launching the System**
Tigra operates on a Client-Server model. You must run the FastAPI backend first.
Start the Backend:
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Start the Frontend:
```bash
streamlit run streamlit_app.py
```

## Features in Depth
**Intelligent Routing**
The `Supervisor` utilizes a specialized prompt to classify queries into `TECHNICAL`, `BILLING`, or `GENERAL` categories. If a query requires physical computation or scheduling, it invokes `custom_tools.py`.

**RAG Pipeline**
Documents in `files/` are automatically chunked and embedded using `sentence-transformers`. The system uses **Contextual Compression** to ensure only the most relevant snippets are passed to the LLM, reducing latency and cost.

**Safety & Escalation**
 - **Guardrails**: Every response is passed through `support_guard`. If a response contains restricted topics or halluncinated data, it is automatically blocked or "fixed" before the user sees it.

- **Escalation Evaluator**: Uses a composite scoring system (Sentiment + Complexity) to detect frustrated users and flag the conversation for human intervention.

## Testing Suite
The project maintains a rigorous testing standard using `pytest`:
 - **Unit Tests**: Verify individual logic in tools, loaders, and state management.
- **Mock Tests**: Simulate LLM and RAG responses to test Graph flow without incurring API costs or hardware overhead.
- **Integration Tests**: End-to-end verification of the FastAPI endpoints and Live LLM connectivity.

**Run tests**:
```bash
pytest tests/
```

## Monitoring & Health
 - **Metrics**: Access /metrics for Prometheus-formatted data on request latency, escalation rates, and guardrail violations.
 - **Health**: Access /health for a status report on the Model, VectorDB, and Environment variables.


## Technical Stack
  - **Framework**: LangChain + LangGraph
  - **LLM**: Hugging Face Meta-Llama-3-8B-Instruct
  - **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
  - **Vector Store**: ChromaDB
  - **Safety & Evaluation**: Guardrails AI and Giskard
  - **Frontend & Observability**: Streamlit(Provide user interface and intereaction)
                                  Prometheus & Prometheus FastAPI Instrumentator(Tracks custom business metrics and API performance)
 - **Language**: Python 3.9+



Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch.
3. Commit your changes and push to your branch.
4. Open a Pull Request with a clear description of your changes.

Please follow our coding conventions and include relevant tests when applicable.

**License**

This project is licensed under the MIT License. See the LICENSE file for details.

## Author
**Ahmad Tigress**

- GitHub: @AhmadTigress
- Course: Ready Tensor Agentic AI Development - Phase II

##  Acknowledgments
- Ready Tensor for the educational framework
- LangChain and LangGraph teams for excellent documentation
- Hugging Face for model access and tools
