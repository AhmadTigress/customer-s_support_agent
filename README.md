# Tigress Tech RAG Multi-Agent Customer Support System

A Retrieval-Augmented Generation (RAG) multi-agent system built with LangChain and LangGraph frameworks to provide intelligent customer support through Matrix communication platform.

## Project Overview
## Overview
A modular, extensible customer support bot for Matrix chat platforms.
- Answers user queries with retrieval-augmented generation (RAG).
- Uses Guardrails to filter toxic/profane/inappropriate content.
- Supports escalation to human agents and customizable workflows.
- Built with Python, LangChain, and modern AI tooling.

## System Architecture
```txt
CUSTOM_SUPPORT/

├── codes/
│   ├── API/
│   │   ├── huggingface_api.py      # Hugging Face LLM integration
│   │
│   ├── config/
│   │   └── prompt_config.yaml      # YAML configuration for prompts
│   ├── graph/
│   │   └── bot_graph.py            # LangGraph workflow definition
│   ├── nodes/
│   │   └── bot_nodes.py            # Graph nodes implementation
│   ├── states/
│   │   └── bot_state.py            # State management
│   ├── custom_tools.py             # Custom tools (calculator, scheduler)
│   ├── document_loader.py          # Document loading and processing
│   ├── escalation_evaluator.py     # Detects need for escalation
|   |-- giskard_eval.py             # Evaluates and tests RAG performance.
│   ├── guardrails_ai.py            # Validates input and output ✅ ADDED
│   ├── initialize.py               # Loads model and services
│   ├── main.py                     # Main application entry point
│   ├── prompt_manager.py           # Prompt management and formatting
│   ├── rag_system.py               # RAG system implementation
│   └── supervisor.py              # Supervisor agent coordination
|--- monitoring/
|   |--- metrics.py                 # Custom prometheus metric
|   |--- health.py                  # Detailed dependency checks
|
|--- app.py               #
├── files/
│   ├── faqs.txt                    # Frequently Asked Questions
│   └── services_policies.txt       # Service policies document
├── tests/
│   ├── unit/
│   │   ├── test_bot_state.py
│   │   ├── test_custom_tools.py
│   │   ├── test_document_loader.py
│   │   ├── test_escalation_evaluator.py
│   │   ├── test_guardrails_ai.py   # ✅ ADDED test file
│   │   ├── test_initialize.py
│   │   ├── test_prompt_manager.py
│   │   ├── test_rag_system.py
│   │   ├── test_prompt_config.py
│   │   ├── test_main.py
│   │   └── test_file_resources.py
│   ├── mock/
│   │   ├── test_bot_nodes.py
│   │   ├── test_bot_graph.py
│   │   ├── test_supervisor.py
│   │   ├── test_huggingface_api.py
│   │   └── test_matrix_api.py
│   ├── integration/
│   │   ├── test_live_huggingface.py
│   │   ├── test_live_matrix.py
│   │   └── test_full_graph.py
│   └── conftest.py
├── .env                            # Environment variables
└── requirements.txt                # Python dependencies
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
    ```sh
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
    ```sh
    pip install -r requirements.txt
    pip install guardrails-ai     # <--- Add this if not in requirements.txt
    ```

4. **Setup Environment Variables**
    - Copy `.env copy` to `.env`:
      ```sh
      cp ".env copy" .env
      ```
    - Fill in `.env` values (Matrix server, user, password, room ID; and if needed, HUGGINGFACE_API_KEY).

    **.env example:**
    ```
    MATRIX_HOMESERVER=https://matrix.example.com
    MATRIX_USER=@bot:example.com
    MATRIX_PASSWORD=yourpassword
    MATRIX_ROOM_ID=!some:example.com
    HUGGINGFACE_API_KEY=your_api_key
    API_KEY="1234567-FAKE-KEY"
    ```

## Usage

Run the main bot:
```sh
python codes/main.py
```
or test guardrails separately
```sh
python codes/guardrails_ai.py
```
The bot will auto-join rooms it's involved to and begin processing/supporting conversation.

**Example Interactions**
- **General Query**: "What are your business hours?"

- **Technical Support**: "My computer won't turn on"

- **Sales Inquiry**: "How much does your service cost?"

- **Complaint**: "I'm unhappy with the service"

- **Tools**: "Calculate 15% of 200" or "Schedule an appointment"

## Configuration
**Core Agents**

- **codes/API/matrix_api.py**: Handles matrix chat API integration(login, sync, message sending).
- **codes/guardrails_ai.py**: Guardrails protection on user input and bot output.
- **codes/rag_system.py**: RAG logic for answer generation.
- **codes/main.py**: Main entry point, loads/links all functionality.
- **codes/custom_tools.py**, **codes/escalation_evaluator.py**, **codes/      document_loader.py**: Custom behaviours, escalation, data loading.
- **.env**: Sensitive configuration(not in public repo!)

**Knowledge Base**
  - FAQ documents

  - Service policies

  - Custom configuration prompts

## Technical Stack
  - **Framework**: LangChain + LangGraph

  - **LLM**: Hugging Face Meta-Llama-3-8B-Instruct

  - **Embeddings**: sentence-transformers/all-MiniLM-L6-v2

  - **Vector Store**: ChromaDB

  - **Communication**: Matrix Protocol

  - **Language**: Python 3.9+


## How it works
1. Connects to your Matrix server as a bot user.
2. Listens for and receives messages.
3. Applies Guardrails to input to prevent bad content.
4. Answers with RAG system (based on documents/data).
5. Applies Guardrails to outgoing replies.
6. Escalates to human if answer cannot be generated.
(See code comments for deeper technical detail.)


## Troubleshooting
- Bot not joining rooms? - Check your
- Matrix credentials and permissions.
- Guardrails errors? - Try updating/
- installing
`guardrails-ai` or check error messages in logs.
- Can't generate a response? -
- Ensure your vector store/db and API keys (Huggingface) are set.
- For logging/debugging: - Adjust
`logging.basicConfig(level=log ging. INFO)` in the code.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/my-feature).
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

- Matrix.org for the communication protocol
