# Tigress Tech RAG Multi-Agent Customer Support System

A Retrieval-Augmented Generation (RAG) multi-agent system built with LangChain and LangGraph frameworks to provide intelligent customer support through Matrix communication platform.

## Project Overview
This project is part of the Ready Tensor Agentic AI Development Course - Phase II requirement. It implements a multi-agent system that handles customer complaints, technical support, sales inquiries, and general queries using advanced AI capabilities with RAG architecture.

## System Architecture
```txt
CUSTOM_SUPPORT/
├── codes/
│   ├── API/
│   │   ├── huggingface_api.py      # Hugging Face LLM integration
│   │   └── matrix_api.py           # Matrix client communication
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
│   ├── initialize.py               # Loads model and services      
│   ├── main.py                     # Main application entry point
│   ├── prompt_manager.py           # Prompt management and formatting
│   ├── rag_system.py               # RAG system implementation
│   └── supervisor.py               # Supervisor agent coordination
├── files/
│   ├── faqs.txt                    # Frequently Asked Questions
│   └── services_policies.txt       # Service policies document
├── .env                            # Environment variables
└── requirements.txt                # Python dependencies
```

For the test files
# Test Architecture
```text
CUSTOM_SUPPORT/
├── tests/
│   ├── unit/
│   │   ├── test_bot_state.py
│   │   ├── test_custom_tools.py
│   │   ├── test_document_loader.py
│   │   ├── test_escalation_evaluator.py
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
```

## Key Features
- **Multi-Agent Architecture**: Coordinated agent system with specialized roles

- **RAG Integration**: Retrieval-Augmented Generation for context-aware responses

- **Matrix Integration**: Real-time messaging through Matrix protocol

- **Custom Tools**: Built-in calculator and appointment scheduling capabilities

- **Intelligent Routing**: Smart query classification and routing

- **Privacy Protection**: Built-in privacy guidelines and data handling

## Installation
1. **Clone the repository**:
```bash
git clone https://github.com/AhmadTigress/customer_support.git
cd customer_support
```
2. **Install dependencies**:
```bash
pip install -r requirements.txt
```
3. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your credentials
```
4. **Prepare documents**:
```bash
mkdir -p files documents config
# Add your faqs.txt, services_policies.txt, and prompt_config.yaml
```

##  Configuration
Environment Variables (.env)
```env
HUGGINGFACE_API_KEY=your_huggingface_api_key
MATRIX_HOMESERVER=https://matrix.example.com
MATRIX_USER=@yourbot:example.com
MATRIX_PASSWORD=your_password
MATRIX_ROOM_ID=!roomid:example.com
```

## Usage
Starting the Bot
```bash
python main.py
```
**Example Interactions**
- **General Query**: "What are your business hours?"

- **Technical Support**: "My computer won't turn on"

- **Sales Inquiry**: "How much does your service cost?"

- **Complaint**: "I'm unhappy with the service"

- **Tools**: "Calculate 15% of 200" or "Schedule an appointment"

## Components
**Core Agents**

1. **Input Node**: Processes incoming Matrix messages

2. **Query Classifier**: Detects query type and routing needs

3. **RAG System**: Retrieves relevant context from knowledge base

4. **LLM Node**: Generates responses using Hugging Face models

5. **Supervisor**: Coordinates complex queries and tool usage

6. **Output Node**: Formats and sends responses

**Custom Tools**
  - **Calculator**: Mathematical expressions and unit conversions

  - **Appointment Scheduler**: Customer meeting scheduling system

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


## Query Processing Flow

  - **Message Reception**: Matrix client receives message

  - **Query Classification**: Detects query type and complexity

  - **Context Retrieval**: RAG system fetches relevant information

  - **Response Generation**: LLM generates context-aware response

  - **Tool Integration**: Optional tool execution for specific queries

  - **Response Delivery**: Formatted response sent via Matrix

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