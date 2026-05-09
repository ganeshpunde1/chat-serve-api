# ChatServeAPI: LangChain-Powered LLM Service

A production-ready AI application providing REST APIs and a web interface for executing LLM-powered text generation workflows using LangChain, LangSmith, FastAPI, and Ollama. The project showcases enterprise-grade architecture for scalable AI services with advanced prompt orchestration, API integration, observability, and monitoring capabilities.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Technology Stack](#technology-stack)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Development Guide](#development-guide)
- [Performance Considerations](#performance-considerations)

## Architecture Overview

ChatServeAPI follows a **client-server architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web Client                      │
│                     (client.py)                              │
│   - User interface for prompt submission                     │
│   - Real-time streaming feedback                            │
│   - Error handling and request management                   │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST
                         ├─ POST /essay/invoke
                         └─ POST /poem/invoke
┌────────────────────────▼────────────────────────────────────┐
│                  FastAPI Backend Server                      │
│                      (app.py)                                │
│   - LangServe routing for LLM chains                        │
│   - Request/response serialization                          │
│   - LangChain integration                                   │
└────────────────────────┬────────────────────────────────────┘
                         │ LangChain Runnable Protocol
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
┌──────────────────┐         ┌──────────────────┐
│  Prompt Template │         │  Prompt Template │
│   (Essay Chain)  │         │   (Poem Chain)   │
└────────┬─────────┘         └────────┬─────────┘
         │                            │
         └────────────┬───────────────┘
                      ▼
           ┌────────────────────────┐
           │  Ollama LLM Engine     │
           │   (llama2 model)       │
           └────────────────────────┘
```

### Data Flow

1. **Client Request**: User submits topic through Streamlit UI
2. **API Call**: Client makes HTTP POST to FastAPI endpoint with JSON payload
3. **Chain Execution**: FastAPI routes request through LangChain prompt template + LLM
4. **LLM Generation**: Ollama processes the prompt using local llama2 model
5. **Response Serialization**: LangServe returns structured JSON response
6. **UI Display**: Streamlit renders the generated content

## Technology Stack

### Backend Technologies

#### **FastAPI** (Web Framework)
- **Purpose**: High-performance async web framework for building REST APIs
- **Version**: Latest stable
- **Key Features**:
  - ASGI-based async request handling for non-blocking I/O
  - Automatic OpenAPI/Swagger documentation at `/docs`
  - Built-in request validation with Pydantic
  - Type hints for self-documenting APIs
- **Usage**: Routes HTTP requests to LangChain chains

#### **LangChain** (LLM Framework)
- **Core Module**: `langchain_core`
  - **Purpose**: Core abstractions for LLM applications
  - **Components Used**:
    - `ChatPromptTemplate`: Structured prompt engineering
    - `Runnable` protocol: Composable chain interfaces
  
- **Community Module**: `langchain_community`
  - **Ollama Integration**: Local LLM execution
  - **Purpose**: Keeps core lightweight while providing extensibility
  
- **LangServe**: 
  - **Purpose**: REST API wrapper for LangChain chains
  - **Functionality**: Automatically converts Runnable objects to HTTP endpoints
  - **Serialization**: Handles JSON serialization/deserialization

#### **Ollama** (Local LLM Runtime)
- **Purpose**: Run large language models locally without cloud dependencies
- **Model**: `llama2` (7B parameter model)
- **Advantages**:
  - Data privacy (no external API calls)
  - Lower latency than cloud APIs
  - Cost-effective for high-volume inference
  - Fully open-source and customizable
- **Architecture**: C++ inference engine with model quantization support

#### **Uvicorn** (ASGI Server)
- **Purpose**: Production-grade ASGI application server
- **Configuration**: Single-threaded, multi-worker deployment ready
- **Features**: Hot-reloading, graceful shutdown, request logging

### Frontend Technologies

#### **Streamlit** (Web UI Framework)
- **Purpose**: Rapid data app development without frontend knowledge
- **Architecture**: Python-only, reactive component model
- **Features Used**:
  - `st.columns()`: Responsive multi-column layouts
  - `st.text_input()`: Form input handling
  - `st.spinner()`: Loading state indicators
  - Error/success messaging
- **Advantages**: 
  - No JavaScript/HTML/CSS required
  - Hot-reload during development
  - Built-in session state management

### Infrastructure & Utilities

#### **python-dotenv**
- **Purpose**: Environment variable management for sensitive configuration
- **Security Pattern**: Externalize secrets from source code via `.env` file
- **Usage**: Load API keys and tracing configuration

#### **Requests** (HTTP Client)
- **Purpose**: Make HTTP calls from Streamlit client to FastAPI backend
- **Error Handling**: Connection management, timeout handling, response validation

#### **LangSmith Integration**
- **Purpose**: Observability and tracing for LLM applications
- **Environment Variables**:
  - `LANGCHAIN_API_KEY`: Authentication token
  - `LANGCHAIN_TRACING_V2`: Enable tracing backend
- **Benefits**: 
  - Monitor chain execution
  - Debug prompt issues
  - Analyze token usage
  - Track latency metrics

## Features

-  **Structured Prompting**: Template-based prompt engineering with variable substitution
-  **Multiple Chains**: Separate LLM chains for different tasks (essay, poem generation)
-  **REST API**: LangServe-powered HTTP endpoints with automatic documentation
-  **Web UI**: Responsive Streamlit interface with real-time feedback
-  **Error Handling**: Comprehensive exception handling with user-friendly messages
-  **Request Timeouts**: 60-second timeout for LLM inference
-  **Observability**: LangSmith integration for monitoring and debugging
-  **Local LLM**: Privacy-preserving inference with Ollama
-  **Type Hints**: Full Python type annotations for IDE support and documentation

## Prerequisites

### System Requirements
- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: 3.9 or higher
- **RAM**: Minimum 8GB (16GB+ recommended for smooth Ollama inference)
- **Disk Space**: 
  - Base dependencies: ~500MB
  - Ollama + llama2 model: ~5-7GB

### External Dependencies
1. **Ollama**: Download from [ollama.ai](https://ollama.ai)
   - After installation, run: `ollama pull llama2`
   - Verify: `ollama list` (should show llama2 available)

2. **OPENAI_API_KEY** (Optional): For potential OpenAI integration
3. **LANGCHAIN_API_KEY** (Optional): For LangSmith observability

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd ChatServeAPI
```

### 2. Create Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Dependency Breakdown:**
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `langchain_core`: LLM abstractions
- `langchain_openai`: OpenAI integrations
- `langchain_community`: Community extensions (Ollama support)
- `langserve`: API routing layer
- `streamlit`: Web UI
- `python-dotenv`: Configuration management
- `requests`: HTTP client
- `sse_starlette`: Server-sent events (streaming support)

### 4. Verify Installation
```bash
pip list | grep -E "langchain|fastapi|streamlit"
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI Configuration (optional)
OPENAI_API_KEY=sk_your_key_here

# LangSmith Configuration (optional but recommended)
LANGCHAIN_API_KEY=ls_your_key_here
LANGCHAIN_TRACING_V2=true
```

**Security Best Practices:**
- Never commit `.env` to version control
- Use `.gitignore` entry: `*.env`
- Rotate keys periodically
- Use service accounts for production

### Server Configuration

Edit `app.py` constants to modify:

```python
HOST = "localhost"      # Change to 0.0.0.0 for network access
PORT = 8000            # Change port if 8000 is in use
OLLAMA_MODEL = "llama2" # Switch to different model (llama2, mistral, etc.)
```

### Prompt Customization

Modify prompt templates in `app.py`:

```python
ESSAY_PROMPT_TEMPLATE = "Write an essay about {topic} with 100 words."
POEM_PROMPT_TEMPLATE = "Write a poem about {topic} for 10-year-old children."
```

## Running the Application

### Step 1: Start Ollama
```bash
# Ollama should be running in background or started explicitly
ollama serve  # Starts on localhost:11434 by default
```

### Step 2: Start FastAPI Backend
```bash
python app.py
# Output: Starting server at http://localhost:8000/docs
```

**API Documentation**: Open http://localhost:8000/docs for interactive Swagger UI

### Step 3: Start Streamlit Client (New Terminal)
```bash
streamlit run client.py
# Output: You can now view your Streamlit app in your browser at: http://localhost:8501
```

### Alternative: Run with Uvicorn (Production)
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Essay Generation
**Endpoint**: `POST /essay/invoke`

**Request**:
```json
{
  "input": {
    "topic": "The future of artificial intelligence"
  }
}
```

**Response**:
```json
{
  "output": "Artificial intelligence represents one of humanity's most transformative technologies..."
}
```

**Status Codes**:
- `200`: Success
- `422`: Validation error (invalid input format)
- `500`: Internal server error

### Poem Generation
**Endpoint**: `POST /poem/invoke`

**Request**:
```json
{
  "input": {
    "topic": "friendship"
  }
}
```

**Response**:
```json
{
  "output": "Friends are treasures rare and true,\nWith hearts so bright and kind..."
}
```

### API Documentation
**Endpoint**: `GET /docs`

Interactive Swagger/OpenAPI documentation auto-generated by FastAPI.

**Endpoint**: `GET /openapi.json`

Raw OpenAPI specification for integration with third-party tools.

## Project Structure

```
ChatServeAPI/
├── app.py                 # FastAPI backend server
├── client.py             # Streamlit web frontend
├── requirements.txt      # Python dependencies
├── .env                  # Environment configuration (not in repo)
├── .gitignore           # Git ignore patterns
└── Readme.md            # This file
```

### File Descriptions

#### **app.py** (Backend, ~80 lines)
**Responsibilities**:
- Configure environment and LangSmith tracing
- Initialize FastAPI application instance
- Define LangChain prompt chains
- Register API routes with LangServe
- Start Uvicorn server

**Key Functions**:
- `configure_environment()`: Load env vars and enable tracing
- `initialize_app()`: Create FastAPI instance with metadata
- `setup_routes()`: Register LLM chains as HTTP endpoints
- `main()`: Orchestration entry point

#### **client.py** (Frontend, ~85 lines)
**Responsibilities**:
- Provide interactive web UI with Streamlit
- Make HTTP requests to backend API
- Handle errors gracefully
- Display results with visual feedback

**Key Functions**:
- `call_api()`: Abstracted HTTP client with error handling
- `main()`: UI layout and event handling

#### **requirements.txt**
Pinned versions of all dependencies for reproducibility.

## Development Guide

### Adding New Chains

1. **Define Prompt Template** in `app.py`:
```python
NEW_PROMPT_TEMPLATE = "Your prompt here with {input_variable}"
```

2. **Create Chain** in `setup_routes()`:
```python
new_chain = ChatPromptTemplate.from_template(NEW_PROMPT_TEMPLATE) | llm
```

3. **Register Route**:
```python
add_routes(app, new_chain, path="/new-endpoint")
```

4. **Update Client** in `client.py`:
```python
NEW_ENDPOINT = f"{API_BASE_URL}/new-endpoint/invoke"
```

### Switching LLM Models

**Ollama Models**:
```bash
ollama pull mistral      # 7B instruction-tuned model
ollama pull neural-chat  # 7B optimized for dialogue
ollama pull orca-mini    # 3B lightweight model
```

Update `OLLAMA_MODEL` in `app.py`:
```python
OLLAMA_MODEL = "mistral"
```

### Debugging

**Enable Verbose Logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check Ollama Status**:
```bash
curl http://localhost:11434/api/tags
```

**Monitor LangSmith**:
Visit [langsmith.com](https://www.langsmith.com) dashboard for execution traces.

## Performance Considerations

### Latency Analysis

| Component | Typical Latency |
|-----------|-----------------|
| Ollama LLM Inference (llama2) | 5-30 seconds |
| API Overhead | 100-500ms |
| Streamlit UI | 100-200ms |
| Network (local) | 10-50ms |
| **Total** | **5-31 seconds** |

### Optimization Strategies

1. **Model Quantization**: Use smaller quantized models for faster inference
   ```bash
   ollama pull llama2:7b-q4  # 4-bit quantized
   ```

2. **Batch Processing**: Process multiple requests in parallel with worker processes
   ```bash
   uvicorn app:app --workers 4
   ```

3. **Caching**: Implement response caching for identical prompts
   ```python
   from functools import lru_cache
   ```

4. **Streaming**: For long outputs, implement server-sent events (SSE)
   - Use `sse_starlette` for streaming responses

5. **Resource Allocation**: Monitor Ollama memory usage
   ```bash
   # macOS/Linux
   ps aux | grep ollama
   ```

### Scaling Considerations

- **Single Instance**: Supports ~10-50 concurrent users (latency-dependent)
- **Multi-Worker Setup**: Use `--workers 4+` with load balancer (nginx)
- **Production Deployment**: Docker containerization recommended
- **Distributed Architecture**: Consider message queue (Celery) for async processing

---

**Last Updated**: May 2026  
**Status**: Production Ready

