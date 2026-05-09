"""
FastAPI server for LangChain-based LLM chains using Ollama.

This module provides REST endpoints for generating essays and poems
using LangChain prompts and the Ollama LLM backend.
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langserve import add_routes
import uvicorn

# Configuration constants
HOST = "localhost"
PORT = 8000
OLLAMA_MODEL = "llama2"

# API metadata
API_TITLE = "Langchain API Server"
API_VERSION = "1.0"
API_DESCRIPTION = "REST API for LangChain-based LLM chains using Ollama"

# Prompt templates
ESSAY_PROMPT_TEMPLATE = "Write an essay about {topic} with 100 words."
POEM_PROMPT_TEMPLATE = "Write a poem about {topic} for 10-year-old children."


def configure_environment() -> None:
    """Load environment variables and configure LangSmith tracing."""
    load_dotenv()
    
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"


def initialize_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
    )


def setup_routes(app: FastAPI, llm: Ollama) -> None:
    """Configure API routes with LangChain chains.
    
    Args:
        app: FastAPI application instance
        llm: Ollama language model instance
    """
    essay_chain = ChatPromptTemplate.from_template(ESSAY_PROMPT_TEMPLATE) | llm
    poem_chain = ChatPromptTemplate.from_template(POEM_PROMPT_TEMPLATE) | llm
    
    add_routes(app, essay_chain, path="/essay")
    add_routes(app, poem_chain, path="/poem")


def main() -> None:
    """Initialize and run the FastAPI server."""
    configure_environment()
    
    app = initialize_app()
    llm = Ollama(model=OLLAMA_MODEL)
    
    setup_routes(app, llm)
    
    print(f"Starting server at http://{HOST}:{PORT}/docs")
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
