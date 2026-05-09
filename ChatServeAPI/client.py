"""
Streamlit client for ChatServeAPI LLM endpoints.

Provides a web interface for generating essays and poems
through the FastAPI backend.
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv

# Configuration
API_BASE_URL = "http://localhost:8000"
ESSAY_ENDPOINT = f"{API_BASE_URL}/essay/invoke"
POEM_ENDPOINT = f"{API_BASE_URL}/poem/invoke"

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"


def call_api(endpoint: str, topic: str) -> str:
    """Make a request to the API endpoint and return the response.
    
    Args:
        endpoint: The API endpoint URL
        topic: The input topic for the LLM chain
        
    Returns:
        The generated content from the LLM
        
    Raises:
        requests.RequestException: If the API call fails
    """
    try:
        response = requests.post(
            endpoint,
            json={"input": {"topic": topic}},
            timeout=60
        )
        response.raise_for_status()
        return response.json()["output"]
    except requests.exceptions.ConnectionError:
        st.error(" Cannot connect to API server. Is it running on localhost:8000?")
        return ""
    except requests.exceptions.Timeout:
        st.error(" API request timed out")
        return ""
    except requests.exceptions.RequestException as e:
        st.error(f" API Error: {str(e)}")
        return ""
    except (KeyError, ValueError) as e:
        st.error(f" API Error: {str(e)}")
        return ""


def main() -> None:
    """Main Streamlit application."""
    st.title(" ChatServeAPI Client")
    st.markdown("Generate essays and poems using LangChain and Ollama")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Essay Generator")
        essay_topic = st.text_input("Enter a topic for an essay:")
        if essay_topic:
            with st.spinner("Generating essay..."):
                essay_output = call_api(ESSAY_ENDPOINT, essay_topic)
                if essay_output:
                    st.success("Essay generated!")
                    st.write(essay_output)
    
    with col2:
        st.subheader(" Poem Generator")
        poem_topic = st.text_input("Enter a topic for a poem:")
        if poem_topic:
            with st.spinner("Generating poem..."):
                poem_output = call_api(POEM_ENDPOINT, poem_topic)
                if poem_output:
                    st.success("Poem generated!")
                    st.write(poem_output)


if __name__ == "__main__":
    main()