import streamlit as st
from groundx import GroundX
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import requests
from typing import List, Dict, Any, Optional
from langchain.llms.base import BaseLLM
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import Field
from langchain.chains import LLMChain
from langchain.schema import Generation
import re
from app_deepseek_agentic_langchain import main as langchain_main
from app_deep_seek_agentic_crewai import main as crewai_main
from groundx_deepseek_search import main as groundx_main
from groundx_openai_search import main as groundx_openai_main

# Fix for SQLite version issues with ChromaDB
os.environ["LANGCHAIN_CHROMA_FORCE_PYSQLITE"] = "1"

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load environment variables
load_dotenv()

# Initialize session state variables
if "current_implementation" not in st.session_state:
    st.session_state.current_implementation = "langchain"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_llm" not in st.session_state:
    st.session_state.selected_llm = "gpt-4"

if "retriever_agent" not in st.session_state:
    st.session_state.retriever_agent = None

if "synthesizer_agent" not in st.session_state:
    st.session_state.synthesizer_agent = None

if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = []

if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None

if "crew" not in st.session_state:
    st.session_state.crew = None

def render_header():
    """Render the app header with navigation"""
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        st.title("AI Document Search & Analysis")
    
    with col2:
        if st.session_state.current_chat is not None:
            if st.button("← Back to Menu", use_container_width=True):
                st.session_state.current_chat = None
                st.session_state.messages = []
                st.rerun()

def main():
    render_header()
    
    st.title("Agentic RAG with Multiple Implementations")
    
    # Sidebar for implementation selection
    with st.sidebar:
        st.header("Select Implementation")
        implementation = st.selectbox(
            "Choose Implementation",
            ["langchain", "crewai", "groundx_deepseek", "groundx_openai"],
            index=["langchain", "crewai", "groundx_deepseek", "groundx_openai"].index(st.session_state.current_implementation)
        )
        
        if implementation != st.session_state.current_implementation:
            st.session_state.current_implementation = implementation
            # Reset relevant session state variables when switching implementations
            if "messages" in st.session_state:
                del st.session_state.messages
            if "retriever_agent" in st.session_state:
                del st.session_state.retriever_agent
            if "synthesizer_agent" in st.session_state:
                del st.session_state.synthesizer_agent
            if "pdf_files" in st.session_state:
                del st.session_state.pdf_files
            if "pdf_tool" in st.session_state:
                del st.session_state.pdf_tool
            if "crew" in st.session_state:
                del st.session_state.crew
            st.rerun()

        # Add back button
        if st.button("← Back to Selection"):
            st.session_state.current_implementation = "langchain"
            st.rerun()

    # Main content area
    try:
        if implementation == "langchain":
            langchain_main()
        elif implementation == "crewai":
            crewai_main()
        elif implementation == "groundx_deepseek":
            groundx_main()
        elif implementation == "groundx_openai":
            groundx_openai_main()
    except Exception as e:
        st.error(f"Error loading chat interface: {str(e)}")
        st.info("Please make sure all required dependencies are installed and environment variables are set.")

if __name__ == "__main__":
    main() 