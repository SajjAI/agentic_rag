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


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load environment variables
load_dotenv()

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_llm" not in st.session_state:
    st.session_state.selected_llm = "gpt-4"

if "implementation" not in st.session_state:
    st.session_state.implementation = None

if "show_main_menu" not in st.session_state:
    st.session_state.show_main_menu = True

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

# Import implementations
import groundx_openai_search
import groundx_deepseek_search
import app_deep_seek_agentic_crewai
import app_deepseek_agentic_langchain



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
    
    if st.session_state.current_chat is None:
        st.write("Select a chat interface to begin:")
        
        chat_options = {
            "OpenAI Search": "groundx_openai_search",
            "Deepseek Search": "groundx_deepseek_search",
            "Agentic CrewAI": "app_deep_seek_agentic_crewai",
            "Agentic LangChain": "app_deepseek_agentic_langchain"
        }
        
        selected_chat = st.selectbox(
            "Choose Chat Interface",
            options=list(chat_options.keys()),
            format_func=lambda x: x,
            index=None,
            placeholder="Select a chat interface..."
        )
        
        if selected_chat:
            st.session_state.current_chat = chat_options[selected_chat]
            st.rerun()
            
    else:
        try:
            # Import and run the selected chat interface
            if st.session_state.current_chat == "groundx_openai_search":
                import groundx_openai_search
                groundx_openai_search.main()
            
            elif st.session_state.current_chat == "groundx_deepseek_search":
                import groundx_deepseek_search
                groundx_deepseek_search.main()
            
            elif st.session_state.current_chat == "app_deep_seek_agentic_crewai":
                import app_deep_seek_agentic_crewai
                app_deep_seek_agentic_crewai.main()
            
            elif st.session_state.current_chat == "app_deepseek_agentic_langchain":
                import app_deepseek_agentic_langchain
                app_deepseek_agentic_langchain.main()
                
        except Exception as e:
            st.error(f"Error loading chat interface: {str(e)}")
            st.error("Please make sure all required dependencies are installed and environment variables are set.")
            
            if st.button("Return to Menu"):
                st.session_state.current_chat = None
                st.rerun()

if __name__ == "__main__":
    main() 