import streamlit as st
from groundx import GroundX
from dotenv import load_dotenv
import os
import requests
import json
import time
import re

# Load environment variables
load_dotenv()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize clients
@st.cache_resource
def init_clients():
    groundx_client = GroundX(
        api_key=os.getenv('GROUNDX_API_KEY')
    )
    return groundx_client

# Cache for GroundX search results
@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_groundx_search(bucket_id: int, query: str):
    """Cache GroundX search results to reduce API calls"""
    groundx = init_clients()
    content_response = groundx.search.content(
        id=bucket_id,
        query=query
    )
    return content_response

def get_deepseek_response(prompt, model_name="deepseek-r1:14b"):
    """Get response from locally running Deepseek model via Ollama"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        return None

def main():
    # Initialize client
    groundx = init_clients()
    st.title("GroundX Deepseek Search")
    # Search configuration
    with st.sidebar:
        st.header("Configuration")
        model_name = "deepseek-r1:14b"
        
        bucket_id = st.number_input(
            "Bucket ID",
            value=int(os.getenv('GROUNDX_BUCKET_ID', 15153)),
            min_value=1,
            key="deepseek_simple_bucket_id"
        )

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.experimental_rerun()

        # Add LLM status indicator
        try:
            # Test Ollama connection
            requests.get("http://localhost:11434/api/tags")
            st.success("✅ Ollama is running")
        except:
            st.error("❌ Ollama is not running. Please start Ollama service.")
            st.info("To start Ollama, run: `ollama serve`")
            st.info(f"To pull the model, run: `ollama pull {model_name}`")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metadata" in message:
                with st.expander("Search Results"):
                    st.text(message["metadata"]["search_results"])

    # Chat input
    query = st.chat_input("Ask a question about your documents...")

    if query:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Get AI response
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching documents..."):
                    # Use cached search results
                    content_response = cached_groundx_search(bucket_id, query)
                    results = content_response.search
                    llm_text = results.text
                # Direct analysis with improved prompt
                # instruction = f"""You are a helpful virtual assistant that answers questions 
                # using ONLY the content provided below. Your task is to:
                # 1. Create a detailed answer based strictly on the provided content
                # 2. Use specific quotes or references from the content to support your answer
                # 3. Clearly state if any part of the question cannot be answered from the content
                # 4. Do not make assumptions or add information not present in the content
                # 5. If the answer is not in the content, say so explicitly
                # Direct analysis with conversation context
                with st.spinner(f"Analyzing results with {model_name}..."):
                    instruction = f"""Answer the question using ONLY the information from the provided content. 
                    Be direct and concise. If the information is not in the content, simply state that.
                    Do not add any external knowledge or assumptions.

                    Previous conversation context:
                    {' '.join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:] if m['role'] == 'user'])}

                    Content:
                    {llm_text}

                    Question: {query}

                    Answer based ONLY on the above content:"""

                    # Create a placeholder for streaming effect
                    message_placeholder = st.empty()
                    full_response = ""

                    completion = get_deepseek_response(instruction, model_name)

                    if completion:
                        # Remove content within <think> tags
                        response_text = completion
                        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
                        response_text = response_text.strip()

                        # Simulate streaming effect
                        for chunk in response_text.split():
                            full_response += chunk + " "
                            message_placeholder.markdown(full_response + "▌")
                            time.sleep(0.01)
                        
                        # Show final response without cursor
                        message_placeholder.markdown(full_response)

                        # Store response in chat history with metadata
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "metadata": {
                                "search_results": llm_text,
                                "model_used": model_name,
                                "search_score": results.score
                            }
                        })
                    else:
                        st.error("Failed to get response from Deepseek model")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Add some helpful information at the bottom
    with st.sidebar:
        st.markdown("---")
        st.markdown("""
        ### How to use:
        1. Type your question in the chat box
        2. The app will search through your documents
        3. Get AI-powered answers based on the content
        4. Continue the conversation with follow-up questions
        5. View search results in the expandable sections
        6. Clear chat history using the button above

        Note: Make sure Ollama is running locally with the Deepseek model installed.
        """)

if __name__ == "__main__":
    main()



