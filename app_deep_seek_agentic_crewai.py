import streamlit as st
import os
import tempfile
import gc
import base64
import time
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from src.agentic_rag.tools.custom_tool import DocumentSearchTool
from Agents_config import create_agents_and_tasks

# Load environment variables at startup
load_dotenv()

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history

if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = []  # Store list of processed PDF files

if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None  # Initialize pdf_tool as None

if "crew" not in st.session_state:
    st.session_state.crew = None      # Store the Crew object

def reset_chat():
    st.session_state.messages = []
    gc.collect()

def display_pdf(file_bytes: bytes, file_name: str):
    """Displays the uploaded PDF in an iframe."""
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f"""
    <iframe 
        src="data:application/pdf;base64,{base64_pdf}" 
        width="100%" 
        height="600px" 
        type="application/pdf"
    >
    </iframe>
    """
    st.markdown(f"### Preview of {file_name}")
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    # ===========================
    #   Sidebar
    # ===========================
    with st.sidebar:
        st.header("Add Your PDF Documents")
        uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
        st.session_state.pdf_tool = DocumentSearchTool()
        
        if uploaded_files:
            # Create PDF tool if not exists
            if st.session_state.pdf_tool is None:
                with st.spinner("Setting up document search..."):
                    st.success("Search system initialized!")

            # Process new files
            for uploaded_file in uploaded_files:
                # Check if file was already processed
                if uploaded_file.name not in [f["name"] for f in st.session_state.pdf_files]:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        with st.spinner(f"Indexing {uploaded_file.name}... Please wait..."):
                            try:
                                process_id = st.session_state.pdf_tool.upload_document(temp_file_path)
                                st.session_state.pdf_files.append({
                                    "name": uploaded_file.name,
                                    "process_id": process_id
                                })
                                st.success(f"{uploaded_file.name} indexed!")
                            except Exception as e:
                                st.error(f"Error indexing {uploaded_file.name}: {str(e)}")

            # Display summary of indexed documents
            if st.session_state.pdf_files:
                st.write("📚 Indexed Documents:")
                for pdf_file in st.session_state.pdf_files:
                    st.write(f"- {pdf_file['name']}")

        if st.button("Clear Chat"):
            reset_chat()
            st.session_state.pdf_files = []  # Also clear the PDF files list

    # ===========================
    #   Main Chat Interface
    # ===========================
    st.title("Agentic RAG with CrewAI")
    st.write("Upload PDFs and ask questions about their content using an AI crew.")

    # Render existing conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Ask a question about your PDF...")

    if prompt:
        # 1. Show user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Build or reuse the Crew (only once after PDF is loaded)
        if st.session_state.crew is None:
            st.session_state.crew = create_agents_and_tasks(st.session_state.pdf_tool)

        # 3. Get the response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Get the complete response first
            with st.spinner("Thinking..."):
                inputs = {"query": prompt}
                result = st.session_state.crew.kickoff(inputs=inputs).raw
            
            # Split by lines first to preserve code blocks and other markdown
            lines = result.split('\n')
            for i, line in enumerate(lines):
                full_response += line
                if i < len(lines) - 1:  # Don't add newline to the last line
                    full_response += '\n'
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.15)  # Adjust the speed as needed
            
            # Show the final response without the cursor
            message_placeholder.markdown(full_response)

        # 4. Save assistant's message to session
        st.session_state.messages.append({"role": "assistant", "content": result})

if __name__ == "__main__":
    main()
