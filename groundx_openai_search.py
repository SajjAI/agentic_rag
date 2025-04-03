import streamlit as st
from openai import OpenAI
from groundx import GroundX
from dotenv import load_dotenv
import os
import tempfile
import gc
from src.agentic_rag.tools.custom_tool import DocumentSearchTool

# Load environment variables
load_dotenv()


def reset_chat():
    st.session_state.messages = []
    gc.collect()

# Initialize clients
@st.cache_resource
def init_clients():
    groundx_client = GroundX(
        api_key=os.getenv('GROUNDX_API_KEY')
    )
    openai_client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY')
    )
    return groundx_client, openai_client

def main():
    # Initialize clients
    groundx, client = init_clients()
    st.title("GroundX OpenAI Search")
    # Search configuration
    with st.sidebar:
        st.header("Configuration")
        completion_model = "gpt-4o"
        
        bucket_id = st.number_input(
            "Bucket ID",
            value=int(os.getenv('GROUNDX_BUCKET_ID', 15153)),
            min_value=1
        )
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
    # Main search interface
    query = st.chat_input("Enter your search query:")

    if query:
        try:
            with st.spinner("Searching documents..."):
                # Get content from GroundX
                content_response = groundx.search.content(
                    id=bucket_id,
                    query=query
                )
                results = content_response.search
                llm_text = results.text

                # Show raw search results in expander
                with st.expander("View Raw Search Results"):
                    st.text(llm_text)

            with st.spinner("Analyzing results with AI..."):
                # OpenAI analysis
                instruction = """You are a helpful virtual assistant that answers questions 
                using the content below. Your task is to create detailed answers to the 
                questions by combining your understanding of the world with the content 
                provided below. Do not share links."""

                completion = client.chat.completions.create(
                    model=completion_model,
                    messages=[
                        {
                            "role": "system",
                            "content": f"{instruction}\n===\n{llm_text}\n===",
                        },
                        {"role": "user", "content": query},
                    ]
                )

                # Display results
                st.subheader("Analysis Results")
                
                # Display metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Search Score", f"{results.score:.2f}")
                with col2:
                    st.metric("Model Used", completion_model)

                # Display the answer
                st.markdown("### Answer")
                st.write(completion.choices[0].message.content)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Add some helpful information at the bottom
    with st.sidebar:
        st.markdown("---")
        st.markdown("""
        ### How to use:
        1. Enter your search query in the chat box
        2. The app will search through your documents using GroundX
        3. The results will be analyzed using OpenAI's language models
        4. You'll get a detailed response based on the document contents
        """)

if __name__ == "__main__":
    main()



