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
    
    # Initialize session state for messages if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
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

    # Display chat history
    st.subheader("Chat History")
    
    # Create a container for the chat history
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            # Display user message
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            
            # Display assistant message
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    # Display the answer
                    st.markdown(message["content"])
                    
                    # Display metadata in expanders
                    if "metadata" in message:
                        # Show search results
                        with st.expander("View Search Results"):
                            st.text(message["metadata"]["search_results"])
                        
                        # Show reasoning process
                        if "reasoning" in message["metadata"]:
                            with st.expander("View Reasoning Process"):
                                st.markdown(message["metadata"]["reasoning"])
                        
                        # Show search score
                        if "search_score" in message["metadata"]:
                            st.caption(f"Search Score: {message['metadata']['search_score']:.2f}")

    # Main search interface
    query = st.chat_input("Enter your search query:")

    if query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(query)
        
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
                # OpenAI analysis with chain of thought reasoning
                instruction = """You are a helpful virtual assistant that answers questions using the content below. 
                Follow these steps to provide a detailed answer:

                1. First, analyze the question:
                   - What are the key concepts and requirements?
                   - What specific information is needed?
                   - What type of answer would be most helpful?

                2. Then, analyze the provided content:
                   - What relevant information was found?
                   - How does it relate to the question?
                   - What supporting details are available?
                   - Are there any gaps in the information?

                3. Finally, synthesize the answer:
                   - How should the information be structured?
                   - What main points should be highlighted?
                   - What supporting details should be included?
                   - How can the answer be made most clear and helpful?

                Your task is to create detailed answers by combining your understanding of the world with the content provided below. 
                Do not share links. Show your reasoning process step by step."""

                completion = client.chat.completions.create(
                    model=completion_model,
                    messages=[
                        {
                            "role": "system",
                            "content": f"{instruction}\n===\n{llm_text}\n===",
                        },
                        {"role": "user", "content": query},
                    ],
                    temperature=0.7,  # Slightly higher temperature for more detailed reasoning
                )

                # Display results
                st.subheader("Analysis Results")
                
                # Display metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Search Score", f"{results.score:.2f}")
                with col2:
                    st.metric("Model Used", completion_model)

                # Display the answer with reasoning
                st.markdown("### Answer")
                
                # Split the response into reasoning and final answer
                response_text = completion.choices[0].message.content
                reasoning_parts = []
                final_answer = ""
                
                # Extract reasoning steps and final answer
                lines = response_text.split('\n')
                in_reasoning = False
                for line in lines:
                    if line.strip().startswith(('1.', '2.', '3.')) or line.strip().startswith(('- ')):
                        in_reasoning = True
                        reasoning_parts.append(line)
                    elif in_reasoning and line.strip() and not line.strip().startswith(('1.', '2.', '3.', '- ')):
                        in_reasoning = False
                        final_answer = line
                    elif not in_reasoning:
                        final_answer += '\n' + line

                # Display reasoning in an expander
                if reasoning_parts:
                    with st.expander("View Reasoning Process"):
                        st.markdown('\n'.join(reasoning_parts))

                # Display the final answer
                st.markdown(final_answer)

                # Store in chat history with metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer,
                    "metadata": {
                        "search_results": llm_text,
                        "reasoning": '\n'.join(reasoning_parts),
                        "search_score": results.score
                    }
                })
                
                # Rerun to update the chat history display
                st.rerun()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Add some helpful information at the bottom
    with st.sidebar:
        st.markdown("---")
        st.markdown("""
        ### How to use:
        1. Enter your search query in the chat box
        2. The app will search through your documents using GroundX
        3. The results will be analyzed using OpenAI's language models with chain of thought reasoning
        4. You'll see:
           - The reasoning process in an expandable section
           - A detailed answer based on the document contents
           - Search metadata and scores
        5. Your chat history will be displayed above, with all questions and answers
        """)

if __name__ == "__main__":
    main()



