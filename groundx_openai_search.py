import streamlit as st
from openai import OpenAI
from groundx import GroundX
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

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



