import streamlit as st
from groundx import GroundX
from dotenv import load_dotenv
import os
import time
from typing import List, Dict, Any, Optional
import requests
from langchain.llms.base import BaseLLM
from langchain.callbacks.base import BaseCallbackHandler
import re

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import Field
from langchain.chains import LLMChain
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents import AgentOutputParser
from langchain.schema.output import LLMResult
from langchain.schema.messages import BaseMessage
from langchain.schema import Generation

# Load environment variables
load_dotenv()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_llm" not in st.session_state:
    st.session_state.selected_llm = "gpt-4"

if "retriever_agent" not in st.session_state:
    st.session_state.retriever_agent = None

if "synthesizer_agent" not in st.session_state:
    st.session_state.synthesizer_agent = None

# Add after existing imports
class DeepseekLLM(BaseLLM):
    model_name: str = Field(default="deepseek-r1:14b")
    temperature: float = Field(default=0.7)
    streaming: bool = Field(default=True)
    callbacks: List[BaseCallbackHandler] = Field(default_factory=list)
    base_url: str = Field(default="http://localhost:11434")
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": 0.9,
                        "top_k": 40
                    }
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            raise Exception(f"Error calling Deepseek LLM: {str(e)}")

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        """Generate text from a list of prompts."""
        generations = []
        for prompt in prompts:
            response = self._call(prompt, stop)
            generations.append([Generation(text=response)])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "streaming": self.streaming
        }

def get_llm(model_name: str, temperature: float = 0.0) -> BaseLLM:
    """Initialize the selected LLM with given parameters"""
    if model_name.startswith("deepseek"):
        return DeepseekLLM(
            model_name=model_name,
            temperature=temperature,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
    else:
        return ChatOpenAI(
            temperature=temperature,
            model_name=model_name,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )

# ===========================
#   Custom Tools
# ===========================
class DocumentSearchTool(BaseTool):
    name: str = Field(default="document_search")
    description: str = Field(default="Search through documents for relevant information. Use specific keywords and phrases from the question.")
    return_direct: bool = Field(default=False)
    client: Any = Field(default=None)
    bucket_id: int = Field(default=None)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = GroundX(api_key=os.getenv('GROUNDX_API_KEY'))
        self.bucket_id = int(os.getenv('GROUNDX_BUCKET_ID', 15153))
    
    def _run(self, query: str) -> str:
        try:
            content_response = self.client.search.content(
                id=self.bucket_id,
                query=query,
                n=100 , # Increase number of results
                verbosity=2,
                relevance=0.3  # Lower relevance threshold to get more results
            )
            results = content_response.search
            if not results.text or results.text.strip() == "":
                return "No results found for this query. Try another search with different keywords."
            return results.text
        except Exception as e:
            return f"Error searching documents: {str(e)}"

# ===========================
#   Custom Prompt Templates
# ===========================
class RetrieverPromptTemplate(BaseChatPromptTemplate):
    template: str = Field(...)
    tools: List[Tool] = Field(default_factory=list)
    input_variables: List[str] = Field(default_factory=lambda: ["input", "intermediate_steps"])
    
    def format_messages(self, **kwargs) -> List[HumanMessage]:
        intermediate_steps = kwargs.pop("intermediate_steps", [])
        thoughts = ""
        
        for action, observation in intermediate_steps:
            thoughts += f"\nAction: {action.log}\nObservation: {observation}\n"
            
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        
        messages = [
            SystemMessage(content="You are an expert information retriever. Your ONLY task is to search through documents to find relevant information."),
            HumanMessage(content=self.template.format(**kwargs))
        ]
        return messages

class SynthesizerPromptTemplate(BaseChatPromptTemplate):
    template: str = Field(...)
    input_variables: List[str] = Field(default_factory=lambda: ["question", "context"])
    
    def format_messages(self, **kwargs) -> List[HumanMessage]:
        messages = [
            SystemMessage(content="You are an expert information synthesizer. Your ONLY task is to create a response using EXACTLY the information provided, with NO additions or assumptions."),
            HumanMessage(content=self.template.format(**kwargs))
        ]
        return messages

# ===========================
#   Custom Output Parser
# ===========================
class CustomAgentOutputParser(AgentOutputParser):
    def parse(self, text: str) -> AgentAction | AgentFinish:
        # Clean the text by removing any <think> tags and extra whitespace
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # First check for Final Answer
        if "Final Answer:" in text:
            answer = text.split("Final Answer:")[-1].strip()
            return AgentFinish(
                return_values={"output": answer},
                log=text
            )
        
        # Extract Action and Action Input using more robust patterns
        action_pattern = r"Action:\s*(.*?)(?:\n|Action Input:)"
        action_input_pattern = r"Action Input:\s*(.*?)(?:\n|$)"
        
        action_match = re.search(action_pattern, text)
        action_input_match = re.search(action_input_pattern, text)
        
        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            
            # Validate that action is in the expected format
            if action == "document_search":
                return AgentAction(
                    tool=action,
                    tool_input=action_input,
                    log=text
                )
        
        # If we can't parse it as an action, treat it as a final answer
        return AgentFinish(
            return_values={"output": text.strip()},
            log=text
        )

    @property
    def _type(self) -> str:
        return "custom_agent_output_parser"

# ===========================
#   Agent Initialization
# ===========================
def init_retriever_agent(tools: List[Tool], model_name: str) -> AgentExecutor:
    llm = get_llm(model_name, temperature=0.0)  # Lower temperature for retrieval
    
    template = """You are an expert information retriever. Your ONLY task is to search through documents to find relevant information.

    AVAILABLE TOOLS:
    {tools}

    QUESTION: {input}
    
    STRICT FORMAT TO FOLLOW:
    Thought: (identify key terms and concepts to search for)
    Action: document_search
    Action Input: (create a focused search query using key terms from the question)
    Observation: (wait for result)
    Thought: (analyze if the search result is relevant; if not, try a different search approach)
    (Repeat the above steps if needed, max 3 times)
    Final Answer: (provide the relevant information found in the searches)

    CRITICAL RULES:
    1. ONLY use the document_search tool
    2. If first search returns no results, try again with different keywords
    3. Make search queries specific and focused
    4. Use exact phrases from the question when possible
    5. If multiple searches are needed, combine all relevant information found
    6. Only say "No relevant information found" after trying at least 2 different search queries
    7. Include ALL relevant information found in the Final Answer

    SEARCH TIPS:
    - Break complex questions into smaller search queries
    - Try both specific and general search terms
    - Use exact phrases in quotes for precise matching
    - Remove unnecessary words from search queries
    - If a search fails, try synonyms or related terms

    CURRENT CONVERSATION:
    {agent_scratchpad}
    """
    
    prompt = RetrieverPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["input", "intermediate_steps"]
    )
    
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    output_parser = CustomAgentOutputParser()
    
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:", "\nThought:", "\nAction:", "\nFinal Answer:"],
        allowed_tools=["document_search"]
    )
    
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=3
    )

def init_synthesizer_agent(model_name: str) -> AgentExecutor:
    llm = get_llm(model_name, temperature=0.0)  # Lower temperature to reduce creativity
    
    template = """You are an expert information synthesizer. Your ONLY task is to create a response using EXACTLY the information provided, with NO additions or assumptions.

    QUESTION: {question}
    
    RETRIEVED INFORMATION: {context}
    
    CRITICAL RULES:
    1. ONLY use information explicitly stated in the RETRIEVED INFORMATION
    2. DO NOT add any external knowledge or assumptions
    3. DO NOT make inferences beyond what is directly stated
    4. If information is missing or unclear, explicitly say so
    5. Use exact quotes where possible
    6. If the retrieved information doesn't answer the question, say "The provided information does not answer this question. Please try asking the question differently."

    STRICT FORMAT TO FOLLOW:
    Thought: (analyze what information from the retrieval is relevant to the question)
    Final Answer: (provide a response using ONLY the retrieved information, with NO external knowledge or assumptions)

    Begin your response now:"""
    
    prompt = SynthesizerPromptTemplate(
        template=template,
        input_variables=["question", "context"]
    )
    
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    output_parser = CustomAgentOutputParser()
    
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:", "\nThought:", "\nAction:", "\nFinal Answer:"],
        allowed_tools=[]
    )
    
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=[],
        verbose=True
    )

# ===========================
#   Main Functions
# ===========================
@st.cache_resource
def init_clients(model_name: str):
    doc_tool = DocumentSearchTool()
    tools = [
        Tool(
            name="document_search",
            func=doc_tool._run,
            description="Search through documents for relevant information"
        )
    ]
    retriever_agent = init_retriever_agent(tools, model_name)
    synthesizer_agent = init_synthesizer_agent(model_name)
    return retriever_agent, synthesizer_agent

def main():
    # ===========================
    #   Sidebar Configuration
    # ===========================
    with st.sidebar:
        st.header("Configuration")
        
        # Add LLM selection
        selected_llm = st.selectbox(
            "Select Language Model",
            ["gpt-4", "deepseek-r1:14b"],
            index=0,
            help="Choose between OpenAI's GPT-4 or locally hosted Deepseek models"
        )
        
        if selected_llm != st.session_state.selected_llm:
            st.session_state.selected_llm = selected_llm
            # Force reinitialization of agents when LLM changes
            if "retriever_agent" in st.session_state:
                del st.session_state.retriever_agent
            if "synthesizer_agent" in st.session_state:
                del st.session_state.synthesizer_agent
            st.rerun()

        # Add LLM status indicator for Deepseek
        if selected_llm.startswith("deepseek"):
            try:
                requests.get("http://localhost:11434/api/tags")
                st.success("✅ Ollama is running")
            except:
                st.error("❌ Ollama is not running. Please start Ollama service.")
                st.info("To start Ollama, run: `ollama serve`")
                st.info(f"To pull the model, run: `ollama pull {selected_llm}`")

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # ===========================
    #   Main Chat Interface
    # ===========================
    st.title("Agentic RAG with LangChain")
    st.write("Ask questions and get AI-powered answers using a retriever-synthesizer agent system.")

    # Initialize agents if needed
    if st.session_state.retriever_agent is None or st.session_state.synthesizer_agent is None:
        doc_tool = DocumentSearchTool()
        tools = [
            Tool(
                name="document_search",
                func=doc_tool._run,
                description="Search through documents for relevant information"
            )
        ]
        st.session_state.retriever_agent = init_retriever_agent(tools, st.session_state.selected_llm)
        st.session_state.synthesizer_agent = init_synthesizer_agent(st.session_state.selected_llm)

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
                # Step 1: Retrieve relevant information
                with st.spinner("Retrieving relevant information..."):
                    retrieval_result = st.session_state.retriever_agent.run(query)

                # Step 2: Synthesize the response
                with st.spinner("Synthesizing response..."):
                    final_response = st.session_state.synthesizer_agent.run({
                        "question": query,
                        "context": retrieval_result
                    })

                # Display response with streaming effect
                message_placeholder = st.empty()
                full_response = ""

                # Stream the response
                for chunk in final_response.split():
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
                        "search_results": retrieval_result
                    }
                })

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Add helpful information in the sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("""
        ### How to use:
        1. Select your preferred language model
        2. Type your question in the chat box
        3. The system will:
           - Search for relevant information
           - Synthesize a comprehensive answer
        4. View search results in expandable sections
        5. Continue the conversation with follow-up questions
        """)

if __name__ == "__main__":
    main()