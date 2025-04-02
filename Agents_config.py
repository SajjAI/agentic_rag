from crewai import Agent, Crew, Process, Task, LLM
import streamlit as st


import os
@st.cache_resource
def load_llm():
    # Load configuration from environment variables with defaults
    model_name = os.getenv('LLM_MODEL', 'ollama/deepseek-r1:14b')
    base_url = os.getenv('LLM_BASE_URL', 'http://localhost:11434')

    llm = LLM(
        model=model_name,
        base_url=base_url
    )
    return llm

# ===========================
#   Define Agents & Tasks
# ===========================
# def create_agents_and_tasks(pdf_tool):
#     """Creates a Crew with the given PDF tool (if any) and a web search tool."""
 
#     retriever_agent = Agent(
#         role="Retrieve relevant information to answer the user query: {query}",
#         goal=(
#             "Retrieve the most relevant information from the available sources "
#             "for the user query: {query}. Always try to use the PDF search tool first. "
#             "If you are not able to retrieve the information from the PDF search tool, "
#             "then try to use the web search tool."
#         ),
#         backstory=(
#             "You're a meticulous analyst with a keen eye for detail. "
#             "You're known for your ability to understand user queries: {query} "
#             "and retrieve knowledge from the most suitable knowledge base."
#         ),
#         verbose=True,
#         tools=[t for t in [pdf_tool] if t],
#         llm=load_llm(),
#         allow_delegation=False,
#         # Add structured format instructions
#         instructions=(
#             "Always structure your responses in this format:\n"
#             "Thought: Think about what information you need to find\n"
#             "Action: Choose the tool to use (e.g., pdf_search)\n"
#             "Action Input: Specify your search query\n"
#             "Observation: Review the search results\n"
#             "Thought: Analyze the results\n"
#             "Final Answer: Provide the retrieved information"
#         )
#     )

#     response_synthesizer_agent = Agent(
#         role="Response synthesizer agent for the user query: {query}",
#         goal=(
#             "Synthesize the retrieved information into a concise and coherent response "
#             "based on the user query: {query}. If you are not able to retrieve the "
#             'information then respond with "I\'m sorry, I couldn\'t find the information '
#             'you\'re looking for."'
#         ),
#         backstory=(
#             "You're a skilled communicator who excels at synthesizing "
#             "complex information into clear responses."
#         ),
#         verbose=True,
#         llm=load_llm(),
#         allow_delegation=False,
#         # Add structured format instructions
#         instructions=(
#             "Always structure your responses in this format:\n"
#             "Thought: Analyze the information provided\n"
#             "Final Answer: Provide the synthesized response\n\n"
#             "If no relevant information is found, respond exactly as follows:\n"
#             '"I\'m sorry, I couldn\'t find the information you\'re looking for."'
#         )
#     )

#     retrieval_task = Task(
#         description=(
#             "Search and retrieve relevant information for the query : {query}"
#             "Always use the following format:\n"
#             "Thought: [Your reasoning]\n"
#             "Action: [Tool name]\n"
#             "Action Input: [Search query]\n"
#             "Observation: [Results]\n"
#             "Final Answer: [Retrieved information]"
#         ),
#         expected_output=(
#             "The most relevant information in the form of text as retrieved "
#             "from the sources."
#         ),
#         agent=retriever_agent
#     )

#     response_task = Task(
#         description=(
#             "Create a final response based on the retrieved information. "
#             "Use this format:\n"
#             "Thought: [Your analysis]\n"
#             "Final Answer: [Your response]"
#         ),
#         expected_output= "A concise and coherent response based on the retrieved information "
#             "from the right source for the user query: {query}. If you are not "
#             "able to retrieve the information, then respond with: "
#             '"I\'m sorry, I couldn\'t find the information you\'re looking for."',
#         agent=response_synthesizer_agent
#     )

#     crew = Crew(
#         agents=[retriever_agent, response_synthesizer_agent],
#         tasks=[retrieval_task, response_task],
#         process=Process.sequential,  # or Process.hierarchical
#         verbose=True
#     )
#     return crew


# ===========================
#   Define Agents & Tasks suggedted by open ai 
# ===========================

# def create_agents_and_tasks(pdf_tool):
#     """Creates a Crew with the given PDF tool (if any) and a web search tool."""

#     retriever_agent = Agent(
#         role="Retrieve relevant information to answer the user query: {query}",
#         goal=(
#             "Retrieve the most relevant information from the available sources "
#             "for the user query: {query}. Always try to use the PDF search tool first. "
#             "If you are not able to retrieve the information from the PDF search tool, "
#             "then try to use the web search tool."
#         ),
#         backstory=(
#             "You're a meticulous analyst with a keen eye for detail. "
#             "You're known for your ability to understand user queries: {query} "
#             "and retrieve knowledge from the most suitable knowledge base."
#         ),
#         verbose=True,
#         tools=[t for t in [pdf_tool] if t],
#         llm=load_llm(),
#         instructions= (
#             "Always structure your responses in this format:\n"
#             "```\n"
#             "Thought: What information do I need to find?\n"
#             "Action: pdf_search\n"
#             "Action Input: {query}\n"
#             "Observation: [Results]\n"
#             "Thought: Analyze the results\n"
#             "Final Answer: Provide the retrieved information\n"
#             "```\n\n"
#             "If no information is found in the PDF search tool, try using the web search tool in the same format."
#         )
#     )

#     response_synthesizer_agent = Agent(
#         role="Response synthesizer agent for the user query: {query}",
#         goal=(
#             "Synthesize the retrieved information into a concise and coherent response "
#             "based on the user query: {query}. If you are not able to retrieve the "
#             'information then respond with "I\'m sorry, I couldn\'t find the information '
#             'you\'re looking for."'
#         ),
#         backstory=(
#             "You're a skilled communicator with a knack for turning "
#             "complex information into clear and concise responses."
#         ),
#         verbose=True,
#         llm=load_llm()
#     )

#     retrieval_task = Task(
#         description=(
#             "Retrieve the most relevant information from the available "
#             "sources for the user query: {query}"
#         ),
#         expected_output=(
#             "The most relevant information in the form of text as retrieved "
#             "from the sources."
#         ),
#         agent=retriever_agent
#     )

#     response_task = Task(
#         description="Synthesize the final response for the user query: {query}",
#         expected_output=(
#             "A concise and coherent response based on the retrieved information "
#             "from the right source for the user query: {query}. If you are not "
#             "able to retrieve the information, then respond with: "
#             '"I\'m sorry, I couldn\'t find the information you\'re looking for."'
#         ),
#         agent=response_synthesizer_agent
#     )

#     crew = Crew(
#         agents=[retriever_agent, response_synthesizer_agent],
#         tasks=[retrieval_task, response_task],
#         process=Process.sequential,  # or Process.hierarchical
#         verbose=True
#     )
#     return crew


# ===========================
#   Define Agents & Tasks suggedted by claude ai for single docuemnt
# ===========================
# def create_agents_and_tasks(pdf_tool):
#     retriever_agent = Agent(
#         role="Information Retrieval Specialist",
#         goal=(
#             "Extract precise and relevant information from documents to answer the query: {query}. "
#             "Use a combination of semantic search and keyword matching to find the most relevant passages."
#         ),
#         backstory=(
#             "You are an expert at information retrieval with deep experience in document analysis. "
#             "You understand context and can identify both explicit and implicit relevant information."
#         ),
#         verbose=True,
#         tools=[t for t in [pdf_tool] if t],
#         llm=load_llm(),
#         instructions=(
#             "Follow this systematic process:\n"
#             "1. Break down the query into key concepts and search terms\n"
#             "2. Search using multiple variations of the query\n"
#             "3. Look for both direct answers and supporting context\n"
#             "4. Validate the relevance of retrieved information\n\n"
#             "Structure your response as:\n"
#             "Thought: Analyze what specific information needs to be found\n"
#             "Action: pdf_search\n"
#             "Action Input: [Carefully crafted search query]\n"
#             "Observation: [Search results]\n"
#             "Thought: Evaluate relevance and need for additional searches\n"
#             "Final Answer: Most relevant information with context\n"
#         )
#     )

#     # 2. Enhanced Response Synthesizer Agent
#     response_synthesizer_agent = Agent(
#         role="Knowledge Integration Expert",
#         goal=(
#             "Create comprehensive, accurate, and well-structured responses by synthesizing "
#             "retrieved information. Ensure responses are directly relevant to the query: {query}"
#         ),
#         backstory=(
#             "You are a skilled information architect who excels at organizing complex "
#             "information into clear, logical, and insightful responses."
#         ),
#         verbose=True,
#         llm=load_llm(),
#         instructions=(
#             "1. Analyze all retrieved information\n"
#             "2. Identify key themes and relationships\n"
#             "3. Structure response with clear sections\n"
#             "4. Include specific evidence and citations\n"
#             "5. Provide context when needed\n"
#             "6. Ensure direct alignment with query\n\n"
#             "Format response with:\n"
#             "- Clear headings for different aspects\n"
#             "- Direct citations from source\n"
#             "- Logical flow of information\n"
#             "- Concrete examples where relevant"
#         )
#     )

#     # 3. Enhanced Tasks with Better Coordination
#     retrieval_task = Task(
#         description=(
#             "Conduct a thorough information retrieval process:\n"
#             "1. Parse query for key concepts\n"
#             "2. Perform multiple targeted searches\n"
#             "3. Evaluate and rank results\n"
#             "4. Collect supporting context"
#         ),
#         expected_output=(
#             "A comprehensive set of relevant information including:\n"
#             "- Direct answers to the query\n"
#             "- Supporting context and evidence\n"
#             "- Source locations for verification\n"
#             "- Confidence levels for retrieved information"
#         ),
#         agent=retriever_agent
#     )

#     response_task = Task(
#         description=(
#             "Create a well-structured response that:\n"
#             "1. Directly addresses the query\n"
#             "2. Integrates all relevant information\n"
#             "3. Provides clear reasoning and evidence\n"
#             "4. Maintains logical flow and readability"
#         ),
#         expected_output=(
#             "A clear, comprehensive response that:\n"
#             "- Directly answers the query\n"
#             "- Includes supporting evidence\n"
#             "- Maintains logical structure\n"
#             "- Cites sources appropriately"
#         ),
#         agent=response_synthesizer_agent
#     )

#     # 4. Enhanced Crew Configuration
#     crew = Crew(
#         agents=[retriever_agent, response_synthesizer_agent],
#         tasks=[retrieval_task, response_task],
#         process=Process.sequential,
#         verbose=True,
#         max_recursion=3  # Allow for multiple rounds if needed
#     )
#     return crew



# ===========================
#   Define Agents & Tasks suggedted by claude ai for multiple docuemnt
# ===========================
# def create_agents_and_tasks(pdf_tool):
#     # Enhanced retriever agent for multi-document search
#     retriever_agent = Agent(
#         role="Cross-Document Research Specialist",
#         goal=(
#             "Find and correlate information across multiple documents to comprehensively "
#             "answer the query: {query}"
#         ),
#         backstory=(
#             "You are an expert researcher skilled at finding and connecting information "
#             "across multiple documents. You know how to identify relevant pieces of "
#             "information and combine them meaningfully."
#         ),
#         verbose=True,
#         tools=[t for t in [pdf_tool] if t],
#         llm=load_llm(),
#         instructions=(
#             "Follow this process for multi-document search:\n\n"
#             "1. Initial Search:\n"
#             "   - Search across all documents with key terms\n"
#             "   - Note which documents contain relevant information\n\n"
#             "2. Deep Dive:\n"
#             "   - For each relevant document, perform focused searches\n"
#             "   - Track document sources for each piece of information\n\n"
#             "3. Cross-Reference:\n"
#             "   - Look for complementary information across documents\n"
#             "   - Identify any contradictions or confirmations\n\n"
#             "Response format:\n"
#             "Thought: What information am I looking for across documents?\n"
#             "Action: pdf_search\n"
#             "Action Input: [Search terms]\n"
#             "Observation: [Results with document sources]\n"
#             "Thought: What other documents might have related information?\n"
#             "Action: pdf_search\n"
#             "Action Input: [Refined search terms]\n"
#             "Observation: [Additional results]\n"
#             "Final Answer: [\n"
#             "  - Information found with document sources\n"
#             "  - Connections between documents\n"
#             "  - Confidence level in findings\n"
#             "]\n\n"
#             "Important:\n"
#             "- Always specify which document each piece of information comes from\n"
#             "- Look for information that might be split across documents\n"
#             "- Consider different terminology across documents"
#         )
#     )

#     # Enhanced synthesizer for multi-document information
#     response_synthesizer_agent = Agent(
#         role="Information Integration Specialist",
#         goal=(
#             "Create a comprehensive response that synthesizes information from multiple "
#             "documents while maintaining clarity and accuracy for query: {query}"
#         ),
#         backstory=(
#             "You are an expert at combining and presenting information from multiple "
#             "sources in a clear and coherent way. You know how to highlight connections "
#             "and resolve conflicts between different sources."
#         ),
#         verbose=True,
#         llm=load_llm(),
#         instructions=(
#             "Follow these steps for multi-document synthesis:\n\n"
#             "1. Information Organization:\n"
#             "   - Group related information from different documents\n"
#             "   - Note source documents for each piece of information\n"
#             "   - Identify any gaps or conflicts\n\n"
#             "2. Response Construction:\n"
#             "   - Start with the most relevant information\n"
#             "   - Clearly indicate when combining information from multiple sources\n"
#             "   - Highlight any corroborating evidence across documents\n"
#             "   - Address any contradictions found\n\n"
#             "3. Source Attribution:\n"
#             "   - Cite specific documents for key information\n"
#             "   - Indicate when information spans multiple sources\n\n"
#             "Response format:\n"
#             "Summary: [Brief overview of findings]\n"
#             "Details: [\n"
#             "  - Main points with document sources\n"
#             "  - Supporting information from other documents\n"
#             "  - Any notable gaps or uncertainties\n"
#             "]\n"
#             "Sources: [List of documents referenced]"
#         )
#     )

#     # Enhanced tasks for multi-document processing
#     retrieval_task = Task(
#         description=(
#             "Search across all documents for information related to: {query}\n"
#             "Identify connections and relationships between information in different documents."
#         ),
#         expected_output=(
#             "A comprehensive collection of relevant information including:\n"
#             "- Primary findings with document sources\n"
#             "- Related information from other documents\n"
#             "- Cross-document connections and confirmations\n"
#             "- Clear source attribution for all information"
#         ),
#         agent=retriever_agent
#     )

#     response_task = Task(
#         description=(
#             "Create a unified response that effectively combines information from all "
#             "relevant documents for query: {query}"
#         ),
#         expected_output=(
#             "A clear, well-structured response that:\n"
#             "- Synthesizes information across documents\n"
#             "- Maintains clear source attribution\n"
#             "- Highlights cross-document connections\n"
#             "- Addresses any conflicts or gaps"
#         ),
#         agent=response_synthesizer_agent
#     )

#     crew = Crew(
#         agents=[retriever_agent, response_synthesizer_agent],
#         tasks=[retrieval_task, response_task],
#         process=Process.sequential,
#         verbose=True
#     )
#     return crew





# ===========================
#   Define Agents & Tasks suggedted by claude ai for multiple docuemnt bounded as well 
# ===========================



# def create_agents_and_tasks(pdf_tool):
#     # Cross-document research specialist for finding connections
#     cross_doc_agent = Agent(
#         role="Cross-Document Research Specialist",
#         goal=(
#             "Find and correlate information across multiple documents while ensuring "
#             "all connections are explicitly supported by document content for query: {query}"
#         ),
#         backstory=(
#             "You are an expert researcher skilled at finding and connecting information "
#             "across multiple documents, while maintaining strict verification of all "
#             "connections through explicit document evidence."
#         ),
#         verbose=True,
#         tools=[t for t in [pdf_tool] if t],
#         llm=load_llm(),
#         instructions=(
#             "Follow this cross-document analysis process:\n\n"
#             "1. Initial Cross-Reference:\n"
#             "   - Identify common topics across documents\n"
#             "   - Track explicit connections and overlapping information\n"
#             "   - Note document-specific terminology variations\n\n"
#             "2. Connection Verification:\n"
#             "   - Verify each connection with direct quotes\n"
#             "   - Document exact locations of connected information\n"
#             "   - Flag potential connections needing verification\n\n"
#             "3. Pattern Recognition:\n"
#             "   - Identify recurring themes across documents\n"
#             "   - Map related information with source tracking\n"
#             "   - Note strength of connections based on evidence\n\n"
#             "Response format:\n"
#             "Cross-Document Findings: [\n"
#             "  - Connected information with source quotes\n"
#             "  - Document-specific terminology mappings\n"
#             "  - Strength of connection assessment\n"
#             "]\n"
#             "Connection Verification: [Evidence for each connection]\n"
#             "Pattern Analysis: [Documented themes with sources]"
#         )
#     )

#     # # Document-bounded specialist for strict content adherence
#     # bounded_doc_agent = Agent(
#     #     role="Document-Bounded Research Specialist",
#     #     goal=(
#     #         "Extract and verify information strictly from provided documents for query: {query}, "
#     #         "ensuring no external information is introduced"
#     #     ),
#     #     backstory=(
#     #         "You are a precise researcher who works exclusively with provided documents, "
#     #         "ensuring strict adherence to document content and explicit source verification."
#     #     ),
#     #     verbose=True,
#     #     tools=[t for t in [pdf_tool] if t],
#     #     llm=load_llm(),
#     #     instructions=(
#     #         "Follow this strict verification process:\n\n"
#     #         "1. Content Extraction:\n"
#     #         "   - Extract exact quotes from documents\n"
#     #         "   - Maintain precise source tracking\n"
#     #         "   - Flag unclear or ambiguous content\n\n"
#     #         "2. Verification:\n"
#     #         "   - Double-check each piece of information\n"
#     #         "   - Verify exact document locations\n"
#     #         "   - Note explicit knowledge gaps\n\n"
#     #         "3. Documentation:\n"
#     #         "   - Record all source references\n"
#     #         "   - Mark confidence levels\n"
#     #         "   - List missing information\n\n"
#     #         "Response format:\n"
#     #         "Verified Content: [\n"
#     #         "  - Direct quotes with sources\n"
#     #         "  - Confidence assessments\n"
#     #         "  - Knowledge gaps\n"
#     #         "]\n"
#     #         "Source Tracking: [Detailed reference list]"
#     #     )
#     # )

#     # Enhanced synthesizer for integrating both specialists' findings
#     response_synthesizer_agent = Agent(
#         role="Document-Bounded Integration Specialist",
#         goal=(
#             "Create a comprehensive response integrating cross-document connections and "
#             "verified content while maintaining strict document adherence for query: {query}"
#         ),
#         backstory=(
#             "You are an expert at combining verified information and cross-document "
#             "connections while maintaining strict boundaries against external knowledge."
#         ),
#         verbose=True,
#         llm=load_llm(),
#         instructions=(
#             "Follow these integration guidelines:\n\n"
#             "1. Information Integration:\n"
#             "   - Combine verified content and cross-document connections\n"
#             "   - Maintain clear source tracking\n"
#             "   - Separate facts from document-based inferences\n\n"
#             "2. Connection Validation:\n"
#             "   - Verify all cross-document connections\n"
#             "   - Rate connection strength with evidence\n"
#             "   - Mark uncertain connections\n\n"
#             "3. Response Assembly:\n"
#             "   - Present verified content first\n"
#             "   - Add validated connections\n"
#             "   - Note knowledge gaps\n\n"
#             "Response format:\n"
#             "Verified Findings: [\n"
#             "  - Document-specific content\n"
#             "  - Cross-document connections\n"
#             "  - Evidence strength assessment\n"
#             "]\n"
#             "Knowledge Gaps: [Documented limitations]\n"
#             "Sources: [Complete reference list]"
#         )
#     )

#     # Tasks for each specialist and synthesis
#     cross_doc_task = Task(
#         description=(
#             "Identify and verify connections across documents for: {query}\n"
#             "Ensure all connections have explicit document support."
#         ),
#         expected_output=(
#             "A verified set of cross-document connections including:\n"
#             "- Connected information with sources\n"
#             "- Connection strength assessment\n"
#             "- Pattern analysis with evidence"
#         ),
#         agent=cross_doc_agent
#     )

#     # bounded_doc_task = Task(
#     #     description=(
#     #         "Extract and verify information strictly from documents for: {query}\n"
#     #         "Maintain strict document adherence."
#     #     ),
#     #     expected_output=(
#     #         "A verified collection of document content including:\n"
#     #         "- Direct quotes with sources\n"
#     #         "- Confidence assessments\n"
#     #         "- Knowledge gaps"
#     #     ),
#     #     agent=bounded_doc_agent
#     # )

#     synthesis_task = Task(
#         description=(
#             "Create an integrated response combining verified content and connections "
#             "for query: {query}"
#         ),
#         expected_output=(
#             "A comprehensive response that:\n"
#             "- Combines verified content and connections\n"
#             "- Maintains source attribution\n"
#             "- Assesses evidence strength\n"
#             "- Notes knowledge limitations"
#         ),
#         agent=response_synthesizer_agent
#     )

#     crew = Crew(
#         # agents=[cross_doc_agent, bounded_doc_agent, response_synthesizer_agent],
#         # tasks=[cross_doc_task, bounded_doc_task, synthesis_task],
#         agents=[cross_doc_agent,  response_synthesizer_agent],
#         tasks=[cross_doc_task, synthesis_task],
#         process=Process.sequential,
#         verbose=True
#     )
#     return crew











# ===========================
#   Define Agents & Tasks suggedted by claude ai for multiple docuemnt 
# ===========================



# def create_agents_and_tasks(pdf_tool):
#     # Enhanced cross-document research specialist with strict verification
#     retriever_agent = Agent(
#         role="Cross-Document Research Specialist",
#         goal=(
#             "Find, verify, and correlate information across documents while ensuring "
#             "all findings are explicitly supported by document content for query: {query}"
#         ),
#         backstory=(
#             "You are an expert researcher skilled at finding and connecting information "
#             "across documents, maintaining strict verification and source tracking for "
#             "all findings and connections."
#         ),
#         verbose=True,
#         tools=[t for t in [pdf_tool] if t],
#         llm=load_llm(),
#         instructions=(
#             "Follow this strict cross-document research process:\n\n"
#             "1. Initial Search:\n"
#             "   - Search using exact query terms\n"
#             "   - Track which documents contain information\n"
#             "   - Note exact quotes and locations\n\n"
#             "2. Cross-Reference:\n"
#             "   - Identify overlapping information\n"
#             "   - Map related concepts across documents\n"
#             "   - Verify connections with direct quotes\n\n"
#             "3. Verification:\n"
#             "   - Double-check all findings in source documents\n"
#             "   - Rate evidence strength for each connection\n"
#             "   - Note any gaps or uncertainties\n\n"
#             "Response format:\n"
#             "Thought: What information am I searching for?\n"
#             "Action: pdf_search\n"
#             "Action Input: [Exact search terms]\n"
#             "Observation: [Results with document locations]\n"
#             "Verification: [\n"
#             "  - Direct quotes supporting findings\n"
#             "  - Cross-document connections with evidence\n"
#             "  - Information gaps identified\n"
#             "]\n"
#             "Final Answer: [\n"
#             "  - Verified findings with sources\n"
#             "  - Connection strength assessment\n"
#             "  - Confidence levels based on evidence\n"
#             "]\n\n"
#             "Important Rules:\n"
#             "- Only report information explicitly in documents\n"
#             "- Support all connections with direct quotes\n"
#             "- Clearly state when information is missing\n"
#             "- Rate confidence based on document evidence"
#         )
#     )

#     # Enhanced synthesizer for verified information
#     response_synthesizer_agent = Agent(
#         role="Information Integration Specialist",
#         goal=(
#             "Create a comprehensive response that synthesizes verified information and "
#             "connections across documents for query: {query}"
#         ),
#         backstory=(
#             "You are an expert at combining verified information from multiple sources, "
#             "ensuring all conclusions are supported by explicit document evidence."
#         ),
#         verbose=True,
#         llm=load_llm(),
#         instructions=(
#             "Follow these synthesis guidelines:\n\n"
#             "1. Information Organization:\n"
#             "   - Group related findings across documents\n"
#             "   - Maintain clear source tracking\n"
#             "   - Rate evidence strength\n\n"
#             "2. Connection Integration:\n"
#             "   - Present strongest connections first\n"
#             "   - Support each connection with quotes\n"
#             "   - Note evidence quality\n\n"
#             "3. Response Construction:\n"
#             "   - Start with best-supported findings\n"
#             "   - Include direct document quotes\n"
#             "   - Note confidence levels\n"
#             "   - Acknowledge information gaps\n\n"
#             "Response format:\n"
#             "Key Findings: [\n"
#             "  - Primary information with sources\n"
#             "  - Supporting quotes from documents\n"
#             "  - Cross-document connections\n"
#             "]\n"
#             "Evidence Assessment: [\n"
#             "  - Connection strength ratings\n"
#             "  - Confidence levels\n"
#             "  - Information gaps\n"
#             "]\n"
#             "Sources: [Document references]\n\n"
#             "Important Rules:\n"
#             "- Use only verified information\n"
#             "- Support all claims with quotes\n"
#             "- Be explicit about uncertainties\n"
#             "- Maintain clear source tracking"
#         )
#     )

#     # Enhanced tasks for cross-document research
#     retrieval_task = Task(
#         description=(
#             "Search and verify information across documents for: {query}\n"
#             "Identify and validate all cross-document connections."
#         ),
#         expected_output=(
#             "A verified set of findings including:\n"
#             "- Primary information with sources\n"
#             "- Cross-document connections with evidence\n"
#             "- Confidence assessments\n"
#             "- Knowledge gaps"
#         ),
#         agent=retriever_agent
#     )

#     response_task = Task(
#         description=(
#             "Create an integrated response combining verified findings and connections "
#             "for query: {query}"
#         ),
#         expected_output=(
#             "A comprehensive response that:\n"
#             "- Presents verified information\n"
#             "- Supports connections with evidence\n"
#             "- Rates confidence levels\n"
#             "- Notes limitations"
#         ),
#         agent=response_synthesizer_agent
#     )

#     crew = Crew(
#         agents=[retriever_agent, response_synthesizer_agent],
#         tasks=[retrieval_task, response_task],
#         process=Process.sequential,
#         verbose=True
#     )
#     return crew






#currently working on this create agents and tasks 15 feb @6:44pm


# def create_agents_and_tasks(pdf_tool):
#     # Cross-document research specialist with strict context boundaries
#     retriever_agent = Agent(
#         role="Cross-Document Research Specialist",
#         goal=(
#             "Find and correlate information ONLY from the retrieved search results, "
#             "without using any external knowledge for query: {query}"
#         ),
#         backstory=(
#             "You are a research specialist who works exclusively with retrieved search "
#             "results. You never use external knowledge and only report information "
#             "that appears in the current search context."
#         ),
#         verbose=True,
#         tools=[t for t in [pdf_tool] if t],
#         llm=load_llm(),
#         instructions=(
#             "Follow these strict context-bounded guidelines:\n\n"
#             "1. Context Rules:\n"
#             "   - ONLY use information from current search results\n"
#             "   - NEVER use external knowledge or assumptions\n"
#             "   - If information isn't in search results, explicitly state it's not found\n\n"
#             "2. Search Process:\n"
#             "   - Search within provided documents\n"
#             "   - Use only returned search results\n"
#             "   - Ignore any external knowledge\n\n"
#             "3. Verification:\n"
#             "   - Verify all information exists in current search results\n"
#             "   - Include exact quotes from search context\n"
#             "   - State when information is not in search results\n\n"
#             "Response format:\n"
#             "Thought: What am I searching for in the current context?\n"
#             "Action: pdf_search\n"
#             "Action Input: [Search terms]\n"
#             "Observation: [Search results]\n"
#             "Context Check: [\n"
#             "  - List information found in search results\n"
#             "  - List information NOT found in search results\n"
#             "]\n"
#             "Final Answer: [\n"
#             "  - ONLY information from search results\n"
#             "  - Direct quotes as evidence\n"
#             "  - Clear statements about missing information\n"
#             "]\n\n"
#             "Critical Rules:\n"
#             "- NEVER use knowledge outside current search results\n"
#             "- ALWAYS quote directly from search results\n"
#             "- Explicitly state 'Not found in search results' when applicable\n"
#             "- Do not make assumptions beyond search context"
#         )
#     )

#     # Context-bounded synthesizer
#     response_synthesizer_agent = Agent(
#         role="Context-Bounded Integration Specialist",
#         goal=(
#             "Create a response using ONLY information from the retriever's search results "
#             "for query: {query}, with no external knowledge"
#         ),
#         backstory=(
#             "You are a specialist who works exclusively with provided search results, "
#             "never adding external knowledge or making assumptions beyond the given context."
#         ),
#         verbose=True,
#         llm=load_llm(),
#         instructions=(
#             "Follow these strict context-bounded synthesis rules:\n\n"
#             "1. Source Restriction:\n"
#             "   - ONLY use information from retriever's results\n"
#             "   - NEVER add external knowledge\n"
#             "   - Keep track of all source quotes\n\n"
#             "2. Response Building:\n"
#             "   - Use only verified search results\n"
#             "   - Include direct quotes as evidence\n"
#             "   - Clearly mark information gaps\n\n"
#             "3. Quality Control:\n"
#             "   - Verify everything against search results\n"
#             "   - State when information is missing\n"
#             "   - No assumptions beyond context\n\n"
#             "Response format:\n"
#             "Context Verification: [\n"
#             "  - Information found in results\n"
#             "  - Information not in results\n"
#             "]\n"
#             "Response: [\n"
#             "  - Only verified information\n"
#             "  - Direct quotes as evidence\n"
#             "  - Clear gaps in context\n"
#             "]\n\n"
#             "Critical Rules:\n"
#             "- ONLY use information from current search results\n"
#             "- NEVER add external knowledge\n"
#             "- Always quote evidence\n"
#             "- Clearly state what's not in search results"
#         )
#     )

#     # Context-bounded tasks
#     retrieval_task = Task(
#         description=(
#             "Search documents and return ONLY information found in search results "
#             "for: {query}. Do not use any external knowledge."
#         ),
#         expected_output=(
#             "A context-bounded result including:\n"
#             "- Only information from search results\n"
#             "- Direct quotes as evidence\n"
#             "- Clear statements about missing information"
#         ),
#         agent=retriever_agent
#     )

#     response_task = Task(
#         description=(
#             "Create a response using ONLY information from retriever's results "
#             "for query: {query}. No external knowledge allowed."
#         ),
#         expected_output=(
#             "A context-bounded response that:\n"
#             "- Uses only retrieved information\n"
#             "- Includes source quotes\n"
#             "- Clearly states what's not in context"
#         ),
#         agent=response_synthesizer_agent
#     )

#     crew = Crew(
#         agents=[retriever_agent, response_synthesizer_agent],
#         tasks=[retrieval_task, response_task],
#         process=Process.sequential,
#         verbose=True
#     )
#     return crew





# ===========================
#   Define Agents & Tasks
# ===========================
def create_agents_and_tasks(pdf_tool):
    """Creates a Crew with agents specialized in handling multiple documents."""
 
    retriever_agent = Agent(
        role="Information Retriever",
        goal=(
            "Search and extract relevant information from multiple documents to answer "
            "the user query. Ensure comprehensive coverage across all documents."
        ),
        backstory=(
            "You are an expert researcher skilled at finding relevant information "
            "across multiple documents. You know how to search effectively and "
            "combine information from different sources."
        ),
        verbose=True,
        tools=[t for t in [pdf_tool] if t],
        llm=load_llm(),
        instructions=(
            "Follow these steps when searching:\n\n"
            "1. Break down the query into key search terms\n"
            "2. Search using the pdf_search tool\n"
            "3. Review results and identify relevant information\n"
            "4. If needed, perform follow-up searches with refined terms\n\n"
            "Always format your response as:\n"
            "Thought: [Your reasoning about what to search]\n"
            "Action: pdf_search\n"
            "Action Input: [Your search query]\n"
            "Observation: [Search results]\n"
            "Thought: [Analysis of results, decide if more searches needed]\n"
            "Final Answer: [Relevant information found, with source details]\n\n"
            "Important:\n"
            "- Include source details (document name, page) when available\n"
            "- Perform multiple searches if needed\n"
            "- Clearly state if information is not found"
        )
    )

    response_synthesizer_agent = Agent(
        role="Response Synthesizer",
        goal=(
            "Create clear, coherent responses by combining and organizing information "
            "from multiple documents. Ensure accuracy and proper source attribution."
        ),
        backstory=(
            "You are a skilled communicator who excels at organizing and presenting "
            "information from multiple sources in a clear, logical manner."
        ),
        verbose=True,
        llm=load_llm(),
        instructions=(
            "Follow these steps to create responses:\n\n"
            "1. Review all retrieved information\n"
            "2. Cross-Reference:\n"
            "   - Identify overlapping information\n"
            "   - Map related concepts across documents\n"
            "3. Organize related information together\n"
            "4. Create a clear, structured response\n\n"
            "Format your response as:\n"
            "Thought: [Analysis of the information]\n"
            "Final Answer: [Your structured response]\n\n"
            "In your final answer:\n"
            "- Start with a clear summary\n"
            "- Include relevant details from each source\n"
            "- Maintain source attribution\n"
            "- Use bullet points for multiple pieces of information\n"
            "- Clearly state if information is incomplete\n"
            "- If no relevant information is found, say so clearly"
        )
    )

    retrieval_task = Task(
        description=(
            "Search through all available documents to find relevant information for: {query}\n"
            "Perform multiple searches if needed to ensure comprehensive coverage."
        ),
        expected_output=(
            "Relevant information from all documents, including:\n"
            "- Direct answers to the query\n"
            "- Supporting context\n"
            "- Source attribution"
        ),
        agent=retriever_agent
    )

    response_task = Task(
        description=(
            "Create a comprehensive response that combines all relevant information found "
            "for: {query}"
        ),
        expected_output=(
            "A clear, well-structured response that:\n"
            "- Directly answers the query\n"
            "- Includes relevant details from all sources\n"
            "- Maintains clear organization\n"
            "- Indicates source documents"
        ),
        agent=response_synthesizer_agent
    )

    crew = Crew(
        agents=[retriever_agent, response_synthesizer_agent],
        tasks=[retrieval_task, response_task],
        process=Process.sequential,  # or Process.hierarchical
        verbose=True
    )
    return crew























            # "Follow this strict cross-document research process:\n\n"
            # "1. Initial Search:\n"
            # "   - Search using exact query terms\n"
            # "   - Track which documents contain information\n"
            # "   - Note exact quotes and locations\n\n"
            # "2. Cross-Reference:\n"
            # "   - Identify overlapping information\n"
            # "   - Map related concepts across documents\n"


# def create_agents_and_tasks(pdf_tool):
#     retriever_agent = Agent(
#         role="Cross-Document Research Specialist",
#         goal=(
#             "Find and synthesize information across multiple documents, ensuring comprehensive "
#             "coverage while maintaining relevance to the query: {query}"
#         ),
#         backstory=(
#             "You are an expert researcher skilled at finding and connecting information "
#             "across multiple documents. You understand how to evaluate source credibility "
#             "and combine information from different sources effectively."
#         ),
#         verbose=True,
#         tools=[t for t in [pdf_tool] if t],
#         llm=load_llm(),
#         instructions=(
#             "Follow this systematic search process:\n\n"
#             "1. Initial Search:\n"
#             "   - Break down the query into key concepts\n"
#             "   - Search using multiple variations of terms\n"
#             "   - Track which documents contain relevant information\n\n"
#             "2. Deep Analysis:\n"
#             "   - Evaluate relevance of each result\n"
#             "   - Note contradictions or confirmations across documents\n"
#             "   - Consider context and source credibility\n\n"
#             "3. Information Synthesis:\n"
#             "   - Combine related information from different sources\n"
#             "   - Maintain clear source attribution\n"
#             "   - Note confidence levels based on evidence\n\n"
#             "Response format:\n"
#             "Thought: [Analysis of search needs]\n"
#             "Action: pdf_search\n"
#             "Action Input: [Carefully crafted search query]\n"
#             "Observation: [Evaluation of results]\n"
#             "Thought: [Analysis of findings]\n"
#             "Final Answer: [\n"
#             "  - Key findings with source attribution\n"
#             "  - Confidence levels\n"
#             "  - Related information across documents\n"
#             "]\n"
#         )
#     )

#     response_synthesizer_agent = Agent(
#         role="Multi-Source Integration Expert",
#         goal=(
#             "Create a coherent and comprehensive response by synthesizing information "
#             "from multiple documents while maintaining accuracy and context"
#         ),
#         backstory=(
#             "You are an expert at combining information from multiple sources into "
#             "clear, well-structured responses that preserve context and source attribution."
#         ),
#         verbose=True,
#         llm=load_llm(),
#         instructions=(
#             "Follow these synthesis guidelines:\n\n"
#             "1. Information Organization:\n"
#             "   - Group related findings across documents\n"
#             "   - Identify key themes and patterns\n"
#             "   - Note source agreement/disagreement\n\n"
#             "2. Response Construction:\n"
#             "   - Start with strongest supported findings\n"
#             "   - Integrate supporting information\n"
#             "   - Maintain clear source attribution\n"
#             "   - Address any contradictions\n\n"
#             "3. Quality Control:\n"
#             "   - Verify information consistency\n"
#             "   - Check source attribution\n"
#             "   - Ensure logical flow\n\n"
#             "Response format:\n"
#             "Thought: [Analysis of information to synthesize]\n"
#             "Final Answer: [\n"
#             "  - Main findings with sources\n"
#             "  - Supporting evidence\n"
#             "  - Confidence assessment\n"
#             "  - Any notable gaps or uncertainties\n"
#             "]\n"
#         )
#     )

#     # Tasks with improved multi-document focus
#     retrieval_task = Task(
#         description=(
#             "Search and analyze information across multiple documents for query: {query}\n"
#             "Identify connections, patterns, and relationships between documents."
#         ),
#         expected_output=(
#             "Comprehensive findings including:\n"
#             "- Key information from each relevant document\n"
#             "- Cross-document connections and patterns\n"
#             "- Source attribution and confidence levels"
#         ),
#         agent=retriever_agent
#     )

#     response_task = Task(
#         description=(
#             "Create a well-structured response that synthesizes information from "
#             "multiple documents while maintaining accuracy and context."
#         ),
#         expected_output=(
#             "A coherent response that:\n"
#             "- Integrates information across documents\n"
#             "- Maintains clear source attribution\n"
#             "- Addresses any contradictions\n"
#             "- Notes confidence levels and gaps"
#         ),
#         agent=response_synthesizer_agent
#     )

#     crew = Crew(
#         agents=[retriever_agent, response_synthesizer_agent],
#         tasks=[retrieval_task, response_task],
#         process=Process.sequential,
#         verbose=True
#     )
#     return crew