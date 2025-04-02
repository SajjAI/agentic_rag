# import os
# from crewai.tools import BaseTool
# from typing import Type, List, Dict
# from pydantic import BaseModel, Field, ConfigDict
# from groundx import Document, GroundX
# from dotenv import load_dotenv
# import streamlit as st
# import uuid

# load_dotenv()


# class DocumentSearchToolInput(BaseModel):
#     """Input schema for DocumentSearchTool."""
#     query: str = Field(..., description="Query to search the document.")

# class DocumentSearchTool(BaseTool):
#     name: str = "DocumentSearchTool"
#     description: str = "Search multiple documents for the given query."
#     args_schema: Type[BaseModel] = DocumentSearchToolInput
    
#     model_config = ConfigDict(extra="allow")
    
#     def __init__(self):
#         """Initialize the searcher and set up the Qdrant collection."""
#         super().__init__()
#         self.client = GroundX(
#             api_key="5762b3c7-bc2a-450a-805d-01e90a3a0797"
#             # os.getenv("GROUNDX_API_KEY")
#         )
#         self.bucket_id = 15153
#         # self.bucket_id = 15151
#         # self._create_bucket()
#         self.documents = {}  # Store document IDs and their content
#         self.search_results = []
    
#     def upload_document(self, file_path: str) -> str:
#         """Upload and index a document, returning a process ID."""
#         try:
#             st.info(f"Starting upload for: {os.path.basename(file_path)}")
#             st.info(f"Using bucket ID: {self.bucket_id}")
            
#             ingest = self.client.ingest(
#                 documents=[
#                     Document(
#                         bucket_id=self.bucket_id,
#                         file_name=os.path.basename(file_path),
#                         file_path=file_path,
#                         file_type="pdf",
#                         search_data=dict(
#                             key="value",
#                         ),
#                     )
#                 ]
#             )
#             process_id = ingest.ingest.process_id
#             st.info(f"Upload successful. Process ID: {process_id}")
#             self.documents[process_id] = file_path
#             return process_id
#         except Exception as e:
#             st.error(f"Error uploading document: {str(e)}")
#             raise

#     def _create_bucket(self):
#         """Create a bucket for storing documents."""
#         response = self.client.buckets.create(
#             name="agentic_rag_multi"
#         )
#         return response.bucket.bucket_id

#     def search(self, query: str, **kwargs) -> str:
#         """
#         Search across all uploaded documents and return combined relevant results.
#         """
#         all_results = []
        
#         for doc_id, file_path in self.documents.items():
#             try:
#                 # Perform search in each document
#                 results = self._search_document(doc_id, query)
#                 if results:
#                     # Add document source information to each result
#                     doc_name = os.path.basename(file_path)
#                     for result in results:
#                         result['source'] = doc_name
#                     all_results.extend(results)
#             except Exception as e:
#                 print(f"Error searching document {doc_id}: {str(e)}")
#                 continue

#         # Sort results by relevance score if available
#         if all_results:
#             all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
#             # Format the results into a string
#             formatted_results = []
#             for result in all_results:
#                 source = result.get('source', 'Unknown')
#                 content = result.get('content', '').strip()
#                 formatted_results.append(f"From {source}:\n{content}\n")
            
#             return "\n".join(formatted_results)
        
#         return "No relevant information found in the documents."

#     def _search_document(self, doc_id: str, query: str) -> List[Dict]:
#         """
#         Search within a specific document.
#         Returns a list of dictionaries containing search results.
#         """
#         # Implement your document search logic here
#         # This could use vector similarity search, keyword matching, etc.
#         # Return format should be list of dicts with at least 'content' and 'score' keys
#         pass

#     def _run(self, query: str) -> str:
#         """Search across all documents in the bucket with improved retrieval."""
#         try:
#             st.info("Getting document list...")
#             documents_response = self.client.documents.list()
#             document_ids = [
#                 doc.document_id 
#                 for doc in documents_response.documents 
#                 if doc.bucket_id == self.bucket_id
#             ]
            
#             st.info(f"Found {len(document_ids)} documents to search")
            
#             if not document_ids:
#                 return "No documents found to search."
            
#             st.info("Executing search...")

#             # Simplified search parameters based on GroundX API
#             search_response = self.client.search.documents(
#                 query=query,
#                 document_ids=document_ids,
#                 n=100,  # Number of results to return
#                 verbosity=2, # Detailed results
#                 relevance= 0.6
#             )
            
#             st.info(f"Found {len(search_response.search.results)} results")
            
#             # Enhanced result formatting with source tracking and relevance scores
#             formatted_results = []
#             for i, result in enumerate(search_response.search.results, 1):
#                 # Get document metadata
#                 doc_info = next(
#                     (doc for doc in documents_response.documents 
#                      if doc.document_id == result.document_id),
#                     None
#                 )
                
#                 # Format result with source and relevance information
#                 result_entry = {
#                     "source": doc_info.file_name if doc_info else "Unknown",
#                     "content": result.text,
#                     "score": getattr(result, 'score', 0),
#                     "page": result.metadata.get("page_number", "Unknown") if hasattr(result, 'metadata') else "Unknown"
#                 }
                
#                 # Create formatted text without line breaks in f-string
#                 formatted_text = "".join([
#                     f"[Source: {result_entry['source']}, ",
#                     f"Page: {result_entry['page']}, ",
#                     f"Relevance: {result_entry['score']:.2f}]\n",
#                     f"{result_entry['content']}\n",
#                     f"{'-' * 40}\n"
#                 ])
#                 formatted_results.append(formatted_text)
            
#             # Sort results by relevance score
#             formatted_results.sort(
#                 key=lambda x: float(x.split("Relevance: ")[1].split("]")[0]),
#                 reverse=True
#             )
            
#             # Combine results with clear separation
#             final_output = "".join([
#                 "🔍 Search Results Summary:\n",
#                 f"Found {len(formatted_results)} relevant passages across {len(document_ids)} documents\n\n",
#                 f"{'=' * 50}\n\n",
#                 "\n".join(formatted_results)
#             ])
            
#             return final_output if formatted_results else "No relevant information found."
            
#         except Exception as e:
#             st.error(f"Error in search: {str(e)}")
#             return f"An error occurred during search: {str(e)}"

# # Update the test implementation
# def test_document_searcher():
#     # Test file paths
#     pdf_paths = [
#         "/path/to/first.pdf",
#         "/path/to/second.pdf"
#     ]
    
#     # Create instance
#     searcher = DocumentSearchTool()
    
#     # Upload documents
#     for pdf_path in pdf_paths:
#         process_id = searcher.upload_document(pdf_path)
#         print(f"Uploaded {pdf_path} with process ID: {process_id}")
    
#     # Test search
#     result = searcher._run("What is the purpose of DSpy?")
#     print("Search Results:", result)

# if __name__ == "__main__":
#     test_document_searcher()












import os
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field, ConfigDict
from groundx import Document, GroundX
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


class DocumentSearchToolInput(BaseModel):
    """Input schema for DocumentSearchTool."""
    query: str = Field(..., description="Query to search the document.")

class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search the document for the given query."
    args_schema: Type[BaseModel] = DocumentSearchToolInput
    
    model_config = ConfigDict(extra="allow")
    
    def __init__(self):
        """Initialize the searcher and set up the GroundX client."""
        super().__init__()
        self.client = GroundX(
            api_key=os.getenv('GROUNDX_API_KEY')
        )
        self.bucket_id = int(os.getenv('GROUNDX_BUCKET_ID'))
    
    def upload_document(self, file_path: str) -> str:
        ingest = self.client.ingest(
                        documents=[
                            Document(
                            bucket_id=self.bucket_id,
                            file_name=os.path.basename(file_path),
                            file_path=file_path,
                            file_type="pdf",
                            search_data=dict(
                                key = "value",
                            ),
                            )
                        ]
                        )
        return ingest.ingest.process_id
    
    def _create_bucket(self):
        response = self.client.buckets.create(
            name="agentic_rag"
        )
        return response.bucket.bucket_id    

    def _run(self, query: str) -> str:
        # Check processing status
        
        # status_response = self.client.documents.get_processing_status_by_id(
        #     process_id=self.process_id
        # )
        
        # # Check if processing is complete
        # if status_response.ingest.status != "complete":
        #     return "Document is still being processed..."
        
        documents_response = self.client.documents.list()
        document_ids = [
            doc.document_id 
            for doc in documents_response.documents 
            if doc.bucket_id == self.bucket_id
        ]
        
        st.info(f"Found {len(document_ids)} documents to search")
        
        if not document_ids:
            return "No documents found to search."
        
        st.info("Executing search...")

        # search_response = self.client.search.documents(
        #     query=query,
        #     document_ids=document_ids,
        #     n=100,  # Number of results to return
        #     verbosity=2, # Detailed results
        #     relevance= 0.6
        # )
            
        # If processing is complete, proceed with search
        search_response = self.client.search.content(
            id=self.bucket_id,
            query=query,
            n=100,
            verbosity=2,
            relevance= 0.6
        )
        
        # Format the results with separators
        formatted_results = ""
        for result in search_response.search.results:
            formatted_results += f"{result.text}\n____\n"
        st.info("Done Executing search...")
        return formatted_results.rstrip('____\n')

# Test the implementation
def test_document_searcher():
    # Test file path
    # pdf_path = "/Users/shkhilji/Documents/AI/ai-engineering-hub-main/agentic_rag_deepseek/knowledge/dspy.pdf"
    
    # Create instance
    # searcher = DocumentSearchTool(file_path=pdf_path)
    searcher = DocumentSearchTool() 
    # Test search
    result = searcher._run("what property is rented by suhail ahmed ?")
    print("Search Results:", result)

if __name__ == "__main__":
    test_document_searcher()
