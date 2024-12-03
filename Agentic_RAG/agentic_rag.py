import os
import json
from typing import List, Dict, Any
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
class AgenticRAG:
    def __init__(self, api_key: str, urls: List[str], top_k: int = 4):
        """
        Initialize the Agentic RAG system.
        
        Args:
            api_key (str): Google API key
            urls (List[str]): URLs to load documents from
            top_k (int): Number of top documents to retrieve
        """
        # Configure API key
        os.environ['GOOGLE_API_KEY'] = api_key
        
        # Configure Gemini models
        genai.configure(api_key=api_key)
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=api_key
        )
        
        # Generation configs
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_output_tokens": 4096
        }
        generation_config_j = {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_output_tokens": 4096,
            "response_mime_type": "application/json"
        }
        self.llm = genai.GenerativeModel(
            model_name='gemini-1.5-flash', 
            generation_config=generation_config
        )
        self.llm_json= genai.GenerativeModel(
            model_name='gemini-1.5-flash', 
            generation_config=generation_config_j)
        # Load and process documents
        self.urls = urls
        self.top_k = top_k
        self.vectorstore = self._load_and_index_documents()
    
    def _load_and_index_documents(self) -> FAISS:
        """
        Load documents from URLs, split them, and create a vector store.
        
        Returns:
            FAISS: Indexed vector store
        """
        try:
            docs = [WebBaseLoader(url).load() for url in self.urls]
            docs_list = [item for sublist in docs for item in sublist]
            
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=200, 
                chunk_overlap=50
            )
            doc_splits = text_splitter.split_documents(docs_list)
            
            return FAISS.from_documents(doc_splits, self.embeddings)
        except Exception as e:
            print(f"Error loading documents: {e}")
            raise
    
    def retrieve_documents(self, query: str) -> str:
        """
        Retrieve most relevant documents for a query.
        
        Args:
            query (str): Search query
        
        Returns:
            str: Retrieved document contents
        """
        try:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": self.top_k}
            )
            retrieved_docs = retriever.invoke(query)
            return " ".join([doc.page_content for doc in retrieved_docs])
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return ""
    
    def grade_documents(self, context: str, question: str) -> bool:
        """
        Grade the relevance of retrieved documents.
        
        Args:
            context (str): Retrieved document context
            question (str): User's original question
        
        Returns:
            bool: Whether documents are relevant
        """
        try:
            prompt = f"""
            Assess the relevance of the following document to the question:
            
            Document: {context}
            Question: {question}
            
            Determine if the document contains information directly related to the question.
            Respond with a JSON: {{"relevant":"yes/no"}}
            """
            
            response = self.llm_json.generate_content(prompt)
            result = json.loads(response.text)
            
            return result["relevant"]
        except Exception as e:
            print(f"Error grading documents: {e}")
            return False
    
    def rewrite_query(self, original_query: str) -> str:
        """
        Attempt to improve the original query.
        
        Args:
            original_query (str): Initial user query
        
        Returns:
            str: Refined query
        """
        try:
            prompt = f"""
            Analyze and improve the semantic clarity of this query:
            Original Query: {original_query}
            
            Refine the query to be more precise and clear. Consider:
            - Adding context
            - Clarifying intent
            - Using more specific language
            
            Refined Query:
            """
            
            response = self.llm.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error rewriting query: {e}")
            return original_query
    
    def generate_response(self, context: str, query: str) -> str:
        """
        Generate a response based on context and query.
        
        Args:
            context (str): Retrieved document context
            query (str): User's query
        
        Returns:
            str: Generated response
        """
        try:
            prompt = f"""Context: {context}
            
            Question: {query}
            
            Provide a clear, concise, and accurate answer based solely on the given context.
            If the context does not contain sufficient information, state that clearly.
            """
            
            response = self.llm.generate_content(prompt,stream=True)
            for chunk in response:
                print(chunk.text)
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm unable to generate a response at this time."
    
    def process_query(self, query: str) -> str:
        """
        Main method to process a user query through the Agentic RAG pipeline.
        
        Args:
            query (str): User's original query
        
        Returns:
            str: Final response
        """
        # Retrieve documents
        context = self.retrieve_documents(query)
        score=self.grade_documents(context, query)
        print(score)
        # Grade document relevance
        if score=="no":
            # If documents are not relevant, try rewriting the query
            query = self.rewrite_query(query)
            print(query)
            context = self.retrieve_documents(query)
            return self.generate_response(context, query)
        else:
            return self.generate_response(context, query)


def main():
    # Replace with your actual Google API key
    api_key = os.getenv("GEMINI_API_KEY")
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
    ]
    
    rag_system = AgenticRAG(api_key, urls)
    
    query = "What does Lilian Weng say about the types of agent memory?"
    rag_system.process_query(query)
    
    

if __name__ == "__main__":
    main()