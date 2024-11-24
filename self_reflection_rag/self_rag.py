from langchain_community.document_loaders import WebBaseLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_core.output_parsers import JsonOutputParser
import json
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")
load_dotenv()
api_key=os.getenv("GEMINI_API_KEY") 
genai.configure(api_key=api_key)
class Self_RAG:
    def __init__(self):
        self.generation_config={
            "temperature": 0.2,
            "top_p": 0.9,
            "max_output_tokens": 4096,
            "response_mime_type": "application/json"
        }
        
        self.llm_json=genai.GenerativeModel(model_name="gemini-1.5-flash",generation_config=self.generation_config,)
        self.llm=genai.GenerativeModel(model_name="gemini-1.5-flash")
        self.embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectorstore=None
        self.pdf_file="2201.08528v3.pdf"
        self.chunk_size=500

    def load_and_split(self):
        load=PyPDFLoader(self.pdf_file)
        data=load.load()
        text=RecursiveCharacterTextSplitter(chunk_size=self.chunk_size)
        doc=text.split_documents(data)
        return doc  

    def vector_store(self,doc):
        self.vectorstore=FAISS.from_documents(embedding=self.embedding,documents=doc)
        self.vectorstore.save_local("smot_index")
        return self.vectorstore
    
    def retrieve_doc(self,query,k=3):
        if not self.vectorstore:
            self.vectorstore=FAISS.load_local("smot_index",embeddings=self.embedding,allow_dangerous_deserialization=True)
        retrieve=self.vectorstore.as_retriever(search_type="similarity",search_kwarg={"k":k})
        relevant_doc=retrieve.invoke(query)
        return "/n".join(doc.page_content for doc in relevant_doc)
    
    def evaluate_response_quality(self, query, response, context):
        """Evaluate the quality of the generated response."""
        eval_prompt = f"""
        Evaluate the quality of this response to the given query.
        Query: {query}
        Context: {context}
        Response: {response}

        Score the following aspects from 1-10:
        1. Relevance to query
        2. Factual accuracy based on context
        3. Completeness of answer
        4. Coherence and clarity

        **Return the evaluation **ONLY as a JSON object.** Do not include any other text or explanation. Strictly adhere to this format:
        *format:*
        {{
            "evaluation": {{
                "relevance": {{
                    "score": 9,
                    "explanation": "Explanation here."
                }},
                "factual_accuracy": {{
                    "score": 8,
                    "explanation": "Explanation here."
                }},
                "completeness": {{
                    "score": 7,
                    "explanation": "Explanation here."
                }},
                "coherence": {{
                    "score": 10,
                    "explanation": "Explanation here."
                }},
                "overall_score": 8.5,
                "overall_explanation":"Explanation here"
            }}
        }}
        """
        try:
            response = self.llm_json.generate_content(eval_prompt)
            
            # Parse the JSON string into a Python dictionary
            evaluation_data = json.loads(response.text)
            

            overall_score = evaluation_data["evaluation"]["overall_score"]
            overall_explanation=evaluation_data["evaluation"]["overall_explanation"]
            
            return overall_score,overall_explanation
        except json.JSONDecodeError as e:
            print("JSONDecodeError:", e)
            print("Response Text for Debugging:", response.text)
            return None
        except Exception as e:
            print("Error occurred:", e)
            return None
    
    def improve_response(self, query, initial_response, evaluation,feedback):
        """Improve the response based on evaluation feedback."""
        improve_prompt = f"""
        Improve this response based on the evaluation feedback:
        Query: {query}
        Initial Response: {initial_response}
        Evaluation: {evaluation}
        feedback:{feedback}
        
        Generate an improved response that addresses the weaknesses identified.
        """
        
        improved_response = self.llm.generate_content(improve_prompt)
        print(f"**{improved_response.text}")
        return improved_response.text
    
    def generate_response(self, query):
        """Generate a response using self-RAG approach."""

        context = self.retrieve_doc(query) #Retrieve relevant chunks
       
        initial_prompt = f"""
        Based on the following context, answer the query:
        Query: {query}
        Context: {context}
        
        Provide a comprehensive and accurate response.
        """
        initial_response = self.llm.generate_content(initial_prompt).text
        evaluation,explanation = self.evaluate_response_quality(query, initial_response, context) #evaluation score and overall_explanation value from json output
        
        average_score=int(evaluation)
        if average_score < 8:  
            improved_response = self.improve_response(query, initial_response, evaluation,explanation)
            return f"""
            "final_response": {improved_response},
            "evaluation":{evaluation},
            "improved": {True}
            """
            
        return f"""
            "final_response": {initial_response},
            "evaluation":{evaluation},
            "improved": {False}
        """

if __name__ == "__main__":
    rag = Self_RAG()
    docs = rag.load_and_split()
    vectorstore = rag.vector_store(docs)
    
    query = "what is SMOTE ?"
    response = rag.generate_response(query)
    print(response)