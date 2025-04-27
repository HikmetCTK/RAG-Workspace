from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
import os 
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json 
import gradio as gr 



load_dotenv()
apkey=os.getenv("GEMINI_API_KEY")
client=genai.Client(api_key=apkey)

def extract_infos(data):
    """
    Extracts table information from the given data structure.

    This function processes a JSON-like data structure, specifically extracting 
    table-related information from the 'analyzeResult' section.

    Args:
        data (dict): A dictionary containing the result of an analysis, typically 
                     generated from an OCR  feature which is used in Azure document analysis.

    Returns:
        list: A list of dictionaries, each representing a cell from the tables, 
              with information such as table number, content, row index, column index, 
              and page number.
    """
    extracted_data=[] # Extract table  information
    table=0

    for doc in data["analyzeResult"]["tables"]:
        for cell in doc.get('cells', []):
            content = cell.get('content')
            row_index = cell.get('rowIndex')
            column_index = cell.get('columnIndex')
            page_number=cell.get('boundingRegions')[0].get("pageNumber")

            if content is not None and row_index is not None and column_index is not None:
                        try:
                            if row_index== 0 and column_index == 0: #specify each table information
                                table+=1
                                extracted_data.append({
                                    'table':table,
                                    'content': content,
                                    'rowIndex': row_index,
                                    'columnIndex': column_index,
                                    "page_label":page_number
                                })
                            else:
                                extracted_data.append({
                                    'table':table,
                                    'content': content,
                                    'rowIndex': row_index,
                                    'columnIndex': column_index,
                                    "page_label":page_number
                                })
                        except Exception as e:
                            print(e)
    return extracted_data



                              
def flatten_table(table_cells):
    """
    Flattens table data into a dictionary organized by table ID, grouping rows and page labels.

    Args:
        table_cells (list): A list of dictionaries, where each dictionary represents a cell in a table 
                             and contains 'rowIndex', 'columnIndex', 'content', 'page_label', and 'table' keys.

    Returns:
        dict: A dictionary where keys are table IDs and values are dictionaries with:
              - 'rows' (list): A list of formatted strings representing rows in the table.
              - 'page_label' (set): A set of page numbers where the table appears.
    """

    tables = {}
    for cell in table_cells:
        row = cell["rowIndex"]  
        col = cell["columnIndex"]
        content = cell["content"]
        page=cell["page_label"]
        table_id=cell["table"]
        line=f"Row {row}, Column {col}: {content} , page:{page}, table:{table_id}"
        if table_id not in tables: # store each table  information according to table_id
            tables[table_id] = {"rows": [], "page_label":set() }  # Use set to avoid duplicate pages
        tables[table_id]["rows"].append(line)
        tables[table_id]["page_label"].add(page)  # Store page numbers separately
            

    return tables
    



def table_document(grouped_table):
    """
    Converts grouped table data into a list of Document objects, each representing a table with its content 
    and metadata (table ID and page label).

    Args:
        grouped_table (dict): A dictionary where keys are table IDs and values are dictionaries containing:
                              - 'rows' (list): A list of formatted strings representing rows of the table.
                              - 'page_label' (set): A set of page numbers where the table appears.

    Returns:
        list: A list of `Document` objects where each document contains the content of a table and its metadata (table ID and page label).
    """
    table_doc=[]
    for id,doc in grouped_table.items():
        content=" ".join([d for d in doc.get("rows")])  # get information for each table 
        metadata={"table_id":id,"page_label":list(doc.get("page_label"))[0]} # metadata informatons ! page_label should convert to list and get page label . before :{1} after :1
        doc=Document(page_content=content,metadata=metadata) #Document object 
        table_doc.append(doc)
    return table_doc




# Get image description with gemini 
def get_image_description(page:int,image_path:str="Screenshot 2025-04-26 234130.jpg"):

    """
    Generates a description of the content in an image using the Gemini API, and stores the description 
    in a Document object with associated metadata (page label).

    Args:
        page (int): The page number associated with the image.
        image_path (str): The file path to the image for which the description is to be generated (default is "Screenshot 2025-04-26 234130.jpg").

    Returns:
        list: A list containing a single `Document` object with the generated image description as its content and the page label as its metadata.
    """
    with open(image_path, 'rb') as f:
        image_bytes = f.read()


    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',  
            ),
            "Describe all information detailed in this image."  #  Prompt
        ]
    )
    image_doc = [
    Document(
        metadata={"page_label":page

        },
        page_content=response.text # store output in  document
    ),

    ]
    return image_doc




def parse_split_pdf(pdf_path:str="22365_3_Prompt Engineering_v7 (1).pdf"):
    """
    Loads a PDF from the specified file path, splits its content into chunks using a text splitter, 
    and returns the split document chunks.

    Args:
        pdf_path (str): The file path of the PDF to be loaded and split (default is "22365_3_Prompt Engineering_v7 (1).pdf").

    Returns:
        list: A list of Document objects, where each document represents a chunk of text from the PDF.
    """

    pdf_path="22365_3_Prompt Engineering_v7 (1).pdf"
    loader = PyPDFLoader(
    file_path=pdf_path
)

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,  # %10-20 of chunk size. It Provides strong contextual chunks based on overlapping number.
    separators=["\n\n", "\n", ".", " ", ""]  # paragraph,newline,sentence,word..
)
    chunked_doc=text_splitter.split_documents(docs)
    return chunked_doc


def ensemble_retriever(chunked_doc:list , table_doc:list , image_doc:list , bm_k=5,vector_k=5):
    """
    Creates a hybrid retriever that combines BM25 (keyword-based) and semantic (embedding-based) retrieval.

    Args:
        chunked_doc (list): List of chunked text documents to be used for retrieval.
        table_doc (list): List of table documents to be included in the retrieval corpus.
        image_doc (list): List of image documents (converted to text) to be included in the retrieval corpus.
        bm_k (int, optional): Number of top documents to retrieve using BM25. Defaults to 5.
        vector_k (int, optional): Number of top documents to retrieve using semantic similarity. Defaults to 5.

    Returns:
        EnsembleRetriever: A hybrid retriever that fuses BM25 and semantic retrievers using rank fusion with equal weights.
    """


    bm25=BM25Retriever.from_documents(chunked_doc + table_doc + image_doc,k=bm_k) #Bm25 retriever(keyword search)
    vectorstore=FAISS.from_documents(documents=chunked_doc + image_doc
                                      ,embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=apkey)) # semantic  retriever
    
    semantic_retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":vector_k}) 
    hybrid_retriever=EnsembleRetriever(retrievers=[bm25,semantic_retriever],weights=[0.5,0.5]) #Hybrid approach uses rank fusion 

    return hybrid_retriever


def get_answer(query:str,hybrid_retriever:EnsembleRetriever):

    """
    Retrieves relevant documents using the hybrid retriever based on the provided query,
    and generates an answer using the retrieved documents and a predefined system prompt.

    Args:
        hybrid_retriever (EnsembleRetriever): The retriever used to fetch relevant documents based on the query.
        query (str): The user's question to be answered.

    Returns:
        str: The generated answer to the query, based on the relevant documents.
    """

    # Retrieve the top k relevant chunks using your hybrid retriever
    relevant_docs = hybrid_retriever.invoke(query)
    if not relevant_docs:
        return "No relevant documents found."
    context = "\n\n".join([f"page_label: {doc.metadata.get('page_label','table_id')} -- {doc.page_content}"   for doc in relevant_docs])


    system_prompt = f"""
DOCUMENT: {context}

QUESTION: {query}

INSTRUCTIONS: Answer the user's QUESTION using the DOCUMENT text above. 
Keep your answer grounded in the facts of the DOCUMENT. 
Your answer should include citation . if user asks question about table specify row and column information. If the DOCUMENT doesn't contain the facts to answer the QUESTION say "I don't know"
"""

    config = types.GenerateContentConfig(
            temperature=0.1,
            top_k=64,
            top_p=0.96,
            system_instruction=system_prompt
        )
    response=client.models.generate_content(model="gemini-2.0-flash",contents="",
                                            config=config)
    
    return response.text


def main():
    
    with open("Hybrid_rag/29-1_schollaert-uz_examples.pdf.json") as json_file:  # json file which includes table information.
        try:
            data = json.load(json_file)
        except json.decoder.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")

    extracted_data=extract_infos(data)

    flattened_table_text_grouped = flatten_table(extracted_data)                         
    table_doc=table_document(flattened_table_text_grouped)
    image_document=get_image_description(image_path="Screenshot 2025-04-26 234130.jpg",page=1)

    chunked_doc=parse_split_pdf(pdf_path="22365_3_Prompt Engineering_v7 (1).pdf")

    hybrid_retriever=ensemble_retriever(table_doc=table_doc,chunked_doc=chunked_doc,image_doc=image_document)  # hybrid retriever. returns vector store initialized for vector embedding

    #Interface
    def gradio_func(query):
        return get_answer(query,hybrid_retriever)
    interface=gr.Interface(fn=gradio_func,
    inputs=gr.Textbox(label="Enter your query"),  # Input type is a Textbox for the query
    outputs=gr.Markdown(label="Answer")  # Output type is a Markdown to display the answer
    
)
    interface.launch()
    



if __name__=="__main__":
    main()
    


