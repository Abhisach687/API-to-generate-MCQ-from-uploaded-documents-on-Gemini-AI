from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from PyPDF2 import PdfReader
import asyncio
import threading
import os
import json
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory dictionary to store vector stores
vector_stores = {}

# Function to get vectorstore from a URL
def get_vectorstore_from_url(url):
    # Load the webpage
    loader = WebBaseLoader(url)
    # Create documents from the webpage
    documents = [Document(text) for text in loader.load()]
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    document_chunks = text_splitter.split_documents([doc.page_content for doc in documents])
    # Generate embeddings for the chunks
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # Create a vector store from the documents and embeddings
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    # Save the vector store locally
    vector_store.save_local("faiss_index")
    return document_chunks, embeddings

# Class to represent a document
class Document:
    def __init__(self, text):
        self.page_content = text
        self.metadata = {}

# Function to get vectorstore from a text
def get_vectorstore_from_text(text):
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    document = Document(text)
    document_chunks = text_splitter.split_documents([document])
    # Generate embeddings for the chunks
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # Create a vector store from the documents and embeddings
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    # Save the vector store locally
    vector_store.save_local("faiss_index")
    return document_chunks, embeddings

# Function to get a conversational chain
def get_conversational_chain():
    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    threading.Thread(target=loop.run_forever).start()
    # Define the prompt template
    prompt_template = """
    Create MCQ based on the context:\n {context}?\n
    The MCQ should be sent back in json format. 
    The JSON format should be like:
   "mcqs": [
            {
                "question": "?",
                "options": [
                    "",
                    "",
                    "",
                    ""
                ],
                "correctAnswer": ""
            },{},{} ... ]
    This json should strictly follow the format above. Do not deviate from the format.
    Only send the json back. Do not write anything else. Nothing else other than json. Peroid. Generate 5-10 MCQs. Do not attach mcqs in json. Just generate the list [].
    """
    # Create a model and a prompt
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    # Load a QA chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to get a MCQ chain
def get_mcq_chain():
    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    threading.Thread(target=loop.run_forever).start()
    # Create a model and a prompt
    mcq_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    mcq_prompt = PromptTemplate(template = "{context}\n\nGenerate MCQs in JSON format:", input_variables = ["context"])
    # Load a QA chain
    mcq_chain = load_qa_chain(mcq_model, chain_type="stuff", prompt=mcq_prompt)
    return mcq_chain

@app.post("/generate_mcq")
async def generate_mcq(website_url: Optional[str] = None, uploaded_file: Optional[UploadFile] = None):
    if website_url is None and uploaded_file is None:
        raise HTTPException(status_code=400, detail="Please provide a website URL or upload a PDF")

    # Check if a vector store exists in the vector_stores dictionary
    if website_url not in vector_stores:
        # If a URL is provided, get the vector store from the URL
        if website_url is not None:
            vector_stores[website_url] = get_vectorstore_from_url(website_url)
        # If a file is uploaded, get the vector store from the text in the file
        elif uploaded_file is not None:
            pdf_file = PdfReader(uploaded_file.file)
            text = ""
            for page in pdf_file.pages:
                text += page.extract_text()
            vector_stores[uploaded_file.filename] = get_vectorstore_from_text(text)

    # Get the document chunks and embeddings from the vector store
    document_chunks, embeddings = vector_stores[website_url if website_url is not None else uploaded_file.filename]
    # Create a new vector store from the documents and embeddings
    new_db = FAISS.from_documents(document_chunks, embeddings)
    # Search for similar documents
    docs = new_db.similarity_search("Generate MCQ based on the document")
    # Get the MCQ chain
    mcq_chain = get_mcq_chain()
    # Get the response from the MCQ chain
    mcq_response = mcq_chain({"input_documents": docs, "question": "Generate MCQ based on the document"}, return_only_outputs=True)

    # Check if the response is not empty
    if mcq_response["output_text"].strip():
        try:
            # Clean the response
            cleaned_response = mcq_response["output_text"].strip().strip("```json").strip("```").strip()
            cleaned_response = cleaned_response.replace('```json', '').replace('```', '').strip()
            # Load the cleaned response as JSON
            mcq_json = json.loads(cleaned_response)
            return mcq_json
        except json.JSONDecodeError as e:
            # If the response is not a valid JSON string, raise an error
            raise HTTPException(status_code=400, detail=f"The response is not a valid JSON string. Error: {str(e)}")
    else:
        # If the response is empty, raise an error
        raise HTTPException(status_code=400, detail="The response is empty.")