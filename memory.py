import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Directory to store the local vector database
DB_PATH = "faiss_db"

# Using Ollama to generate embeddings locally
embeddings = OllamaEmbeddings(model="llama3.1:8b")

def save_to_memory(text_content: str, metadata: dict):
    """Chunks the final report and saves it to a local FAISS database."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([text_content], metadatas=[metadata])
    
    if os.path.exists(DB_PATH):
        # Load existing DB and add new data
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(docs)
        db.save_local(DB_PATH)
    else:
        # Create new DB
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(DB_PATH)
    
    return True