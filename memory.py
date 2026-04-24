import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_PATH = "faiss_db"

# Optimized: Use a dedicated embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

def save_to_memory(text_content: str, metadata: dict):
    """Chunks and saves report to FAISS with automatic dimension-mismatch recovery."""
    print(f"--- Archiving research on '{metadata.get('topic')}' ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([text_content], metadatas=[metadata])
    
    try:
        if os.path.exists(DB_PATH):
            db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
            db.add_documents(docs)
            db.save_local(DB_PATH)
        else:
            db = FAISS.from_documents(docs, embeddings)
            db.save_local(DB_PATH)
    except Exception as e:
        # If dimensions mismatch (AssertionError), delete and rebuild
        print(f"⚠️ Memory Mismatch Detected: {e}. Resetting database...")
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(DB_PATH)
    
    print("✅ Memory updated.")
    return True