import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- DOMAIN NAMESPACE MAP ---
# Each domain gets its own FAISS index so retrieval stays sharp
# as the index grows — no cross-domain dilution
DOMAIN_KEYWORDS = {
    "aws":       ["lakeformation", "lake formation", "glue", "s3", "iam", "ec2",
                  "vpc", "lambda", "rds", "sqs", "sns", "macie", "athena",
                  "iceberg", "hudi", "lf-tag", "data catalog", "aws", "amazon"],
    "terraform": ["terraform", "hcl", "provider", "resource", "module", ".tf"],
    "python":    ["python", "boto3", "pandas", "fastapi", "flask", "django",
                  "asyncio", "pip", "lambda function"],
    "java":      ["java", "spring", "maven", "gradle", "jvm"],
    "react":     ["react", "jsx", "component", "hook", "next.js", "vite"],
    "nodejs":    ["node", "express", "javascript", "npm", "typescript"],
    "go":        ["golang", "go ", "goroutine", "gin", "gorm"],
}

DEFAULT_DOMAIN = "general"


def _resolve_domain(text: str) -> str:
    """Infers the best domain bucket from topic/metadata text."""
    lowered = text.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return domain
    return DEFAULT_DOMAIN


def _db_path(domain: str) -> str:
    return f"faiss_db_{domain}"


def save_to_memory(text_content: str, metadata: dict):
    """
    Chunks and saves report to the correct domain FAISS index.
    Each domain has its own isolated index — no cross-domain dilution.
    """
    topic = metadata.get("topic", "")
    language = metadata.get("language", "")
    domain = _resolve_domain(f"{topic} {language}")
    db_path = _db_path(domain)

    print(f"💾 Archiving to domain='{domain}' | topic='{topic}'")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([text_content], metadatas=[{**metadata, "domain": domain}])

    try:
        if os.path.exists(db_path):
            db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
            db.add_documents(docs)
            db.save_local(db_path)
        else:
            db = FAISS.from_documents(docs, embeddings)
            db.save_local(db_path)
        print(f"✅ Memory updated [{domain}].")
    except Exception as e:
        print(f"⚠️ Memory mismatch in '{domain}': {e} — resetting index")
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(db_path)

    return True


def query_memory(topic: str, k: int = 3) -> str:
    """
    Queries the correct domain index first.
    Falls back to 'general' index if domain index is empty.
    Returns combined relevant chunks.
    """
    domain = _resolve_domain(topic)
    results = _query_index(topic, _db_path(domain), k=k)

    # If domain index came up short, supplement from general
    if not results and domain != DEFAULT_DOMAIN:
        results = _query_index(topic, _db_path(DEFAULT_DOMAIN), k=k)

    if results:
        print(f"🧠 Recovered {len(results)} chunks from '{domain}' memory.")
        return "\n".join(results)
    return ""


def _query_index(topic: str, db_path: str, k: int) -> list[str]:
    """Helper: queries a single FAISS index. Returns list of page content strings."""
    if not os.path.exists(db_path):
        return []
    try:
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(topic, k=k)
        return [d.page_content for d in docs]
    except Exception as e:
        print(f"⚠️ Memory read error at '{db_path}': {e}")
        return []


def list_domains() -> list[str]:
    """Returns all domains that have a saved FAISS index."""
    all_domains = list(DOMAIN_KEYWORDS.keys()) + [DEFAULT_DOMAIN]
    return [d for d in all_domains if os.path.exists(_db_path(d))]