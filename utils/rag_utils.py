# utils/rag_utils.py (updated for Chroma >= 0.5)

import os
import logging
import sys

# Patch the standard sqlite3 module with pysqlite3 to ensure compatibility with ChromaDB, which requires SQLite version >= 3.35.0 (often not available in default Python builds) for streamlit cloud.

try:
    import pysqlite3 # type: ignore
    sys.modules["sqlite3"] = sys.modules["pysqlite3"]
except ImportError:
    pass

import chromadb
from models.embeddings import EmbeddingModel
from config.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

def get_chroma_client(persist_directory=CHROMA_PERSIST_DIR):
    """Return a Chroma PersistentClient (new API)."""
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_directory)
    return client

def chunk_text(text, chunk_size=500, overlap=50):
    """Split long text into overlapping chunks."""
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def index_documents(docs, collection_name="docs"):
    """Index a list of dicts into ChromaDB."""
    client = get_chroma_client()
    try:
        coll = client.get_or_create_collection(name=collection_name)
    except Exception as e:
        raise RuntimeError(f"Failed to create/get collection: {e}")

    emb = EmbeddingModel(EMBEDDING_MODEL_NAME)
    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    metadatas = [d.get("meta", {}) for d in docs]
    vectors = [vec.tolist() for vec in emb.embed_texts(texts)]

    try:
        coll.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=vectors,
        )
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to add docs to Chroma: {e}")

def retrieve(query, k=4, collection_name="docs"):
    """Retrieve top-k documents for a query."""
    client = get_chroma_client()
    try:
        coll = client.get_collection(name=collection_name)
    except Exception:
        return []

    emb = EmbeddingModel(EMBEDDING_MODEL_NAME)
    q_vec = emb.embed_texts([query])[0].tolist()  # ✅ ensure list

    results = coll.query(
        query_embeddings=[q_vec],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    print("DEBUG - Raw retrieval results:", results)  # ✅ extra debug

    docs = []
    if results and results.get("documents"):
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            docs.append({"text": doc, "meta": meta, "distance": dist})
    return docs

