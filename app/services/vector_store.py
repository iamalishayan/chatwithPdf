import logging
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from app.config import settings

logger = logging.getLogger("vector-store")

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(
    path=settings.CHROMA_DATA_PATH,
    settings=Settings(anonymized_telemetry=False)
)

# Get or create the collection
collection = chroma_client.get_or_create_collection(
    name="pdf_documents",
    metadata={"hnsw:space": "cosine"}
)

def add_documents(
    doc_id: str,
    documents: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]]
):
    """Saves documents, embeddings, and metadata to ChromaDB."""
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(documents))]
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    logger.info(f"Stored {len(documents)} chunks in ChromaDB for doc_id={doc_id}")

def query_documents(
    doc_id: str,
    query_embedding: List[float],
    n_results: int = 3
) -> Dict[str, Any]:
    """Queries documents based on cosine similarity and strictly filters by doc_id."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"doc_id": doc_id}
    )
    return results
