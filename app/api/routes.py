import logging
import uuid
import asyncio
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest, UploadResponse, DocumentMetadata
from app.services.pdf_extractor import extract_text_from_pdf
from app.services.chunking import chunk_text
from app.services.vector_store import add_documents, query_documents
from app.services.database import create_document, get_all_documents, get_next_version, archive_old_versions
from app.llm.gemini_client import get_embedding, stream_answer

logger = logging.getLogger("routes")
router = APIRouter(prefix="/api")

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...)
):
    """Ingests a PDF file, increments its version if it exists, archives old versions, chunks, embeds, and stores it."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    try:
        filename = file.filename
        
        # Calculate next version
        version = get_next_version(filename)
        
        # Archive older versions in SQLite
        archive_old_versions(filename)
        
        # Generate a unique document ID for this specific version upload
        doc_id = str(uuid.uuid4())
        
        # Read the file
        file_bytes = await file.read()
        
        # Extract text
        raw_text = extract_text_from_pdf(file_bytes)
        
        # Chunk text
        chunks = chunk_text(raw_text)
        if not chunks:
            raise HTTPException(status_code=400, detail="The PDF file contains no readable text.")
        
        # Embed chunks concurrently
        embeddings = await asyncio.gather(
            *(get_embedding(chunk) for chunk in chunks)
        )
        
        # Prepare Metadata for ChromaDB
        metadatas = [{"doc_id": doc_id} for _ in chunks]
        
        # Store in ChromaDB
        add_documents(
            doc_id=doc_id,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        # Store metadata in SQLite database
        create_document(doc_id=doc_id, filename=filename, version=version)
        
        logger.info(f"Successfully processed and stored document version. ID: {doc_id}, Name: {filename}, Version: {version}")
        
        return UploadResponse(
            message=f"Successfully processed {len(chunks)} chunks.",
            doc_id=doc_id,
            filename=filename,
            version=version
        )
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents", response_model=List[DocumentMetadata])
async def list_documents():
    """Retrieves all uploaded documents stored in the SQLite database, with their version and status."""
    try:
        return get_all_documents()
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents.")

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Queries ChromaDB using the question's embedding, retrieves context, and streams the answer with metadata and performance metrics."""
    try:
        # 1. Embed query
        query_embedding = await get_embedding(request.message)
        
        # 2. Query ChromaDB filtered strictly by doc_id
        query_results = query_documents(
            doc_id=request.doc_id,
            query_embedding=query_embedding,
            n_results=3
        )
        
        documents = query_results.get("documents", [[]])[0]
        ids = query_results.get("ids", [[]])[0]
        
        # Validate documents exist
        if not documents:
            raise HTTPException(
                status_code=404,
                detail=f"No relevant context found in document with doc_id '{request.doc_id}'."
            )
        
        # Structure source data to pass down the stream
        sources: List[Dict[str, Any]] = []
        for chunk_id, text in zip(ids, documents):
            sources.append({
                "id": chunk_id,
                "text": text[:200] + "..." if len(text) > 200 else text  # Send snippet to keep response lightweight
            })
        
        # Combine documents into context
        context = "\n\n---\n\n".join(documents)
        
        # 3. Return StreamingResponse
        return StreamingResponse(
            stream_answer(context, request.message, sources),
            media_type="text/event-stream"
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error querying stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))
