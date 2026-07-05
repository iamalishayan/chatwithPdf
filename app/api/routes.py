import logging
import uuid
import asyncio
from typing import List, Dict
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from app.config import settings
from app.models.schemas import (
    ChatRequest,
    UploadResponse,
    DocumentMetadata,
    GenericMessageResponse,
    ChunkListResponse,
    ChunkItem,
    DocumentType,
)
from app.services.pdf_extractor import extract_text_from_pdf
from app.services.chunking import chunk_text
from app.services.vector_store import (
    add_documents,
    query_documents,
    delete_document_vectors,
    get_document_chunks,
)
from app.services.database import (
    create_document,
    get_all_documents,
    get_next_version,
    archive_old_versions,
    delete_document_record,
    get_documents_metadata,
)
from app.llm.gemini_client import get_embedding, stream_answer

logger = logging.getLogger("routes")
router = APIRouter(prefix="/api")


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    doc_type: DocumentType = Form(DocumentType.STANDARD),
):
    """Ingests a PDF, resolves its semantic chunking profile, archives older versions, chunks, embeds, and stores it."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Resolve chunking profiles safely from server settings config
    profile = settings.CHUNK_PROFILES.get(
        doc_type.value, settings.CHUNK_PROFILES["standard"]
    )
    chunk_size = profile["size"]
    chunk_overlap = profile["overlap"]

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

        # Chunk text based on profile settings
        chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            raise HTTPException(
                status_code=400, detail="The PDF file contains no readable text."
            )

        # Embed chunks concurrently
        embeddings = await asyncio.gather(*(get_embedding(chunk) for chunk in chunks))

        # Prepare Metadata for ChromaDB
        metadatas = [{"doc_id": doc_id} for _ in chunks]

        # Store in ChromaDB
        add_documents(
            doc_id=doc_id,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        # Store metadata in SQLite database
        create_document(doc_id=doc_id, filename=filename, version=version)

        logger.info(
            f"Successfully processed and stored document version. ID: {doc_id}, Name: {filename}, Version: {version}, Profile: {doc_type}"
        )

        return UploadResponse(
            message=f"Successfully processed {len(chunks)} chunks using '{doc_type}' profile (size={chunk_size}, overlap={chunk_overlap}).",
            doc_id=doc_id,
            filename=filename,
            version=version,
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


@router.get("/documents/{doc_id}/chunks", response_model=ChunkListResponse)
async def list_document_chunks(doc_id: str):
    """Retrieves all vector chunks and associated text segments matching the document ID."""
    try:
        chunks_data = get_document_chunks(doc_id)
        if not chunks_data:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for document ID '{doc_id}'. Check if it exists.",
            )

        chunks = [
            ChunkItem(id=c["id"], text=c["text"], metadata=c["metadata"])
            for c in chunks_data
        ]
        return ChunkListResponse(chunks=chunks)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve document chunks."
        )


@router.delete("/documents/{doc_id}", response_model=GenericMessageResponse)
async def delete_document(doc_id: str):
    """Deletes a document from the system, wiping SQLite records and associated ChromaDB vectors."""
    try:
        # 1. Delete from SQLite
        deleted_from_db = delete_document_record(doc_id)
        if not deleted_from_db:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID '{doc_id}' not found in registry.",
            )

        # 2. Delete vectors from ChromaDB
        delete_document_vectors(doc_id)

        return GenericMessageResponse(
            message=f"Successfully deleted document '{doc_id}' from the system."
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to delete document from the system."
        )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Queries ChromaDB using the question's embedding, retrieves context, and streams the answer."""
    try:
        # 1. Embed query
        query_embedding = await get_embedding(request.message)

        # 2. Query ChromaDB filtered strictly by doc_ids (increased default top_k search limit for multi-doc candidate pools)
        query_results = query_documents(
            doc_ids=request.doc_ids,
            query_embedding=query_embedding,
            n_results=6,
        )

        documents = query_results.get("documents", [[]])[0]
        ids = query_results.get("ids", [[]])[0]
        metadatas = query_results.get("metadatas", [[]])[0]

        # Validate documents exist
        if not documents:
            raise HTTPException(
                status_code=404,
                detail=f"No relevant context found in documents with IDs {request.doc_ids}.",
            )

        # Resolve document metadata from SQLite for context aggregation
        metadata_map = get_documents_metadata(request.doc_ids)

        # Group chunks by document ID (ContextAggregator pattern)
        grouped_chunks: Dict[str, List[str]] = {}
        for chunk_id, text, metadata in zip(ids, documents, metadatas):
            doc_id = metadata.get("doc_id")
            if doc_id not in grouped_chunks:
                grouped_chunks[doc_id] = []
            grouped_chunks[doc_id].append(text)

        # Build clean formatted context string
        context_parts = []
        for doc_id, chunk_texts in grouped_chunks.items():
            doc_meta = metadata_map.get(doc_id, {})
            filename = doc_meta.get("filename", "Unknown Document")
            version = doc_meta.get("version", "?")
            header = f"DOCUMENT: {filename} (Version {version})"
            divider = "-" * len(header)
            grouped_text = "\n".join(f"* {t}" for t in chunk_texts)
            context_parts.append(f"{header}\n{divider}\n{grouped_text}")
        context = "\n\n".join(context_parts)

        # Structure source data to pass down the stream
        sources = []
        for chunk_id, text in zip(ids, documents):
            sources.append(
                {
                    "id": chunk_id,
                    "text": (text[:200] + "..." if len(text) > 200 else text),
                }
            )

        # 3. Return StreamingResponse
        return StreamingResponse(
            stream_answer(context, request.message, sources),
            media_type="text/event-stream",
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error querying stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))
