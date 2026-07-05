# ChatWithPdf Backend

An enterprise-grade, containerized RAG (Retrieval-Augmented Generation) backend built with **FastAPI**, **ChromaDB**, **SQLite**, and the **Google Gemini API**. This project demonstrates professional GenAI architecture patterns, including modular service layers, database-driven document versioning, metadata catalogs, and structured Server-Sent Events (SSE) streaming.

---

## Key Features

*   **Modular Clean Architecture:** Strict separation of API routes, configurations, LLM clients, and business service layers (chunking, extraction, vector store, DB metadata).
*   **Automatic Document Versioning:** Seamlessly tracks uploads of the same filename. It assigns version numbers (e.g., `v1 -> v2`) and flags older versions as `archived` in SQLite while keeping the latest version `active`.
*   **ChromaDB Vector Integration:** Generates embeddings using `gemini-embedding-001` and filters retrieval matches strictly by version IDs.
*   **Rich SSE Stream:** Streams responses token-by-token with structured event packages:
    *   `sources`: Citations of the referenced text chunks.
    *   `token`: Generated text tokens.
    *   `metrics`: Time-to-First-Token (TTFT) and total generation duration.

---

## Directory Structure

```text
.
├── app
│   ├── api
│   │   └── routes.py          # FastAPI route handlers (/upload, /documents, /chat/stream)
│   ├── config.py              # Environment configuration & Pydantic settings
│   ├── llm
│   │   └── gemini_client.py   # Client for embeddings and streaming responses
│   ├── models
│   │   └── schemas.py         # Pydantic validation schemas
│   └── services
│       ├── chunking.py        # Overlapping text chunker
│       ├── database.py        # SQLite metadata manager (versioning & archival)
│       ├── pdf_extractor.py   # In-memory PDF text parser
│       └── vector_store.py    # ChromaDB database layer
├── Dockerfile                 # Slim multi-stage container file
├── docker-compose.yml         # Container compose setup with local volumes
├── requirements.txt           # Project python requirements
├── main.py                    # App entry point
└── README.md
```

---

## Technology Stack

*   **Web Framework:** FastAPI (Uvicorn ASGI)
*   **Vector Database:** ChromaDB (Cosine similarity space)
*   **Relational Database:** SQLite (Metadata & state cataloging)
*   **LLM API:** Google Gemini REST API (`gemini-2.5-flash` for streaming, `gemini-embedding-001` for embeddings)
*   **Containerization:** Docker & Docker Compose

---

## Local Setup & Running

### Prerequisites
*   Docker & Docker Compose installed.
*   A Gemini API Key.

### 1. Configure Environment
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
PORT=8000
HOST=0.0.0.0
```

### 2. Build & Launch Containers
Spin up the service and local database volume:
```bash
docker compose up --build -d
```
The server will start at `http://localhost:8000`. You can access the interactive OpenAPI docs at `http://localhost:8000/docs`.

---

## Verification & Endpoint Usage

### 1. Health Check
```bash
curl http://localhost:8000/
```
**Response:**
```json
{"status": "healthy", "service": "Chat with PDF Backend"}
```

### 2. Upload a PDF Document
Upload any PDF file. The API automatically assigns a UUID and sets `version=1` (or increments it if the file already exists):
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@/path/to/your/document.pdf"
```
**Response:**
```json
{
  "message": "Successfully processed 12 chunks.",
  "doc_id": "c2586b3c-32b3-4143-aebc-76e81a678408",
  "filename": "document.pdf",
  "version": 1
}
```

### 3. List Stored Documents
Fetch the metadata catalogue showing version histories and status states:
```bash
curl http://localhost:8000/api/documents
```
**Response:**
```json
[
  {
    "id": "c2586b3c-32b3-4143-aebc-76e81a678408",
    "filename": "document.pdf",
    "version": 1,
    "status": "active",
    "uploaded_at": "2026-07-05T11:03:35.655496"
  }
]
```

### 4. Stream RAG Chat Query
Initiate an SSE connection to query the specific document ID:
```bash
curl -X POST "http://localhost:8000/api/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "c2586b3c-32b3-4143-aebc-76e81a678408",
    "message": "What is this document about?"
  }'
```
**Streamed SSE Output:**
```text
data: {"event": "sources", "data": [{"id": "c2586b3c-32b3-4143-aebc-76e81a678408_chunk_0", "text": "Snippet..."}]}

data: {"event": "token", "text": "This document is..."}

data: {"event": "metrics", "ttft_seconds": 1.258, "total_duration_seconds": 5.156}
```
