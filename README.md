# ChatWithPdf Backend

An enterprise-grade, containerized RAG (Retrieval-Augmented Generation) backend built with **FastAPI**, **ChromaDB**, **SQLite**, and the **Google Gemini API**. This project demonstrates professional GenAI architecture patterns, including modular service layers, database-driven document versioning, metadata catalogs, and structured Server-Sent Events (SSE) streaming.

---

## Live Demo & Testing

**Public Interactive API Docs:** Anyone can test this application live at [http://168.144.95.33/docs](http://168.144.95.33/docs).

You can upload files, retrieve records, and test streaming queries directly from the Swagger UI.

---

## Key Features

*   **Modular Clean Architecture:** Strict separation of API routes, configurations, LLM clients, and business service layers (chunking, extraction, vector store, DB metadata).
*   **Automatic Document Versioning:** Seamlessly tracks uploads of the same filename. It assigns version numbers (e.g., `v1 -> v2`) and flags older versions as `archived` in SQLite while keeping the latest version `active`.
*   **Multi-Document Chunking Profiles:** Accepts a `doc_type` parameter on upload (`standard`, `legal`, `technical`) to resolve semantic chunking profiles (size/overlap) optimized for that document category.
*   **Multi-Document Vector Integration:** Generates embeddings using `gemini-embedding-001` and filters retrieval matches across multiple document IDs using ChromaDB `$in` query filtering.
*   **Context Aggregator:** Groups retrieved chunks dynamically by document name and version, presenting them with clear headers to the LLM to minimize hallucinations.
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
curl http://168.144.95.33/
```
**Response:**
```json
{"status": "healthy", "service": "Chat with PDF Backend"}
```

### 2. Upload a PDF Document
Upload any PDF file. You can optionally specify a `doc_type` (`standard`, `legal`, or `technical`) as form data to adjust chunk configurations:
```bash
curl -X POST "http://168.144.95.33/api/upload" \
  -F "file=@/path/to/your/document.pdf" \
  -F "doc_type=legal"
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
curl http://168.144.95.33/api/documents
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

### 4. Stream RAG Chat Query (Multi-Document)
Initiate an SSE connection to query multiple document IDs:
```bash
curl -X POST "http://168.144.95.33/api/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_ids": ["c2586b3c-32b3-4143-aebc-76e81a678408", "b3901b3c-12c3-4243-bebc-76e81a678409"],
    "message": "What do these documents say about travel expenses?"
  }'
```
**Streamed SSE Output:**
```text
data: {"event": "sources", "data": [{"id": "c2586b3c-32b3-4143-aebc-76e81a678408_chunk_0", "text": "Snippet..."}]}

data: {"event": "token", "text": "Based on the travel policy document..."}

data: {"event": "metrics", "ttft_seconds": 1.258, "total_duration_seconds": 5.156}
```

---

## Design Tradeoffs

### Managed API (Gemini) vs. Self-Hosted LLM
For the **ChatWithPdf** backend, we explicitly chose a managed API layer (**Google Gemini**) over self-hosting an open-weight LLM (e.g., Llama 3 or Mistral via Ollama/vLLM). The core tradeoffs evaluated were:

1. **Resource Constraints:** Hosting an open-weight model requires significant VRAM/GPU resources or a massive RAM footprint on CPU VMs. A basic DigitalOcean droplet (e.g., 1-2 GB RAM / 1 vCPU) cannot host or run self-hosted LLMs reliably. By using Gemini API, our backend droplet operates with a minuscule memory footprint (~30% of 1 GB RAM).
2. **Latency & Cold Starts:** Managed model endpoints have highly optimized hardware setups yielding fast Time-to-First-Token (TTFT) metrics, bypassing VM memory bottleneck limitations.
3. **Operational Overhead:** Gemini API handles concurrency, context scaling, and safety filtering out of the box, letting the application focus strictly on the clean architecture pipeline and database-driven version controls rather than managing infrastructure and model configurations.

