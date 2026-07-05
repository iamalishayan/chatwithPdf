import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.services.database import init_db

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# Initialize SQLite metadata database
try:
    init_db()
    logger.info("Database initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize metadata database: {e}")

# Initialize FastAPI App
app = FastAPI(
    title="Chat with PDF Backend",
    description="Enterprise-grade local backend for document ingestion and retrieval using ChromaDB and Gemini.",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.get("/")
def read_root():
    return {"status": "healthy", "service": "Chat with PDF Backend"}
