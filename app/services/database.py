import sqlite3
import os
from typing import List, Dict, Any
from datetime import datetime
from app.config import settings

DB_PATH = os.path.join(settings.CHROMA_DATA_PATH, "metadata.db")


def init_db():
    """Creates the document metadata table if it does not exist, handles schema migrations locally."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Verify schema versioning migrations
        cursor.execute("PRAGMA table_info(documents)")
        columns = [info[1] for info in cursor.fetchall()]
        if columns and "version" not in columns:
            cursor.execute("DROP TABLE IF EXISTS documents")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                version INTEGER NOT NULL,
                status TEXT NOT NULL,
                uploaded_at TEXT NOT NULL
            )
        """)
        conn.commit()


def get_next_version(filename: str) -> int:
    """Calculates the next version number for a filename."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MAX(version) FROM documents WHERE filename = ?", (filename,)
        )
        row = cursor.fetchone()
        if row and row[0] is not None:
            return row[0] + 1
        return 1


def archive_old_versions(filename: str):
    """Sets all previous versions of a document to 'archived' status."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE documents SET status = 'archived' WHERE filename = ? AND status = 'active'",
            (filename,),
        )
        conn.commit()


def create_document(doc_id: str, filename: str, version: int) -> Dict[str, Any]:
    """Inserts a new active document record into SQLite."""
    uploaded_at = datetime.utcnow().isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO documents (id, filename, version, status, uploaded_at) VALUES (?, ?, ?, ?, ?)",
            (doc_id, filename, version, "active", uploaded_at),
        )
        conn.commit()
    return {
        "id": doc_id,
        "filename": filename,
        "version": version,
        "status": "active",
        "uploaded_at": uploaded_at,
    }


def delete_document_record(doc_id: str) -> bool:
    """Deletes a document metadata record from SQLite by its unique ID. Returns True if a record was deleted."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_all_documents() -> List[Dict[str, Any]]:
    """Fetches all documents ordered by upload date."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, filename, version, status, uploaded_at FROM documents ORDER BY uploaded_at DESC"
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_active_document_id(filename: str) -> str:
    """Gets the active document UUID for a filename."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM documents WHERE filename = ? AND status = 'active'",
            (filename,),
        )
        row = cursor.fetchone()
        return row[0] if row else None
