import io
import logging
from PyPDF2 import PdfReader

logger = logging.getLogger("pdf-extractor")

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Reads a PDF in memory and extracts all text page by page."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        full_text = []
        for i, page in enumerate(reader.pages): # i can be removed......
            extracted = page.extract_text()
            if extracted:
                full_text.append(extracted)
        return "\n\n".join(full_text)
    except Exception as e:
        logger.error(f"Failed to parse PDF: {e}")
        raise ValueError(f"Invalid or corrupted PDF file: {str(e)}")
