from typing import List


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Splits text into overlapping chunks of a specific character length."""
    chunks = []
    start = 0
    text_len = len(text)
    if text_len == 0:
        return []

    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= text_len:
            break
        start += chunk_size - overlap
    return chunks
