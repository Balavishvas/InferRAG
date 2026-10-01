import re


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 80,
    source: str | None = None,
) -> list[dict]:
    """Split text into overlapping chunks while keeping traceable metadata."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    paragraphs = [p.strip() for p in re.split(r"\\n\\s*\\n", text) if p.strip()]
    words = " ".join(paragraphs).split()
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append({
                "text": chunk,
                "chunk_id": chunk_id,
                "source": source,
                "start_word": start,
                "end_word": end,
            })
            chunk_id += 1

        if end == len(words):
            break
        start = end - overlap

    return chunks
