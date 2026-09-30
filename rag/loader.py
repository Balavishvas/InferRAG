from pathlib import Path
import fitz

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

def load_document(path: str) -> str:
    file_path = Path(path)
    if file_path.suffix.lower() in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".pdf":
        with fitz.open(file_path) as doc:
            return "\n".join(page.get_text() for page in doc)
    raise ValueError(f"Unsupported file type: {file_path.suffix}")
