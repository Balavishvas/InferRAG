from pathlib import Path
import json

from .chunker import chunk_text
from .embeddings import Embedder
from .loader import load_document
from .vector_store import VectorStore


class RAGPipeline:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_root: str = "data/index",
    ):
        self.embedder = Embedder(embedding_model)
        self.embedding_model = embedding_model
        self.index_root = Path(index_root)
        self.store = None
        self.document_path = None

    def _index_dir(self, path: str) -> Path:
        return self.index_root / Path(path).stem

    def _metadata_matches(self, directory: Path, path: str, chunk_size: int, overlap: int) -> bool:
        try:
            metadata = VectorStore.load_metadata(str(directory))
        except (FileNotFoundError, json.JSONDecodeError):
            return False

        file_path = Path(path)
        return (
            metadata.get("source") == str(file_path)
            and metadata.get("modified_time") == file_path.stat().st_mtime
            and metadata.get("chunk_size") == chunk_size
            and metadata.get("overlap") == overlap
            and metadata.get("embedding_model") == self.embedding_model
        )

    def ingest(self, path: str, chunk_size: int = 500, overlap: int = 80):
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        self.document_path = str(file_path)
        index_dir = self._index_dir(path)

        if (
            (index_dir / "index.faiss").exists()
            and (index_dir / "documents.json").exists()
            and (index_dir / "metadata.json").exists()
            and self._metadata_matches(index_dir, path, chunk_size, overlap)
        ):
            self.store = VectorStore.load(str(index_dir))
            return len(self.store.documents)

        text = load_document(path)
        chunks = chunk_text(text, chunk_size, overlap, source=file_path.name)
        if not chunks:
            raise ValueError("The document contains no extractable text.")

        embeddings = self.embedder.encode([chunk["text"] for chunk in chunks])
        self.store = VectorStore(embeddings.shape[1])
        self.store.add(embeddings, chunks)
        self.store.save(
            str(index_dir),
            metadata={
                "source": str(file_path),
                "modified_time": file_path.stat().st_mtime,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "embedding_model": self.embedding_model,
                "chunk_count": len(chunks),
            },
        )
        return len(chunks)

    def retrieve(self, question: str, top_k: int = 5):
        if self.store is None:
            raise RuntimeError("Ingest a document before retrieving.")
        query_embedding = self.embedder.encode([question])[0]
        return self.store.search(query_embedding, top_k)
