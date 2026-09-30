from .chunker import chunk_text
from .embeddings import Embedder
from .loader import load_document
from .vector_store import VectorStore

class RAGPipeline:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedder = Embedder(embedding_model)
        self.store = None

    def ingest(self, path: str, chunk_size: int = 500, overlap: int = 80):
        text = load_document(path)
        chunks = chunk_text(text, chunk_size, overlap)
        if not chunks:
            raise ValueError("The document contains no extractable text.")

        embeddings = self.embedder.encode(chunks)
        self.store = VectorStore(embeddings.shape[1])
        self.store.add(embeddings, chunks)
        return len(chunks)

    def retrieve(self, question: str, top_k: int = 5):
        if self.store is None:
            raise RuntimeError("Ingest a document before retrieving.")
        query_embedding = self.embedder.encode([question])[0]
        return self.store.search(query_embedding, top_k)
