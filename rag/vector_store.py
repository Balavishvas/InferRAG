from pathlib import Path
import json
import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: list[dict] = []

    def add(self, embeddings, documents: list[dict]):
        vectors = np.asarray(embeddings, dtype="float32")
        self.index.add(vectors)
        self.documents.extend(documents)

    def search(self, query_embedding, top_k: int = 5):
        if not self.documents:
            return []

        vector = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        scores, indices = self.index.search(vector, min(top_k, len(self.documents)))
        return [
            {**self.documents[i], "score": float(score)}
            for score, i in zip(scores[0], indices[0])
            if i >= 0
        ]

    def save(self, directory: str, metadata: dict | None = None):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        (path / "documents.json").write_text(
            json.dumps(self.documents, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if metadata is not None:
            (path / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    @classmethod
    def load(cls, directory: str):
        path = Path(directory)
        index = faiss.read_index(str(path / "index.faiss"))
        documents = json.loads(
            (path / "documents.json").read_text(encoding="utf-8")
        )
        store = cls(index.d)
        store.index = index
        store.documents = documents
        return store

    @staticmethod
    def load_metadata(directory: str) -> dict:
        path = Path(directory) / "metadata.json"
        return json.loads(path.read_text(encoding="utf-8"))
