from pathlib import Path
import json
import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: list[str] = []

    def add(self, embeddings, documents: list[str]):
        vectors = np.asarray(embeddings, dtype="float32")
        self.index.add(vectors)
        self.documents.extend(documents)

    def search(self, query_embedding, top_k: int = 5):
        vector = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        scores, indices = self.index.search(vector, min(top_k, len(self.documents)))
        return [
            {"text": self.documents[i], "score": float(score)}
            for score, i in zip(scores[0], indices[0])
            if i >= 0
        ]

    def save(self, directory: str):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        (path / "documents.json").write_text(
            json.dumps(self.documents, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
