from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        if not documents:
            return []

        pairs = [(query, document["text"]) for document in documents]
        scores = self.model.predict(pairs)

        ranked = []
        for document, score in zip(documents, scores):
            ranked.append({
                **document,
                "rerank_score": float(score),
            })

        ranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )
        return ranked[:top_k]
