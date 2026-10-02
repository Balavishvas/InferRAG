from collections.abc import Iterable


def _relevant_ids(results: list[dict]) -> list[int]:
    return [int(item["chunk_id"]) for item in results]


def precision_at_k(results: list[dict], relevant_chunk_ids: set[int], k: int) -> float:
    top = _relevant_ids(results[:k])
    if not top:
        return 0.0
    return sum(chunk_id in relevant_chunk_ids for chunk_id in top) / len(top)


def recall_at_k(
    results: list[dict],
    relevant_chunk_ids: set[int],
    k: int,
) -> float:
    if not relevant_chunk_ids:
        return 0.0
    top = _relevant_ids(results[:k])
    return sum(chunk_id in relevant_chunk_ids for chunk_id in top) / len(relevant_chunk_ids)


def reciprocal_rank(results: list[dict], relevant_chunk_ids: set[int]) -> float:
    for rank, chunk_id in enumerate(_relevant_ids(results), 1):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    results: list[dict],
    relevant_chunk_ids: Iterable[int],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    relevant = {int(chunk_id) for chunk_id in relevant_chunk_ids}

    return {
        "precision": {k: precision_at_k(results, relevant, k) for k in ks},
        "recall": {k: recall_at_k(results, relevant, k) for k in ks},
        "mrr": reciprocal_rank(results, relevant),
    }
