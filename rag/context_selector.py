def select_context(
    results: list[dict],
    max_chunks: int = 5,
    max_characters: int = 12000,
) -> list[dict]:
    """
    Select the highest-ranked evidence that fits within a character budget.

    The reranker has already ordered the candidates, so this stage focuses on
    controlling the amount of context sent to the LLM.
    """
    selected = []
    used = 0

    for result in results:
        if len(selected) >= max_chunks:
            break

        text = result["text"].strip()
        if not text:
            continue

        extra = len(text) + (2 if selected else 0)

        if selected and used + extra > max_characters:
            continue

        if not selected and len(text) > max_characters:
            text = text[:max_characters]

        item = {**result, "text": text}
        selected.append(item)
        used += len(text) + (2 if len(selected) > 1 else 0)

    return selected
