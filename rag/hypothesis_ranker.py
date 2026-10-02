def rank_hypotheses(
    hypotheses: list[dict],
    verifications: list[dict],
) -> list[dict]:
    """Combine generation and verification metadata for final ordering.

    Verification status is the primary signal. Scores are only heuristic
    signals and must not be interpreted as calibrated probabilities.
    """
    verification_by_index = {
        int(item.get("hypothesis_index", 0)): item
        for item in verifications
        if item.get("hypothesis_index") is not None
    }

    status_weight = {
        "supported": 3,
        "uncertain": 2,
        "contradicted": 1,
    }

    ranked = []
    for index, hypothesis in enumerate(hypotheses, 1):
        verification = verification_by_index.get(index, {})
        score = float(verification.get("verification_score", 0.0) or 0.0)
        status = verification.get("status", "uncertain")

        ranked.append({
            **hypothesis,
            "verification": verification,
            "_rank": status_weight.get(status, 0) + score,
        })

    ranked.sort(key=lambda item: item["_rank"], reverse=True)

    for item in ranked:
        item.pop("_rank", None)

    return ranked
