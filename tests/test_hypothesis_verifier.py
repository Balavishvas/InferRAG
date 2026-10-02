from rag.hypothesis_verifier import (
    _aggregate,
    DIRECT,
    INDIRECT,
    CONTRADICTED,
)


def verification(status, support=None, contradiction=None, claim_index=1):
    return {
        "hypothesis_index": 1,
        "claim_index": claim_index,
        "status": status,
        "supporting_evidence": support or [],
        "contradicting_evidence": contradiction or [],
        "explanation": status,
        "directness": "direct" if status == DIRECT else "indirect",
    }


def test_related_evidence_is_uncertain():
    claims = [{
        "hypothesis_index": 1,
        "claim_index": 1,
        "claim": "Semantic splitting enables RAG.",
        "importance": "core",
    }]
    result = _aggregate(1, claims, [verification(INDIRECT, support=[1])])
    assert result["status"] == "uncertain"
    assert result["directness"] == "indirect"
    assert result["verification_score"] == 0.4


def test_direct_support_is_supported():
    claims = [{
        "hypothesis_index": 1,
        "claim_index": 1,
        "claim": "Semantic splitting improves retrieval.",
        "importance": "core",
    }]
    result = _aggregate(1, claims, [verification(DIRECT, support=[2])])
    assert result["status"] == "supported"
    assert result["directness"] == "direct"
    assert result["verification_score"] == 1.0


def test_contradiction_wins():
    claims = [
        {
            "hypothesis_index": 1,
            "claim_index": 1,
            "claim": "The method improves retrieval.",
            "importance": "core",
        },
        {
            "hypothesis_index": 1,
            "claim_index": 2,
            "claim": "The method is model-agnostic.",
            "importance": "core",
        },
    ]
    result = _aggregate(
        1,
        claims,
        [
            verification(DIRECT, support=[1], claim_index=1),
            verification(CONTRADICTED, contradiction=[2], claim_index=2),
        ],
    )
    assert result["status"] == "contradicted"


def test_missing_verification_is_uncertain():
    claims = [{
        "hypothesis_index": 1,
        "claim_index": 1,
        "claim": "Unknown claim.",
        "importance": "core",
    }]
    result = _aggregate(1, claims, [])
    assert result["status"] == "uncertain"
    assert result["verification_score"] == 0.15
