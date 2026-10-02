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


def test_single_hypothesis_verification_filters_invalid_claims(monkeypatch):
    from rag import hypothesis_verifier as hv

    def fake_ollama(prompt, model, base_url):
        return {
            "verifications": [
                {
                    "claim_index": 1,
                    "status": DIRECT,
                    "supporting_evidence": [1],
                    "contradicting_evidence": [],
                    "explanation": "The evidence explicitly states the claim.",
                },
                {
                    "claim_index": 99,
                    "status": DIRECT,
                    "supporting_evidence": [1],
                    "contradicting_evidence": [],
                    "explanation": "Invalid claim.",
                },
            ]
        }

    monkeypatch.setattr(hv, "_ollama_json", fake_ollama)

    result = hv._verify_hypothesis_claims(
        question="Why?",
        hypothesis_index=1,
        hypothesis="The method helps.",
        claims=[{
            "hypothesis_index": 1,
            "claim_index": 1,
            "claim": "The method helps.",
            "importance": "core",
        }],
        evidence_by_claim={
            "1:1": [{"chunk_id": "chunk-1", "text": "The method helps."}]
        },
        model="qwen2.5:3b",
        base_url="http://localhost:11434",
    )

    assert len(result) == 1
    assert result[0]["claim_index"] == 1
    assert result[0]["status"] == DIRECT


def test_single_hypothesis_verification_failure_returns_empty(monkeypatch):
    from rag import hypothesis_verifier as hv

    def fake_ollama(prompt, model, base_url):
        raise ValueError("bad model JSON")

    monkeypatch.setattr(hv, "_ollama_json", fake_ollama)

    result = hv._verify_hypothesis_claims(
        question="Why?",
        hypothesis_index=1,
        hypothesis="Unknown.",
        claims=[{
            "hypothesis_index": 1,
            "claim_index": 1,
            "claim": "Unknown.",
            "importance": "core",
        }],
        evidence_by_claim={"1:1": []},
        model="qwen2.5:3b",
        base_url="http://localhost:11434",
    )

    assert result == []
