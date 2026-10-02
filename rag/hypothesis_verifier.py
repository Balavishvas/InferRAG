import json
import re

import requests


DIRECT = "direct_support"
INDIRECT = "indirect_support"
CONTRADICTED = "contradicted"
INSUFFICIENT = "insufficient"

_STATUS_WEIGHT = {
    DIRECT: 1.0,
    INDIRECT: 0.4,
    INSUFFICIENT: 0.15,
    CONTRADICTED: 0.0,
}


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("The model did not return valid JSON.")

    return json.loads(match.group(0))


def _ollama_json(prompt: str, model: str, base_url: str) -> dict:
    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        },
        timeout=600,
    )
    response.raise_for_status()
    return _extract_json(response.json().get("response", ""))


def _extract_claims(
    question: str,
    hypotheses: list[dict],
    model: str,
    base_url: str,
) -> list[dict]:
    hypothesis_text = "\n\n".join(
        f"[Hypothesis {i}] {item.get('hypothesis', '')}"
        for i, item in enumerate(hypotheses, 1)
    )

    prompt = f"""You are the claim-decomposition component of InferRAG.

Break each hypothesis into the smallest factual or inferential claims that
must be checked against the source document.

Rules:
- Use ONLY the wording of each hypothesis and the question.
- Do not add outside facts.
- Prefer 1-3 atomic claims per hypothesis.
- A claim should be independently verifiable from evidence.
- Preserve the hypothesis meaning.
- Mark each claim as "core" or "supporting". Core claims are necessary for
  the hypothesis to be true.

Return ONLY valid JSON:
{{
  "claims": [
    {{
      "hypothesis_index": 1,
      "claim_index": 1,
      "claim": "atomic claim",
      "importance": "core"
    }}
  ]
}}

Question:
{question}

Hypotheses:
{hypothesis_text}
"""

    data = _ollama_json(prompt, model, base_url)
    claims = data.get("claims", [])
    if not isinstance(claims, list):
        raise ValueError("Model response contains an invalid claims list.")

    cleaned = []
    for item in claims:
        try:
            h = int(item.get("hypothesis_index"))
            c = int(item.get("claim_index"))
        except (TypeError, ValueError):
            continue

        claim = str(item.get("claim", "")).strip()
        if not claim or h < 1 or h > len(hypotheses):
            continue

        importance = item.get("importance", "core")
        if importance not in {"core", "supporting"}:
            importance = "core"

        cleaned.append({
            "hypothesis_index": h,
            "claim_index": c,
            "claim": claim,
            "importance": importance,
        })

    return cleaned


def _verify_hypothesis_claims(
    question: str,
    hypothesis_index: int,
    hypothesis: str,
    claims: list[dict],
    evidence_by_claim: dict[str, list[dict]],
    model: str,
    base_url: str,
) -> list[dict]:
    """Verify each claim with a tiny one-claim-at-a-time model call."""
    results = []

    for claim in claims:
        claim_index = int(claim["claim_index"])
        key = f"{hypothesis_index}:{claim_index}"
        evidence = evidence_by_claim.get(key, [])

        evidence_text = "\n\n".join(
            f"[Evidence {i}] chunk_id={item.get('chunk_id')}\n{item['text']}"
            for i, item in enumerate(evidence, 1)
        ) or "[No evidence retrieved]"

        prompt = f"""You verify ONE claim for InferRAG.

Question: {question}
Hypothesis: {hypothesis}
Claim: {claim['claim']}

Evidence:
{evidence_text}

Choose exactly one status:
direct_support = evidence explicitly states the claim or logically entails it.
indirect_support = evidence is related but does not establish the claim.
contradicted = evidence explicitly conflicts with the claim.
insufficient = evidence is not enough either way.

Rules:
- Do not use outside knowledge.
- Topic overlap is not direct support.
- Absence of evidence is not contradiction.
- Be conservative.
- Return ONLY this JSON object and nothing else:
{{"status":"insufficient","supporting_evidence":[],"contradicting_evidence":[],"explanation":"brief reason"}}
"""

        try:
            data = _ollama_json(prompt, model, base_url)
        except (requests.RequestException, ValueError, json.JSONDecodeError):
            data = {}

        allowed = {DIRECT, INDIRECT, CONTRADICTED, INSUFFICIENT}
        status = data.get("status", INSUFFICIENT)
        if status not in allowed:
            status = INSUFFICIENT

        def _numbers(value):
            if not isinstance(value, list):
                return []
            out = []
            for n in value:
                try:
                    n = int(n)
                except (TypeError, ValueError):
                    continue
                if 0 < n <= len(evidence):
                    out.append(n)
            return sorted(set(out))

        explanation = str(data.get("explanation", "")).strip()
        if not explanation:
            explanation = (
                "No verification was returned for this claim."
                if not data
                else "The verifier did not provide a usable explanation."
            )

        results.append({
            "hypothesis_index": hypothesis_index,
            "claim_index": claim_index,
            "status": status,
            "supporting_evidence": _numbers(data.get("supporting_evidence", [])),
            "contradicting_evidence": _numbers(
                data.get("contradicting_evidence", [])
            ),
            "explanation": explanation,
            "directness": (
                "direct" if status == DIRECT
                else "contradicted" if status == CONTRADICTED
                else "indirect"
            ),
        })

    return results


def _verify_claims(
    question: str,
    hypotheses: list[dict],
    claims: list[dict],
    evidence_by_claim: dict[str, list[dict]],
    model: str,
    base_url: str,
) -> list[dict]:
    """Verify each hypothesis separately for a smaller, more reliable prompt."""
    all_verifications = []

    for hypothesis_index, hypothesis in enumerate(hypotheses, 1):
        hypothesis_claims = [
            claim for claim in claims
            if claim["hypothesis_index"] == hypothesis_index
        ]
        if not hypothesis_claims:
            continue

        all_verifications.extend(
            _verify_hypothesis_claims(
                question,
                hypothesis_index,
                str(hypothesis.get("hypothesis", "")),
                hypothesis_claims,
                evidence_by_claim,
                model,
                base_url,
            )
        )

    return all_verifications


def _aggregate(
    hypothesis_index: int,
    claims: list[dict],
    verifications: list[dict],
) -> dict:
    verification_by_key = {
        (int(item["hypothesis_index"]), int(item["claim_index"])): item
        for item in verifications
    }

    claim_results = []
    scores = []
    has_contradiction = False
    all_core_direct = True
    direct_evidence = []
    indirect_evidence = []
    contradicting_evidence = []

    for claim in claims:
        key = (hypothesis_index, int(claim["claim_index"]))
        verification = verification_by_key.get(key, {
            "status": INSUFFICIENT,
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "explanation": "No verification was returned for this claim.",
            "directness": "unknown",
        })

        status = verification["status"]
        scores.append(_STATUS_WEIGHT[status])

        if status == CONTRADICTED:
            has_contradiction = True
        if claim["importance"] == "core" and status != DIRECT:
            all_core_direct = False

        if status == DIRECT:
            direct_evidence.extend(verification.get("supporting_evidence", []))
        elif status == INDIRECT:
            indirect_evidence.extend(verification.get("supporting_evidence", []))
        if status == CONTRADICTED:
            contradicting_evidence.extend(
                verification.get("contradicting_evidence", [])
            )

        claim_results.append({
            **claim,
            "verification": verification,
        })

    if has_contradiction:
        status = "contradicted"
    elif claims and all_core_direct:
        status = "supported"
    else:
        status = "uncertain"

    score = round(sum(scores) / len(scores), 3) if scores else 0.0

    explanations = [
        item["verification"].get("explanation", "")
        for item in claim_results
        if item["verification"].get("explanation")
    ]

    directness = "direct" if all_core_direct and claims else (
        "contradicted" if has_contradiction else "indirect"
    )

    return {
        "hypothesis_index": hypothesis_index,
        "status": status,
        "supporting_evidence": sorted(set(direct_evidence)),
        "indirect_evidence": sorted(set(indirect_evidence)),
        "contradicting_evidence": sorted(set(contradicting_evidence)),
        "directness": directness,
        "explanation": " ".join(explanations),
        "verification_score": score,
        "claim_results": claim_results,
    }


def verify_hypotheses(
    question: str,
    hypotheses: list[dict],
    rag_pipeline,
    model: str = "qwen2.5:3b",
    base_url: str = "http://localhost:11434",
    claim_candidate_k: int = 10,
    claim_top_k: int = 3,
) -> list[dict]:
    """V4 claim-level verification with claim-specific retrieval.

    The LLM never decides that a generic related passage is direct support.
    Claims are retrieved independently, verified independently, and then
    aggregated conservatively into a hypothesis-level result.
    """
    if not hypotheses:
        return []

    claims = _extract_claims(question, hypotheses, model, base_url)
    if not claims:
        return [
            {
                "hypothesis_index": i,
                "status": "uncertain",
                "supporting_evidence": [],
                "indirect_evidence": [],
                "contradicting_evidence": [],
                "directness": "unknown",
                "explanation": "No verifiable claims were extracted.",
                "verification_score": 0.0,
                "claim_results": [],
            }
            for i in range(1, len(hypotheses) + 1)
        ]

    evidence_by_claim = {}
    for claim in claims:
        key = f"{claim['hypothesis_index']}:{claim['claim_index']}"
        evidence_by_claim[key] = rag_pipeline.retrieve(
            claim["claim"],
            top_k=claim_top_k,
            candidate_k=claim_candidate_k,
        )

    verifications = _verify_claims(
        question,
        hypotheses,
        claims,
        evidence_by_claim,
        model,
        base_url,
    )

    return [
        _aggregate(i, [c for c in claims if c["hypothesis_index"] == i], verifications)
        for i in range(1, len(hypotheses) + 1)
    ]
