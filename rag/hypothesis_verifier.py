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
- Use ONLY information already expressed by the hypothesis and question.
- Do not invent new benefits, mechanisms, metrics, causes, or conclusions.
- Preserve the hypothesis meaning; do not make a claim stronger than the hypothesis.
- Prefer 1-3 atomic claims per hypothesis.
- A claim must be independently verifiable from evidence.
- Mark each claim as "core" or "supporting".
- Core claims are necessary for the hypothesis to be true.
- Supporting claims explain or qualify the hypothesis but are not necessary.

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
    seen = set()
    for item in claims:
        try:
            h = int(item.get("hypothesis_index"))
            c = int(item.get("claim_index"))
        except (TypeError, ValueError):
            continue

        claim = str(item.get("claim", "")).strip()
        if not claim or h < 1 or h > len(hypotheses):
            continue

        # Prevent duplicate claims from consuming extra verification calls.
        key = (h, claim.lower())
        if key in seen:
            continue
        seen.add(key)

        importance = item.get("importance", "core")
        if importance not in {"core", "supporting"}:
            importance = "core"

        cleaned.append({
            "hypothesis_index": h,
            "claim_index": c,
            "claim": claim,
            "importance": importance,
        })

    # Keep claim indices contiguous for deterministic evidence lookup/output.
    by_hypothesis = {}
    for item in cleaned:
        by_hypothesis.setdefault(item["hypothesis_index"], []).append(item)

    normalized = []
    for h, items in by_hypothesis.items():
        for index, item in enumerate(items, 1):
            item["claim_index"] = index
            normalized.append(item)

    return normalized

def _clean_explanation(value) -> str:
    """Remove common small-model placeholder/garbage explanations."""
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    placeholders = {
        "brief reason",
        "brief reason brief reason",
        "reason",
        "n/a",
        "none",
    }
    if lowered in placeholders:
        return ""
    return text


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

        prompt = f"""You are InferRAG's evidence verifier.

Verify ONE claim against ONLY the supplied evidence.

Question: {question}
Hypothesis: {hypothesis}
Claim: {claim['claim']}

Evidence:
{evidence_text}

Choose exactly one:
- direct_support: the evidence explicitly states the claim OR logically entails
  the same proposition using clear paraphrase/synonym/terminology differences.
- indirect_support: the evidence is relevant and suggests the claim, but does
  not establish the proposition.
- contradicted: the evidence explicitly conflicts with the claim.
- insufficient: the evidence cannot establish or contradict the claim.

Important:
- Do NOT reject direct support merely because the source uses different words.
  Example: "semantic splitting" can be directly supported by "breaking a query
  into sub-topics" when the surrounding evidence describes that exact operation.
- Do NOT treat merely related concepts as direct support.
- Do NOT add outside knowledge.
- Absence of evidence is not contradiction.
- A direct_support or contradicted result MUST cite at least one evidence item.
- Be conservative about benefits, causality, metrics, and performance claims.

Return ONLY this JSON object:
{{"status":"insufficient","supporting_evidence":[],"contradicting_evidence":[],"explanation":"brief reason"}}
"""

        data = {}
        try:
            data = _ollama_json(prompt, model, base_url)
        except (requests.RequestException, ValueError, json.JSONDecodeError):
            data = {}

        allowed = {DIRECT, INDIRECT, CONTRADICTED, INSUFFICIENT}

        # Retry once with a shorter prompt. This specifically targets Qwen's
        # occasional empty/partial JSON responses.
        status = data.get("status") if isinstance(data, dict) else None
        if status not in allowed:
            retry_prompt = f"""Check this claim using only the evidence.

CLAIM:
{claim['claim']}

EVIDENCE:
{evidence_text}

Rules:
direct_support = evidence explicitly states or logically entails the claim,
including clear paraphrases.
indirect_support = relevant but not enough.
contradicted = evidence explicitly conflicts.
insufficient = cannot tell.
No outside knowledge. Direct/contradicted requires cited evidence.

Return ONLY JSON:
{{"status":"insufficient","supporting_evidence":[],"contradicting_evidence":[],"explanation":"reason"}}
"""
            try:
                data = _ollama_json(retry_prompt, model, base_url)
            except (requests.RequestException, ValueError, json.JSONDecodeError):
                data = {}

        status = data.get("status", INSUFFICIENT) if isinstance(data, dict) else INSUFFICIENT
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

        supporting = _numbers(data.get("supporting_evidence", []))
        contradicting = _numbers(data.get("contradicting_evidence", []))

        # Never allow a model to claim direct/contradicted without an actual
        # cited passage. This is a deterministic safety rail.
        if status == DIRECT and not supporting:
            status = INSUFFICIENT
        elif status == CONTRADICTED and not contradicting:
            status = INSUFFICIENT

        explanation = _clean_explanation(
            data.get("explanation", "") if isinstance(data, dict) else ""
        )
        if not explanation:
            if status == DIRECT:
                explanation = "The cited evidence directly supports the claim."
            elif status == CONTRADICTED:
                explanation = "The cited evidence contradicts the claim."
            elif status == INDIRECT:
                explanation = "The cited evidence is relevant but does not fully establish the claim."
            else:
                explanation = "The available evidence is insufficient to verify the claim."

        results.append({
            "hypothesis_index": hypothesis_index,
            "claim_index": claim_index,
            "status": status,
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting,
            "explanation": explanation,
            "directness": (
                "direct" if status == DIRECT
                else "contradicted" if status == CONTRADICTED
                else "indirect" if status == INDIRECT
                else "insufficient"
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

    if has_contradiction:
        directness = "contradicted"
    elif claims and all_core_direct:
        directness = "direct"
    elif any(
        item["verification"].get("status") == INDIRECT
        for item in claim_results
    ):
        directness = "indirect"
    else:
        directness = "insufficient"

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
