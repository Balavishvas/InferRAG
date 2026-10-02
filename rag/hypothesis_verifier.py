import json
import re

import requests


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


def verify_hypotheses(
    question: str,
    hypotheses: list[dict],
    contexts: list[dict],
    model: str = "qwen2.5:3b",
    base_url: str = "http://localhost:11434",
) -> list[dict]:
    """Challenge hypotheses against retrieved evidence.

    The verifier looks for both supporting and contradicting evidence.
    It does not treat model-generated plausibility as proof.
    """
    if not hypotheses or not contexts:
        return []

    evidence = "\n\n".join(
        f"[Evidence {i}] chunk_id={item.get('chunk_id')}\n{item['text']}"
        for i, item in enumerate(contexts, 1)
    )
    hypothesis_text = "\n\n".join(
        f"[Hypothesis {i}] {item.get('hypothesis', '')}\n"
        f"Reasoning: {item.get('reasoning', '')}"
        for i, item in enumerate(hypotheses, 1)
    )

    prompt = f"""You are the evidence-verification component of InferRAG.

Challenge each hypothesis using ONLY the supplied evidence.

For every hypothesis:
- identify evidence that supports it
- identify evidence that contradicts it, if any
- distinguish direct evidence from weak/indirect evidence
- classify it as exactly one of: supported, contradicted, uncertain
- explain the classification briefly
- provide a verification score from 0.0 to 1.0 as a heuristic, NOT a probability

Do not invent facts or evidence.
If the evidence is insufficient, use "uncertain".

Return ONLY valid JSON:
{{
  "verifications": [
    {{
      "hypothesis_index": 1,
      "status": "supported",
      "supporting_evidence": [1],
      "contradicting_evidence": [],
      "directness": "direct",
      "explanation": "brief evidence-based explanation",
      "verification_score": 0.0
    }}
  ]
}}

Question:
{question}

Hypotheses:
{hypothesis_text}

Evidence:
{evidence}
"""

    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        },
        timeout=120,
    )
    response.raise_for_status()

    data = _extract_json(response.json().get("response", ""))
    verifications = data.get("verifications", [])
    if not isinstance(verifications, list):
        raise ValueError("Model response contains an invalid verifications list.")

    return verifications
