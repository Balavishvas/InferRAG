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


def generate_hypotheses(
    question: str,
    contexts: list[dict],
    model: str = "qwen2.5:3b",
    base_url: str = "http://localhost:11434",
    max_hypotheses: int = 3,
) -> list[dict]:
    """Generate possible explanations from retrieved evidence.

    These are hypotheses, not verified conclusions. Evidence verification is
    intentionally reserved for the next reasoning stage.
    """
    if not contexts:
        return []

    evidence = "\n\n".join(
        f"[Evidence {i}] chunk_id={item.get('chunk_id')}\n{item['text']}"
        for i, item in enumerate(contexts, 1)
    )

    prompt = f"""You are the hypothesis-generation component of InferRAG.

Your task is to propose possible explanations for the user's question using
ONLY the supplied evidence.

Important:
- A hypothesis is a possible explanation, not a fact.
- Do not invent evidence.
- Do not claim that a hypothesis is proven.
- Keep facts from the evidence separate from inference.
- Generate up to {max_hypotheses} distinct hypotheses.
- For each hypothesis, list the evidence numbers that support it.
- Also list what additional evidence would be useful to verify it.
- Use a preliminary plausibility value from 0.0 to 1.0 only as a heuristic;
  it is NOT a verified confidence score.

Return ONLY valid JSON in this shape:
{{
  "hypotheses": [
    {{
      "hypothesis": "possible explanation",
      "reasoning": "why the evidence suggests this possibility",
      "supporting_evidence": [1, 2],
      "missing_evidence": ["evidence that would help verify it"],
      "preliminary_plausibility": 0.0
    }}
  ]
}}

Question:
{question}

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

    payload = response.json()
    raw = payload.get("response", "")
    data = _extract_json(raw)

    hypotheses = data.get("hypotheses", [])
    if not isinstance(hypotheses, list):
        raise ValueError("Model response contains an invalid hypotheses list.")

    return hypotheses[:max_hypotheses]
