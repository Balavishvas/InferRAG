import requests

def generate_answer(question: str, contexts: list[dict], model: str = "qwen2.5:3b",
                    base_url: str = "http://localhost:11434") -> str:
    context = "\n\n".join(
        f"[Evidence {i}] {item['text']}" for i, item in enumerate(contexts, 1)
    )
    prompt = f"""You are InferRAG, a grounded question-answering assistant.
Answer using only the evidence below. If the evidence is insufficient, say so.
Do not invent facts.

Question:
{question}

Evidence:
{context}

Answer:"""
    response = requests.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()
