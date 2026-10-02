import argparse

from rag.context_selector import select_context
from rag.hypothesis import generate_hypotheses
from rag.llm import generate_answer
from rag.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Ask InferRAG")
    parser.add_argument("document")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--hypotheses", type=int, default=3)
    args = parser.parse_args()

    rag = RAGPipeline()
    count = rag.ingest(args.document)

    results = rag.retrieve(
        args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )
    contexts = select_context(
        results,
        max_chunks=args.top_k,
        max_characters=args.max_context_chars,
    )

    print(f"Retrieved {len(results)} chunks from {count} indexed chunks.")
    print(f"Selected {len(contexts)} chunks for reasoning.\n")

    print("--- Answer ---")
    print(generate_answer(args.query, contexts, args.model))

    print("\n--- Hypotheses (unverified) ---")
    try:
        hypotheses = generate_hypotheses(
            args.query,
            contexts,
            model=args.model,
            max_hypotheses=args.hypotheses,
        )

        if not hypotheses:
            print("No hypotheses generated.")
        else:
            for i, hypothesis in enumerate(hypotheses, 1):
                print(f"\n[{i}] {hypothesis.get('hypothesis', 'N/A')}")
                print(f"Reasoning: {hypothesis.get('reasoning', 'N/A')}")
                print(
                    "Supporting evidence: "
                    f"{hypothesis.get('supporting_evidence', [])}"
                )
                print(
                    "Missing evidence: "
                    f"{hypothesis.get('missing_evidence', [])}"
                )
                print(
                    "Preliminary plausibility: "
                    f"{hypothesis.get('preliminary_plausibility', 'N/A')}"
                )
    except (ValueError, requests.RequestException) as exc:
        print(f"Hypothesis generation failed: {exc}")

    print("\n--- Evidence ---")
    for i, item in enumerate(contexts, 1):
        print(
            f"\n[{i}] chunk_id={item.get('chunk_id')} "
            f"retrieval={item.get('score', 0.0):.4f} "
            f"rerank={item.get('rerank_score', 0.0):.4f}"
        )
        print(item["text"])


if __name__ == "__main__":
    main()
