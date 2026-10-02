import argparse

import requests

from rag.context_selector import select_context
from rag.hypothesis import generate_hypotheses
from rag.hypothesis_ranker import rank_hypotheses
from rag.hypothesis_verifier import verify_hypotheses
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
    parser.add_argument("--claim-top-k", type=int, default=3)
    parser.add_argument("--claim-candidate-k", type=int, default=10)
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

    print("\n--- Hypotheses (V4 Claim Verification) ---")
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
            verifications = verify_hypotheses(
                args.query,
                hypotheses,
                rag_pipeline=rag,
                model=args.model,
                claim_top_k=args.claim_top_k,
                claim_candidate_k=args.claim_candidate_k,
            )
            ranked = rank_hypotheses(hypotheses, verifications)

            for i, hypothesis in enumerate(ranked, 1):
                verification = hypothesis.get("verification", {})
                print(f"\n[{i}] {hypothesis.get('hypothesis', 'N/A')}")
                print(f"Reasoning: {hypothesis.get('reasoning', 'N/A')}")
                print(
                    "Generation evidence: "
                    f"{hypothesis.get('supporting_evidence', [])}"
                )
                print(
                    "Missing evidence: "
                    f"{hypothesis.get('missing_evidence', [])}"
                )
                print(f"Status: {verification.get('status', 'uncertain')}")
                print(
                    "Direct evidence: "
                    f"{verification.get('supporting_evidence', [])}"
                )
                print(
                    "Indirect evidence: "
                    f"{verification.get('indirect_evidence', [])}"
                )
                print(
                    "Contradicting evidence: "
                    f"{verification.get('contradicting_evidence', [])}"
                )
                print(
                    f"Directness: {verification.get('directness', 'unknown')}"
                )
                print(
                    "Verification score (heuristic): "
                    f"{verification.get('verification_score', 0.0)}"
                )
                print(f"Verification: {verification.get('explanation', 'N/A')}")

                claim_results = verification.get("claim_results", [])
                if claim_results:
                    print("Claims:")
                    for claim in claim_results:
                        claim_verification = claim.get("verification", {})
                        print(
                            f"  - [{claim.get('importance', 'core')}] "
                            f"{claim.get('claim', 'N/A')}"
                        )
                        print(
                            f"    Status: "
                            f"{claim_verification.get('status', 'insufficient')}"
                        )
                        print(
                            f"    Evidence: "
                            f"{claim_verification.get('supporting_evidence', [])}"
                        )
                        print(
                            f"    Explanation: "
                            f"{claim_verification.get('explanation', 'N/A')}"
                        )
    except (ValueError, requests.RequestException) as exc:
        print(f"Hypothesis reasoning failed: {exc}")

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
