import argparse
import json

from rag.evaluator import evaluate_retrieval
from rag.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate InferRAG retrieval against labeled chunk IDs."
    )
    parser.add_argument("document")
    parser.add_argument("dataset", help="JSON file containing evaluation questions")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    dataset = json.loads(open(args.dataset, encoding="utf-8").read())

    rag = RAGPipeline()
    count = rag.ingest(args.document)

    totals = {"precision": {}, "recall": {}, "mrr": 0.0}

    for case in dataset:
        results = rag.retrieve(
            case["question"],
            top_k=args.top_k,
            candidate_k=args.candidate_k,
        )
        metrics = evaluate_retrieval(
            results,
            case["relevant_chunk_ids"],
        )

        print(f"\nQuestion: {case['question']}")
        print(f"Relevant chunks: {case['relevant_chunk_ids']}")
        print(f"MRR: {metrics['mrr']:.4f}")

        for metric_name in ("precision", "recall"):
            for k, value in metrics[metric_name].items():
                print(f"{metric_name}@{k}: {value:.4f}")

        totals["mrr"] += metrics["mrr"]
        for metric_name in ("precision", "recall"):
            for k, value in metrics[metric_name].items():
                totals[metric_name][k] = (
                    totals[metric_name].get(k, 0.0) + value
                )

    n = len(dataset)
    if n:
        print(f"\nEvaluated {n} questions across {count} document chunks.")
        print(f"Average MRR: {totals['mrr'] / n:.4f}")
        for metric_name in ("precision", "recall"):
            for k, value in sorted(totals[metric_name].items()):
                print(f"Average {metric_name}@{k}: {value / n:.4f}")


if __name__ == "__main__":
    main()
