import argparse
import json

from rag.evaluator import evaluate_retrieval
from rag.reranker import Reranker
from rag.pipeline import RAGPipeline


def _accumulate(totals, metrics):
    totals["mrr"] += metrics["mrr"]

    for metric_name in ("precision", "recall"):
        for k, value in metrics[metric_name].items():
            totals[metric_name][k] = (
                totals[metric_name].get(k, 0.0) + value
            )


def _average(totals, count):
    result = {"mrr": totals["mrr"] / count}

    for metric_name in ("precision", "recall"):
        result[metric_name] = {
            k: value / count
            for k, value in totals[metric_name].items()
        }

    return result


def _print_metrics(label, metrics):
    print(f"  {label}")
    print(f"    MRR: {metrics['mrr']:.4f}")

    for metric_name in ("precision", "recall"):
        for k, value in sorted(metrics[metric_name].items()):
            print(f"    {metric_name}@{k}: {value:.4f}")


def _print_delta(metric_name, k, baseline, reranked):
    delta = reranked - baseline
    sign = "+" if delta >= 0 else ""
    print(
        f"  {metric_name}@{k}: "
        f"{baseline:.4f} -> {reranked:.4f} ({sign}{delta:.4f})"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare FAISS retrieval against BGE reranking "
            "using labeled chunk IDs."
        )
    )
    parser.add_argument("document")
    parser.add_argument(
        "dataset",
        help="JSON file containing evaluation questions",
    )
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--rerank-model",
        default="BAAI/bge-reranker-base",
    )
    args = parser.parse_args()

    if args.top_k > args.candidate_k:
        parser.error("--top-k cannot be greater than --candidate-k")

    dataset = json.loads(
        open(args.dataset, encoding="utf-8").read()
    )

    rag = RAGPipeline(reranker_model=args.rerank_model)
    count = rag.ingest(args.document)

    baseline_totals = {
        "precision": {},
        "recall": {},
        "mrr": 0.0,
    }
    reranked_totals = {
        "precision": {},
        "recall": {},
        "mrr": 0.0,
    }

    reranker = Reranker(args.rerank_model)

    for case in dataset:
        question = case["question"]
        relevant = case["relevant_chunk_ids"]

        candidates = rag.retrieve_candidates(
            question,
            candidate_k=args.candidate_k,
        )

        baseline_results = candidates[:args.top_k]
        reranked_results = reranker.rerank(
            question,
            candidates,
            top_k=args.top_k,
        )

        baseline_metrics = evaluate_retrieval(
            baseline_results,
            relevant,
        )
        reranked_metrics = evaluate_retrieval(
            reranked_results,
            relevant,
        )

        _accumulate(baseline_totals, baseline_metrics)
        _accumulate(reranked_totals, reranked_metrics)

        print(f"\nQuestion: {question}")
        print(f"Relevant chunks: {relevant}")
        print(
            "FAISS top-{}: {}".format(
                args.top_k,
                [item["chunk_id"] for item in baseline_results],
            )
        )
        print(
            "BGE top-{}:    {}".format(
                args.top_k,
                [item["chunk_id"] for item in reranked_results],
            )
        )

        _print_metrics("FAISS baseline", baseline_metrics)
        _print_metrics("BGE reranked", reranked_metrics)

        print("  Change after reranking")
        _print_delta(
            "MRR",
            "",
            baseline_metrics["mrr"],
            reranked_metrics["mrr"],
        )
        for metric_name in ("precision", "recall"):
            for k in sorted(baseline_metrics[metric_name]):
                _print_delta(
                    metric_name,
                    k,
                    baseline_metrics[metric_name][k],
                    reranked_metrics[metric_name][k],
                )

    n = len(dataset)
    if not n:
        return

    baseline_average = _average(baseline_totals, n)
    reranked_average = _average(reranked_totals, n)

    print(f"\n{'=' * 64}")
    print(f"Evaluated {n} questions across {count} document chunks.")
    print("\nOverall comparison")
    _print_metrics("FAISS baseline", baseline_average)
    _print_metrics("BGE reranked", reranked_average)

    print("\nOverall improvement from BGE reranking")
    _print_delta(
        "MRR",
        "",
        baseline_average["mrr"],
        reranked_average["mrr"],
    )
    for metric_name in ("precision", "recall"):
        for k in sorted(baseline_average[metric_name]):
            _print_delta(
                metric_name,
                k,
                baseline_average[metric_name][k],
                reranked_average[metric_name][k],
            )

    print("\nInterpretation")
    recall5_delta = (
        reranked_average["recall"].get(args.top_k, 0.0)
        - baseline_average["recall"].get(args.top_k, 0.0)
    )
    mrr_delta = reranked_average["mrr"] - baseline_average["mrr"]

    if recall5_delta > 0:
        print(
            f"- BGE increases Recall@{args.top_k} by "
            f"{recall5_delta:.4f}."
        )
    elif recall5_delta < 0:
        print(
            f"- BGE decreases Recall@{args.top_k} by "
            f"{abs(recall5_delta):.4f}."
        )
    else:
        print(
            f"- BGE leaves Recall@{args.top_k} unchanged."
        )

    if mrr_delta > 0:
        print(f"- BGE improves MRR by {mrr_delta:.4f}.")
    elif mrr_delta < 0:
        print(f"- BGE decreases MRR by {abs(mrr_delta):.4f}.")
    else:
        print("- BGE leaves MRR unchanged.")


if __name__ == "__main__":
    main()
