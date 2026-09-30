import argparse
from rag.llm import generate_answer
from rag.pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser(description="Ask InferRAG V1")
    parser.add_argument("document")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default="qwen2.5:3b")
    args = parser.parse_args()

    rag = RAGPipeline()
    count = rag.ingest(args.document)
    results = rag.retrieve(args.query, args.top_k)

    print(f"Retrieved {len(results)} of {count} chunks.\n")
    print(generate_answer(args.query, results, args.model))
    print("\n--- Evidence ---")
    for i, item in enumerate(results, 1):
        print(f"\n[{i}] score={item['score']:.4f}\n{item['text']}")

if __name__ == "__main__":
    main()
