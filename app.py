import argparse
from rag.pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser(description="InferRAG V1 local retrieval demo")
    parser.add_argument("document", help="Path to a PDF, TXT, or Markdown file")
    parser.add_argument("--query", required=True, help="Question to retrieve evidence for")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    rag = RAGPipeline()
    count = rag.ingest(args.document)
    print(f"Ingested {count} chunks.\n")
    results = rag.retrieve(args.query, args.top_k)

    for i, result in enumerate(results, 1):
        print(f"[{i}] score={result['score']:.4f}")
        print(result["text"])
        print()

if __name__ == "__main__":
    main()
