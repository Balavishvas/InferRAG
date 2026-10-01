# InferRAG 🧠

> An evidence-driven RAG system that retrieves relevant knowledge and produces grounded answers.
> **V2.1:** local document ingestion → metadata-aware chunking → persistent FAISS retrieval → optional local LLM.

## V2.1 architecture

```
PDF / TXT / Markdown
        ↓
  Document Loader
        ↓
Metadata-aware Chunking
        ↓
Local Embeddings
        ↓
Persistent FAISS Index
        ↓
 Semantic Retrieval
        ↓
   Relevant Evidence
        ↓
   Optional Ollama
        ↓
      Answer
```

InferRAG is being built in stages. V1 intentionally focuses on a clean, understandable RAG pipeline before adding reranking, caching, graphs, hypothesis generation, and evidence verification.

## V2.1 features

- PDF, TXT and Markdown ingestion
- Overlapping chunks with source/chunk metadata
- Persistent FAISS indexes saved under `data/index/`
- Automatic index reuse when the document and retrieval settings are unchanged
- Local sentence-transformer embeddings
- FAISS semantic retrieval
- Top-k evidence retrieval
- Optional local Ollama generation
- Evidence shown with every answer
- No paid API required

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows:
```bash
.venv\\Scripts\\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The first use downloads `all-MiniLM-L6-v2` and then runs embeddings locally.

### 3. Test retrieval without an LLM

Put a PDF, TXT, or Markdown file inside `data/documents/`:

```bash
python app.py data/documents/example.pdf --query "What is this document about?"
```

### 4. Optional local LLM with Ollama

Install Ollama, then:

```bash
ollama pull qwen2.5:3b
```

Then:

```bash
python ask.py data/documents/example.pdf --query "What is this document about?"
```

## Project structure

```
InferRAG/
├── app.py
├── ask.py
├── requirements.txt
├── rag/
│   ├── loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── llm.py
│   └── pipeline.py
├── data/
│   └── documents/
└── tests/
```

## Roadmap

- [x] V1 — Basic local RAG
- [x] V2.1 — Better chunk metadata and persistent retrieval index
- [ ] V2.2 — Reranking
- [ ] V2.3 — Retrieval evaluation and context selection
- [ ] V4 — Hypothesis generation
- [ ] V5 — Evidence verification
- [ ] V6 — Knowledge graph
- [ ] V7 — Redis and semantic caching
- [ ] V8 — Confidence and uncertainty modeling
- [ ] V9 — Evaluation benchmarks
- [ ] V10 — Cloud deployment

## Design principle

Every component should solve a real problem. We will not add infrastructure merely to make the architecture look complex.

InferRAG is designed to remain local-first and free to develop wherever practical.

## Status

**V2.1 — Retrieval foundation in development.**
