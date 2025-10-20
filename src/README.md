# RegulAId Source Code

Source code for the EU AI ACT Q&A bot.

## Directory Structure

```
src/
├── preprocessing/              # Data preprocessing
│   ├── strip_aiact.py         # Extract text from PDF
│   └── chunking-paragraphs.py # Chunk text into paragraphs
│
├── database/                   # Database setup
│   ├── connection.py          # Connection helpers
│   ├── setup.py              # Table/index creation, document insertion
│   ├── init_db.py            # Database initialization script
│   └── reset_db.py           # Drop table script
│
├── retrievers/                 # Retrieval methods
│   ├── bm25.py               # BM25 keyword search
│   ├── vector.py             # Vector semantic search (HNSW)
│   ├── hybrid.py             # Hybrid retriever (TODO)
│   └── test_retrievers.py    # Test script
│
└── rag/                        # RAG pipeline
    └── pipeline.py           # RAG pipeline (TODO)
```

## Setup

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env with your values
```

### 2. Preprocess Data (One-time)

```bash
# Extract and clean PDF
uv run --with pymupdf python src/preprocessing/strip_aiact.py

# Chunk into paragraphs
uv run python src/preprocessing/chunking-paragraphs.py
```

Creates `data/processed/aiact-chunks.json`.

### 3. Initialize Database (One-time)

```bash
# Insert all documents
uv run python src/database/init_db.py

# Or insert only first 100 for testing
uv run python src/database/init_db.py --num-docs 100
```

This creates tables, indexes (HNSW, BM25), and inserts documents with embeddings.

## Usage

### Test Retrievers

```bash
uv run python src/retrievers/test_retrievers.py
```

### Use Retrievers in Code

```python
from retrievers import BM25Retriever, VectorRetriever

# BM25 search
bm25 = BM25Retriever()
results = bm25.search("What are high-risk AI systems?", k=5)

# Vector search
vector = VectorRetriever()
results = vector.search("What are high-risk AI systems?", k=5)
```

### RAG Pipeline (TODO)

```bash
# When implemented
uv run python src/rag/pipeline.py "Your question" --retriever bm25
```

## Reset Database

If you need to reset the database (e.g., schema changed):

```bash
uv run python src/database/reset_db.py
```

Drops the table. Re-run `init_db.py` to recreate.

## Next Steps

1. Implement `retrievers/hybrid.py` using RRF
2. Implement `rag/pipeline.py` with LLM integration
3. Ragas
