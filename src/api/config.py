# ADD AND/OR MODIFY STUFF HERE
OPENAI_MODEL = "gpt-4o"

# Retriever configuration
# Options: "bm25" (keyword), "vector" (semantic), "hybrid" (combines both with RRF)
RETRIEVER_TYPE = "hybrid" 
RETRIEVER_TOP_K = 10  # Number of documents to retrieve

# Hybrid retriever configuration (only used if RETRIEVER_TYPE = "hybrid")
HYBRID_BM25_TOP_K = 100  # Number of candidates from BM25
HYBRID_VECTOR_TOP_K = 100  # Number of candidates from vector search
HYBRID_BM25_WEIGHT = 1.0  # Weight for BM25 contribution
HYBRID_VECTOR_WEIGHT = 1.0  # Weight for vector contribution
HYBRID_RRF_K = 60.0  # RRF constant (lower = more aggressive re-ranking)
