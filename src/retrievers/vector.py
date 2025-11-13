"""Vector/semantic retriever using embeddings and HNSW."""

import os
import sys
import psycopg
from typing import List, Dict, Any
# from langchain_ollama import OllamaEmbeddings # remove comment if you want to use OllamaEmbeddings
try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    OpenAIEmbeddings = None
from dotenv import load_dotenv
from database.connection import get_psycopg_connection_string

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .filters import build_where_clauses

load_dotenv()


class LocalEmbeddings:
    """Deterministic lightweight embeddings fallback for testing when OpenAI key is not available."""

    def __init__(self, model: str | None = None, dim: int = 8):
        self.dim = dim

    def embed_query(self, text: str) -> List[float]:
        # Produce a deterministic vector from the query text via hashing
        import hashlib

        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand bytes to float values in range [-1,1]
        vals = []
        for i in range(self.dim):
            byte = h[i % len(h)]
            vals.append((byte / 255.0) * 2 - 1)
        return vals


class VectorRetriever:
    """Vector retriever using ParadeDB HNSW index."""

    def __init__(self, **_):
        """Initialize vector retriever. Accepts and ignores extra kwargs."""
        self.table_name = os.getenv("COLLECTION_NAME")
        self.column_name = "embedding"
        self.metric_operator = "<=>"  # Cosine distance

        # Initialize embeddings model
        #self.embeddings = OllamaEmbeddings( if ollamaEmbedding
        # Prefer OpenAIEmbeddings if available and API key present; otherwise use LocalEmbeddings fallback
        use_openai = bool(os.getenv("OPENAI_KEY")) and OpenAIEmbeddings is not None
        if use_openai:
            try:
                self.embeddings = OpenAIEmbeddings(
                    # base_url=os.getenv("EMBEDDING_API_BASE_URL"), Include if Ollama
                    model=os.getenv("EMBEDDING_MODEL_NAME"),
                )
            except Exception as e:
                print(f"Warning: OpenAIEmbeddings initialization failed: {e}. Falling back to LocalEmbeddings.")
                self.embeddings = LocalEmbeddings()
        else:
            if OpenAIEmbeddings is None and not os.getenv("OPENAI_KEY"):
                print("Info: OPENAI_KEY not set or OpenAIEmbeddings unavailable — using LocalEmbeddings for testing.")
            self.embeddings = LocalEmbeddings()

    def search(self, query: str, k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search with optional metadata filtering.

        Args:
            query: Search query text
            k: Number of results to return
            filters: Optional dict of metadata filters (see filters.build_where_clauses)

        Returns:
            List of documents with their similarity scores
        """
        # Generate embedding for query
        embedding = self.embeddings.embed_query(query)
        embedding_str = str(embedding)

        base_select = f"SELECT id, page_content, metadata, {self.column_name} {self.metric_operator} %s AS similarity FROM {self.table_name}"

        where_clauses: List[str] = []
        params: List[Any] = [embedding_str]

        # For vector search we need to include the embedding as a parameter for the similarity computation
        # Many vector SQL variants expect the vector twice (in ORDER BY as well); we'll include it twice below.

        # Build metadata filters
        where_clauses.extend(build_where_clauses(filters or {}, params))

        if where_clauses:
            where_sql = " WHERE " + " AND ".join(where_clauses)
        else:
            where_sql = ""

        order_sql = f" ORDER BY {self.column_name} {self.metric_operator} %s"
        # second embedding param for ORDER BY
        params.append(embedding_str)

        limit_sql = " LIMIT %s"
        params.append(k)

        sql = base_select + where_sql + order_sql + limit_sql

        try:
            with psycopg.connect(get_psycopg_connection_string()) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, tuple(params))
                    results = cur.fetchall()

            # Format results
            formatted_results = []
            for row in results:
                formatted_results.append({
                    "id": row[0],
                    "content": row[1],
                    "metadata": row[2],
                    "similarity": row[3],
                })

            return formatted_results

        except Exception as e:
            print(f"ERROR in vector search: {str(e)}")
            return []

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Convenience method expected by evaluator: return only contexts.

        Args:
            query: Query text
            k: Number of contexts to return

        Returns:
            List of page contents (strings)
        """
        results = self.search(query, k=k)
        return [str(r.get("content", "")) for r in results if r.get("content")]

    def __repr__(self) -> str:
        return f"VectorRetriever(table='{self.table_name}')"
