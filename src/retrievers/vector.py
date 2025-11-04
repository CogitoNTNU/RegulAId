"""Vector/semantic retriever using embeddings and HNSW."""

import os
import sys
import psycopg
from typing import List, Dict, Any
# from langchain_ollama import OllamaEmbeddings # remove comment if you want to use OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from database.connection import get_psycopg_connection_string

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

class VectorRetriever:
    """Vector retriever using ParadeDB HNSW index."""

    def __init__(self, **_):
        """Initialize vector retriever. Uses env-configured embeddings to match DB."""
        self.table_name = os.getenv("COLLECTION_NAME")
        self.column_name = "embedding"
        self.metric_operator = "<=>"  # Cosine distance

        # Initialize embeddings model
        # Always use the same model as used for ingestion to avoid dim mismatch
        self.embeddings = OpenAIEmbeddings(
            # base_url=os.getenv("EMBEDDING_API_BASE_URL"), Include if Ollama
            model=os.getenv("EMBEDDING_MODEL_NAME"),
        )

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of documents with their similarity scores
        """
        # Generate embedding for query
        embedding = self.embeddings.embed_query(query)
        embedding_str = str(embedding)

        sql = f"""
            SELECT id, page_content, metadata, {self.column_name} {self.metric_operator} %s AS similarity
            FROM {self.table_name}
            ORDER BY {self.column_name} {self.metric_operator} %s
            LIMIT %s
        """

        try:
            with psycopg.connect(get_psycopg_connection_string()) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (embedding_str, embedding_str, k))
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
