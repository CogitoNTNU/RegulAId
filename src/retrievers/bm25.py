"""BM25 retriever for keyword-based search."""

import os
import sys
import psycopg
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import get_psycopg_connection_string

load_dotenv()


class BM25Retriever:
    """BM25 retriever using ParadeDB."""

    def __init__(self):
        """Initialize BM25 retriever."""
        self.table_name = os.getenv("COLLECTION_NAME")
        self.column_name = "page_content"

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform BM25 search.

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of documents with their BM25 scores
        """
        sql = f"""
            SELECT id, {self.column_name}, metadata, paradedb.score(id) AS score
            FROM {self.table_name}
            WHERE {self.column_name} @@@ %s
            ORDER BY score DESC
            LIMIT %s
        """
        try:
            with psycopg.connect(get_psycopg_connection_string()) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (query, k))
                    results = cur.fetchall()

            # Format results
            formatted_results = []
            for row in results:
                formatted_results.append({
                    "id": row[0],
                    "content": row[1],
                    "metadata": row[2],
                    "score": row[3],
                })

            return formatted_results

        except Exception as e:
            print(f"ERROR in BM25 search: {str(e)}")
            return []

    def __repr__(self) -> str:
        return f"BM25Retriever(table='{self.table_name}')"
