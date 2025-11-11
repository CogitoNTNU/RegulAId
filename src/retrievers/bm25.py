"""BM25 retriever for keyword-based search."""

import os
import sys
import psycopg
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import get_psycopg_connection_string
from .filters import build_where_clauses

load_dotenv()


class BM25Retriever:
    """BM25 retriever using ParadeDB."""

    def __init__(self):
        """Initialize BM25 retriever."""
        self.table_name = os.getenv("COLLECTION_NAME")
        self.column_name = "page_content"

    def search(self, query: str, k: int = 10, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Perform BM25 search with optional metadata filtering.

        Args:
            query: Search query text
            k: Number of results to return
            filters: Optional dict of metadata filters (see filters.build_where_clauses)

        Returns:
            List of documents with their BM25 scores
        """
        base_select = f"SELECT id, {self.column_name}, metadata, paradedb.score(id) AS score FROM {self.table_name}"

        where_clauses: List[str] = []
        params: List[Any] = []

        # Full text query condition
        if query is not None and str(query).strip() != "":
            where_clauses.append(f"{self.column_name} @@@ %s")
            params.append(query)

        # Build metadata where clauses using shared helper
        where_clauses.extend(build_where_clauses(filters or {}, params))

        # Combine clauses
        if where_clauses:
            where_sql = " WHERE " + " AND ".join(where_clauses)
        else:
            where_sql = ""

        # If there's no full-text query, paradedb.score(id) may be NULL; sort nulls last
        order_sql = " ORDER BY score DESC NULLS LAST"

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
                    "score": row[3],
                })

            return formatted_results

        except Exception as e:
            print(f"ERROR in BM25 search: {str(e)}")
            return []

    def __repr__(self) -> str:
        return f"BM25Retriever(table='{self.table_name}')"
