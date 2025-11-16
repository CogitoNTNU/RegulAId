"""BM25 retriever for keyword-based search."""

import os
import sys
import re
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

    def __init__(self, **_):
        """Initialize BM25 retriever. Accepts and ignores extra kwargs."""
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
            # Sanitize query for ParadeDB: extract only alphanumeric words
            # ParadeDB's @@@ operator requires special characters to be escaped
            # By extracting only words, we avoid all special character issues
            query_text = str(query).strip()
            # Extract words (alphanumeric sequences only - no apostrophes to avoid issues)
            # Convert to lowercase for consistent matching
            words = re.findall(r"\b[a-z0-9]+\b", query_text.lower())
            # Join words with spaces - creates a clean query without any special syntax
            sanitized_query = " ".join(words) if words else ""
            
            # If we have no words after extraction, try a fallback approach
            if not sanitized_query:
                # Remove all non-alphanumeric characters except spaces
                sanitized_query = re.sub(r"[^a-z0-9\s]", "", query_text.lower())
                sanitized_query = re.sub(r'\s+', ' ', sanitized_query).strip()
            
            # Only add query if we have something to search for
            if sanitized_query:
                # Use plain text search - ParadeDB's @@@ should handle plain space-separated words
                where_clauses.append(f"{self.column_name} @@@ %s")
                params.append(sanitized_query)

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
            error_msg = str(e)
            # If it's a parsing error, try a fallback with even simpler query
            if "could not parse query" in error_msg.lower() and where_clauses:
                # Try with just the first few words as a fallback
                try:
                    query_param_idx = 0
                    for i, clause in enumerate(where_clauses):
                        if "@@@" in clause:
                            query_param_idx = i
                            break
                    
                    if query_param_idx < len(params):
                        original_query = params[query_param_idx]
                        # Use only first 10 words as fallback
                        words = original_query.split()[:10]
                        fallback_query = " ".join(words)
                        params[query_param_idx] = fallback_query
                        
                        # Retry with simplified query
                        with psycopg.connect(get_psycopg_connection_string()) as conn:
                            with conn.cursor() as cur:
                                cur.execute(sql, tuple(params))
                                results = cur.fetchall()
                        
                        formatted_results = []
                        for row in results:
                            formatted_results.append({
                                "id": row[0],
                                "content": row[1],
                                "metadata": row[2],
                                "score": row[3],
                            })
                        return formatted_results
                except Exception:
                    pass  # Fall through to return empty list
            
            print(f"ERROR in BM25 search: {error_msg}")
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
        return f"BM25Retriever(table='{self.table_name}')"
