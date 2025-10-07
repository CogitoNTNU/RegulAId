#!/usr/bin/env python3

from typing import Optional, Dict, Any
from langchain_core.documents import Document

from dotenv import load_dotenv

from pydantic import BaseModel, Field

load_dotenv()

from ..setup import *
from ..data_preprocessing.chunking import *


class DocumentMetadata(BaseModel):
    """
    Metadata for each chunk
    """
    article_id: Optional[int] = Field()
    chunk_id: Optional[int] = Field()


class QueryParams(BaseModel):
    k: int = Field(default=5, description="Number of similar documents to retrieve for context", ge=1, le=10)
    filter: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Filtering for querying")
    query: str = Field(..., description="The query to process against the document")


vector_store = get_vector_store()


def create_hnsw_index(table_name=None, column_name="embedding", metric="vector_cosine_ops", m=16, ef_construction=64):
    """
    Create HNSW index for fast vector search in ParadeDB.
    """
    if table_name is None:
        table_name = os.getenv("COLLECTION_NAME")
    query = f"""
        CREATE INDEX IF NOT EXISTS ON {table_name}
        USING hnsw ({column_name} {metric})
        WITH (m = {m}, ef_construction = {ef_construction});
    """
    try:
        with psycopg.connect(psycopg_connection_string()) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                conn.commit()
        print(f"HNSW index created (if not exists) on {table_name}.{column_name} with metric {metric}.")
    except Exception as e:
        print("Error creating HNSW index:", str(e))


def create_bm25_index(table_name=None, columns=None):
    """
    Create a ParadeDB BM25 index on the specified columns, including a key_field.
    """
    if table_name is None:
        table_name = os.getenv("COLLECTION_NAME")
    if columns is None:
        columns = ["id", "page_content", "metadata"]

    columns_str = ", ".join(columns)
    index_name = f"bm25_idx_on_{table_name}"

    # Assuming 'id' is the primary key and suitable for key_field
    sql = f"""
        CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}
        USING bm25 ({columns_str})
        WITH (key_field='id');
    """
    try:
        with psycopg.connect(psycopg_connection_string()) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                conn.commit()
        print(
            f"BM25 index '{index_name}' created (if not exists) on {table_name} with columns ({columns_str}) and key_field='id'.")
    except Exception as e:
        print("Error creating BM25 index:", str(e))


def parade_bm25_search(query: str, k: int = 5, table_name=None, column_name="page_content") -> list:
    """
    Perform BM25 search using ParadeDB's BM25 operator and scoring function.
    """
    if table_name is None:
        table_name = os.getenv("COLLECTION_NAME")
    sql = f"""
        SELECT id, {column_name}, metadata, paradedb.score(id) AS score
        FROM {table_name}
        WHERE {column_name} @@@ %s
        ORDER BY score DESC
        LIMIT %s
    """
    try:
        with psycopg.connect(psycopg_connection_string()) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (query, k))
                results = cur.fetchall()
        return results
    except Exception as e:
        print("Error in ParadeDB BM25 search:", str(e))
        return []


def parade_similarity_search(query_text: str, k: int = 5, table_name=None, column_name="embedding",
                             metric_operator="<=>") -> list:
    """
    Perform similarity search using ParadeDB's vector operator and HNSW index.
    """
    if table_name is None:
        table_name = os.getenv("COLLECTION_NAME")
    # Get embedding for the query
    embedding = vector_store.embed_query(query_text)
    # ParadeDB expects embedding as a string representation of a vector
    embedding_str = str(embedding)
    sql = f"""
        SELECT *, {column_name} {metric_operator} %s AS similarity
        FROM {table_name}
        ORDER BY {column_name} {metric_operator} %s
        LIMIT %s
    """
    try:
        with psycopg.connect(psycopg_connection_string()) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (embedding_str, embedding_str, k))
                results = cur.fetchall()
        return results
    except Exception as e:
        print("Error in ParadeDB similarity search:", str(e))
        return []


def create_collection_table(table_name=None, column_name="embedding"):
    """
    Create the collection table if it does not exist, with a vector column for ParadeDB.
    """
    if table_name is None:
        table_name = os.getenv("COLLECTION_NAME")
    # Use jsonb for metadata if supported, otherwise text
    sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            page_content TEXT NOT NULL,
            metadata JSONB,
            {column_name} VECTOR
        );
    """
    try:
        with psycopg.connect(psycopg_connection_string()) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                conn.commit()
        print(f"Table '{table_name}' ensured.")
    except Exception as e:
        print("Error creating table:", str(e))


if __name__ == "__main__":
    path = "../data/processed/AIACT-Serina.md"

    check_database_connection()
    # Ensure all operations use the same table name
    collection_table_name = "langchain_collection"  # Define it once
    create_collection_table(table_name=collection_table_name)
    create_hnsw_index(table_name=collection_table_name)
    create_bm25_index(table_name=collection_table_name)

    text = load_text(path)
    articles = split_articles_by_header(text)
    verify_articles(articles)

    if articles:
        last_num, last_chunk = articles[-1]
        print("\nLast chunk header number:", last_num)
        print("Last chunk first line:", last_chunk.splitlines()[0] if last_chunk.splitlines() else "<empty>")

        documents: List[Document] = articles_to_chunks(articles)

        print("Starting embedding")
        # Ensure documents are added to the correct table
        add_documents(documents[:10], table_name=collection_table_name)
        print("Finished embedding")

        print("ParadeDB BM25 search:")
        # Pass the consistent table name here
        print(parade_bm25_search("Regulation applies", k=5, table_name=collection_table_name))

        qurey = QueryParams(query="Regulation applies")
        # Pass the consistent table name here
        print(get_similar_documents(qurey, table_name=collection_table_name))
        print("ParadeDB HNSW similarity search:")
        # Pass the consistent table name here
        print(parade_similarity_search("Regulation applies", k=5, table_name=collection_table_name))
