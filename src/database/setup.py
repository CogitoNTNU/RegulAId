"""Database table and index setup utilities."""
import os
import sys
import json
import psycopg
from typing import List
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import get_psycopg_connection_string

load_dotenv()


def get_embeddings_model() -> OllamaEmbeddings:
    """Get the embeddings model."""
    return OllamaEmbeddings(
        base_url=os.getenv("EMBEDDING_API_BASE_URL"),
        model=os.getenv("EMBEDDING_MODEL_NAME"),
    )


def create_table(table_name: str = None, column_name: str = "embedding", vector_dim: int = None) -> None:
    """Create the collection table if it does not exist."""
    if table_name is None:
        table_name = os.getenv("COLLECTION_NAME")

    # Get the actual embedding dimension from the model
    if vector_dim is None:
        embeddings = get_embeddings_model()
        test_embedding = embeddings.embed_query("test")
        vector_dim = len(test_embedding)

    sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            page_content TEXT NOT NULL,
            metadata JSONB,
            {column_name} VECTOR({vector_dim})
        );
    """
    try:
        with psycopg.connect(get_psycopg_connection_string()) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                conn.commit()
        print(f"Table '{table_name}' created/verified")
    except Exception as e:
        print(f"ERROR creating table: {str(e)}")


def create_hnsw_index(
    table_name: str = None,
    column_name: str = "embedding",
    metric: str = "vector_cosine_ops",
    m: int = 16,
    ef_construction: int = 64
) -> None:
    """Create HNSW index for fast vector search. HNSW = Fast approximate nearest neighbor search """
    if table_name is None:
        table_name = os.getenv("COLLECTION_NAME")

    query = f"""
        CREATE INDEX IF NOT EXISTS {table_name}_hnsw_idx ON {table_name}
        USING hnsw ({column_name} {metric})
        WITH (m = {m}, ef_construction = {ef_construction});
    """
    try:
        with psycopg.connect(get_psycopg_connection_string()) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                conn.commit()
        print(f"HNSW index created on {table_name}.{column_name}")
    except Exception as e:
        print(f"ERROR creating HNSW index: {str(e)}")


def create_bm25_index(table_name: str = None, columns: List[str] = None) -> None:
    """Create BM25 index for text search.
        Creates BM25 index on page_content column"""
    if table_name is None:
        table_name = os.getenv("COLLECTION_NAME")
    if columns is None:
        columns = ["id", "page_content", "metadata"]

    columns_str = ", ".join(columns)
    index_name = f"{table_name}_bm25_idx"

    sql = f"""
        CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}
        USING bm25 ({columns_str})
        WITH (key_field='id');
    """
    try:
        with psycopg.connect(get_psycopg_connection_string()) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                conn.commit()
        print(f"BM25 index '{index_name}' created on {table_name}")
    except Exception as e:
        print(f"ERROR creating BM25 index: {str(e)}")


def add_documents(
    documents: List[Document],
    table_name: str = None,
    column_name: str = "embedding"
) -> None:
    """Insert documents and their embeddings into the database."""
    if table_name is None:
        table_name = os.getenv("COLLECTION_NAME")

    embeddings = get_embeddings_model()

    try:
        with psycopg.connect(get_psycopg_connection_string()) as conn:
            with conn.cursor() as cur:
                for i, doc in enumerate(documents, 1):
                    embedding = embeddings.embed_query(doc.page_content)
                    embedding_str = str(embedding)
                    metadata_json = json.dumps(doc.metadata)

                    cur.execute(
                        f"INSERT INTO {table_name} (page_content, metadata, {column_name}) VALUES (%s, %s, %s)",
                        (doc.page_content, metadata_json, embedding_str)
                    )

                    if i % 10 == 0:
                        print(f"Processed {i}/{len(documents)} documents...")

                conn.commit()
        print(f"Inserted {len(documents)} documents into {table_name}")
    except Exception as e:
        print(f"ERROR inserting documents: {str(e)}")
