#!/usr/bin/env python3
import os
import re
import unicodedata
import json
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_ollama import OllamaEmbeddings
import psycopg
from dotenv import load_dotenv

from pydantic import BaseModel, Field

load_dotenv()


def pg_connection_string() -> str:
    return f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"


def psycopg_connection_string() -> str:
    return f"dbname='{os.getenv('DB_NAME')}' user='{os.getenv('DB_USER')}' password='{os.getenv('DB_PASSWORD')}' host='{os.getenv('DB_HOST')}' port='{os.getenv('DB_PORT')}'"


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


def get_vector_store() -> OllamaEmbeddings:
    # https://huggingface.co/spaces/mteb/leaderboard

    """
    print(f"OLLAMA_API_BASE_URL{os.getenv('EMBEDDING_API_BASE_URL')}")
    print(f"EMBEDDING_MODEL_NAME{os.getenv('EMBEDDING_MODEL_NAME')}")

    """

    embeddings = OllamaEmbeddings(
        base_url=os.getenv("EMBEDDING_API_BASE_URL"),
        model=os.getenv("EMBEDDING_MODEL_NAME"),
    )

    """   embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL_NAME,
        api_key=settings.EMBEDDING_API_KEY,
        base_url=settings.EMBEDDING_API_BASE_URL,
        headers={
            "HTTP-Referer": "localhost",
            "X-Title": "AgenticCompliance"
        })
    """

    return embeddings


vector_store = get_vector_store()


def add_documents(documents: List[Document], table_name=None, column_name="embedding") -> None:
    """
    Insert documents and their embeddings into the database using raw SQL.
    """
    if table_name is None:
        table_name = os.getenv("COLLECTION_NAME")
    # Get embedding model
    embeddings = get_vector_store()
    try:
        with psycopg.connect(psycopg_connection_string()) as conn:
            with conn.cursor() as cur:
                for doc in documents:
                    embedding = embeddings.embed_query(doc.page_content)
                    embedding_str = str(embedding)
                    metadata_json = json.dumps(doc.metadata)  # Convert metadata to valid JSON
                    # Insert document and embedding
                    cur.execute(
                        f"INSERT INTO {table_name} (page_content, metadata, {column_name}) VALUES (%s, %s, %s)",
                        (doc.page_content, metadata_json, embedding_str)
                    )
                conn.commit()
        print(f"Inserted {len(documents)} documents into {table_name}.")
    except Exception as e:
        print("Error inserting documents:", str(e))


def get_similar_documents(query_request: QueryParams, table_name=None, column_name="embedding",
                          metric_operator="<=>") -> list:
    """
    Perform similarity search using ParadeDB's vector operator and HNSW index via raw SQL.
    """
    if table_name is None:
        table_name = os.getenv("COLLECTION_NAME")

    embedding = get_vector_store().embed_query(query_request.query)
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
                cur.execute(sql, (embedding_str, embedding_str, query_request.k))
                results = cur.fetchall()
        return results
    except Exception as e:
        print("Error in ParadeDB similarity search:", str(e))
        return []


def articles_to_chunks(articles) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=0)

    chunks: List[Document] = []  # This is the list of chunks of the whole document

    for articleID, article in articles:  # Chapter => Article => Paragraph etc
        print(articleID)

        # https://python.langchain.com/docs/concepts/text_splitters/

        chunks_of_an_article = text_splitter.split_text(article)

        chunk_id = 1

        for chunk in chunks_of_an_article:
            chunk_metadata = DocumentMetadata(
                article_id=articleID,
                chunk_id=chunk_id
            ).__dict__

            chunk_id += 1
            chunks.append(Document(page_content=chunk, metadata=chunk_metadata))

    return chunks


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # normalize unicode and replace non-breaking spaces which often break regex
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00A0", " ")
    return text


def split_articles_by_header(text: str) -> List[Tuple[int, str]]:  # TODO add Ines and Ingunn's code
    """
    Return list of (article_number, chunk_text).
    The header regex is anchored to line-start (MULTILINE) to avoid accidental matches inside paragraphs.
    """

    # header_re = re.compile(r'\nArticle\s*(\d+)\n | ##\s*Article\s*(\d+)\n', flags=re.I | re.M)

    header_re = re.compile(r'(?:\n|##\s*)Article\s*(\d+)\s*\n', flags=re.I | re.M)
    matches = list(header_re.finditer(text))
    if not matches:
        return []

    articles = []
    for i, m in enumerate(matches):
        num = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].rstrip()
        articles.append((num, chunk))

    return articles


def check_database_connection():
    try:
        conn = psycopg.connect(psycopg_connection_string())
        conn.close()
        print("Database connection was successful")
    except Exception as e:
        print('Database Connection Error:', str(e))


def verify_articles(articles: List[Tuple[int, str]]):
    if not articles:
        print("No article headers found.")
        return

    nums = [n for n, _ in articles]
    cnt = Counter(nums)
    duplicates = [n for n, c in cnt.items() if c > 1]

    mn, mx = min(nums), max(nums)
    expected = list(range(mn, mx + 1))
    missing = [n for n in expected if n not in cnt]

    order_issues = []
    for i in range(1, len(nums)):
        prev, cur = nums[i - 1], nums[i]
        if cur != prev + 1:
            order_issues.append((i, prev, cur))

    print(f"Found {len(articles)} chunks. Header numbers span {mn} .. {mx}.")
    if duplicates:
        print("Duplicate article numbers:", duplicates)
    if missing:
        print("Missing article numbers:", missing)
    if order_issues:
        print("Order problems (chunk_index, previous_number -> current_number):")
        for idx, prev, cur in order_issues[:50]:
            prev_title = articles[idx - 1][1].splitlines()[0][:80]
            cur_title = articles[idx][1].splitlines()[0][:80]
            print(
                f"  chunk {idx}: {prev} -> {cur}; prev header starts: {prev_title!r}; cur header starts: {cur_title!r}")
    if not (duplicates or missing or order_issues):
        print("All article headers look sequential and in order.")


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
