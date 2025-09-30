#!/usr/bin/env python3
import os
import re
import unicodedata
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
import psycopg
from dotenv import load_dotenv

from pydantic import BaseModel, Field

documents_from_db: List[Document] = []  # TODO This should be saved in the database

load_dotenv()


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


def get_vector_store() -> PGVector:
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

    return PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("COLLECTION_NAME"),
        connection=os.getenv("pg_connection_string"),
        use_jsonb=True
    )


vector_store = get_vector_store()


def add_documents(documents: List[Document]) -> list[str]:
    langchain_docs = []
    ids = []
    for doc in documents:
        langchain_docs.append(Document(
            id=doc.metadata.id,
            page_content=doc.page_content,
            metadata=doc.metadata.model_dump(exclude_unset=True)
        ))
        ids.append(doc.metadata.id)

    return vector_store.add_documents(langchain_docs, ids=ids)


def get_similar_documents(query_request: QueryParams) -> tuple[Any, List[dict[str, Any]]]:
    retrieved_docs = vector_store.similarity_search_with_relevance_scores(query=query_request.query,
                                                                          k=query_request.k,
                                                                          filter=query_request.filter)

    sources = []
    for doc, score in retrieved_docs:
        sources.append({
            "score": score,
            "article_id": doc.metadata.get('article_id', 'unknown'),
            "chunk_id": doc.metadata.get('chunk_id', 'unknown')
        })

    return retrieved_docs, sources


def articles_to_chunks(articles):
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


def bf25(query, k=10):
    retriever = BM25Retriever.from_documents(documents_from_db,
                                             k=k,
                                             )
    return retriever.invoke(query)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # normalize unicode and replace non-breaking spaces which often break regex
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00A0", " ")
    return text


def split_articles_by_header(text: str) -> List[Tuple[int, str]]:
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


def pg_connection_string() -> str:
    return f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"


def psycopg_connection_string() -> str:
    return f"dbname='{os.getenv('DB_NAME')}' user='{os.getenv('DB_USER')}' password='{os.getenv('DB_PASSWORD')}' host='{os.getenv('DB_HOST')}' port='{os.getenv('DB_PORT')}'"


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


if __name__ == "__main__":
    path = "../data/processed/AIACT-Serina.md"

    check_database_connection()

    text = load_text(path)
    articles = split_articles_by_header(text)
    verify_articles(articles)

    # optional: show the last chunk header and first line for quick inspection
    if articles:
        last_num, last_chunk = articles[-1]
        print("\nLast chunk header number:", last_num)
        print("Last chunk first line:", last_chunk.splitlines()[0] if last_chunk.splitlines() else "<empty>")

    documents_from_db: List[Document] = articles_to_chunks(articles)  # TODO This should be saved in the database

    print(bf25("Regulation applies"))

    print("Starting embedding")
    add_documents(documents_from_db)
    print("Finished embedding")

    qurey = QueryParams(query="Regulation applies")
    print(get_similar_documents(qurey))
