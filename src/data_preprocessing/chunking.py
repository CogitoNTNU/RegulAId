import re
import unicodedata
import json
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


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
