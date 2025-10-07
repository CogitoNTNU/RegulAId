import os
import psycopg
from langchain_ollama import OllamaEmbeddings


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
