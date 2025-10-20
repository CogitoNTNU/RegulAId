"""Database connection utilities."""

import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

def get_connection_string() -> str:
    """Get PostgreSQL connection string for SQLAlchemy."""
    return f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"


def get_psycopg_connection_string() -> str:
    """Get PostgreSQL connection string for psycopg."""
    return f"dbname='{os.getenv('DB_NAME')}' user='{os.getenv('DB_USER')}' password='{os.getenv('DB_PASSWORD')}' host='{os.getenv('DB_HOST')}' port='{os.getenv('DB_PORT')}'"


def check_connection() -> bool:
    """Check if database connection is successful."""
    try:
        conn = psycopg.connect(get_psycopg_connection_string())
        conn.close()
        print("Database connection successful")
        return True
    except Exception as e:
        print(f"Database connection ERROR: {str(e)}")
        return False
