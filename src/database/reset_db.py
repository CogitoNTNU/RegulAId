#!/usr/bin/env python3
"""
Helper script to drop the database table.
Use this if you need to reset the database schema.
"""

import os
import sys
import psycopg
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import get_psycopg_connection_string

load_dotenv()


def reset_table(table_name: str = None):
    """Drop and recreate the table."""
    if table_name is None:
        table_name = os.getenv("COLLECTION_NAME")

    try:
        with psycopg.connect(get_psycopg_connection_string()) as conn:
            with conn.cursor() as cur:
                # Drop table if exists
                cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
                conn.commit()
        print(f"Dropped table '{table_name}' (if it existed)")
        return True
    except Exception as e:
        print(f"ERROR dropping table: {str(e)}")
        return False


if __name__ == "__main__":
    table_name = os.getenv("COLLECTION_NAME")

    if not table_name:
        print("ERROR: COLLECTION_NAME not found in .env file")
        sys.exit(1)

    print(f"\nThis will DROP the table '{table_name}' and all its data.")
    response = input("Are you sure you want to continue? (yes/no): ")

    if response.lower() == "yes":
        if reset_table(table_name):
            print("\nTable reset complete!")
            print("Run 'uv run python src/database/init_db.py' to recreate with correct schema.")
    else:
        print("\nCancelled.")
