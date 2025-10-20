#!/usr/bin/env python3
"""
Database initialization script
"""

import os
import sys
import json
from typing import List
from langchain_core.documents import Document
import argparse
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import (
    check_connection,
    create_table,
    create_hnsw_index,
    create_bm25_index,
    add_documents,
)

def load_chunks(json_path: str) -> List[Document]:
    """
    Load preprocessed chunks from aiact-chunks.json and convert to LangChain Documents.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = []
    for chunk_data in data["chunks"]:
        # Create metadata from the chunk data
        metadata = {
            "id": chunk_data["id"],
            "type": chunk_data["type"],
            "paragraph_number": chunk_data.get("paragraph_number"),
            "page_range": chunk_data["page_range"],
            "chapter_number": chunk_data.get("chapter_number"),
            "chapter_name": chunk_data.get("chapter_name"),
            "section_number": chunk_data.get("section_number"),
            "section_name": chunk_data.get("section_name"),
            "article_number": chunk_data.get("article_number"),
            "article_name": chunk_data.get("article_name"),
            "annex_number": chunk_data.get("annex_number"),
            "annex_name": chunk_data.get("annex_name"),
        }

        # Create Document with text as page_content and metadata
        chunks.append(Document(page_content=chunk_data["text"], metadata=metadata))

    return chunks


def init_database(num_docs: int = None):
    """
    Initialize the database with all necessary setup.

    Args:
        num_docs: Number of documents to insert (None = all documents)
    """
    chunks_path = "data/processed/aiact-chunks.json"
    print("Database Initialization")

    # Step 1: Check connection
    print("\nChecking database connection...")
    if not check_connection():
        print("\n ERROR: Failed to connect to database. Please check your .env file.")
        return False

    # Step 2: Create table
    print("\nCreating table...")
    create_table()

    # Step 3: Create HNSW index
    print("\nCreating HNSW index for vector search...")
    create_hnsw_index()

    # Step 4: Create BM25 index
    print("\nCreating BM25 index for keyword search...")
    create_bm25_index()

    # Step 5: Load and insert documents
    print(f"\nLoading and inserting documents...")
    documents = load_chunks(chunks_path)
    print(f"Loaded {len(documents)} chunks")

    if num_docs is not None:
        documents = documents[:num_docs]
        print(f"Inserting first {num_docs} documents...")
    else:
        print(f"Inserting all {len(documents)} documents...")

    add_documents(documents)

    print("\n" + "=" * 60)
    print("Database initialization complete!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize the database with documents")
    parser.add_argument(
        "--num-docs",
        type=int,
        default=None,
        help="Number of documents to insert (default: all documents)"
    )

    args = parser.parse_args()

    # Run initialization
    success = init_database(num_docs=args.num_docs)

    if not success:
        sys.exit(1)
