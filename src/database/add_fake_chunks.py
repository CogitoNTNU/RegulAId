#!/usr/bin/env python3
"""
Script to add fake chunks to the existing database
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
    add_documents,
)


def load_fake_chunks(json_path: str) -> List[Document]:
    """
    Load fake chunks from fake-chunks.json and convert to LangChain Documents.
    Handles direct array format (no "chunks" wrapper).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # fake-chunks.json is a direct array
    if not isinstance(data, list):
        raise ValueError(f"Expected array format in {json_path}, got {type(data)}")

    chunks = []
    for chunk_data in data:
        # Create metadata from the chunk data
        metadata = {
            "id": chunk_data["id"],
            "type": chunk_data["type"],
            "paragraph_number": chunk_data.get("paragraph_number"),
            "page_range": chunk_data.get("page_range"),
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


def add_fake_chunks_to_database(num_docs: int = None):
    """
    Add fake chunks to the existing database.

    Args:
        num_docs: Number of documents to insert (None = all documents)
    """
    chunks_path = "data/processed/fake-chunks.json"
    print("Adding Fake Chunks to Database")

    # Step 1: Check connection
    print("\nChecking database connection...")
    if not check_connection():
        print("\n ERROR: Failed to connect to database. Please check your .env file.")
        return False

    # Step 2: Load and insert documents
    print(f"\nLoading fake chunks from {chunks_path}...")
    documents = load_fake_chunks(chunks_path)
    print(f"Loaded {len(documents)} fake chunks")

    if num_docs is not None:
        documents = documents[:num_docs]
        print(f"Inserting first {num_docs} documents...")
    else:
        print(f"Inserting all {len(documents)} documents...")

    # Step 3: Add documents (this will generate embeddings and add to existing indexes)
    add_documents(documents)

    print("\n" + "=" * 60)
    print("Fake chunks added to database successfully!")
    print("They now have embeddings for vector search and are indexed for BM25 search.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add fake chunks to the existing database")
    parser.add_argument(
        "--num-docs",
        type=int,
        default=None,
        help="Number of documents to insert (default: all documents)"
    )

    args = parser.parse_args()

    # Run the addition
    success = add_fake_chunks_to_database(num_docs=args.num_docs)

    if not success:
        sys.exit(1)