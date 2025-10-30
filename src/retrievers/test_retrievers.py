#!/usr/bin/env python3
"""
Test script for retrievers.
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import check_connection
from retrievers import BM25Retriever, VectorRetriever, HybridRetriever


def test_retrievers(query: str = "What are high-risk AI systems?", k: int = 3):
    print("Testing Retrievers")
    print(f"\nQuery: '{query}'")
    print(f"Retrieving top {k} results\n")

    # Test BM25 retriever
    print("BM25 Retriever:")
    bm25 = BM25Retriever()
    bm25_results = bm25.search(query, k=k)

    if bm25_results:
        for i, result in enumerate(bm25_results, 1):
            print(f"\n{i}. Score: {result['score']:.4f}")
            print(f"   Content: {result['content'][:300]}...")
            print(f"   Metadata: {result['metadata'].get('id', 'N/A')}")

    else:
        print("No results found")

    # Test Vector retriever
    print("\nSemantic Search:")
    vector = VectorRetriever()
    vector_results = vector.search(query, k=k)

    if vector_results:
        for i, result in enumerate(vector_results, 1):
            print(f"\n{i}. Similarity: {result['similarity']:.4f}")
            print(f"   Content: {result['content'][:300]}...")
            print(f"   Metadata: {result['metadata'].get('id', 'N/A')}")
    else:
        print("No results found")

    # Test Hybrid retriever
    print("\nHybrid Search (RRF):")
    hybrid = HybridRetriever()
    hybrid_results = hybrid.search(
        query,
        k=k,
        rrf_k=60,
        bm25_weight=1.0,
        vector_weight=1.0,
        bm25_top_k=100,
        vector_top_k=100,
    )

    if hybrid_results:
        for i, result in enumerate(hybrid_results, 1):
            print(f"\n{i}. RRF Score: {result['rrf_score']:.4f}")
            print(f"   Content: {result['content'][:300]}...")
            print(f"   Metadata: {result.get('id', 'N/A')}")
    else:
        print("No results found")


def main():
    """Main function."""
    print("Retriever Testing")

    # Check database connection
    print("\nChecking database connection...")
    if not check_connection():
        print("\nERROR Database connection failed!")
        return

    # Test with sample queries
    test_retrievers(
        query="What is recital 34 is in the EU AI ACT?",
        k=3
    )

if __name__ == "__main__":
    main()
