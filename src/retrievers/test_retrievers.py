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



def test_retrievers(query: str = "What are high-risk AI systems?", k: int = 30):
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
            print(f"   Metadata: {result['metadata']}")

    else:
        print("No results found")

    # --- New: test filtering by metadata article_number and paragraph_number ---
    print("\nBM25 Filter Test: metadata article_number == '1' and paragraph_number == '1'")
    article_id = "1"
    paragraph_id = "1"
    try:
        filtered_results = bm25.search(query, k=k, filters={"article_number": article_id, "paragraph_number": paragraph_id})
        if filtered_results:
            for i, result in enumerate(filtered_results, 1):
                print(f"\n{i}. Score: {result['score']:.4f}")
                print(f"   Content: {result['content'][:300]}...")
                md = result.get('metadata', {})
                print(f"   Metadata article_number: {md.get('article_number', 'N/A')}, paragraph_number: {md.get('paragraph_number', 'N/A')}")

            # Basic sanity check: at least one returned result should have the expected article_number and paragraph_number
            assert any(
                (r.get('metadata', {}).get('article_number') == article_id and r.get('metadata', {}).get('paragraph_number') == paragraph_id)
                for r in filtered_results
            ), (
                f"No filtered result contained the expected article_number '{article_id}' and paragraph_number '{paragraph_id}'"
            )
        else:
            print(f"No filtered results found for article_number: {article_id} and paragraph_number: {paragraph_id}")
    except Exception as e:
        print(f"Error while running filtered BM25 search: {e}")

    # --- New: test filtering by metadata article_number and paragraph_number for semantic/vector search ---
    print("\nVector Filter Test: metadata article_number == '1' and paragraph_number == '1'")
    try:
        vector = VectorRetriever()
        v_filtered = vector.search(query, k=k, filters={"article_number": article_id, "paragraph_number": paragraph_id})
        if v_filtered:
            for i, result in enumerate(v_filtered, 1):
                print(f"\n{i}. Similarity: {result.get('similarity', 0):.4f}")
                print(f"   Content: {result['content'][:300]}...")
                md = result.get('metadata', {})
                print(f"   Metadata article_number: {md.get('article_number', 'N/A')}, paragraph_number: {md.get('paragraph_number', 'N/A')}")

            # Basic sanity check: at least one returned vector result should have the expected article_number and paragraph_number
            assert any(
                (r.get('metadata', {}).get('article_number') == article_id and r.get('metadata', {}).get('paragraph_number') == paragraph_id)
                for r in v_filtered
            ), (
                f"No filtered vector result contained the expected article_number '{article_id}' and paragraph_number '{paragraph_id}'"
            )
        else:
            print(f"No filtered vector results found for article_number: {article_id} and paragraph_number: {paragraph_id}")
    except Exception as e:
        print(f"Error while running filtered vector search: {e}")

    # Test Vector retriever (unfiltered)
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
