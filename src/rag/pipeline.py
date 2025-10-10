"""RAG Pipeline for EU AI ACT Q&A bot"""

import os
import sys
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrievers import BM25Retriever, VectorRetriever


class RAGPipeline:
    """RAG pipeline for answering questions about the EU AI Act."""

    def __init__(self, retriever_type: str = "bm25", top_k: int = 5):
        """
        Initialize the RAG pipeline.

        Args:
            retriever_type: Type of retriever ("bm25", "vector", or "hybrid")
            top_k: Number of documents to retrieve
        """
        self.retriever_type = retriever_type
        self.top_k = top_k
        self.retriever = self._init_retriever(retriever_type)

    def _init_retriever(self, retriever_type: str):
        """Initialize the retriever."""
        if retriever_type == "bm25":
            return BM25Retriever()
        elif retriever_type == "vector":
            return VectorRetriever()
        elif retriever_type == "hybrid":
            raise NotImplementedError("HybridRetriever not yet implemented")
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")

    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string.

        TODO: Implement context formatting
        - Take list of retrieved documents
        - Format them into a string for the LLM prompt
        - Include document IDs for citations
        """
        raise NotImplementedError("format_context not yet implemented")

    def generate_prompt(self, query: str, context: str) -> str:
        """
        Generate prompt for the LLM.

        TODO: Implement prompt generation
        - Combine context and user query
        - Add instructions for the LLM
        - Return complete prompt string
        """
        raise NotImplementedError("generate_prompt not yet implemented")

    def generate_answer(self, prompt: str) -> str:
        """
        Generate answer using LLM.

        TODO: Implement LLM call
        - Send prompt to LLM (Ollama, OpenAI, etc.)
        - Get response
        - Return answer text
        """
        raise NotImplementedError("generate_answer not yet implemented")

    def answer(self, query: str) -> Dict[str, Any]:
        """
        Answer a question using RAG.

        TODO: Implement RAG pipeline
        1. Retrieve relevant documents using self.retriever.search()
        2. Format documents into context using format_context()
        3. Generate prompt using generate_prompt()
        4. Get answer from LLM using generate_answer()
        5. Return dict with answer and sources
        """
        raise NotImplementedError("answer not yet implemented")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EU AI ACT Q&A")
    parser.add_argument("query", type=str, help="Your question")
    parser.add_argument(
        "--retriever",
        type=str,
        default="bm25",
        choices=["bm25", "vector", "hybrid"],
        help="Retriever type (default: bm25)"
    )

    args = parser.parse_args()

    # TODO: Implement CLI
    # 1. Initialize RAGPipeline with args.retriever
    # 2. Call rag.answer(args.query)
    # 3. Print the answer and sources

    print(f"Query: {args.query}")
    print(f"Retriever: {args.retriever}")
    print("\nTODO: Implement CLI")
