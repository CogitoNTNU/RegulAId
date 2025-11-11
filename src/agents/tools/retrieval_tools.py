"""LangChain tools for RAG retrieval from EU AI Act database."""

from langchain.tools import tool
from typing import Any


def create_retrieval_tools(retriever: Any, top_k: int = 5):
    """
    Create LangChain tools for RAG retrieval.

    This factory function creates tools that are bound to a specific retriever instance.
    The retriever can be BM25, Vector, or Hybrid - all have the same interface.

    Args:
        retriever: The retriever instance (BM25Retriever, VectorRetriever, or HybridRetriever)
        top_k: Default number of documents to retrieve

    Returns:
        List of LangChain tools
    """

    @tool
    def retrieve_eu_ai_act(query: str, k: int = top_k) -> str:
        """
        Retrieves relevant articles and information from the EU AI Act database.

        Use this tool to search for information about:
        - Risk classifications (prohibited, high-risk, limited-risk, minimal-risk)
        - AI system definitions and categories
        - Requirements and obligations for different AI systems
        - Specific articles and regulations

        Args:
            query: The search query describing what information you need
            k: Number of documents to retrieve (default: 5)

        Returns:
            Formatted string with retrieved articles and their metadata
        """
        results = retriever.search(query=query, k=k)

        if not results:
            return "No relevant information found in the EU AI Act database."

        formatted_output = "Retrieved EU AI Act information:\n\n"
        for i, doc in enumerate(results, 1):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})

            # Format metadata for better readability
            article_num = metadata.get('article_number', 'N/A')
            article_name = metadata.get('article_name', '')
            chapter = metadata.get('chapter_name', '')

            formatted_output += f"[{i}] Article {article_num}"
            if article_name:
                formatted_output += f" - {article_name}"
            if chapter:
                formatted_output += f" (Chapter: {chapter})"
            formatted_output += f"\n{content}\n\n"

        return formatted_output

    @tool
    def retrieve_risk_requirements(risk_level: str, k: int = 10) -> str:
        """
        Retrieves compliance requirements for a specific risk level.

        Use this tool to find out what requirements apply to AI systems
        classified at a particular risk level.

        Args:
            risk_level: One of: 'prohibited', 'high-risk', 'limited-risk', or 'minimal-risk'
            k: Number of documents to retrieve (default: 10)

        Returns:
            Formatted string with requirements and obligations
        """
        # Create a targeted query for requirements
        query = f"{risk_level} AI systems requirements obligations compliance"

        results = retriever.search(query=query, k=k)

        if not results:
            return f"No specific requirements found for {risk_level} AI systems."

        formatted_output = f"Requirements for {risk_level} AI systems:\n\n"
        for i, doc in enumerate(results, 1):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})

            article_num = metadata.get('article_number', 'N/A')
            article_name = metadata.get('article_name', '')

            formatted_output += f"[{i}] Article {article_num}"
            if article_name:
                formatted_output += f" - {article_name}"
            formatted_output += f"\n{content}\n\n"

        return formatted_output

    @tool
    def retrieve_system_type_info(system_type: str, k: int = 8) -> str:
        """
        Retrieves information about a specific type of AI system.

        Use this tool to learn about specific AI system categories like:
        - Biometric systems
        - Critical infrastructure
        - Education and vocational training
        - Employment and worker management
        - Law enforcement
        - General purpose AI

        Args:
            system_type: The type of AI system to search for
            k: Number of documents to retrieve (default: 8)

        Returns:
            Formatted string with information about that system type
        """
        query = f"{system_type} AI system definition requirements classification"

        results = retriever.search(query=query, k=k)

        if not results:
            return f"No specific information found for {system_type} AI systems."

        formatted_output = f"Information about {system_type} AI systems:\n\n"
        for i, doc in enumerate(results, 1):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})

            article_num = metadata.get('article_number', 'N/A')
            article_name = metadata.get('article_name', '')

            formatted_output += f"[{i}] Article {article_num}"
            if article_name:
                formatted_output += f" - {article_name}"
            formatted_output += f"\n{content}\n\n"

        return formatted_output

    return [
        retrieve_eu_ai_act,
        retrieve_risk_requirements,
        retrieve_system_type_info
    ]
