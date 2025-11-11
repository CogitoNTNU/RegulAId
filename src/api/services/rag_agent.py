"""RAG Agent for EU AI Act question answering using LangChain."""

import os
from dotenv import load_dotenv
import logging
from time import perf_counter
import json
import ast

from langchain.agents import create_agent
from langchain.tools import tool

from langchain_openai import ChatOpenAI

from src.schemas.search_schemas import LLMResponse

# Load environment variables from a .env file if present
load_dotenv()

logger = logging.getLogger(__name__)


class RAGAgent:
    """RAG Agent that uses a LangChain agent and a `search` tool backed by the project's retriever."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-5") -> None:
        # Accepts 'model' kwarg to match caller in src/api/main.py
        self.api_key = api_key or os.getenv("OPENAI_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing OPENAI_KEY environment variable or api_key."
            )
        # Keep model string in self.model for downstream agent creation
        self.model = model
        # Ensure ChatOpenAI is given the resolved API key
        self.client_llm = ChatOpenAI(
            model=self.model,
            temperature=0.1,
            max_tokens=1000,
            api_key= self.api_key,
            timeout=30)

    def generate_text(self, prompt: str, history, retriever, top_k: int = 5) -> LLMResponse:
        """Always use a LangChain agent with a `search` tool that calls the provided retriever.

        Args:
            prompt: The user prompt (context already applied by caller if desired).
            history: Optional list of prior messages (strings) to append to the prompt.
            retriever: Required retriever object exposing a .search(query, k, filters?) method.
            top_k: Number of documents the tool should return when called.

        Returns:
            LLMResponse with content and openai_elapsed_ms (ms spent invoking the agent).
        """
        if retriever is None:
            raise ValueError("retriever is required for LangChain agent execution")

        # Compose history safely into a single string to send to the agent
        s = "\n".join(history) if history else ""

        # Build a simple search tool that calls the retriever
        captured_sources: list[dict] = []

        @tool(
            "search_ai_act",
            description=(
                    "Search the EU AI ACT documents. Accepts either a plain text query string OR a JSON/dict with"
                    " keys 'query' and optional 'filters'. Use 'filters' to restrict results by metadata (see docstring)."
            ),
        )
        def _search_tool(payload) -> str:
            """Search the EU AI ACT documents using the project's retriever.

            How the LLM should use this tool (explicit instructions):
            - The tool accepts EITHER:
              1) A plain text string containing the user query, e.g. "What are the risk levels in the AI Act?";
              2) A JSON object (or Python dict) with the shape:
                 {
                   "query": "your query string",
                   "filters": { <metadata filter spec> }
                 }

            - The `filters` object uses the same operators the retrievers expect (see `src/retrievers/filters.py`):
                * equality (default / alias 'is'):  "article_number": "1"
                * contains:                         "title": {"op": "contains", "value": "accuracy"}
                * is one of:                        "tags": {"op": "is one of", "value": ["recital", "article"]}
                * numeric comparisons:              "score": {"op": ">", "value": 3}

            - Examples the LLM can use when deciding to call the tool:
                - Plain query: "What is the risk level in the AI Act?"
                - With metadata filter (article 15):
                  {"query": "accuracy metrics", "filters": {"article_number": "15"}}

            - Notes for the LLM:
                * If you need to restrict results to a specific article or paragraph, include a `filters` object
                  with the appropriate metadata keys (e.g., 'article_number', 'paragraph_number').
                * Operator names are case-insensitive strings; the simplest form is to pass a plain value for
                  equality checks.

            Args:
                payload: plain string query, a JSON string representing the dict above, or a Python dict.

            Returns:
                Concatenated snippets and metadata from top-k matching documents.
            """

            nonlocal captured_sources
            captured_sources.clear()

            # Normalize input to (query, filters)
            query_text = ""
            filters = None

            # If the tool received a dict-like payload
            if isinstance(payload, dict):
                query_text = payload.get("query") or payload.get("q") or ""
                filters = payload.get("filters")

                # Safely log values (avoid string concatenation with non-strings)
                logger.debug("search tool received dict payload - query_text=%r filters=%r", query_text, filters)

            elif isinstance(payload, str):
                s = payload.strip()
                # Try to parse JSON payloads produced by the agent
                if (s.startswith("{") and s.endswith("}")) or s.startswith("["):
                    parsed = None
                    try:
                        parsed = json.loads(s)
                    except Exception:
                        # Many LLMs emit Python-style dicts with single quotes; try ast.literal_eval as a fallback
                        try:
                            parsed = ast.literal_eval(s)
                        except Exception:
                            parsed = None

                    if isinstance(parsed, dict):
                        query_text = parsed.get("query") or parsed.get("q") or ""
                        filters = parsed.get("filters")
                    else:
                        # Unexpected JSON/Python shape; fall back to raw string
                        query_text = s
                else:
                    query_text = s
            else:
                # Fallback for other input types
                query_text = str(payload)

            # Call the retriever with filters when provided
            try:
                results = retriever.search(query=query_text, k=top_k, filters=filters)
            except TypeError:
                # Some retriever implementations may expect positional args
                try:
                    results = retriever.search(query_text, top_k, filters)
                except TypeError:
                    # Old retriever signature without filters
                    results = retriever.search(query_text, top_k)

            if not results:
                return "No relevant documents found."

            out = []
            for i, r in enumerate(results, 1):
                md = r.get("metadata") or {}
                content_snippet = r.get("content", "")
                # Keep a structured representation for callers
                captured_sources.append({"index": i, "content": content_snippet, "metadata": md})
                out.append(f"[{i}] {content_snippet} \nMETADATA: {md}")
            return "\n\n".join(out)

        agent = create_agent(model=self.client_llm, tools=[_search_tool], system_prompt="You are a helpful assistant")

        # Invoke the agent with a single user message
        start = perf_counter()
        try:
            agent_response = agent.invoke(
                {"messages": [{"role": "user", "content": prompt + ("\n\n" + s if s else "")}]})

        except Exception as e:
            # Log the exception and return a safe response so callers don't get 500s
            logger.exception("OpenAI agent invocation failed: %s", e)
            return LLMResponse(content=f"OpenAI agent error: {e}", openai_elapsed_ms=0.0, sources=[])
        openai_elapsed_ms = (perf_counter() - start) * 1000.0

        result = agent_response['messages'][-1].content

        logger.info("RAGAgent.generate_text: agent call took %.2f ms", openai_elapsed_ms)
        # Return any captured sources from tool calls so routers can expose them to clients
        return LLMResponse(content=result, openai_elapsed_ms=openai_elapsed_ms, sources=captured_sources)
