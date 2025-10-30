import os
from dotenv import load_dotenv
import logging
from time import perf_counter

from langchain.agents import create_agent
from langchain.tools import tool

from langchain_openai import ChatOpenAI

from src.schemas.search_schemas import LLMResponse

# Load environment variables from a .env file if present
load_dotenv()

logger = logging.getLogger(__name__)


class OpenAIService:
    """Wrapper that uses a LangChain agent and a `search` tool backed by the project's retriever."""

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
            api_key=self.api_key,
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
        @tool("search_ai_act", description="Search the EU AI ACT documents")
        def _search_tool(query: str) -> str:
            """Search the EU AI ACT documents using the project's retriever.

            Args:
                query: user query string

            Returns:
                Concatenated snippets and metadata from top-k matching documents.
            """
            try:
                results = retriever.search(query=query, k=top_k)
            except TypeError:
                # Some retriever implementations expect (query, k) positional args
                results = retriever.search(query, top_k)
            if not results:
                return "No relevant documents found."
            out = []
            for i, r in enumerate(results, 1):
                md = r.get("metadata") or {}
                out.append(f"[{i}] {r.get('content', '')} \nMETADATA: {md}")
            return "\n\n".join(out)

        agent = create_agent(model=self.client_llm, tools=[_search_tool], system_prompt="You are a helpful assistant")

        # Invoke the agent with a single user message
        start = perf_counter()
        agent_response = agent.invoke({"messages": [{"role": "user", "content": prompt + ("\n\n" + s if s else "")} ]})
        openai_elapsed_ms = (perf_counter() - start) * 1000.0

        # Extract text from agent response (support dict-like or object responses)
        content = None
        if isinstance(agent_response, dict):
            content = agent_response.get("output") or agent_response.get("text") or agent_response.get("result")
        else:
            content = getattr(agent_response, "output", None) or getattr(agent_response, "text", None)

        if content is None:
            content = str(agent_response)

        result = content.strip() if content is not None else ""

        logger.info("OpenAIService.generate_text (agent): agent call took %.2f ms", openai_elapsed_ms)
        return LLMResponse(content=result, openai_elapsed_ms=openai_elapsed_ms)
