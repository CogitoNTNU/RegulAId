from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
from time import perf_counter

# Load environment variables from a .env file if present
load_dotenv()

logger = logging.getLogger(__name__)

class OpenAIService:
    """Simple wrapper around OpenAI Chat Completions API."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-5") -> None:
        self.api_key = api_key or os.getenv("OPENAI_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing OpenAI API key. Set OPENAI_KEY environment variable or pass api_key."
            )
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        logger.info("OpenAIService initialized")

    def generate_text(self, prompt: str, history) -> str:
        """Generate a chat completion response for a prompt."""
        s = "these are the previously sent messages:".join(history)

        # Time the OpenAI API call itself (network + processing by OpenAI)
        start = perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Always respond in English only, regardless of the language of the question."},
                {"role": "user", "content": prompt + s }
            ],
        )
        openai_elapsed_ms = (perf_counter() - start) * 1000.0

        content = response.choices[0].message.content
        result = content.strip() if content is not None else ""

        logger.info("OpenAIService.generate_text: openai call took %.2f ms", openai_elapsed_ms)

        # Return the content plus the OpenAI elapsed time in milliseconds
        return result, openai_elapsed_ms
