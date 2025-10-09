from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

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

    def generate_text(self, prompt: str) -> str:
        """Generate a chat completion response for a prompt."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
