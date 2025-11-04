# ADD AND/OR MODIFY STUFF HERE
OPENAI_MODEL = "gpt-4o"

# System prompt used by OpenAIService (can be iterated for evaluation)
SYSTEM_PROMPT = (
    "You are a helpful assistant. Always respond in English only, regardless of the language of the question."
)

# Retriever configuration
# Options: "bm25" for keyword search, "vector" for semantic search
# TODO: add hybrid retriever
RETRIEVER_TYPE = "bm25"
RETRIEVER_TOP_K = 5  # Number of documents to retrieve
