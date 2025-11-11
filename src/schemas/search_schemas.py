from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    history: List[str]


class SearchResponse(BaseModel):
    result: str
    sources: Optional[List[Dict[str, Any]]]


class LLMResponse(BaseModel):
    content: str = Field(..., description="Search query")
    openai_elapsed_ms: float
    sources: Optional[List[Dict[str, Any]]] = None
