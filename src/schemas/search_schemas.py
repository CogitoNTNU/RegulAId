from pydantic import BaseModel, Field
from typing import List

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    history: List[str]


class SearchResponse(BaseModel):
    result: str
