from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query") 

class SearchResponse(BaseModel):
    query: str 
    result: str 
