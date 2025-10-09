from fastapi import APIRouter
from src.schemas.search_schemas import SearchRequest, SearchResponse
from src.api.services.openai_service import OpenAIService

router = APIRouter(prefix="/search", tags=["search"])

@router.post("/", response_model=SearchResponse, summary="Query OpenAI")
def search_documents(payload: SearchRequest):
    oa_service = OpenAIService()
    result = oa_service.generate_text(prompt=payload.query)
    response = SearchResponse(query=payload.query,result=result)
    return response
