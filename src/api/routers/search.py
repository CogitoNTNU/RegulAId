from fastapi import APIRouter, Request
from src.schemas.search_schemas import SearchRequest, SearchResponse
from time import perf_counter
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/", response_model=SearchResponse, summary="Query OpenAI")
def search_documents(payload: SearchRequest, request: Request):
    # Measure total backend time for handling this request (including OpenAI call)
    backend_start = perf_counter()

    oa_service = request.app.state.openai
    # OpenAIService.generate_text returns (result, openai_elapsed_ms)
    result, openai_elapsed_ms = oa_service.generate_text(prompt=payload.query, history=payload.history)

    backend_elapsed_ms = (perf_counter() - backend_start) * 1000.0

    # Compute individual times in seconds (OpenAI call and rest of backend)
    openai_s = openai_elapsed_ms / 1000.0
    rest_ms = backend_elapsed_ms - openai_elapsed_ms
    # guard against negative due to float noise
    if rest_ms < 0:
        rest_ms = 0.0
    rest_s = rest_ms / 1000.0

    # Print to terminal in seconds with two decimals as requested
    print(f"OpenAI call: {openai_s:.2f} s; Rest of backend: {rest_s:.2f} s")
    logger.info("search_documents timings - openai: %.2f s, rest: %.2f s", openai_s, rest_s)

    # Return only the result to the client
    return SearchResponse(result=result)
