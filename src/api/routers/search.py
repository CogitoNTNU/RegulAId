from fastapi import APIRouter, Request
from ...schemas.search_schemas import SearchRequest, SearchResponse
from time import perf_counter
import logging
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/", response_model=SearchResponse, summary="Query OpenAI")
def search_documents(payload: SearchRequest, request: Request):
    # Measure total backend time for handling this request (including OpenAI call)
    print("I am in here")
    backend_start = perf_counter()

    # Don't prefetch documents here — let the LLM agent call the `search` tool itself.
    retriever = request.app.state.retriever
    top_k = request.app.state.top_k

    oa_service = request.app.state.openai
    # Pass the retriever so the service can register a `search` tool the agent will call
    llmResponse = oa_service.generate_text(prompt=payload.query, history=payload.history, retriever=retriever, top_k=top_k)

    # If the OpenAIService returned an error placeholder, return a 502 so clients get a meaningful status
    if llmResponse.content.startswith("OpenAI agent error:"):
        logging.getLogger(__name__).error("OpenAIService reported an error: %s", llmResponse.content)
        return JSONResponse(status_code=502, content={"detail": llmResponse.content})

    backend_elapsed_ms = (perf_counter() - backend_start) * 1000.0

    # Compute individual times in seconds (OpenAI call and rest of backend)
    openai_s = llmResponse.openai_elapsed_ms / 1000.0
    rest_ms = backend_elapsed_ms - llmResponse.openai_elapsed_ms
    # guard against negative due to float noise
    if rest_ms < 0:
        rest_ms = 0.0
    rest_s = rest_ms / 1000.0

    # Print to terminal in seconds with two decimals as requested
    print(f"OpenAI call: {openai_s:.2f} s; Rest of backend: {rest_s:.2f} s")
    logger.info("search_documents timings - openai: %.2f s, rest: %.2f s", openai_s, rest_s)

    # Return captured sources from the tool (if any)
    return SearchResponse(result=llmResponse.content, sources=llmResponse.sources or [])
