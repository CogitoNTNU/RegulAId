from fastapi import APIRouter, Request
from ...schemas.search_schemas import SearchRequest, SearchResponse
from time import perf_counter
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/", response_model=SearchResponse, summary="Query OpenAI")
def search_documents(payload: SearchRequest, request: Request):
    # Measure total backend time for handling this request (including OpenAI call)
    print("I am in here")
    backend_start = perf_counter()

    # Retrieve relevant documents using retriever
    retriever = request.app.state.retriever
    top_k = request.app.state.top_k
    retrieved_docs = retriever.search(query=payload.query, k=top_k)

    # Format retrieved documents as context
    context = ""
    if retrieved_docs:
        context = "Context from EU AI Act documents:\n\n"
        for i, doc in enumerate(retrieved_docs, 1):
            context += f"[{i}] {doc['content']}\n\n"
        context += "---\n\n"
    print(context)
    # Prepend context to the query
    enhanced_query = context + payload.query if context else payload.query

    oa_service = request.app.state.openai
    # OpenAIService.generate_text returns (result, openai_elapsed_ms)
    # result, openai_elapsed_ms = oa_service.generate_text(prompt=enhanced_query, history=payload.history)

    llmResponse = oa_service.generate_text(prompt=enhanced_query, history=payload.history)

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

    print(f"{type(retrieved_docs)} {retrieved_docs}")

    # Return only the result to the client
    return SearchResponse(result=llmResponse.content, sources=retrieved_docs)
