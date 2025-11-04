# src/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .routers import health, search
from .services.openai_service import OpenAIService
from .config import OPENAI_MODEL, RETRIEVER_TYPE, RETRIEVER_TOP_K, SYSTEM_PROMPT
from ..retrievers import BM25Retriever, VectorRetriever
import logging
from time import perf_counter

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init services
    logger.info("Wait for the service")
    app.state.openai = OpenAIService(model=OPENAI_MODEL, system_prompt=SYSTEM_PROMPT)

    # Initialize retriever based on config
    if RETRIEVER_TYPE == "bm25":
        app.state.retriever = BM25Retriever()
        logger.info("Initialized BM25 (keyword) retriever")
    elif RETRIEVER_TYPE == "vector":
        app.state.retriever = VectorRetriever()
        logger.info("Initialized Vector (semantic) retriever")
    else:
        raise ValueError(f"Unknown RETRIEVER_TYPE: {RETRIEVER_TYPE}. Use 'bm25' or 'vector'")

    app.state.top_k = RETRIEVER_TOP_K

    try:
        yield
    finally:
        # Clean the program
        svc = app.state.openai
        if hasattr(svc, "aclose"):
            await svc.aclose()
        elif hasattr(svc, "close"):
            svc.close()

app = FastAPI(lifespan=lifespan)


# Middleware to measure total request time and add timing headers when available
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        # ensure we still measure time for exceptions
        elapsed_ms = (perf_counter() - start) * 1000.0
        logging.getLogger(__name__).exception("Unhandled exception in request (%.2f ms): %s", elapsed_ms, exc)
        print(f"Request exception - elapsed: {elapsed_ms:.2f} ms")
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
    elapsed_ms = (perf_counter() - start) * 1000.0
    # Log total elapsed time for the request to terminal
    logging.getLogger(__name__).info("Request total elapsed: %.2f ms", elapsed_ms)
    print(f"Request total elapsed: {elapsed_ms:.2f} ms")

    return response

# Root
@app.get("/")
def read_root():
    return {"message": "Welcome to RegulAId API", "docs": "/docs"}

# Routers
app.include_router(health.router)
app.include_router(search.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, loop="asyncio")
