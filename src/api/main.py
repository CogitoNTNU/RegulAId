# src/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.routers import health, search
from src.api.services.openai_service import OpenAIService
from src.api.config import OPENAI_MODEL
import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init services
    logger.info("Wait for the service")
    app.state.openai = OpenAIService(model=OPENAI_MODEL)
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
