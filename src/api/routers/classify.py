"""Classification endpoint for AI system risk assessment."""

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from src.schemas.agent_schemas import ClassificationRequest, ClassificationResponse
from time import perf_counter
import logging
import json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/classify", tags=["agents"])


@router.post("/", response_model=ClassificationResponse, summary="Classify AI System")
def classify_ai_system(payload: ClassificationRequest, request: Request):
    """
    Classify an AI system according to EU AI Act risk levels.

    This endpoint uses a LangChain agent that:
    1. Analyzes the AI system description
    2. Retrieves relevant EU AI Act articles using RAG
    3. Determines if more information is needed
    4. Classifies the system into risk categories:
       - Prohibited
       - High-risk
       - Limited-risk
       - Minimal-risk

    If the agent needs more information, it will return clarifying questions.
    """
    # Measure total backend time
    backend_start = perf_counter()

    # Get classification agent from app state
    classification_agent = request.app.state.classification_agent

    # Run classification
    logger.info("Starting AI system classification")
    result = classification_agent.classify(payload)

    backend_elapsed_ms = (perf_counter() - backend_start) * 1000.0
    backend_elapsed_s = backend_elapsed_ms / 1000.0

    logger.info("Classification completed in %.2f s", backend_elapsed_s)
    print(f"Classification completed in {backend_elapsed_s:.2f} s")

    return result


@router.post("/stream", summary="Classify AI System with Streaming")
async def classify_ai_system_stream(payload: ClassificationRequest, request: Request):
    """
    Classify an AI system with real-time streaming updates.

    This endpoint streams task updates as the agent works:
    - Searching EU AI Act database
    - Analyzing risk requirements
    - Computing classification

    Returns Server-Sent Events (SSE) with updates.
    """
    classification_agent = request.app.state.classification_agent

    async def event_generator():
        """Generate SSE events from agent streaming updates."""
        try:
            logger.info("Starting streaming classification")

            async for update in classification_agent.classify_streaming(payload):
                # Format as Server-Sent Event
                event_data = json.dumps(update)
                yield f"data: {event_data}\n\n"

            logger.info("Streaming classification completed")

        except Exception as e:
            logger.error(f"Error in streaming classification: {str(e)}")
            error_data = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )
