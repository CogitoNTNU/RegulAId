"""Checklist endpoint for compliance requirements generation."""

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from src.schemas.agent_schemas import ChecklistRequest, ChecklistResponse
from time import perf_counter
import logging
import json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/checklist", tags=["agents"])


@router.post("/", response_model=ChecklistResponse, summary="Generate Compliance Checklist")
def generate_checklist(payload: ChecklistRequest, request: Request):
    """
    Generate a comprehensive compliance checklist for an AI system.

    This endpoint uses a LangChain agent that:
    1. Takes the risk level classification as input
    2. Retrieves relevant compliance requirements from the EU AI Act
    3. Generates a structured checklist with actionable items
    4. Organizes items by category (documentation, technical, governance, etc.)
    5. Prioritizes requirements by importance

    Each checklist item includes:
    - Requirement description
    - Applicable EU AI Act articles
    - Priority level (high/medium/low)
    - Category
    """
    # Measure total backend time
    backend_start = perf_counter()

    # Get checklist agent from app state
    checklist_agent = request.app.state.checklist_agent

    # Generate checklist
    logger.info("Starting compliance checklist generation for risk level: %s", payload.risk_level)
    result = checklist_agent.generate_checklist(payload)

    backend_elapsed_ms = (perf_counter() - backend_start) * 1000.0
    backend_elapsed_s = backend_elapsed_ms / 1000.0

    logger.info("Checklist generation completed in %.2f s", backend_elapsed_s)
    print(f"Checklist generation completed in {backend_elapsed_s:.2f} s")

    return result


@router.post("/stream", summary="Generate Compliance Checklist with Streaming")
async def generate_checklist_stream(payload: ChecklistRequest, request: Request):
    """
    Generate a compliance checklist with real-time streaming updates.

    This endpoint streams task updates as the agent works:
    - Analyzing risk requirements
    - Searching system-specific information
    - Generating checklist items

    Returns Server-Sent Events (SSE) with updates.
    """
    checklist_agent = request.app.state.checklist_agent

    async def event_generator():
        """Generate SSE events from agent streaming updates."""
        try:
            logger.info("Starting streaming checklist generation for risk level: %s", payload.risk_level)

            async for update in checklist_agent.generate_checklist_streaming(payload):
                # Format as Server-Sent Event
                event_data = json.dumps(update)
                yield f"data: {event_data}\n\n"

            logger.info("Streaming checklist generation completed")

        except Exception as e:
            logger.error(f"Error in streaming checklist generation: {str(e)}")
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
