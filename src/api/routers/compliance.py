"""Unified compliance workflow endpoint."""

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import logging
import json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/compliance", tags=["compliance"])


class ComplianceWorkflowRequest(BaseModel):
    """Request for the full compliance workflow."""
    ai_system_description: str
    additional_info: Optional[str] = None


@router.post("/workflow/stream", summary="Run Full Compliance Workflow")
async def run_compliance_workflow(payload: ComplianceWorkflowRequest, request: Request):
    """
    Run the complete compliance workflow with streaming updates.

    This endpoint orchestrates:
    1. AI system classification
    2. Compliance checklist generation (if classification succeeds)

    The workflow handles all logic in the backend - frontend just displays events.

    Returns Server-Sent Events (SSE) with workflow progress.
    """
    print("\n" + "="*80)
    print("COMPLIANCE WORKFLOW ENDPOINT HIT!!!")
    print(f"Payload: {payload.ai_system_description[:100]}...")
    print("="*80 + "\n")

    logger.info("========== /compliance/workflow/stream ENDPOINT HIT ==========")
    logger.info(f"Request payload: {payload.ai_system_description[:100]}...")

    # Get workflow from app state
    compliance_workflow = request.app.state.compliance_workflow
    logger.info(f"Got workflow from app state: {compliance_workflow}")
    print(f"Workflow object: {compliance_workflow}")

    async def event_generator():
        """Generate SSE events from workflow."""
        try:
            print("EVENT GENERATOR STARTED!!!")
            logger.info("Starting compliance workflow")

            print(f"About to call run_streaming on {compliance_workflow}")
            async for event in compliance_workflow.run_streaming(
                system_description=payload.ai_system_description,
                additional_info=payload.additional_info
            ):
                # Format as Server-Sent Event
                print(f"Got event from workflow: {event.get('type', 'unknown')}")
                event_data = json.dumps(event)
                yield f"data: {event_data}\n\n"

            print("EVENT GENERATOR COMPLETED!!!")
            logger.info("Compliance workflow completed")

        except Exception as e:
            logger.error(f"Error in compliance workflow: {str(e)}")
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
