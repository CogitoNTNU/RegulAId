"""Checklist endpoint for compliance requirements generation."""

from fastapi import APIRouter, Request
from src.schemas.agent_schemas import ChecklistRequest, ChecklistResponse
from time import perf_counter
import logging

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
