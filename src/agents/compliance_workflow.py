"""LangGraph workflow for compliance agents orchestration."""

from typing import TypedDict, Optional, AsyncGenerator, Any
from langgraph.graph import StateGraph, END
import logging

from src.schemas.agent_schemas import (
    ClassificationRequest,
    ChecklistRequest,
)

logger = logging.getLogger(__name__)


# Define workflow state
class ComplianceWorkflowState(TypedDict):
    """State for the compliance workflow."""
    system_description: str
    additional_info: Optional[str]
    classification: Optional[dict]
    checklist: Optional[dict]
    needs_more_info: bool
    questions: Optional[list]


class ComplianceWorkflow:
    """Simple LangGraph workflow for compliance agents."""

    def __init__(self, classification_agent: Any, checklist_agent: Any):
        """
        Initialize the compliance workflow.

        Args:
            classification_agent: ClassificationAgent instance
            checklist_agent: ChecklistAgent instance
        """
        self.classification_agent = classification_agent
        self.checklist_agent = checklist_agent
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create graph
        workflow = StateGraph(ComplianceWorkflowState)

        # Add nodes
        workflow.add_node("classify", self._classify_node)
        workflow.add_node("checklist", self._checklist_node)

        # Set entry point
        workflow.set_entry_point("classify")

        # Add conditional edge: if needs more info, stop. Otherwise, continue.
        workflow.add_conditional_edges(
            "classify",
            self._should_continue,
            {
                "checklist": "checklist",
                "end": END
            }
        )

        # Checklist always ends
        workflow.add_edge("checklist", END)

        return workflow.compile()

    def _should_continue(self, state: ComplianceWorkflowState) -> str:
        """Decide whether to continue to checklist or stop."""
        if state.get("needs_more_info", False):
            return "end"
        return "checklist"

    async def _classify_node(self, state: ComplianceWorkflowState) -> dict:
        """Classification node - runs the classification agent."""
        logger.info("Running classification node")

        # Create request
        request = ClassificationRequest(
            ai_system_description=state["system_description"],
            additional_info=state.get("additional_info")
        )

        # Run classification synchronously (workflow handles streaming separately)
        result = self.classification_agent.classify(request)

        # Update state
        return {
            "classification": result.model_dump(),
            "needs_more_info": result.needs_more_info,
            "questions": result.questions if result.needs_more_info else None
        }

    async def _checklist_node(self, state: ComplianceWorkflowState) -> dict:
        """Checklist node - runs the checklist agent."""
        logger.info("Running checklist node")

        classification = state["classification"]

        # Create request
        request = ChecklistRequest(
            risk_level=classification["risk_level"],
            system_type=classification.get("system_type"),
            system_description=state["system_description"]
        )

        # Run checklist generation synchronously
        result = self.checklist_agent.generate_checklist(request)

        # Update state
        return {
            "checklist": result.model_dump()
        }

    async def run_streaming(
        self,
        system_description: str,
        additional_info: Optional[str] = None
    ) -> AsyncGenerator[dict, None]:
        """
        Run the workflow with streaming events.

        Yields events:
        - {"type": "step_start", "step": "classification|checklist"}
        - {"type": "tool_start", "tool_name": "...", "tool_input": {...}}
        - {"type": "tool_complete", "tool_name": "...", "tool_output": "..."}
        - {"type": "step_complete", "step": "...", "data": {...}}
        - {"type": "workflow_pause", "reason": "needs_info", "questions": [...]}
        - {"type": "workflow_complete", "result": {...}}
        """
        try:
            # Step 1: Classification with streaming
            yield {"type": "step_start", "step": "classification"}

            request = ClassificationRequest(
                ai_system_description=system_description,
                additional_info=additional_info
            )

            classification_result = None

            async for event in self.classification_agent.classify_streaming(request):
                # Pass through tool events, content streaming, and task events
                if event["type"] in ["tool_start", "tool_complete", "content_stream", "task"]:
                    yield event
                elif event["type"] == "final_result":
                    classification_result = event["data"]

            if not classification_result:
                yield {"type": "error", "message": "No classification result"}
                return

            # Check if needs more info
            if classification_result.get("needs_more_info"):
                yield {
                    "type": "workflow_pause",
                    "reason": "needs_info",
                    "questions": classification_result.get("questions", [])
                }
                return

            yield {
                "type": "step_complete",
                "step": "classification",
                "data": classification_result
            }

            # Step 2: Checklist with streaming
            yield {"type": "step_start", "step": "checklist"}

            checklist_request = ChecklistRequest(
                risk_level=classification_result["risk_level"],
                system_type=classification_result.get("system_type"),
                system_description=system_description
            )

            checklist_result = None

            async for event in self.checklist_agent.generate_checklist_streaming(checklist_request):
                # Pass through tool events, content streaming, and task events
                if event["type"] in ["tool_start", "tool_complete", "content_stream", "task"]:
                    yield event
                elif event["type"] == "final_result":
                    checklist_result = event["data"]

            if not checklist_result:
                yield {"type": "error", "message": "No checklist result"}
                return

            yield {
                "type": "step_complete",
                "step": "checklist",
                "data": checklist_result
            }

            # Workflow complete
            yield {
                "type": "workflow_complete",
                "result": {
                    "classification": classification_result,
                    "checklist": checklist_result
                }
            }

        except Exception as e:
            logger.error(f"Error in workflow: {str(e)}")
            yield {
                "type": "error",
                "message": str(e)
            }
