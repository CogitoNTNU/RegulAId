from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# ==================== Classification Schemas ====================

class ClassificationRequest(BaseModel):
    """Request for AI system classification"""
    ai_system_description: str = Field(
        ...,
        description="Description of the AI system to classify",
        examples=["We are developing a facial recognition system for airport security"]
    )
    additional_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional structured information about the AI system"
    )


class ClassificationResponse(BaseModel):
    """Response from classification agent"""
    risk_level: Optional[str] = Field(
        default=None,
        description="Risk level: 'prohibited', 'high-risk', 'limited-risk', or 'minimal-risk'"
    )
    system_type: Optional[str] = Field(
        default=None,
        description="Type of AI system (e.g., 'biometric', 'critical infrastructure', 'general purpose AI')"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score of the classification (0-1)"
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why this classification was given"
    )
    needs_more_info: bool = Field(
        default=False,
        description="Whether the agent needs more information before classifying"
    )
    questions: Optional[List[str]] = Field(
        default=None,
        description="Follow-up questions to ask the user if more information is needed"
    )
    relevant_articles: List[str] = Field(
        default_factory=list,
        description="EU AI Act articles that were used for classification"
    )


# ==================== Checklist Schemas ====================

class ChecklistItem(BaseModel):
    """A single item in the compliance checklist"""
    requirement: str = Field(
        ...,
        description="Description of what needs to be done"
    )
    applicable_articles: List[str] = Field(
        default_factory=list,
        description="EU AI Act articles that mandate this requirement"
    )
    priority: str = Field(
        default="medium",
        description="Priority level: 'high', 'medium', or 'low'"
    )
    category: str = Field(
        default="general",
        description="Category of requirement (e.g., 'documentation', 'technical', 'governance', 'testing')"
    )


class ChecklistRequest(BaseModel):
    """Request for compliance checklist generation"""
    risk_level: str = Field(
        ...,
        description="Risk level from classification: 'prohibited', 'high-risk', 'limited-risk', or 'minimal-risk'"
    )
    system_type: Optional[str] = Field(
        default=None,
        description="Type of AI system (from classification)"
    )
    system_description: Optional[str] = Field(
        default=None,
        description="Original description of the AI system"
    )


class ChecklistResponse(BaseModel):
    """Response from checklist agent"""
    risk_level: str = Field(
        ...,
        description="Risk level this checklist is for"
    )
    checklist_items: List[ChecklistItem] = Field(
        default_factory=list,
        description="List of compliance requirements"
    )
    total_items: int = Field(
        ...,
        description="Total number of checklist items"
    )
    summary: str = Field(
        ...,
        description="Summary of the compliance requirements"
    )
