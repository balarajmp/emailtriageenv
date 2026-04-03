from __future__ import annotations

from typing import Dict, List, Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


Classification = Literal["spam", "urgent", "normal"]
TriageActionType = Literal["reply", "ignore", "escalate", "schedule"]


class EmailItem(BaseModel):
    id: str = Field(..., description="Unique email ID")
    sender: str = Field(..., description="Sender email address")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body content")
    is_vip: bool = Field(..., description="Whether sender is a VIP")
    timestamp: str = Field(..., description="ISO timestamp")


class EmailTriageAction(Action):
    email_id: str = Field(..., description="ID of the email being triaged")
    classification: Classification = Field(..., description="spam, urgent, or normal")
    priority: int = Field(..., ge=1, le=3, description="Priority from 1 (low) to 3 (high)")
    action: TriageActionType = Field(..., description="reply, ignore, escalate, or schedule")


class EmailTriageObservation(Observation):
    emails: List[EmailItem] = Field(default_factory=list, description="Emails visible to the agent")
    processed_email_ids: List[str] = Field(
        default_factory=list,
        description="IDs already triaged in this episode",
    )
    pending_email_ids: List[str] = Field(
        default_factory=list,
        description="IDs still needing triage",
    )
    per_email_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-email weighted scores in [0, 1]",
    )
    message: str = Field(default="", description="Human-readable step result")


# Backward-compatible aliases for scaffold imports
MyAction = EmailTriageAction
MyObservation = EmailTriageObservation