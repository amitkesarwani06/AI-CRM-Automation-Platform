from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        description="User message to the CRM AI",
        min_length=1,
        max_length=2000,
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for conversation continuity",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Add Rahul from TechCorp as a lead",
                "session_id": "user_123_session_1",
            }
        }


class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None
    agents_used: Optional[list[str]] = None
    intent: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Lead #1 Rahul created from TechCorp!",
                "agents_used": ["sales_agent"],
                "intent": "sales",
            }
        }
