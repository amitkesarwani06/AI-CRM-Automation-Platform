from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from enum import Enum


class LeadStatus(str, Enum):
    NEW = "new"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    UNQUALIFIED = "unqualified"


class LeadInfo(BaseModel):
    """Structured lead data extracted from raw text."""

    name: str = Field(description="Full name of the lead")
    email: Optional[EmailStr] = Field(None, description="Email address if mentioned")
    phone: Optional[str] = Field(None, description="Phone number if mentioned")
    company: Optional[str] = Field(None, description="Company or organization name")
    city: Optional[str] = Field(None, description="City or location if mentioned")
    intent: str = Field(description="One sentence describing what the lead wants")
    score: int = Field(
        description="Lead quality score from 1 (cold) to 10 (hot), based on specificity and urgency",
        ge=1, le=10
    )
    status: LeadStatus = Field(default=LeadStatus.NEW, description="Current lead status")

