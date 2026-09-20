from pydantic import BaseModel, Field, EmailStr
from typing import Optional


class LeadCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    city: Optional[str] = None
    plan_interest: Optional[str] = None
    notes: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Rahul Sharma",
                "email": "rahul@techcorp.com",
                "phone": "+91 9876543210",
                "company": "TechCorp",
                "city": "Bengaluru",
                "plan_interest": "growth",
                "notes": "Met at SaaS conference",
            }
        }


class LeadUpdate(BaseModel):
    status: Optional[str] = None
    notes: Optional[str] = None
    plan_interest: Optional[str] = None
    score: Optional[int] = Field(default=None, ge=1, le=10)
    city: Optional[str] = None


class LeadResponse(BaseModel):
    id: int
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    city: Optional[str] = None
    plan_interest: Optional[str] = None
    status: str
    score: Optional[int] = None
    notes: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True


class LeadListResponse(BaseModel):
    leads: list[LeadResponse]
    total: int


class LeadExtractRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        description="Raw unstructured text (email, chat transcript, note) containing lead details",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hi, I'm Rahul from TechCorp, reach me at rahul@techcorp.com or 9876543210. Interested in Growth plan."
            }
        }
