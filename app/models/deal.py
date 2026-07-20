from pydantic import BaseModel, Field
from typing import Optional


class DealInfo(BaseModel):
    """Structured deal data extracted from raw text."""

    deal_name: str = Field(description="Short name or title for the deal")
    value: float = Field(description="Monetary value of the deal")
    currency: str = Field(default="INR", description="Currency code, e.g. INR, USD")
    expected_close_date: Optional[str] = Field(None, description="Expected close date if mentioned")
    product_interest: list[str] = Field(description="List of products or modules the lead is interested in")
