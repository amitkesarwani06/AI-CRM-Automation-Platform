from datetime import datetime
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class DateTimeInput(BaseModel):
    format: str = Field(
        default="%Y-%m-%d %H:%M",
        description="Date format string"
    )


@tool("get_current_datetime", args_schema=DateTimeInput)
def get_current_datetime(format: str = "%Y-%m-%d %H:%M") -> str:
    """
    Get the current date and time.

    Use this tool when:
    - User asks what time or date it is
    - Need to timestamp a note or lead
    - Calculating deadlines or follow-up dates
    """
    return datetime.now().strftime(format)


@tool("calculate_discount")
def calculate_discount(original_price: float, discount_percent: float) -> str:
    """
    Calculate discounted price for CRM plans.

    Use when:
    - User asks about annual billing discount (20% off)
    - Calculating custom pricing for enterprise
    - Comparing plan costs with discount applied
    """
    discount_amount = original_price * (discount_percent / 100)
    final_price = original_price - discount_amount
    return (
        f"Original: Rs.{original_price:,.0f}\n"
        f"Discount: {discount_percent}% = Rs.{discount_amount:,.0f}\n"
        f"Final Price: Rs.{final_price:,.0f}"
    )
