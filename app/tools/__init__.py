from app.tools.knowledge_base_tool import search_knowledge_base, get_pricing
from app.tools.lead_tools import create_lead, get_lead, update_lead
from app.tools.utility_tools import get_current_datetime, calculate_discount
from app.tools.email_tools import send_email, draft_email, email_lead

# All CRM tools in one list — agents will use this list (10 tools total)
ALL_CRM_TOOLS = [
    search_knowledge_base,
    get_pricing,
    create_lead,
    get_lead,
    update_lead,
    get_current_datetime,
    calculate_discount,
    send_email,    # Day 19
    draft_email,   # Day 19
    email_lead,    # Day 19
]

__all__ = [
    "search_knowledge_base",
    "get_pricing",
    "create_lead",
    "get_lead",
    "update_lead",
    "get_current_datetime",
    "calculate_discount",
    "send_email",
    "draft_email",
    "email_lead",
    "ALL_CRM_TOOLS",
]
