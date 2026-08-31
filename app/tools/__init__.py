from app.tools.knowledge_base_tool import search_knowledge_base, get_pricing
from app.tools.lead_tools import create_lead, get_lead, update_lead
from app.tools.utility_tools import get_current_datetime, calculate_discount

# All CRM tools in one list — agents will use this list
ALL_CRM_TOOLS = [
    search_knowledge_base,
    get_pricing,
    create_lead,
    get_lead,
    update_lead,
    get_current_datetime,
    calculate_discount,
]

__all__ = [
    "search_knowledge_base",
    "get_pricing",
    "create_lead",
    "get_lead",
    "update_lead",
    "get_current_datetime",
    "calculate_discount",
    "ALL_CRM_TOOLS",
]
