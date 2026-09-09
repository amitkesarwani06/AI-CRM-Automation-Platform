from app.tools.knowledge_base_tool import search_knowledge_base, get_pricing
from app.tools.utility_tools import get_current_datetime, calculate_discount
from app.tools.email_tools import send_email, draft_email, email_lead

# Day 20: DB-powered lead tools (replaces in-memory lead_tools)
from app.tools.lead_tools_db import (
    create_lead_db,
    get_lead_db,
    update_lead_db,
    search_leads,
    get_lead_stats,
)

# All CRM tools — 12 tools total
ALL_CRM_TOOLS = [
    search_knowledge_base,   # RAG knowledge base
    get_pricing,             # pricing info
    create_lead_db,          # Day 20: DB version
    get_lead_db,             # Day 20: DB version
    update_lead_db,          # Day 20: DB version
    search_leads,            # Day 20: full-text search
    get_lead_stats,          # Day 20: analytics
    get_current_datetime,
    calculate_discount,
    send_email,              # Day 19
    draft_email,             # Day 19
    email_lead,              # Day 19
]

__all__ = [
    "search_knowledge_base",
    "get_pricing",
    "create_lead_db",
    "get_lead_db",
    "update_lead_db",
    "search_leads",
    "get_lead_stats",
    "get_current_datetime",
    "calculate_discount",
    "send_email",
    "draft_email",
    "email_lead",
    "ALL_CRM_TOOLS",
]
