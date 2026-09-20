import os
import sys
import logging
from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Fix Windows terminal emoji encoding
sys.stdout.reconfigure(encoding='utf-8')

from app.database.connection import SessionLocal, init_db
from app.database.lead_repository import LeadRepository

logger = logging.getLogger(__name__)

# Initialize DB tables on import
init_db()


# ── Helper ────────────────────────────────────────────────────────────────────
def _format_lead(lead) -> str:
    """Format Lead model as readable string."""
    d = lead.to_dict()
    lines = [f"Lead #{d['id']}: {d['name']}"]
    if d.get("company"):       lines.append(f"  Company: {d['company']}")
    if d.get("email"):         lines.append(f"  Email:   {d['email']}")
    if d.get("phone"):         lines.append(f"  Phone:   {d['phone']}")
    if d.get("plan_interest"): lines.append(f"  Plan:    {d['plan_interest']}")
    if d.get("notes"):         lines.append(f"  Notes:   {d['notes']}")
    lines.append(f"  Status:  {d['status']}")
    lines.append(f"  Created: {d['created_at']}")
    return "\n".join(lines)


# ── Input Schemas ─────────────────────────────────────────────────────────────
class CreateLeadInput(BaseModel):
    name:          str            = Field(description="Full name of the lead")
    email:         Optional[str]  = Field(default=None, description="Email address")
    phone:         Optional[str]  = Field(default=None, description="Phone number")
    company:       Optional[str]  = Field(default=None, description="Company name")
    plan_interest: Optional[str]  = Field(default=None, description="Plan interested in")
    notes:         Optional[str]  = Field(default=None, description="Additional notes")


class GetLeadInput(BaseModel):
    lead_id: Optional[int] = Field(default=None, description="Lead ID to look up")
    name:    Optional[str] = Field(default=None, description="Name to search")
    email:   Optional[str] = Field(default=None, description="Email to search")
    status:  Optional[str] = Field(default=None, description="Filter by status")


class UpdateLeadInput(BaseModel):
    lead_id:       int           = Field(description="ID of lead to update")
    status:        Optional[str] = Field(default=None, description="New status: new/contacted/qualified/unqualified/closed")
    notes:         Optional[str] = Field(default=None, description="Updated notes")
    plan_interest: Optional[str] = Field(default=None, description="Updated plan interest")


class SearchLeadInput(BaseModel):
    query: str = Field(description="Search term — searches name, company, email, and notes")


# ── Tools ─────────────────────────────────────────────────────────────────────
@tool("create_lead_db", args_schema=CreateLeadInput)
def create_lead_db(
    name: str,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    company: Optional[str] = None,
    plan_interest: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """
    Create a new lead in the PostgreSQL/SQLite database.
    Data persists permanently — survives server restarts.

    Use when: a new potential customer needs to be added to CRM.
    Unlike in-memory storage, data is never lost.
    """
    try:
        with SessionLocal() as db:
            repo = LeadRepository(db)
            lead = repo.create(
                name=name, email=email, phone=phone,
                company=company, plan_interest=plan_interest, notes=notes,
            )
            return (
                f"Lead saved to database!\n"
                f"ID: {lead.id}\n"
                f"Name: {lead.name}\n"
                f"Company: {lead.company or 'N/A'}\n"
                f"Email: {lead.email or 'N/A'}\n"
                f"Status: {lead.status}\n"
                f"Created: {lead.created_at}"
            )
    except Exception as e:
        logger.error(f"create_lead_db error: {e}")
        return f"Database error: {str(e)}"


@tool("get_lead_db", args_schema=GetLeadInput)
def get_lead_db(
    lead_id: Optional[int] = None,
    name:    Optional[str] = None,
    email:   Optional[str] = None,
    status:  Optional[str] = None,
) -> str:
    """
    Look up leads in the database.
    Search by ID, name, email, or filter by status.
    Leave all blank to get all leads.

    Use when: need to find customer/lead information from CRM.
    """
    try:
        with SessionLocal() as db:
            repo = LeadRepository(db)

            if lead_id:
                lead = repo.get_by_id(lead_id)
                return _format_lead(lead) if lead else f"No lead found with ID {lead_id}"

            if name:
                leads = repo.get_by_name(name)
                if not leads:
                    return f"No leads found with name containing '{name}'"
                return "\n\n".join([_format_lead(l) for l in leads])

            if email:
                lead = repo.get_by_email(email)
                return _format_lead(lead) if lead else f"No lead with email '{email}'"

            if status:
                leads = repo.get_by_status(status)
                if not leads:
                    return f"No leads with status '{status}'"
                return f"Leads with status '{status}':\n\n" + \
                       "\n\n".join([_format_lead(l) for l in leads])

            # All leads
            leads = repo.get_all(limit=20)
            if not leads:
                return "No leads in database yet."

            total = repo.count()
            result = f"All leads ({total} total):\n"
            for l in leads:
                result += (
                    f"  [{l.id}] {l.name}"
                    f"{f' ({l.company})' if l.company else ''}"
                    f" - {l.status}\n"
                )
            return result

    except Exception as e:
        logger.error(f"get_lead_db error: {e}")
        return f"Database error: {str(e)}"


@tool("update_lead_db", args_schema=UpdateLeadInput)
def update_lead_db(
    lead_id:       int,
    status:        Optional[str] = None,
    notes:         Optional[str] = None,
    plan_interest: Optional[str] = None,
) -> str:
    """
    Update a lead's information in the database.

    Use when: lead status changes, notes need updating,
    or plan interest changes.
    Valid statuses: new, contacted, qualified, unqualified, closed
    """
    try:
        with SessionLocal() as db:
            repo = LeadRepository(db)
            lead = repo.update(lead_id, status, notes, plan_interest)

            if not lead:
                return f"Lead #{lead_id} not found in database."

            updates = []
            if status:        updates.append(f"status -> {status}")
            if notes:         updates.append("notes updated")
            if plan_interest: updates.append(f"plan -> {plan_interest}")

            return (
                f"Lead #{lead_id} ({lead.name}) updated in database:\n"
                + "\n".join(f"  - {u}" for u in updates)
            )
    except Exception as e:
        logger.error(f"update_lead_db error: {e}")
        return f"Database error: {str(e)}"


@tool("search_leads", args_schema=SearchLeadInput)
def search_leads(query: str) -> str:
    """
    Search leads across ALL fields: name, company, email, and notes.

    Use when: user asks to find leads with a keyword,
    or searching for a lead without knowing their ID.
    Example: search_leads("TechCorp") returns all TechCorp leads.
    """
    try:
        with SessionLocal() as db:
            repo = LeadRepository(db)
            leads = repo.search(query)

            if not leads:
                return f"No leads found matching '{query}'"

            result = f"Search results for '{query}' ({len(leads)} found):\n\n"
            result += "\n\n".join([_format_lead(l) for l in leads])
            return result

    except Exception as e:
        return f"Database error: {str(e)}"


@tool("get_lead_stats")
def get_lead_stats() -> str:
    """
    Get CRM lead statistics from the database.

    Use when: user asks for a summary, analytics, or overview
    of all leads. Returns total count and breakdown by status.
    """
    try:
        with SessionLocal() as db:
            repo = LeadRepository(db)
            stats = repo.get_stats()

            result = "CRM Lead Statistics:\n"
            result += f"Total Leads: {stats['total_leads']}\n\n"
            result += "By Status:\n"
            for status, count in stats["by_status"].items():
                bar = "#" * min(count, 20)
                result += f"  {status:<12} {count:>3}  {bar}\n"
            return result

    except Exception as e:
        return f"Database error: {str(e)}"


# ── Tests ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    logging.basicConfig(level=logging.ERROR)

    print("\n" + "="*60)
    print("DAY 20 — PostgreSQL/SQLite Database Tool Tests")
    print("="*60)

    # ── TEST 1: Create Leads ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 1: create_lead_db — save to database")
    print("="*60)

    for lead_data in [
        {"name": "Rahul Sharma",  "email": "rahul@techcorp.com",   "company": "TechCorp",   "plan_interest": "growth"},
        {"name": "Priya Mehta",   "email": "priya@startupxyz.com", "company": "StartupXYZ", "plan_interest": "starter"},
        {"name": "Amit Kumar",    "email": "amit@infosys.com",     "company": "InfoSys",    "plan_interest": "enterprise"},
    ]:
        print(create_lead_db.invoke(lead_data))
        print("-" * 40)

    # ── TEST 2: Get Leads ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 2: get_lead_db — query from database")
    print("="*60)

    print("By ID (1):")
    print(get_lead_db.invoke({"lead_id": 1}))

    print("\nBy name ('Priya'):")
    print(get_lead_db.invoke({"name": "Priya"}))

    print("\nAll leads:")
    print(get_lead_db.invoke({}))

    # ── TEST 3: Update Leads ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 3: update_lead_db — modify in database")
    print("="*60)

    print(update_lead_db.invoke({
        "lead_id": 1,
        "status": "contacted",
        "notes": "Called Monday, demo scheduled for Friday",
    }))
    print()
    print(update_lead_db.invoke({"lead_id": 2, "status": "qualified"}))

    # ── TEST 4: Stats ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 4: get_lead_stats — database analytics")
    print("="*60)
    print(get_lead_stats.invoke({}))

    # ── TEST 5: Persistence proof ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 5: Persistence — data survives new session")
    print("="*60)

    print("Opening a brand new DB session (simulates server restart)...")
    with SessionLocal() as new_db:
        repo = LeadRepository(new_db)
        count = repo.count()
        print(f"Leads found in new session: {count}")
        if count > 0:
            print("Data persisted! (in-memory dict would show 0 after restart)")

    # ── TEST 6: Search across all fields ──────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 6: search_leads — full-text search")
    print("="*60)

    print("Search 'TechCorp':")
    print(search_leads.invoke({"query": "TechCorp"}))

    print("\nSearch 'starter':")
    print(search_leads.invoke({"query": "starter"}))

    # ── TEST 7: Filter by status ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 7: Filter leads by status")
    print("="*60)

    print("Contacted leads:")
    print(get_lead_db.invoke({"status": "contacted"}))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Day 20 COMPLETE — Database working!")
    print("="*60)
    print("""
Before (Days 15-19):           After (Day 20):
  _leads_db = {}                 SQLite/PostgreSQL
  RAM only                       Disk persistence
  Lost on restart                Survives restarts
  No filtering                   Full SQL queries
  Single process                 Multi-process safe

DB file: crm_dev.db (SQLite dev mode)
Production: set DATABASE_URL=postgresql://... in .env

New tools (replaces old lead_tools):
  create_lead_db   - save lead to database
  get_lead_db      - query leads from database
  update_lead_db   - update lead in database
  search_leads     - full-text search across all fields
  get_lead_stats   - analytics / lead count by status

Total CRM Tools: 12
Next -> Day 21: Sales Agent!
""")
    print("="*60)
