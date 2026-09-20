import json
from datetime import datetime
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional

# In-memory lead storage — Day 20 will replace with PostgreSQL
# Must be module-level so it persists between tool calls
_leads_db: dict = {}
_lead_counter: int = 0

class CreateLeadInput(BaseModel):
    name: str = Field(description="Full name of the lead")
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    company: Optional[str] = Field(default=None, description="Company name")
    plan_interest: Optional[str] = Field(
        default=None,
        description="Which plan they are interested in: starter, growth, or enterprise"
    )
    notes: Optional[str] = Field(default=None, description="Additional notes")


class GetLeadInput(BaseModel):
    lead_id: Optional[int] = Field(default=None, description="Lead ID to look up")
    name: Optional[str] = Field(default=None, description="Lead name to search by")
    email: Optional[str] = Field(default=None, description="Lead email to search by")


class UpdateLeadInput(BaseModel):
    lead_id: int = Field(description="ID of the lead to update")
    status: Optional[str] = Field(
        default=None,
        description="New status: new, contacted, qualified, unqualified, closed"
    )
    notes: Optional[str] = Field(default=None, description="Updated notes")
    plan_interest: Optional[str] = Field(default=None, description="Updated plan interest")

@tool("create_lead", args_schema=CreateLeadInput)
def create_lead(
    name: str,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    company: Optional[str] = None,
    plan_interest: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """
    Create a new lead in the CRM system.

    Use this tool when:
    - A user mentions a new potential customer
    - Someone wants to add a contact to the CRM
    - A new sales opportunity is mentioned
    - After a discovery call or meeting with a prospect
    """
    global _lead_counter, _leads_db
    _lead_counter += 1
    lead_id = _lead_counter

    lead = {
        "id": lead_id,
        "name": name,
        "email": email,
        "phone": phone,
        "company": company,
        "plan_interest": plan_interest,
        "notes": notes,
        "status": "new",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    _leads_db[lead_id] = lead

    result = f"Lead created successfully!\n"
    result += f"ID: {lead_id}\n"
    result += f"Name: {name}\n"
    if company:       result += f"Company: {company}\n"
    if email:         result += f"Email: {email}\n"
    if phone:         result += f"Phone: {phone}\n"
    if plan_interest: result += f"Plan Interest: {plan_interest}\n"
    if notes:         result += f"Notes: {notes}\n"
    result += f"Status: new\n"
    result += f"Created: {lead['created_at']}"
    return result

@tool("get_lead", args_schema=GetLeadInput)
def get_lead(
    lead_id: Optional[int] = None,
    name: Optional[str] = None,
    email: Optional[str] = None,
) -> str:
    """
    Look up a lead in the CRM system.

    Use this tool when:
    - User asks about a specific customer or lead
    - Need to check lead status or details
    - Want to see if a contact already exists

    Search by lead_id, name, or email. Leave all blank to list all leads.
    """
    global _leads_db

    if not _leads_db:
        return "No leads found in the CRM. The database is empty."

    if lead_id and lead_id in _leads_db:
        return _format_lead(_leads_db[lead_id])

    if name:
        matches = [l for l in _leads_db.values() if name.lower() in l["name"].lower()]
        if matches:
            return "\n\n".join([_format_lead(l) for l in matches])
        return f"No lead found with name containing '{name}'"

    if email:
        matches = [
            l for l in _leads_db.values()
            if l.get("email") and email.lower() in l["email"].lower()
        ]
        if matches:
            return "\n\n".join([_format_lead(l) for l in matches])
        return f"No lead found with email '{email}'"

    # List all
    result = f"All leads ({len(_leads_db)} total):\n"
    for lead in _leads_db.values():
        result += f"• [{lead['id']}] {lead['name']}"
        if lead.get('company'):
            result += f" ({lead['company']})"
        result += f" — {lead['status']}\n"
    return result

@tool("update_lead", args_schema=UpdateLeadInput)
def update_lead(
    lead_id: int,
    status: Optional[str] = None,
    notes: Optional[str] = None,
    plan_interest: Optional[str] = None,
) -> str:
    """
    Update an existing lead's status or information.

    Use this tool when:
    - Lead status changes (contacted, qualified, closed)
    - New information about a lead is available
    - After a follow-up call or meeting

    Valid statuses: new, contacted, qualified, unqualified, closed
    """
    global _leads_db

    if lead_id not in _leads_db:
        return f"Lead with ID {lead_id} not found."

    lead = _leads_db[lead_id]
    updates = []

    if status:
        valid = ["new", "contacted", "qualified", "unqualified", "closed"]
        if status.lower() not in valid:
            return f"Invalid status '{status}'. Use: {', '.join(valid)}"
        lead["status"] = status.lower()
        updates.append(f"status → {status}")

    if notes:
        lead["notes"] = notes
        updates.append("notes updated")

    if plan_interest:
        lead["plan_interest"] = plan_interest
        updates.append(f"plan interest → {plan_interest}")

    lead["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    if updates:
        return f"Lead #{lead_id} ({lead['name']}) updated:\n" + "\n".join(f"  - {u}" for u in updates)
    return f"No updates provided for lead #{lead_id}."


def _format_lead(lead: dict) -> str:
    """Format a lead dict as readable string."""
    result = f"Lead #{lead['id']}: {lead['name']}\n"
    if lead.get('company'):       result += f"  Company: {lead['company']}\n"
    if lead.get('email'):         result += f"  Email: {lead['email']}\n"
    if lead.get('phone'):         result += f"  Phone: {lead['phone']}\n"
    if lead.get('plan_interest'): result += f"  Plan: {lead['plan_interest']}\n"
    if lead.get('notes'):         result += f"  Notes: {lead['notes']}\n"
    result += f"  Status: {lead['status']}\n"
    result += f"  Created: {lead['created_at']}"
    if lead.get('updated_at'):
        result += f"\n  Updated: {lead['updated_at']}"
    return result
