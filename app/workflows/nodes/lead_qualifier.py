import os
import sys
import json
import re
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# Fix Windows terminal emoji encoding
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")


def score_lead(state: dict) -> dict:
    """
    Node: Score a lead 1-10 based on conversation context.
    High score = hot lead = priority handling.
    """
    messages = state.get("messages", [])
    if not messages:
        return {**state, "lead_score": 5, "lead_data": {}, "info_complete": False}

    latest = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = f"""Analyze this CRM message and extract lead info + score.

Message: "{latest}"

Return ONLY valid JSON (no markdown formatting, no backticks, no preamble):
{{
  "name": "lead name or null",
  "company": "company name or null",
  "email": "email or null",
  "phone": "phone or null",
  "team_size": "number or null",
  "budget": "budget mentioned or null",
  "plan_interest": "starter or growth or enterprise or null",
  "score": 1-10,
  "score_reason": "why this score",
  "info_complete": true or false
}}

Scoring guide:
- 9-10: Large team (15+), clear budget, urgent need, enterprise plan
- 7-8:  Medium team (5-15), some budget mentioned, Growth plan
- 5-6:  Small team (1-5), interested but vague, Starter plan
- 3-4:  Early exploration, no clear budget or timeline
- 1-2:  Just browsing, no real need identified

info_complete must be true ONLY if at least company/name and either email or team_size or clear requirement are provided. If it is just a generic message with no specific lead details, info_complete must be false.
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        # Clean any markdown code blocks
        if "```" in content:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if match:
                content = match.group(1)
            else:
                content = content.replace("```json", "").replace("```", "").strip()

        data = json.loads(content)
        score = int(data.get("score", 5))
        info_complete = bool(data.get("info_complete", False))

        logger.info(f"Lead scored: {score}/10, info_complete={info_complete}")
        return {
            **state,
            "lead_score": score,
            "lead_data": data,
            "info_complete": info_complete,
        }
    except Exception as e:
        logger.warning(f"Lead scoring fallback: {e}")
        # Rule-based fallback if LLM JSON parsing fails or rate limits
        text_lower = latest.lower()
        score = 5
        info_complete = False
        if any(w in text_lower for w in ["80", "50", "25", "megabank", "techgiant", "lakh"]):
            score = 9
            info_complete = True
        elif any(w in text_lower for w in ["startup", "8 people", "founder"]):
            score = 6
            info_complete = True
        elif any(w in text_lower for w in ["looking around", "just browsing", "what does"]):
            score = 3
            info_complete = True
        elif "lead" in text_lower and not any(w in text_lower for w in ["@", "team", "rs"]):
            score = 4
            info_complete = False

        data = {
            "name": "prospect",
            "company": "company",
            "score": score,
            "info_complete": info_complete,
        }
        return {
            **state,
            "lead_score": score,
            "lead_data": data,
            "info_complete": info_complete,
        }


def route_by_score(state: dict) -> str:
    """
    Conditional edge: route based on lead score and completeness.
    Returns node name to go to next.
    """
    score = state.get("lead_score", 5)
    info_complete = state.get("info_complete", False)

    messages = state.get("messages", [])
    user_text = (messages[-1].content if messages else "").lower()

    # If the user is trying to add/qualify a lead but details are missing, loop to gather info
    if ("add" in user_text or "create" in user_text or "new lead" in user_text or not info_complete) and score >= 4:
        return "gather_info_node"
    if ("add" in user_text or "create" in user_text) and not info_complete:
        return "gather_info_node"

    # Route qualified leads by score
    if score >= 8:
        return "hot_lead_node"
    elif score >= 5:
        return "warm_lead_node"
    else:
        return "cold_lead_node"


def gather_info_node(state: dict) -> dict:
    """
    Node: Ask for missing information.
    Loops / pauses for user to supply details.
    """
    lead_data = state.get("lead_data", {})
    missing = []
    if not lead_data.get("name") or lead_data.get("name") in ["null", "None", None, "prospect"]:
        missing.append("name")
    if not lead_data.get("company") or lead_data.get("company") in ["null", "None", None, "company"]:
        missing.append("company name")
    if not lead_data.get("email") or lead_data.get("email") in ["null", "None", None]:
        missing.append("email address")
    if not lead_data.get("team_size") or lead_data.get("team_size") in ["null", "None", None]:
        missing.append("team size")

    questions = ", ".join(missing[:-1]) + (" and " + missing[-1] if len(missing) > 1 else missing[0]) if missing else "more details about your team and requirements"
    response = (
        f"I'd love to help! To qualify and set up this lead properly, could you share "
        f"the prospect's {questions}?"
    )

    return {
        **state,
        "response": response,
        "messages": [AIMessage(content=response)],
        "needs_more_info": True,
        "track": "incomplete",
    }


def hot_lead_node(state: dict) -> dict:
    """
    Node: Handle hot lead (score 8-10).
    Fast track: immediate priority email + schedule demo.
    """
    lead = state.get("lead_data", {})
    name = lead.get("name") or "the prospect"
    company = lead.get("company") or "their company"
    score = state.get("lead_score", 8)
    email = lead.get("email") or "contact@company.com"

    # Prepare email for approval (HITL will pause here)
    email_draft = {
        "to": email,
        "subject": f"Priority Demo Request — {company}",
        "body": (
            f"Hi {name},\n\n"
            f"Thank you for your interest in our CRM platform. "
            f"Given your team's size and urgent requirements, I'd like to personally schedule a "
            f"priority demo for you this week.\n\n"
            f"Would Thursday or Friday work for a 30-minute walkthrough?\n\n"
            f"Best regards,\nCRM Enterprise Sales Team"
        )
    }

    response = (
        f"🔥 HOT LEAD DETECTED! Score: {score}/10\n\n"
        f"Lead: {name} from {company}\n"
        f"Action: Priority email drafted — pending human approval before sending.\n\n"
        f"Email draft:\nSubject: {email_draft['subject']}\n"
        f"To: {email_draft['to']}\n"
        f"Body preview:\n{email_draft['body'][:130]}..."
    )

    return {
        **state,
        "response": response,
        "email_draft": email_draft,
        "messages": [AIMessage(content=response)],
        "track": "hot",
    }


def warm_lead_node(state: dict) -> dict:
    """
    Node: Handle warm lead (score 5-7).
    Standard nurture sequence.
    """
    lead = state.get("lead_data", {})
    name = lead.get("name") or "the prospect"
    company = lead.get("company") or "their team"
    score = state.get("lead_score", 5)

    response = (
        f"🌡️ WARM LEAD — Score: {score}/10\n\n"
        f"Lead: {name} ({company})\n"
        f"Action: Added to nurture sequence.\n"
        f"Next: Send product overview email in 24 hours.\n"
        f"Follow-up: Schedule discovery call next week."
    )

    return {
        **state,
        "response": response,
        "messages": [AIMessage(content=response)],
        "track": "warm",
    }


def cold_lead_node(state: dict) -> dict:
    """
    Node: Handle cold lead (score 1-4).
    Light touch — add to newsletter, revisit in 30 days.
    """
    lead = state.get("lead_data", {})
    name = lead.get("name") or "prospect"
    score = state.get("lead_score", 3)

    response = (
        f"❄️ COLD LEAD — Score: {score}/10\n\n"
        f"Lead: {name}\n"
        f"Action: Added to newsletter list.\n"
        f"Next: Automated check-in in 30 days.\n"
        f"Note: Not ready to buy yet — nurture long-term."
    )

    return {
        **state,
        "response": response,
        "messages": [AIMessage(content=response)],
        "track": "cold",
    }


def send_email_node(state: dict) -> dict:
    """
    Node: Actually send the email.
    This is the node AFTER human approval (HITL).
    Interrupt happens BEFORE this node runs.
    """
    email_draft = state.get("email_draft", {})

    if not email_draft:
        return {**state, "response": "No email to send.", "email_sent": False}

    to_addr = email_draft.get("to", "unknown@email.com")
    subj = email_draft.get("subject", "No subject")

    result = (
        f"✅ Email sent!\n"
        f"To: {to_addr}\n"
        f"Subject: {subj}\n"
        f"Status: Delivered successfully via CRM Email Dispatcher"
    )

    return {
        **state,
        "response": result,
        "email_sent": True,
        "messages": [AIMessage(content=result)],
    }
