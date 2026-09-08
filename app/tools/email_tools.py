import os
import sys
import smtplib
import logging
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Fix Windows terminal emoji encoding
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_MODEL = "openai/gpt-oss-20b"


# ── Input Schemas ─────────────────────────────────────────────────────────────
class SendEmailInput(BaseModel):
    to_email: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content (plain text)")
    from_name: Optional[str] = Field(
        default="CRM Platform",
        description="Sender display name",
    )


class DraftEmailInput(BaseModel):
    recipient_name: str = Field(description="Name of the person receiving the email")
    company: Optional[str] = Field(default=None, description="Recipient's company")
    email_type: str = Field(
        description=(
            "Type of email to draft: "
            "'follow_up', 'welcome', 'pricing', "
            "'demo_confirmation', 'thank_you'"
        )
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context to personalize the email",
    )


class EmailLeadInput(BaseModel):
    lead_id: int = Field(description="ID of the lead to email")
    email_type: str = Field(
        description="Type of email: follow_up, welcome, pricing, demo_confirmation"
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context for personalizing the email",
    )


# ── Email Template Library (Mini Assignment) ──────────────────────────────────
EMAIL_TEMPLATES = {
    "follow_up": {
        "subject": "Following up — {name}",
        "body": (
            "Hi {name},\n\n"
            "I wanted to follow up on our recent conversation. "
            "We'd love to help {company} get started with our platform.\n\n"
            "Do you have 15 minutes this week to connect?\n\n"
            "Best regards,\nThe CRM Team"
        ),
    },
    "welcome": {
        "subject": "Welcome to CRM Platform — {name}!",
        "body": (
            "Hi {name},\n\n"
            "Welcome! We're thrilled to have you explore our platform.\n\n"
            "Here's how to get started:\n"
            "1. Book your onboarding call: calendly.com/crmplatform/onboarding\n"
            "2. Import your leads from CSV in Settings > Import Data\n"
            "3. Reach us anytime at support@crmplatform.com\n\n"
            "Best regards,\nThe CRM Team"
        ),
    },
    "pricing": {
        "subject": "CRM Platform Pricing — {name}",
        "body": (
            "Hi {name},\n\n"
            "Here's a quick overview of our plans:\n\n"
            "Starter:    Rs.999/month  — 3 users, 500 leads\n"
            "Growth:     Rs.2,999/month — 15 users, unlimited leads + AI\n"
            "Enterprise: Custom pricing  — unlimited users, dedicated support\n\n"
            "All plans include a 14-day free trial.\n"
            "Let me know which fits {company} best!\n\n"
            "Best regards,\nThe CRM Team"
        ),
    },
    "demo_confirmation": {
        "subject": "Demo Confirmed — {name}",
        "body": (
            "Hi {name},\n\n"
            "Your demo is confirmed! We're looking forward to showing you "
            "how our platform can help {company}.\n\n"
            "Please have your team ready and feel free to prepare any questions.\n\n"
            "Best regards,\nThe CRM Team"
        ),
    },
    "re_engagement": {
        "subject": "We miss you — {name}!",
        "body": (
            "Hi {name},\n\n"
            "It's been a while since we connected. "
            "We've added several new features that might interest {company}.\n\n"
            "Would you like to reconnect for a quick 10-minute catch-up?\n\n"
            "Best regards,\nThe CRM Team"
        ),
    },
}


# ── LLM Email Drafting ────────────────────────────────────────────────────────
def _draft_email_with_llm(
    recipient_name: str,
    email_type: str,
    company: Optional[str] = None,
    context: Optional[str] = None,
) -> dict:
    """
    Use LLM to personalize a template into a professional CRM email.
    Falls back to template if LLM fails.
    """
    # Get base template
    template = EMAIL_TEMPLATES.get(email_type, EMAIL_TEMPLATES["follow_up"])
    company_str = company or "your company"
    base_subject = template["subject"].format(name=recipient_name, company=company_str)
    base_body    = template["body"].format(name=recipient_name, company=company_str)

    prompt = f"""You are a professional CRM email writer.

Personalize this base email for the recipient. Keep it under 120 words.
Return ONLY valid JSON with keys "subject" and "body". No markdown.

Recipient: {recipient_name}
Company: {company_str}
Email type: {email_type}
{f'Context: {context}' if context else ''}

Base subject: {base_subject}
Base body:
{base_body}

Return JSON: {{"subject": "...", "body": "..."}}"""

    try:
        llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.4,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Clean JSON if wrapped in markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(content)
        return {
            "subject": parsed.get("subject", base_subject),
            "body":    parsed.get("body", base_body),
        }

    except Exception as e:
        logger.warning(f"LLM draft failed, using template: {e}")
        # Fallback to template
        return {"subject": base_subject, "body": base_body}


# ── SMTP Sending ──────────────────────────────────────────────────────────────
def _send_via_smtp(to_email: str, subject: str, body: str, from_name: str) -> str:
    """
    Send email via SMTP (Gmail).

    Gmail setup:
    1. Google Account > Security > 2-Step Verification (enable)
    2. Security > App Passwords > generate for "Mail"
    3. Use the 16-char code as SMTP_PASSWORD (NOT your real password!)

    .env:
        SMTP_EMAIL=you@gmail.com
        SMTP_PASSWORD=xxxx xxxx xxxx xxxx
        SMTP_HOST=smtp.gmail.com
        SMTP_PORT=587
    """
    smtp_email    = os.getenv("SMTP_EMAIL")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_host     = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port     = int(os.getenv("SMTP_PORT", "587"))

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"{from_name} <{smtp_email}>"
        msg["To"]      = to_email
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_email, smtp_password)
            server.sendmail(smtp_email, to_email, msg.as_string())

        logger.info(f"Email sent to {to_email}")
        return f"Email sent successfully to {to_email}\nSubject: {subject}"

    except smtplib.SMTPAuthenticationError:
        return (
            "Email failed: Authentication error.\n"
            "Check SMTP_EMAIL and SMTP_PASSWORD.\n"
            "For Gmail: use App Password, not your real password."
        )
    except smtplib.SMTPException as e:
        return f"Email failed: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def _send_via_sendgrid(to_email: str, subject: str, body: str, from_name: str) -> str:
    """
    Send email via SendGrid API.

    .env:
        SENDGRID_API_KEY=SG.xxxxx
        SENDGRID_FROM_EMAIL=verified@yourdomain.com
    """
    api_key    = os.getenv("SENDGRID_API_KEY")
    from_email = os.getenv("SENDGRID_FROM_EMAIL")

    try:
        import sendgrid
        from sendgrid.helpers.mail import Mail

        sg = sendgrid.SendGridAPIClient(api_key=api_key)
        message = Mail(
            from_email=f"{from_name} <{from_email}>",
            to_emails=to_email,
            subject=subject,
            plain_text_content=body,
        )
        response = sg.send(message)
        if response.status_code in [200, 202]:
            return f"Email sent via SendGrid to {to_email}"
        return f"SendGrid response: {response.status_code}"

    except ImportError:
        return "SendGrid not installed. Run: pip install sendgrid"
    except Exception as e:
        return f"SendGrid error: {str(e)}"


# ── Tools ─────────────────────────────────────────────────────────────────────
@tool("send_email", args_schema=SendEmailInput)
def send_email(
    to_email: str,
    subject: str,
    body: str,
    from_name: str = "CRM Platform",
) -> str:
    """
    Send an email to a lead or customer.

    Use this tool when:
    - A follow-up email needs to be sent
    - Sharing pricing or product information via email
    - Sending demo confirmations
    - Any time an email needs to be delivered

    Prefers Gmail SMTP. Falls back to SendGrid. Uses simulation if neither is configured.
    """
    if os.getenv("SMTP_EMAIL") and os.getenv("SMTP_PASSWORD"):
        return _send_via_smtp(to_email, subject, body, from_name)
    elif os.getenv("SENDGRID_API_KEY") and os.getenv("SENDGRID_FROM_EMAIL"):
        return _send_via_sendgrid(to_email, subject, body, from_name)
    else:
        # Simulation mode — safe for development
        logger.info(f"[SIM] Email to {to_email}: {subject}")
        return (
            f"[SIMULATION MODE] Email prepared:\n"
            f"To:      {to_email}\n"
            f"Subject: {subject}\n"
            f"Preview: {body[:120]}...\n"
            f"(Add SMTP_EMAIL + SMTP_PASSWORD to .env to send real emails)"
        )


@tool("draft_email", args_schema=DraftEmailInput)
def draft_email(
    recipient_name: str,
    email_type: str,
    company: Optional[str] = None,
    context: Optional[str] = None,
) -> str:
    """
    Draft a professional CRM email using AI.

    Use this tool to:
    - Create personalized follow-up emails
    - Write welcome emails for new leads
    - Draft pricing information emails
    - Generate demo confirmation emails

    Available email_type values:
    follow_up, welcome, pricing, demo_confirmation, thank_you, re_engagement

    Returns the drafted subject and body — use send_email to actually send it.
    """
    draft = _draft_email_with_llm(recipient_name, email_type, company, context)

    return (
        f"Email Draft Ready:\n"
        f"Subject: {draft['subject']}\n\n"
        f"Body:\n{draft['body']}\n\n"
        f"(Use send_email tool to send this)"
    )


@tool("email_lead", args_schema=EmailLeadInput)
def email_lead(
    lead_id: int,
    email_type: str,
    context: Optional[str] = None,
) -> str:
    """
    Draft AND send an email to a CRM lead by their ID in one step.

    Use this tool when:
    - You want to email a specific lead in one action
    - The lead already exists in the CRM with an email address

    Combines draft_email + send_email automatically.
    Requires the lead to have an email address in the CRM.
    """
    from app.tools.lead_tools import _leads_db

    if lead_id not in _leads_db:
        return f"Lead #{lead_id} not found in CRM."

    lead     = _leads_db[lead_id]
    to_email = lead.get("email")

    if not to_email:
        return (
            f"Lead #{lead_id} ({lead['name']}) has no email address.\n"
            f"Update the lead with an email first using update_lead."
        )

    # Draft with LLM
    draft = _draft_email_with_llm(
        recipient_name=lead["name"],
        email_type=email_type,
        company=lead.get("company"),
        context=context,
    )

    # Send
    send_result = send_email.invoke({
        "to_email":  to_email,
        "subject":   draft["subject"],
        "body":      draft["body"],
    })

    return (
        f"Lead #{lead_id} ({lead['name']}):\n"
        f"{send_result}\n\n"
        f"Subject: {draft['subject']}\n"
        f"Preview: {draft['body'][:150]}..."
    )


# ── Tests ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    logging.basicConfig(level=logging.ERROR)

    from app.tools.lead_tools import create_lead

    print("\n" + "="*60)
    print("DAY 19 — Email Automation Tool Tests")
    print("="*60)

    # ── TEST 1: Draft Email (LLM) ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 1: draft_email — AI writes the email")
    print("="*60)

    draft_cases = [
        ("Rahul Sharma", "follow_up",  "TechCorp",   "Demo scheduled for Friday"),
        ("Priya Mehta",  "pricing",    "StartupXYZ", "Interested in Starter plan"),
        ("Amit Kumar",   "welcome",    "InfoSys",    None),
    ]

    for name, etype, company, ctx in draft_cases:
        print(f"\nDrafting '{etype}' email for {name} ({company}):")
        result = draft_email.invoke({
            "recipient_name": name,
            "email_type":     etype,
            "company":        company,
            "context":        ctx,
        })
        print(result)
        print("-" * 40)

    # ── TEST 2: Send Email (simulation) ───────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 2: send_email — simulation mode")
    print("="*60)

    result = send_email.invoke({
        "to_email": "rahul@techcorp.com",
        "subject":  "Following up on our Growth Plan Discussion",
        "body": (
            "Hi Rahul,\n\n"
            "Great speaking with you today about the Growth Plan.\n"
            "Please let me know if you have any questions.\n\n"
            "Best regards,\nThe CRM Team"
        ),
    })
    print(result)

    # ── TEST 3: email_lead combined ───────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 3: email_lead — draft + send in one step")
    print("="*60)

    create_lead.invoke({
        "name":         "Rahul Sharma",
        "email":        "rahul@techcorp.com",
        "company":      "TechCorp Solutions",
        "plan_interest": "growth",
    })
    print("Lead #1 created.")

    result = email_lead.invoke({
        "lead_id":    1,
        "email_type": "follow_up",
        "context":    "Demo scheduled for this Friday",
    })
    print(result)

    # ── TEST 4: Tool schemas ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 4: Tool schemas — what agent sees")
    print("="*60)

    for t in [send_email, draft_email, email_lead]:
        print(f"\nTool: {t.name}")
        print(f"Args: {list(t.args.keys())}")
        print(f"Desc: {t.description[:80]}...")

    # ── TEST 5: No email address ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 5: email_lead — no email address (error handling)")
    print("="*60)

    create_lead.invoke({"name": "No Email Lead", "company": "TestCorp"})
    result = email_lead.invoke({"lead_id": 2, "email_type": "welcome"})
    print(result)

    # ── TEST 6: All 5 template types ─────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 6: All template types")
    print("="*60)

    for etype in ["follow_up", "welcome", "pricing", "demo_confirmation", "re_engagement"]:
        result = draft_email.invoke({
            "recipient_name": "Test User",
            "email_type":     etype,
            "company":        "TestCorp",
        })
        # Just print subject line
        subject_line = [l for l in result.split("\n") if l.startswith("Subject:")][0]
        print(f"  {etype:<20} -> {subject_line}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Day 19 COMPLETE — Email Tools working!")
    print("="*60)
    print("""
Email Tools added:
  draft_email  -> AI writes professional email content
  send_email   -> Delivers via SMTP / SendGrid / Simulation
  email_lead   -> Draft + Send in one step for a CRM lead

To send REAL emails add to .env:
  Gmail:      SMTP_EMAIL + SMTP_PASSWORD (App Password)
  SendGrid:   SENDGRID_API_KEY + SENDGRID_FROM_EMAIL
  Neither:    Simulation mode (safe for development)

Total CRM Tools: 10
Next -> Day 20: PostgreSQL Database Tool
""")
    print("="*60)
