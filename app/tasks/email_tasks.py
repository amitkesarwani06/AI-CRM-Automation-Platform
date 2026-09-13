import os
import logging
import time
from typing import Optional
from app.tasks.celery_app import celery_app
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    name="tasks.send_single_email",
)
def send_single_email_task(
    self,
    to_email: str,
    subject: str,
    body: str,
    lead_id: Optional[int] = None,
):
    """Background task: Send a single email."""
    try:
        logger.info(f"[Task {self.request.id}] Sending email to {to_email}")
        self.update_state(
            state="STARTED",
            meta={"status": f"Sending to {to_email}..."}
        )

        from app.tools.email_tools import _send_via_smtp
        result = _send_via_smtp(to_email, subject, body)

        logger.info(f"[Task {self.request.id}] Email sent: {result}")
        return {
            "status": "SUCCESS",
            "to": to_email,
            "subject": subject,
            "result": result,
            "lead_id": lead_id,
        }
    except Exception as exc:
        logger.error(f"[Task {self.request.id}] Email failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(
    bind=True,
    max_retries=2,
    name="tasks.send_bulk_email",
)
def send_bulk_email_task(
    self,
    lead_ids: list[int],
    email_type: str = "follow_up",
    context: Optional[str] = None,
):
    """Background task: Send emails to multiple leads with progress updates."""
    total = len(lead_ids)
    sent = 0
    failed = 0
    errors = []

    logger.info(f"[Task {self.request.id}] Bulk email to {total} leads")

    for i, lead_id in enumerate(lead_ids):
        try:
            progress = int((i / total) * 100)
            self.update_state(
                state="PROGRESS",
                meta={
                    "current": i + 1,
                    "total": total,
                    "percent": progress,
                    "status": f"Sending to lead #{lead_id}..."
                }
            )

            from app.database.connection import SessionLocal
            from app.database.lead_repository import LeadRepository

            with SessionLocal() as db:
                repo = LeadRepository(db)
                lead = repo.get_by_id(lead_id)
                if not lead or not lead.email:
                    failed += 1
                    errors.append(f"Lead #{lead_id}: no email")
                    continue

            from app.tools.email_tools import _draft_email_with_llm, _send_via_smtp
            draft = _draft_email_with_llm(
                recipient_name=lead.name,
                email_type=email_type,
                company=lead.company,
                context=context,
            )

            _send_via_smtp(lead.email, draft["subject"], draft["body"])
            sent += 1
            time.sleep(0.5)

        except Exception as e:
            failed += 1
            errors.append(f"Lead #{lead_id}: {str(e)[:50]}")
            logger.warning(f"Failed to email lead #{lead_id}: {e}")

    return {
        "status": "SUCCESS",
        "total": total,
        "sent": sent,
        "failed": failed,
        "errors": errors[:10],
    }
