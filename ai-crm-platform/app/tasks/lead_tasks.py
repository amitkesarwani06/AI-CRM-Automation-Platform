import logging
import csv
import io
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="tasks.import_leads_csv",
)
def import_leads_csv_task(self, csv_content: str, source: str = "csv_import"):
    """Background task: Bulk import leads from CSV without blocking API."""
    from app.database.connection import SessionLocal
    from app.database.lead_repository import LeadRepository

    reader = csv.DictReader(io.StringIO(csv_content))
    rows = list(reader)
    total = len(rows)
    imported = 0
    skipped = 0
    errors = []

    logger.info(f"[Task {self.request.id}] Importing {total} leads from CSV")

    for i, row in enumerate(rows):
        try:
            self.update_state(
                state="PROGRESS",
                meta={
                    "current": i + 1,
                    "total": total,
                    "percent": int((i / total) * 100),
                    "status": f"Importing {row.get('name', 'unknown')}..."
                }
            )

            name = row.get("name", "").strip()
            if not name:
                skipped += 1
                continue

            with SessionLocal() as db:
                repo = LeadRepository(db)
                email = row.get("email", "").strip() or None
                if email and repo.get_by_email(email):
                    skipped += 1
                    errors.append(f"Duplicate: {email}")
                    continue

                repo.create(
                    name=name,
                    email=email,
                    phone=row.get("phone", "").strip() or None,
                    company=row.get("company", "").strip() or None,
                    plan_interest=row.get("plan_interest", "").strip() or None,
                    notes=f"Imported from {source}",
                )
                imported += 1

        except Exception as e:
            errors.append(f"Row {i+1}: {str(e)[:50]}")

    return {
        "status": "SUCCESS",
        "total": total,
        "imported": imported,
        "skipped": skipped,
        "errors": errors[:10],
    }


@celery_app.task(name="tasks.score_all_leads")
def score_all_leads_task():
    """Background task: Batch score all leads."""
    from app.database.connection import SessionLocal
    from app.database.lead_repository import LeadRepository

    with SessionLocal() as db:
        repo = LeadRepository(db)
        leads = repo.get_all(limit=500)

    scored = 0
    for lead in leads:
        try:
            score = 5
            if lead.email: score += 1
            if lead.phone: score += 1
            if lead.company: score += 1
            if lead.plan_interest == "enterprise": score += 2
            elif lead.plan_interest == "growth": score += 1

            score = min(score, 10)
            with SessionLocal() as db:
                repo = LeadRepository(db)
                repo.update(lead.id, score=score)
            scored += 1
        except Exception as e:
            logger.warning(f"Score failed for lead #{lead.id}: {e}")

    return {"status": "SUCCESS", "leads_scored": scored}
