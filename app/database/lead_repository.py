import logging
from typing import Optional
from sqlalchemy.orm import Session
from app.database.models import Lead

logger = logging.getLogger(__name__)


class LeadRepository:
    """
    All database operations for Leads in one place.
    Agents never write SQL directly — they call these methods.

    Usage:
        with SessionLocal() as db:
            repo = LeadRepository(db)
            lead = repo.create(name="Rahul", company="TechCorp")
    """

    def __init__(self, db: Session):
        self.db = db

    # ── Create ────────────────────────────────────────────────────────────────
    def create(
        self,
        name: str,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        company: Optional[str] = None,
        plan_interest: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Lead:
        """Create a new lead and persist to database."""
        lead = Lead(
            name=name,
            email=email,
            phone=phone,
            company=company,
            plan_interest=plan_interest,
            notes=notes,
            status="new",
        )
        self.db.add(lead)
        self.db.commit()        # writes to disk
        self.db.refresh(lead)   # loads DB-generated id and created_at
        logger.info(f"Lead created: #{lead.id} {lead.name}")
        return lead

    # ── Read ──────────────────────────────────────────────────────────────────
    def get_by_id(self, lead_id: int) -> Optional[Lead]:
        return self.db.query(Lead).filter(Lead.id == lead_id).first()

    def get_by_name(self, name: str) -> list[Lead]:
        """Case-insensitive partial name search."""
        return (
            self.db.query(Lead)
            .filter(Lead.name.ilike(f"%{name}%"))
            .all()
        )

    def get_by_email(self, email: str) -> Optional[Lead]:
        return (
            self.db.query(Lead)
            .filter(Lead.email.ilike(f"%{email}%"))
            .first()
        )

    def get_by_status(self, status: str) -> list[Lead]:
        return (
            self.db.query(Lead)
            .filter(Lead.status == status)
            .all()
        )

    def get_all(self, limit: int = 50) -> list[Lead]:
        return self.db.query(Lead).limit(limit).all()

    def count(self) -> int:
        return self.db.query(Lead).count()

    # ── Search across all fields (Mini Assignment) ────────────────────────────
    def search(self, query: str) -> list[Lead]:
        """Full-text search across name, company, email, notes."""
        pattern = f"%{query}%"
        return (
            self.db.query(Lead)
            .filter(
                Lead.name.ilike(pattern)    |
                Lead.company.ilike(pattern) |
                Lead.email.ilike(pattern)   |
                Lead.notes.ilike(pattern)
            )
            .all()
        )

    # ── Update ────────────────────────────────────────────────────────────────
    def update(
        self,
        lead_id: int,
        status: Optional[str] = None,
        notes: Optional[str] = None,
        plan_interest: Optional[str] = None,
        score: Optional[int] = None,
    ) -> Optional[Lead]:
        lead = self.get_by_id(lead_id)
        if not lead:
            return None

        if status is not None:        lead.status = status
        if notes is not None:         lead.notes = notes
        if plan_interest is not None: lead.plan_interest = plan_interest
        if score is not None:         lead.score = score

        self.db.commit()
        self.db.refresh(lead)
        logger.info(f"Lead #{lead_id} updated")
        return lead

    # ── Delete ────────────────────────────────────────────────────────────────
    def delete(self, lead_id: int) -> bool:
        lead = self.get_by_id(lead_id)
        if not lead:
            return False
        self.db.delete(lead)
        self.db.commit()
        logger.info(f"Lead #{lead_id} deleted")
        return True

    # ── Stats ─────────────────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        """Lead statistics for analytics."""
        total = self.count()
        statuses = ["new", "contacted", "qualified", "unqualified", "closed"]
        by_status = {}
        for s in statuses:
            by_status[s] = (
                self.db.query(Lead).filter(Lead.status == s).count()
            )
        return {"total_leads": total, "by_status": by_status}
