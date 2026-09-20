import enum
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.sql import func
from app.database.connection import Base


class LeadStatus(str, enum.Enum):
    NEW         = "new"
    CONTACTED   = "contacted"
    QUALIFIED   = "qualified"
    UNQUALIFIED = "unqualified"
    CLOSED      = "closed"


class Lead(Base):
    """
    CRM Lead table — replaces _leads_db dict from Days 15-19.
    Data persists on disk, survives server restarts.
    """
    __tablename__ = "leads"

    # Primary key (auto-incremented by DB)
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Lead details
    name          = Column(String(255), nullable=False)
    email         = Column(String(255), nullable=True)
    phone         = Column(String(50),  nullable=True)
    company       = Column(String(255), nullable=True)
    city          = Column(String(100), nullable=True)

    # CRM fields
    plan_interest = Column(String(100), nullable=True)
    status        = Column(String(50),  default="new", nullable=False)
    score         = Column(Integer,     default=5,     nullable=True)  # 1-10
    notes         = Column(Text,        nullable=True)

    # Timestamps (set automatically by DB)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    def to_dict(self) -> dict:
        """Convert SQLAlchemy model to plain dict (needed for JSON serialization)."""
        return {
            "id":            self.id,
            "name":          self.name,
            "email":         self.email,
            "phone":         self.phone,
            "company":       self.company,
            "city":          self.city,
            "plan_interest": self.plan_interest,
            "status":        self.status,
            "score":         self.score,
            "notes":         self.notes,
            "created_at":    str(self.created_at) if self.created_at else None,
            "updated_at":    str(self.updated_at) if self.updated_at else None,
        }

    def __repr__(self):
        return f"<Lead #{self.id}: {self.name} ({self.company}) — {self.status}>"
