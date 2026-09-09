import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Database URL ──────────────────────────────────────────────────────────────
# SQLite (default dev): no setup needed — crm_dev.db file created automatically
# PostgreSQL (production): postgresql://user:password@localhost:5432/crm_db
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./crm_dev.db"
)

# ── Engine ────────────────────────────────────────────────────────────────────
# pool_pre_ping: test connection before using — handles DB restarts gracefully
connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    connect_args=connect_args,
)

# ── Session Factory ───────────────────────────────────────────────────────────
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# ── Base Class (all models inherit from this) ─────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Initialize DB ─────────────────────────────────────────────────────────────
def init_db() -> None:
    """
    Create all tables if they don't exist.
    Safe to call multiple times — won't drop existing data.
    """
    from app.database.models import Lead  # noqa — needed for Base to register models
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database initialized: {DATABASE_URL}")
    print(f"[DB] Tables ready: {DATABASE_URL}")
