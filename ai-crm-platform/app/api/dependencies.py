import os
from fastapi import Header, HTTPException, status
from sqlalchemy.orm import Session
from app.database.connection import SessionLocal


# ── API Key Auth ──────────────────────────────────────────────────────────────
API_KEY = os.getenv("API_KEY", "crm-dev-key-2024")


def verify_api_key(x_api_key: str = Header(default=None)):
    """
    Simple API key authentication.
    Pass 'X-API-Key: your-key' header in requests.
    Skip for Swagger UI testing (set API_KEY=none in .env to disable).
    """
    current_key = os.getenv("API_KEY", API_KEY)
    if current_key == "none":
        return True  # auth disabled in dev

    if not x_api_key or x_api_key != current_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Pass X-API-Key header.",
        )
    return True


# ── Database Session ───────────────────────────────────────────────────────────
def get_db():
    """Dependency: provides a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
