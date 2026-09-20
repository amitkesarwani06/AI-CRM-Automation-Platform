import os
import logging
from pymongo import MongoClient
from pymongo.database import Database
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB", "crm_db")

_client: MongoClient = None
_db: Database = None


def get_mongo_client() -> MongoClient:
    """Get or create MongoDB client (singleton)."""
    global _client
    if _client is None:
        _client = MongoClient(
            MONGO_URL,
            serverSelectionTimeoutMS=5000,
        )
        logger.info(f"MongoDB client created: {MONGO_URL}")
    return _client


def get_mongo_db() -> Database:
    """Get CRM database and ensure indexes exist."""
    global _db
    if _db is None:
        client = get_mongo_client()
        _db = client[MONGO_DB]
        # Create indexes for fast querying
        _db.conversations.create_index("session_id", unique=True)
        _db.conversations.create_index("user_id")
        _db.conversations.create_index("created_at")
        _db.agent_logs.create_index("session_id")
        _db.agent_logs.create_index("created_at")
        logger.info(f"MongoDB database ready: {MONGO_DB}")
    return _db


def ping_mongo() -> bool:
    """Check if MongoDB is reachable."""
    try:
        get_mongo_client().admin.command("ping")
        return True
    except Exception as e:
        logger.warning(f"MongoDB not reachable: {e}")
        return False
