"""
Chat History Service — MongoDB-backed conversation persistence.
Saves/loads conversation history and agent logs per session.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from app.database.mongo_connection import get_mongo_db

logger = logging.getLogger(__name__)


def _now() -> datetime:
    """UTC-aware timestamp."""
    return datetime.now(timezone.utc)


# ── Conversation History ──────────────────────────────────────────────────────

def save_message(session_id: str, role: str, content: str, user_id: str = "anonymous") -> dict:
    """
    Append a single message to a conversation session.
    Creates the session document if it does not exist yet.

    Returns the updated session document.
    """
    db = get_mongo_db()
    message = {
        "role": role,          # "user" | "assistant" | "system"
        "content": content,
        "timestamp": _now(),
    }

    result = db.conversations.find_one_and_update(
        {"session_id": session_id},
        {
            "$push": {"messages": message},
            "$set": {
                "user_id": user_id,
                "updated_at": _now(),
            },
            "$setOnInsert": {
                "session_id": session_id,
                "created_at": _now(),
            },
        },
        upsert=True,
        return_document=True,
    )
    logger.debug(f"Message saved to session {session_id}")
    return result


def get_history(session_id: str, limit: int = 50) -> list[dict]:
    """
    Return the last *limit* messages for a session (chronological order).
    """
    db = get_mongo_db()
    doc = db.conversations.find_one({"session_id": session_id})
    if not doc:
        return []
    messages = doc.get("messages", [])
    # Return last N messages, oldest first
    return messages[-limit:]


def get_session_info(session_id: str) -> Optional[dict]:
    """Return session metadata (without the full message array)."""
    db = get_mongo_db()
    doc = db.conversations.find_one(
        {"session_id": session_id},
        {"messages": 0},   # exclude heavy messages field
    )
    if not doc:
        return None
    doc.pop("_id", None)
    return doc


def list_sessions(user_id: str = None, limit: int = 20) -> list[dict]:
    """List recent sessions, optionally filtered by user_id."""
    db = get_mongo_db()
    query = {}
    if user_id:
        query["user_id"] = user_id

    cursor = (
        db.conversations.find(query, {"messages": 0})
        .sort("updated_at", -1)
        .limit(limit)
    )
    sessions = []
    for doc in cursor:
        doc.pop("_id", None)
        sessions.append(doc)
    return sessions


def delete_session(session_id: str) -> bool:
    """Delete an entire conversation session. Returns True if deleted."""
    db = get_mongo_db()
    result = db.conversations.delete_one({"session_id": session_id})
    return result.deleted_count > 0


# ── Agent Logs ────────────────────────────────────────────────────────────────

def log_agent_action(
    session_id: str,
    agent_name: str,
    action: str,
    input_data: dict = None,
    output_data: dict = None,
    duration_ms: int = None,
) -> None:
    """Insert a structured agent-action log entry."""
    db = get_mongo_db()
    entry = {
        "session_id": session_id,
        "agent_name": agent_name,
        "action": action,
        "input": input_data or {},
        "output": output_data or {},
        "duration_ms": duration_ms,
        "created_at": _now(),
    }
    db.agent_logs.insert_one(entry)
    logger.debug(f"Agent log saved: {agent_name}.{action} for session {session_id}")


def get_agent_logs(session_id: str, limit: int = 100) -> list[dict]:
    """Return agent logs for a session, newest first."""
    db = get_mongo_db()
    cursor = (
        db.agent_logs.find({"session_id": session_id}, {"_id": 0})
        .sort("created_at", -1)
        .limit(limit)
    )
    return list(cursor)


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    """High-level stats about conversations and logs stored in MongoDB."""
    db = get_mongo_db()
    total_sessions = db.conversations.count_documents({})
    total_logs = db.agent_logs.count_documents({})

    # Message count via aggregation
    pipeline = [{"$project": {"count": {"$size": "$messages"}}},
                {"$group": {"_id": None, "total": {"$sum": "$count"}}}]
    agg = list(db.conversations.aggregate(pipeline))
    total_messages = agg[0]["total"] if agg else 0

    return {
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "total_agent_logs": total_logs,
    }
