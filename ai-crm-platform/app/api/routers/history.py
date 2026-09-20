"""
History Router — Chat history & agent log endpoints.
Prefix: /history
"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.chat_history_service import (
    save_message,
    get_history,
    get_session_info,
    list_sessions,
    delete_session,
    log_agent_action,
    get_agent_logs,
    get_stats,
)

router = APIRouter(prefix="/history", tags=["Chat History"])
logger = logging.getLogger(__name__)


# ── Request / Response Schemas ────────────────────────────────────────────────

class MessageRequest(BaseModel):
    session_id: str
    role: str = "user"       # "user" | "assistant" | "system"
    content: str
    user_id: str = "anonymous"


class AgentLogRequest(BaseModel):
    session_id: str
    agent_name: str
    action: str
    input_data: dict = {}
    output_data: dict = {}
    duration_ms: Optional[int] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/message", summary="Save a chat message to a session")
async def post_message(req: MessageRequest):
    """
    Append a message to a conversation session.
    Creates the session automatically if it doesn't exist.
    """
    try:
        save_message(
            session_id=req.session_id,
            role=req.role,
            content=req.content,
            user_id=req.user_id,
        )
        return {"status": "saved", "session_id": req.session_id}
    except Exception as e:
        logger.error(f"Failed to save message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}", summary="Get full conversation history")
async def fetch_history(
    session_id: str,
    limit: int = Query(50, ge=1, le=200, description="Max messages to return"),
):
    """Return the last N messages for a session (oldest-first)."""
    messages = get_history(session_id, limit=limit)
    return {
        "session_id": session_id,
        "message_count": len(messages),
        "messages": messages,
    }


@router.get("/session/{session_id}/info", summary="Get session metadata")
async def fetch_session_info(session_id: str):
    """Return session metadata without the full message array."""
    info = get_session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return info


@router.get("/sessions", summary="List recent sessions")
async def fetch_sessions(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    limit: int = Query(20, ge=1, le=100),
):
    """List recent conversation sessions, newest first."""
    sessions = list_sessions(user_id=user_id, limit=limit)
    return {"total": len(sessions), "sessions": sessions}


@router.delete("/session/{session_id}", summary="Delete a session")
async def remove_session(session_id: str):
    """Permanently delete a conversation session and all its messages."""
    deleted = delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"status": "deleted", "session_id": session_id}


# ── Agent Logs ────────────────────────────────────────────────────────────────

@router.post("/log", summary="Log an agent action")
async def post_agent_log(req: AgentLogRequest):
    """Store a structured agent-action log entry for observability."""
    try:
        log_agent_action(
            session_id=req.session_id,
            agent_name=req.agent_name,
            action=req.action,
            input_data=req.input_data,
            output_data=req.output_data,
            duration_ms=req.duration_ms,
        )
        return {"status": "logged"}
    except Exception as e:
        logger.error(f"Failed to log agent action: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/{session_id}", summary="Get agent logs for a session")
async def fetch_agent_logs(
    session_id: str,
    limit: int = Query(100, ge=1, le=500),
):
    """Return agent action logs for a session, newest first."""
    logs = get_agent_logs(session_id, limit=limit)
    return {"session_id": session_id, "log_count": len(logs), "logs": logs}


# ── Stats ─────────────────────────────────────────────────────────────────────

@router.get("/stats", summary="MongoDB chat stats")
async def fetch_stats():
    """Return high-level stats: total sessions, messages, and agent logs."""
    try:
        return get_stats()
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
