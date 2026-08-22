"""Session metadata endpoints (sidebar history): list, fetch messages, delete.

All endpoints require a Bearer JWT (the same token used by /v1/chat/*).
Sessions are scoped to the authenticated user; cross-user access returns 404.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException

from api.controllers.deps import verify_chat_token
from api.models.session_store import SessionStore

router = APIRouter(tags=["sessions"])


@lru_cache
def _get_session_store() -> SessionStore:
    """Singleton SessionStore — avoids re-running init_db / migrations on
    every request and keeps a single SQLAlchemy session factory in use.
    """
    return SessionStore()


@router.get("/v1/sessions")
def list_sessions(email: str = Depends(verify_chat_token)) -> dict:
    """Return the current user's chat sessions for the sidebar.

    Each row carries title (goods_name captured on the first user message,
    falling back to the first 20 chars of the first user message), creation
    timestamp, last-activity timestamp, and total message count. Ordered by
    most-recent activity.
    """
    store = _get_session_store()
    return {"sessions": store.list_sessions_for_user(email)}


@router.get("/v1/sessions/{session_id}/messages")
def list_session_messages(
    session_id: str,
    email: str = Depends(verify_chat_token),
) -> dict:
    """Return full chronological history for one session owned by the user."""
    store = _get_session_store()
    messages = store.load_full(session_id, email)
    if messages is None:
        raise HTTPException(status_code=404, detail="session not found")
    return {"session_id": session_id, "messages": messages}


@router.delete("/v1/sessions/{session_id}", status_code=204)
def delete_session(
    session_id: str,
    email: str = Depends(verify_chat_token),
) -> None:
    """Delete all messages of a session the user owns."""
    store = _get_session_store()
    if not store.delete_session(session_id, email):
        raise HTTPException(status_code=404, detail="session not found")