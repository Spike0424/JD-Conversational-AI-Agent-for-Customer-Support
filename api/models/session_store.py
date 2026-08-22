import logging

from sqlalchemy import select

from api.models.business import AgentMessage
from api.models.db import create_session, init_db

logger = logging.getLogger(__name__)


class SessionStore:
    """Session message persistence backed by agent_messages table (SQLModel)."""

    def __init__(self, max_history_turns: int = 6) -> None:
        self._max_messages = max_history_turns * 2
        init_db()

    def append(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: str | None = None,
        goods_name: str | None = None,
    ) -> None:
        session = create_session()
        try:
            msg = AgentMessage(
                session_id=session_id,
                user_id=user_id,
                role=role,
                content=content,
                goods_name=goods_name if role == "user" else None,
            )
            session.add(msg)
            session.commit()

            # Prune excess messages beyond the window
            if self._max_messages > 0:
                from sqlalchemy import text

                session.exec(
                    text(
                        """
                        DELETE FROM agent_messages
                        WHERE id IN (
                            SELECT id FROM agent_messages
                            WHERE session_id = :sid
                            ORDER BY id DESC
                            OFFSET :max_items
                        )
                        """
                    ),
                    params={"sid": session_id, "max_items": self._max_messages},
                )
                session.commit()
        finally:
            session.close()

    def load(self, session_id: str, limit: int | None = None) -> list[dict[str, str]]:
        max_items = limit or self._max_messages
        session = create_session()
        try:
            stmt = (
                select(AgentMessage)
                .where(AgentMessage.session_id == session_id)
                .order_by(AgentMessage.id.desc())
                .limit(max_items)
            )
            rows = session.exec(stmt).scalars().all()
            # Reverse to chronological order
            return [{"role": r.role, "content": r.content or ""} for r in reversed(rows)]
        finally:
            session.close()

    def recent_user_messages(self, session_id: str, n: int = 3) -> list[str]:
        """Return the contents of the most recent `n` user messages (chronological order)."""
        session = create_session()
        try:
            stmt = (
                select(AgentMessage)
                .where(AgentMessage.session_id == session_id, AgentMessage.role == "user")
                .order_by(AgentMessage.id.desc())
                .limit(n)
            )
            rows = session.exec(stmt).scalars().all()
            return [r.content or "" for r in reversed(rows)]
        finally:
            session.close()

    def load_last_assistant(self, session_id: str) -> str | None:
        """Return content of the most recent assistant message for dedup."""
        session = create_session()
        try:
            stmt = (
                select(AgentMessage)
                .where(
                    AgentMessage.session_id == session_id,
                    AgentMessage.role == "assistant",
                )
                .order_by(AgentMessage.id.desc())
                .limit(1)
            )
            row = session.exec(stmt).scalars().first()
            return row.content if row else None
        finally:
            session.close()

    # ── Per-user session listing (sidebar) ─────────────────────────────

    def list_sessions_for_user(self, user_id: str) -> list[dict]:
        """Return session metadata for the sidebar.

        Each row: ``{session_id, title, created_at, last_message_at,
        message_count}``. Title is the goods_name captured on the first user
        message; if missing we fall back to the first 20 chars of the first
        user message. Sessions are ordered by most-recent activity.
        """
        from sqlalchemy import func, text

        session = create_session()
        try:
            stmt = text(
                """
                SELECT
                    m.session_id,
                    MAX(m.timestamp) AS last_message_at,
                    MIN(m.timestamp) AS created_at,
                    COUNT(*) AS message_count,
                    (
                        SELECT COALESCE(
                            NULLIF(goods_name, ''),
                            SUBSTRING(content FROM 1 FOR 20)
                        )
                        FROM agent_messages
                        WHERE session_id = m.session_id
                          AND role = 'user'
                          AND user_id = :uid
                        ORDER BY id ASC
                        LIMIT 1
                    ) AS title
                FROM agent_messages m
                WHERE m.user_id = :uid
                GROUP BY m.session_id
                ORDER BY last_message_at DESC
                """
            )
            rows = session.exec(stmt, params={"uid": user_id}).all()
            return [
                {
                    "session_id": r.session_id,
                    "title": r.title or "(空对话)",
                    "created_at": r.created_at,
                    "last_message_at": r.last_message_at,
                    "message_count": r.message_count,
                }
                for r in rows
            ]
        finally:
            session.close()

    def load_full(self, session_id: str, user_id: str) -> list[dict] | None:
        """Return full chronological history for a session the user owns.

        Returns ``None`` if the session doesn't belong to ``user_id`` (or
        doesn't exist). Caller should treat that as 404.
        """
        session = create_session()
        try:
            owner = session.exec(
                select(AgentMessage.user_id)
                .where(AgentMessage.session_id == session_id)
                .limit(1)
            ).scalars().first()
            if owner is None or owner != user_id:
                return None
            # Order by timestamp (not id) so a historical insert-order bug
            # where assistant rows landed before user rows doesn't flip
            # the conversation visually on refresh.
            stmt = (
                select(AgentMessage)
                .where(AgentMessage.session_id == session_id)
                .order_by(AgentMessage.timestamp.asc(), AgentMessage.id.asc())
            )
            rows = session.exec(stmt).scalars().all()
            return [
                {
                    "role": r.role,
                    "content": r.content or "",
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in rows
            ]
        finally:
            session.close()

    def delete_session(self, session_id: str, user_id: str) -> bool:
        """Delete all messages of a session if it belongs to user_id.

        Returns True if rows were deleted, False if the session didn't exist
        or wasn't owned by user_id.
        """
        from sqlalchemy import delete

        session = create_session()
        try:
            stmt = (
                delete(AgentMessage)
                .where(
                    AgentMessage.session_id == session_id,
                    AgentMessage.user_id == user_id,
                )
            )
            result = session.exec(stmt)
            session.commit()
            return result.rowcount > 0  # type: ignore[attr-defined]
        finally:
            session.close()
