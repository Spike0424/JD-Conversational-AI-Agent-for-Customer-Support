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

    def append(self, session_id: str, role: str, content: str) -> None:
        session = create_session()
        try:
            msg = AgentMessage(session_id=session_id, role=role, content=content)
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
