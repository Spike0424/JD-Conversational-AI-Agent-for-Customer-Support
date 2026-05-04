import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from redis import Redis
from redis.exceptions import RedisError

from app.config import Settings


@dataclass
class SessionMessage:
    role: str
    content: str


class SessionStore:
    def __init__(self, settings: Settings) -> None:
        self._max_messages = settings.max_history_turns * 2
        self._redis: Redis | None = None
        self._sqlite_path = self._parse_sqlite_path(settings.database_url)
        self._init_sqlite()

        if settings.redis_url.strip():
            try:
                client = Redis.from_url(settings.redis_url, decode_responses=True)
                client.ping()
                self._redis = client
            except RedisError:
                self._redis = None

    @staticmethod
    def _parse_sqlite_path(database_url: str) -> Path:
        prefix = "sqlite:///"
        if not database_url.startswith(prefix):
            raise ValueError("DATABASE_URL must be sqlite:///path/to/db.sqlite")
        raw_path = database_url[len(prefix) :]
        return Path(raw_path).resolve()

    def _init_sqlite(self) -> None:
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._sqlite_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    @staticmethod
    def _redis_key(session_id: str) -> str:
        return f"agent:session:{session_id}"

    def append(self, session_id: str, role: str, content: str) -> None:
        if self._redis:
            try:
                key = self._redis_key(session_id)
                self._redis.rpush(key, json.dumps({"role": role, "content": content}))
                self._redis.ltrim(key, -self._max_messages, -1)
                return
            except RedisError:
                self._redis = None

        with sqlite3.connect(self._sqlite_path) as conn:
            conn.execute(
                "INSERT INTO session_messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )
            conn.execute(
                """
                DELETE FROM session_messages
                WHERE id IN (
                    SELECT id FROM session_messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (session_id, self._max_messages),
            )
            conn.commit()

    def load(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        max_items = limit or self._max_messages

        if self._redis:
            try:
                raw_items = self._redis.lrange(self._redis_key(session_id), -max_items, -1)
                records = [json.loads(item) for item in raw_items]
                return [{"role": str(r["role"]), "content": str(r["content"])} for r in records]
            except RedisError:
                self._redis = None

        with sqlite3.connect(self._sqlite_path) as conn:
            cursor = conn.execute(
                """
                SELECT role, content FROM (
                    SELECT role, content, id
                    FROM session_messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) sub
                ORDER BY id ASC
                """,
                (session_id, max_items),
            )
            return [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
