"""In-memory user registry (email -> password_hash). Replace with DB later.

Process-local only - not safe for multi-worker / multi-process deployments.
For production, swap this for a real user table (Postgres / etc).
"""

from __future__ import annotations

import hashlib


class InMemoryUserStore:
    def __init__(self) -> None:
        self._users: dict[str, str] = {}

    def is_registered(self, email: str) -> bool:
        return email in self._users

    def register(self, email: str, password: str) -> None:
        self._users[email] = self._hash(password)

    def verify(self, email: str, password: str) -> bool:
        stored = self._users.get(email)
        if stored is None:
            return False
        return stored == self._hash(password)

    @staticmethod
    def _hash(password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()


_users = InMemoryUserStore()


def get_user_store() -> InMemoryUserStore:
    return _users
