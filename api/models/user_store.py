"""Postgres-backed user registry (email -> password_hash).

Replaces the previous in-memory dict so registered users survive process
restarts. Kept the same public API (is_registered / register / verify) so
auth.py callers don't change.
"""

from __future__ import annotations

import hashlib

from sqlalchemy import select

from api.models.business import User
from api.models.db import create_session


class UserStore:
    def is_registered(self, email: str) -> bool:
        session = create_session()
        try:
            stmt = select(User.id).where(User.email == email).limit(1)
            return session.exec(stmt).scalars().first() is not None
        finally:
            session.close()

    def register(self, email: str, password: str) -> None:
        session = create_session()
        try:
            session.add(User(email=email, password_hash=self._hash(password)))
            session.commit()
        finally:
            session.close()

    def verify(self, email: str, password: str) -> bool:
        session = create_session()
        try:
            stmt = select(User.password_hash).where(User.email == email).limit(1)
            stored = session.exec(stmt).scalars().first()
            if stored is None:
                return False
            return stored == self._hash(password)
        finally:
            session.close()

    @staticmethod
    def _hash(password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()


_user_store = UserStore()


def get_user_store() -> UserStore:
    return _user_store
