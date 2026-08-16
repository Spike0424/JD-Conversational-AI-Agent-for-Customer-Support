"""JWT signing/verification + registration validation for chat endpoints."""

from __future__ import annotations

import re
import time
from typing import Any

import jwt

from api.core.config import get_settings


_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

MIN_PASSWORD_LEN = 8
MAX_PASSWORD_LEN = 64


def validate_email(email: str) -> None:
    """Raise ValueError if email is not a valid format.

    args:
        email: input string to validate.

    raises:
        ValueError: if email is empty or not a valid email format.
    """
    if not email or not _EMAIL_RE.match(email):
        raise ValueError(f"invalid email format: {email!r}")


def validate_password(password: str) -> None:
    """Raise ValueError if password length is out of bounds.

    args:
        password: input string to validate.

    raises:
        ValueError: if password is shorter than MIN or longer than MAX.
    """
    if not password or len(password) < MIN_PASSWORD_LEN:
        raise ValueError(f"password too short (min {MIN_PASSWORD_LEN} chars)")
    if len(password) > MAX_PASSWORD_LEN:
        raise ValueError(f"password too long (max {MAX_PASSWORD_LEN} chars)")


def issue_token(email: str) -> tuple[str, int]:
    """Sign a JWT for the user. Returns (token, expires_in_seconds)."""
    settings = get_settings()
    expires_in = settings.jwt_expires_seconds
    now = int(time.time())
    payload: dict[str, Any] = {"sub": email, "iat": now, "exp": now + expires_in}
    token = jwt.encode(payload, settings.jwt_secret, algorithm="HS256")
    return token, expires_in


def verify_token(token: str) -> dict[str, Any]:
    """Verify JWT and return payload. Raises jwt.InvalidTokenError on failure."""
    settings = get_settings()
    return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
