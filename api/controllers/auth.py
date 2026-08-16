"""Auth endpoints: register + login. Both validate input via validate_* helpers."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.core.auth import (
    issue_token,
    validate_email,
    validate_password,
)
from api.models.user_store import get_user_store


router = APIRouter(tags=["auth"])

logger = logging.getLogger(__name__)


class AuthRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


@router.post("/v1/auth/register", response_model=TokenResponse, status_code=201)
def register(body: AuthRequest) -> TokenResponse:
    """Register a new user; auto-login (issue JWT) on success."""
    store = get_user_store()
    try:
        validate_email(body.email)
        validate_password(body.password)
        if store.is_registered(body.email):
            raise HTTPException(status_code=400, detail="email already registered")
        store.register(body.email, body.password)
        token, expires_in = issue_token(body.email)
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("register validation failed email=%s err=%s", body.email, exc)
        raise HTTPException(status_code=422, detail=str(exc))
    return TokenResponse(access_token=token, expires_in=expires_in)


@router.post("/v1/auth/login", response_model=TokenResponse)
def login(body: AuthRequest) -> TokenResponse:
    """Verify credentials and issue JWT."""
    store = get_user_store()
    try:
        validate_email(body.email)
        validate_password(body.password)
        if not store.verify(body.email, body.password):
            raise HTTPException(status_code=401, detail="invalid credentials")
        token, expires_in = issue_token(body.email)
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("login validation failed email=%s err=%s", body.email, exc)
        raise HTTPException(status_code=422, detail=str(exc))
    return TokenResponse(access_token=token, expires_in=expires_in)
