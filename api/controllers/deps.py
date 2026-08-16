from functools import lru_cache

import jwt
from fastapi import Header, HTTPException
from jwt import DecodeError, ExpiredSignatureError, InvalidTokenError

from api.core.config import get_settings
from api.core.orchestrator import ChatOrchestrator


@lru_cache
def get_orchestrator() -> ChatOrchestrator:
    return ChatOrchestrator()


async def verify_chat_token(authorization: str = Header(...)) -> str:
    """Verify the Bearer JWT and return the user email.

    raises:
        HTTPException: 401 if token is missing, expired, or invalid.
    """
    try:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        token = authorization[7:]
        settings = get_settings()
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except (InvalidTokenError, DecodeError):
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload["sub"]
