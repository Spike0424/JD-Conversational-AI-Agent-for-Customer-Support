"""Admin / debug endpoints for platform context (TurnContext + ContextType).

Endpoints:
  POST /v1/admin/turn-context/dry-run — parse a raw_query, return TurnContext JSON
  POST /v1/admin/context/validate     — validate a Context payload
  GET  /v1/admin/sessions/{sid}      — list recent agent_messages for a session
  POST /v1/admin/scene-cache/clear   — clear SceneClassifier cache (single or all)
  GET  /v1/admin/context-types       — list all ContextType values + flags
"""

from __future__ import annotations

import hmac
import os
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from api.models.context import ChannelType, Context, ContextType
from api.models.db import create_session
from api.core.scene_classifier import SceneClassifier
from api.core.turn_context import parse_turn_context, turn_context_to_dict
from api.models.business import AgentMessage
from sqlalchemy import select

router = APIRouter(prefix="/v1/admin", tags=["admin"])


def _admin_token() -> str | None:
    """Return the configured admin token, or None if not set (auth disabled)."""
    return os.environ.get("ADMIN_API_TOKEN") or None


async def require_admin_token(request: Request) -> None:
    """FastAPI dependency: 401 unless request Authorization matches ADMIN_API_TOKEN.

    If ADMIN_API_TOKEN is not set, auth is disabled (dev convenience).
    """
    expected = _admin_token()
    if not expected:
        return
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header",
        )
    presented = auth[7:].strip()
    if not hmac.compare_digest(presented, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token",
        )


_AdminAuth = Annotated[None, Depends(require_admin_token)]


# ── 1. dry-run ────────────────────────────────────────────────────────


class DryRunRequest(BaseModel):
    raw_query: str = Field(..., min_length=1, description="原始客户 turn 文本")


@router.post("/turn-context/dry-run", dependencies=[Depends(require_admin_token)])
async def turn_context_dry_run(request: DryRunRequest) -> dict[str, Any]:
    """解析 raw_query，返回结构化 TurnContext JSON。"""
    tc = parse_turn_context(request.raw_query)
    return turn_context_to_dict(tc)


# ── 2. validate ───────────────────────────────────────────────────────


class ValidateContextRequest(BaseModel):
    context: Context


@router.post("/context/validate", dependencies=[Depends(require_admin_token)])
async def validate_context(request: ValidateContextRequest) -> dict[str, Any]:
    """校验 Context 字段是否齐全，返回 ok / missing 列表。"""
    ctx = request.context
    missing: list[str] = []
    if ctx.type is ContextType.GOODS_CARD:
        if ctx.kwargs.goods_id is None:
            missing.append("kwargs.goods_id")
        if not ctx.kwargs.goods_name:
            missing.append("kwargs.goods_name")
    if ctx.type in {ContextType.IMAGE, ContextType.VIDEO}:
        if not ctx.kwargs.media_url:
            missing.append("kwargs.media_url")
    if ctx.type is ContextType.ORDER_INFO and not ctx.kwargs.order_sn:
        missing.append("kwargs.order_sn")
    return {
        "ok": len(missing) == 0,
        "missing": missing,
        "context_type": ctx.type.value,
        "is_silent": ctx.type.is_silent,
        "requires_human": ctx.type.requires_human,
    }


# ── 3. session history ────────────────────────────────────────────────


@router.get("/sessions/{session_id}", dependencies=[Depends(require_admin_token)])
async def get_session_history(
    session_id: str,
    limit: int = Query(default=20, ge=1, le=200),
) -> dict[str, Any]:
    """列出指定 session 最近 N 条 agent_messages。"""
    session = create_session()
    try:
        stmt = (
            select(AgentMessage)
            .where(AgentMessage.session_id == session_id)
            .order_by(AgentMessage.id.desc())
            .limit(limit)
        )
        rows = session.exec(stmt).all()
        messages = [
            {
                "id": r.id,
                "role": r.role,
                "content": r.content,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            }
            for r in reversed(rows)
        ]
        return {"session_id": session_id, "count": len(messages), "messages": messages}
    finally:
        session.close()


# ── 4. scene cache clear ──────────────────────────────────────────────


class ClearSceneCacheRequest(BaseModel):
    session_id: str | None = Field(default=None, description="清指定 session；不传则全清")


@router.post("/scene-cache/clear", dependencies=[Depends(require_admin_token)])
async def clear_scene_cache(request: ClearSceneCacheRequest) -> dict[str, Any]:
    """清掉 SceneClassifier 的内存缓存（默认 30 分钟 TTL）。"""
    SceneClassifier.clear_cache(session_id=request.session_id)
    return {"cleared": True, "session_id": request.session_id or "all"}


# ── 5. context-types catalog ──────────────────────────────────────────


@router.get("/context-types", dependencies=[Depends(require_admin_token)])
async def list_context_types() -> dict[str, Any]:
    """返回所有 ContextType 值 + is_silent / requires_human 标记，给前端表单用。"""
    return {
        "channel_types": [{"value": c.value, "name": c.name} for c in ChannelType],
        "context_types": [
            {
                "value": c.value,
                "name": c.name,
                "is_silent": c.is_silent,
                "requires_human": c.requires_human,
            }
            for c in ContextType
        ],
    }