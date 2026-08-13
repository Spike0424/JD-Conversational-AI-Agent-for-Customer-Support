"""Tool: send_product_card — LLM 可调用的商品卡发送工具。

LLM 传入 session_id + goods_id + message，此工具从 ProductKnowledge 表查出商品信息
并暂存到全局 list（每张卡自带 session_id 标识）。orchestrator 在 ask() 末尾按
session_id filter 取出卡片，塞到 ChatResponse.product_cards 中返回给前端渲染。
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from pydantic import BaseModel, Field

from api.models.business import ProductKnowledge
from api.models.db import create_session
from api.services.rag_logger import RAGLogger
from api.services.knowledge import get_search_knowledge
from util.agent_tool import agent_tool

logger = logging.getLogger(__name__)

# ── ZOL spec value cleanup ───────────────────────────────────────────


def _clean_spec_value(val: Any) -> str:
    """Remove ZOL website artifacts from spec values.

    Handles: trailing '>', '更多xxx>', '查看外观图>', 'xxx>5.2万张照片...'
    """
    if not isinstance(val, str):
        return val
    # Remove "更多xxx>" patterns (ZOL internal links)
    val = re.sub(r'更多.*$', '', val)
    # Remove "查看外观图>" / "查看外观>" patterns
    val = re.sub(r'\s*查看外观.*$', '', val)
    # Remove ">数字..." comparison data (e.g., "256GB>5.2万张照片2.2万首歌曲")
    val = re.sub(r'>[\d].*$', '', val)
    # Remove trailing ">"
    val = val.rstrip('>')
    return val.strip()


# ── Card buffer (plain list; each card carries its own session_id) ────

_CARDS: list[dict[str, Any]] = []


def pop_cards_for_session(session_id: str) -> list[dict[str, Any]]:
    """取出属于该 session 的所有卡片，并从全局列表中移除。"""
    global _CARDS
    matched = [c for c in _CARDS if c.get("session_id") == session_id]
    _CARDS = [c for c in _CARDS if c.get("session_id") != session_id]
    return matched


# ── Params model ──────────────────────────────────────────────────────


class SendProductCardParams(BaseModel):
    session_id: str = Field(..., description="当前会话 ID")
    goods_id: int = Field(..., description="要发送的商品 ID")
    message: str = Field(default="", description="附带在卡片上的文字说明")


# ── Tool registration ─────────────────────────────────────────────────

_send_description = (
    "向客户发送一张商品卡片。当客户询问商品详情、希望推荐类似商品、"
    "或询问是否有更便宜/更贵的选项时，使用此工具。"
    "返回的商品信息来自知识库，不要自行编造。"
)


@agent_tool(
    name="send_product_card",
    description=_send_description,
    param_model=SendProductCardParams,
)
def send_product_card(**kwargs: object) -> str:
    params = SendProductCardParams(**{k: v for k, v in kwargs.items() if v is not None})

    sk = get_search_knowledge()
    shop_id = sk.get_shop_id() if sk else None
    if shop_id is None:
        return "错误：无法确定当前店铺。请先确认会话中的 shop_id。"

    tid = uuid.uuid4().hex[:12]
    RAGLogger.log_query(tid, f"send_product_card(goods_id={params.goods_id})")

    session = create_session()
    try:
        row = session.query(ProductKnowledge).filter(
            ProductKnowledge.shop_id == shop_id,
            ProductKnowledge.goods_id == params.goods_id,
        ).first()
    finally:
        session.close()

    if row is None:
        RAGLogger.log_trace(
            tid, phase="retrieval", user_query=f"send_product_card(goods_id={params.goods_id})",
            scene="product_card", shop_id=shop_id, goods_id=params.goods_id,
            filters={"shop_id": shop_id, "goods_id": params.goods_id},
            top_k=0, retrieved_chunks=[], rerank_result=[],
            retrieval_hit=False,
        )
        return "提示：店铺中暂无该商品记录，请如实告知客人暂无此商品信息。"

    specs_json = row.specifications or {}
    # Clean ZOL artifacts from spec values (safety net for uncleaned data)
    specs_json = {k: _clean_spec_value(v) for k, v in specs_json.items()}

    _CARDS.append({
        "session_id": params.session_id,
        "shop_id": shop_id,
        "goods_id": params.goods_id,
        "goods_name": row.goods_name,
        "price": row.price,
        "price_min": row.price_min,
        "price_max": row.price_max,
        "thumb_url": row.thumb_url,
        "specifications": specs_json,
        "message": params.message,
    })

    # Log to RAG so the retrieval is traceable
    RAGLogger.log_trace(
        tid, phase="retrieval", user_query=f"send_product_card(goods_id={params.goods_id})",
        scene="product_card", shop_id=shop_id, goods_id=params.goods_id,
        filters={"shop_id": shop_id, "goods_id": params.goods_id},
        top_k=1,
        retrieved_chunks=[{"source": row.goods_name, "score": 1.0, "snippet": str(specs_json)[:200]}],
        rerank_result=[{"rank": 1, "source": row.goods_name, "match_type": "exact", "final_score": 1.0}],
        retrieval_hit=True,
    )

    # Return real specs to LLM as structured key-value list (already cleaned at line 121).
    specs_text = ""
    if specs_json:
        lines = ["", "【商品参数（来自数据库）】"]
        for key, val in specs_json.items():
            if val and val != "未提供":
                lines.append(f"- {key}: {val}")
        specs_text = "\n".join(lines)[:3000]

    return f"已为客户发送商品卡片【{row.goods_name}】（{row.price or ''}）{params.message}{specs_text}"