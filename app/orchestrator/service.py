import logging
import uuid
from collections.abc import Iterator
from typing import Any

from app.context_models import ChannelKwargs, Context, ContextType
from app.llm.agent_runtime import ReActQAAgent
from app.orchestrator.intent_router import CUSTOMER_SCENE_LABELS, normalize_customer_scene
from app.orchestrator.scene_classifier import SceneClassifier
from app.orchestrator.turn_context import parse_turn_context, turn_context_to_dict
from app.schemas import ChatResponse, DocumentIngestResponse, HandoffResponse
from app.tools.search_knowledge import get_search_knowledge

logger = logging.getLogger(__name__)

_GOODS_CARD_ONLY_REPLY = "亲，您想了解这款商品的哪方面呢？"


def _context_to_dependencies(context: Context, raw_query: str) -> dict[str, Any]:
    """把 Context + 解析后的 raw_query 拼成依赖字典，给 scene_classifier 和 agent 用。"""
    kw: ChannelKwargs = context.kwargs
    deps: dict[str, Any] = {
        "shop_id": kw.shop_id,
        "shop_name": kw.shop_name,
        "user_id": kw.user_id,
        "customer_uid": kw.from_uid or kw.recipient_uid,
        "from_uid": kw.from_uid,
        "recipient_uid": kw.recipient_uid or kw.from_uid,
        "goods_id": kw.goods_id,
        "goods_name": kw.goods_name,
        "order_sn": kw.order_sn,
        "context_type": context.type.value,
        "channel_type": context.channel_type.value if context.channel_type else "",
        "media_url": kw.media_url,
        "media_type": kw.media_type,
        "raw_query": raw_query,
    }
    return deps


class ChatOrchestrator:
    def __init__(self) -> None:
        self._agent = ReActQAAgent()
        self._scene_classifier = SceneClassifier()

    @staticmethod
    def _new_trace_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def _format_header(scene: str) -> str:
        scene_label = CUSTOMER_SCENE_LABELS.get(scene, scene)
        return f"说明：已识别为「{scene_label}」场景（scene={scene}）。\n\n回答：\n"

    async def chat(
        self,
        session_id: str,
        question: str,
        context: Context | None = None,
        dependencies: dict[str, Any] | None = None,
    ) -> ChatResponse:
        trace_id = self._new_trace_id()

        if context is None:
            context = Context(content=question, kwargs=ChannelKwargs())
        elif context.content is None:
            context = context.model_copy(update={"content": question})

        # ── 1. 静默类：撤回 / 认证 / 系统推送 → 直接返回空 ──
        if context.type.is_silent:
            logger.info(
                "Silent context_type=%s session_id=%s — skipping LLM",
                context.type.value, session_id,
            )
            return ChatResponse(
                session_id=session_id,
                answer="",
                trace_id=trace_id,
                intent="silent",
                context_type=context.type.value,
                actions=[],
            )

        # ── 2. 解析 turncontext（即使纯文本也走一遍归一化）──
        raw_query = context.content or question
        turn_ctx = parse_turn_context(raw_query)
        deps = _context_to_dependencies(context, raw_query)
        if dependencies:
            deps.update(dependencies)

        # ── 3. 图片 / 视频：转人工占位 ──
        if context.type.requires_human:
            logger.info(
                "Human-required context_type=%s session_id=%s — placeholder handoff",
                context.type.value, session_id,
            )
            return ChatResponse(
                session_id=session_id,
                answer="",
                trace_id=trace_id,
                intent="media_handoff",
                context_type=context.type.value,
                actions=["transfer_to_human"],
                need_handoff=True,
            )

        # ── 4. 纯商品卡无文字 → 追问 ──
        if (
            turn_ctx.turn_type.has_product_card
            and not turn_ctx.turn_type.has_text
        ):
            logger.info(
                "Product-card-only turn session_id=%s goods_id=%s — asking for intent",
                session_id, turn_ctx.product_card.goods_id,
            )
            return ChatResponse(
                session_id=session_id,
                answer=_GOODS_CARD_ONLY_REPLY,
                trace_id=trace_id,
                intent="goods_card_only",
                context_type=context.type.value,
            )

        # ── 5. 正常流：scene classifier + agent ──
        # 当 context.content 已经被 TurnContext 归一化过，优先用提取出来的 customer_text
        # （商品卡 / 订单卡 / 元数据已经被剥离）作为 question，避免把空 question 喂给 LLM。
        effective_question = turn_ctx.customer_text.strip() or question
        raw_scene = await self._scene_classifier.classify(deps, effective_question, session_id)
        scene = normalize_customer_scene(raw_scene) or raw_scene

        if self._scene_classifier.last_scene_hint == "mixed_orders":
            scene = "mixed"

        shop_id = deps.get("shop_id")
        goods_id = deps.get("goods_id")
        sk = get_search_knowledge()
        if sk:
            sk.set_context(shop_id=shop_id, scene=scene, goods_id=goods_id)

        answer = self._agent.ask(
            session_id=session_id,
            question=effective_question,
            scene=scene,
            dependencies=deps,
        )
        formatted = self._format_header(scene) + answer.strip()
        return ChatResponse(
            session_id=session_id,
            answer=formatted,
            trace_id=trace_id,
            intent=scene,
            context_type=context.type.value,
            need_handoff=False,
        )

    async def stream_chat(
        self,
        session_id: str,
        question: str,
        context: Context | None = None,
        dependencies: dict[str, Any] | None = None,
    ) -> tuple[str, str, list, list[str], Iterator[str], str]:
        trace_id = self._new_trace_id()

        if context is None:
            context = Context(content=question, kwargs=ChannelKwargs())
        elif context.content is None:
            context = context.model_copy(update={"content": question})

        # silent / image / video / goods_card_only 在 stream 路径下也走短路由。
        # 这里只对 silent 直接返空，其余交由前端读首条 SSE delta 的 actions 字段决定。
        if context.type.is_silent:
            def _empty_stream() -> Iterator[str]:
                if False:
                    yield ""
            return (
                trace_id,
                "silent",
                [],
                [],
                _empty_stream(),
                "LLM",
            )

        raw_query = context.content or question
        turn_ctx = parse_turn_context(raw_query)
        deps = _context_to_dependencies(context, raw_query)
        if dependencies:
            deps.update(dependencies)

        raw_scene = await self._scene_classifier.classify(deps, question, session_id)
        scene = normalize_customer_scene(raw_scene) or raw_scene

        if self._scene_classifier.last_scene_hint == "mixed_orders":
            scene = "mixed"

        shop_id = deps.get("shop_id")
        goods_id = deps.get("goods_id")
        sk = get_search_knowledge()
        if sk:
            sk.set_context(shop_id=shop_id, scene=scene, goods_id=goods_id)

        return (
            trace_id,
            scene,
            [],
            [],
            self._agent.ask_stream(
                session_id=session_id, question=question, scene=scene, dependencies=deps
            ),
            "LLM",
        )

    def handoff(self, session_id: str, reason: str, priority: str) -> HandoffResponse:
        ticket_id = f"TKT-{uuid.uuid4().hex[:10].upper()}"
        queue = "vip" if priority.lower() == "high" else "standard"
        return HandoffResponse(
            session_id=session_id,
            ticket_id=ticket_id,
            queue=queue,
            status=f"queued ({reason})",
        )

    def ingest_document(self, source: str, content: str) -> DocumentIngestResponse:
        chunks_added = self._agent.ingest_document(source=source, content=content)
        return DocumentIngestResponse(source=source, chunks_added=chunks_added)

    def ingest_pdf_bytes(self, source: str, filename: str, pdf_bytes: bytes) -> DocumentIngestResponse:
        chunks_added = self._agent.ingest_pdf_bytes(source=source, filename=filename, pdf_bytes=pdf_bytes)
        return DocumentIngestResponse(source=source, chunks_added=chunks_added)

    def ingest_pdf_path(self, pdf_path: str, source: str | None = None) -> DocumentIngestResponse:
        chunks_added = self._agent.ingest_pdf_path(pdf_path=pdf_path, source=source)
        label = (source or "").strip() or pdf_path
        return DocumentIngestResponse(source=label, chunks_added=chunks_added)