import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from api.models.context import ChannelKwargs, Context, ContextType
from api.core.agent_runtime import ReActQAAgent
from api.core.intent_router import CUSTOMER_SCENE_LABELS, normalize_customer_scene
from api.core.request_state import RequestState, RequestTracker
from api.core.scene_classifier import SceneClassifier
from api.core.turn_context import parse_turn_context, turn_context_to_dict
from api.models.schemas import ChatResponse, HandoffResponse
from api.services.knowledge import get_search_knowledge

logger = logging.getLogger(__name__)

_GOODS_CARD_ONLY_REPLY = "亲，您想了解这款商品的哪方面呢？"

GREETINGS = {"hi", "hello", "在吗", "你好", "有人吗", "在不在", "喂", "哈喽", "下午好", "早上好", "您好"}


def _is_greeting(text: str) -> bool:
    """Check if text is a simple greeting (hi/你好/etc).

    args:
        text: user input.

    returns: True if text matches a greeting in the GREETINGS set.
    """
    return text.strip().lower() in GREETINGS


_WELCOME_MESSAGE = """您好！终于等到您，欢迎光临本店！

【今日特惠福利播报】：
• 关注领券：点击店铺右上角"关注"，即可领取 [满XXX减XX] 粉丝立减券！
• 爆款现货：目前 [热门手机型号] 少量现货已到仓，余量紧俏，建议尽早锁库！
• 加赠保障：今日下单额外赠送 [一年店铺免费碎屏保/延保]，给新机双重防护！

请问您看中了哪款配色和内存规格？我来帮您查询库存和今天最划算的到手价！"""


def _context_to_dependencies(context: Context, raw_query: str) -> dict[str, Any]:
    """Build the dependencies dict from Context + parsed raw query.

    args:
        context: parsed Context from turn parsing.
        raw_query: cleaned customer text.

    returns: dict for scene_classifier and agent.
    """
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
    def _scene_metadata(scene: str) -> dict[str, str]:
        """Look up scene display label.

        args:
            scene: scene key.

        returns: {"scene": str, "scene_label": str} dict.
        """
        return {"scene": scene, "scene_label": CUSTOMER_SCENE_LABELS.get(scene, scene)}

    def _wire_search_context(self, session_id: str, scene: str, deps: dict[str, Any]) -> None:
        """Set shop/product context on the global SearchKnowledge.

        args:
            session_id: for retrieving recent user messages.
            scene: current scene.
            deps: {"shop_id", "goods_id", "goods_name"} for RAG.
        """
        shop_id = deps.get("shop_id")
        goods_id = deps.get("goods_id")
        sk = get_search_knowledge()
        if sk:
            sk.set_context(shop_id=shop_id, scene=scene, goods_id=goods_id, goods_name=deps.get("goods_name", ""))
            sk.set_history(self._agent.recent_user_messages(session_id))

    @staticmethod
    def _rewrite_question(question: str, deps: dict[str, Any]) -> str:
        """Replace vague references (这手机) with the actual goods_name.

        args:
            question: raw user input.
            deps: context dict (must contain goods_name).

        returns: question with vague refs replaced.
        """
        goods_name = deps.get("goods_name")
        if not goods_name:
            return question
        for pattern in ("这款手机", "这部手机", "这手机", "该手机", "这个手机", "这台手机"):
            question = question.replace(pattern, goods_name)
        return question

    async def chat(
        self,
        session_id: str,
        question: str,
        context: Context | None = None,
        dependencies: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Process one non-streaming chat turn (short-circuit + LLM).

        args:
            session_id: conversation ID.
            question: raw user input.
            context: parsed Context (type/kwargs).
            dependencies: pre-extracted context (shop_id, etc).

        returns: full ChatResponse (answer, intent, citations, etc).
        """
        trace_id = uuid.uuid4().hex
        tracker = RequestTracker(session_id, trace_id)
        tracker.transition(RequestState.RECEIVED, context_type=context.type.value if context else "text")

        if context is None:
            context = Context(content=question, kwargs=ChannelKwargs())
        elif context.content is None:
            context = context.model_copy(update={"content": question})

        # ── 1. 静默类：撤回 / 认证 / 系统推送 -> 直接返回空 ──
        if context.type.is_silent:
            tracker.transition(RequestState.SILENT_REPLY)
            logger.info(
                "Silent context_type=%s session_id=%s - skipping LLM",
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
        tracker.transition(RequestState.CONTEXT_PARSED, scene_hint=turn_ctx.raw_scene_hint)
        deps = _context_to_dependencies(context, raw_query)
        if dependencies:
            deps.update(dependencies)

        # ── 3. 图片 / 视频：转人工占位 ──
        if context.type.requires_human:
            tracker.transition(RequestState.MEDIA_HANDOFF)
            logger.info(
                "Human-required context_type=%s session_id=%s - placeholder handoff",
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

        # ── 4. 纯商品卡无文字 -> 追问或欢迎语（首 turn） ──
        if (
            turn_ctx.turn_type.has_product_card
            and not turn_ctx.turn_type.has_text
        ):
            is_new_session = not self._agent.recent_user_messages(session_id, n=1)
            tracker.transition(RequestState.WELCOME if is_new_session else RequestState.GOODS_CARD_PROMPT)
            logger.info(
                "Product-card-only turn session_id=%s goods_id=%s is_new=%s",
                session_id, turn_ctx.product_card.goods_id, is_new_session,
            )
            return ChatResponse(
                session_id=session_id,
                answer=_WELCOME_MESSAGE if is_new_session else _GOODS_CARD_ONLY_REPLY,
                trace_id=trace_id,
                intent="greeting" if is_new_session else "goods_card_only",
                context_type=context.type.value,
            )

        # ── 5. 问候语命中 -> 直接返回欢迎语，跳过 scene_classifier 和 LLM ──
        effective_question = turn_ctx.customer_text.strip() or question
        if _is_greeting(effective_question):
            logger.info(
                "Greeting detected session_id=%s question=%s - short-circuit",
                session_id, effective_question,
            )
            return ChatResponse(
                session_id=session_id,
                answer=_WELCOME_MESSAGE,
                trace_id=trace_id,
                intent="greeting",
                context_type=context.type.value,
            )

        # ── 6. 正常流：scene classifier + agent ──
        # 当 context.content 已经被 TurnContext 归一化过，优先用提取出来的 customer_text
        # （商品卡 / 订单卡 / 元数据已经被剥离）作为 question，避免把空 question 喂给 LLM。
        effective_question = self._rewrite_question(effective_question, deps)
        raw_scene = await self._scene_classifier.classify(deps, effective_question, session_id)
        scene = normalize_customer_scene(raw_scene) or raw_scene
        tracker.transition(RequestState.SCENE_CLASSIFIED, scene=scene)

        if self._scene_classifier.last_scene_hint == "mixed_orders":
            scene = "mixed"

        self._wire_search_context(session_id, scene, deps)

        answer = await self._agent.ask(
            session_id=session_id,
            question=effective_question,
            scene=scene,
            dependencies=deps, tracker=tracker,
        )
        tracker.transition(RequestState.RESPONSE_SENT)
        product_cards = self._agent.pop_product_cards(session_id)
        return ChatResponse(
            session_id=session_id,
            answer=answer.strip(),
            trace_id=trace_id,
            intent=scene,
            context_type=context.type.value,
            need_handoff=False,
            product_cards=product_cards or None,
            metadata=self._scene_metadata(scene),
        )

    async def stream_chat(
        self,
        session_id: str,
        question: str,
        context: Context | None = None,
        dependencies: dict[str, Any] | None = None,
    ) -> tuple[str, str, list, list[str], AsyncIterator[str], str]:
        """Process one streaming chat turn (short-circuit + LLM stream).

        args:
            session_id: conversation ID.
            question: raw user input.
            context: parsed Context (type/kwargs).
            dependencies: pre-extracted context (shop_id, etc).

        returns: (trace_id, scene, citations, actions, chunk_iter, layer).
        """
        trace_id = uuid.uuid4().hex
        tracker = RequestTracker(session_id, trace_id)
        tracker.transition(RequestState.RECEIVED, context_type=context.type.value if context else "text")

        if context is None:
            context = Context(content=question, kwargs=ChannelKwargs())
        elif context.content is None:
            context = context.model_copy(update={"content": question})

        # silent / image / video / goods_card_only 在 stream 路径下也走短路由。
        # 这里只对 silent 直接返空，其余交由前端读首条 SSE delta 的 actions 字段决定。
        if context.type.is_silent:
            tracker.transition(RequestState.SILENT_REPLY)
            return (
                trace_id,
                "silent",
                [],
                [],
                _empty_async_iter(),
                "LLM",
            )

        raw_query = context.content or question
        turn_ctx = parse_turn_context(raw_query)
        tracker.transition(RequestState.CONTEXT_PARSED, scene_hint=turn_ctx.raw_scene_hint)
        deps = _context_to_dependencies(context, raw_query)
        if dependencies:
            deps.update(dependencies)

        # ── 问候语命中 -> 直接返回，跳过 scene_classifier 和 LLM ──
        effective_question = turn_ctx.customer_text.strip() or question
        if _is_greeting(effective_question):
            async def _greeting_stream() -> AsyncIterator[str]:
                yield _WELCOME_MESSAGE
            return (
                trace_id,
                "greeting",
                [],
                [],
                _greeting_stream(),
                "GREETING",
            )

        effective_question = self._rewrite_question(effective_question, deps)
        raw_scene = await self._scene_classifier.classify(deps, effective_question, session_id)
        scene = normalize_customer_scene(raw_scene) or raw_scene
        tracker.transition(RequestState.SCENE_CLASSIFIED, scene=scene)

        if self._scene_classifier.last_scene_hint == "mixed_orders":
            scene = "mixed"

        self._wire_search_context(session_id, scene, deps)

        return (
            trace_id,
            scene,
            [],
            [],
            self._agent.ask_stream(
                session_id=session_id, question=effective_question, scene=scene, dependencies=deps, tracker=tracker,
            ),
            "LLM",
        )

    def handoff(self, session_id: str, reason: str, priority: str) -> HandoffResponse:
        """Transfer the conversation to a human agent (mocked).

        args:
            session_id: conversation ID.
            reason: why transferring.
            priority: 'low' | 'normal' | 'high'.

        returns: HandoffResponse with ticket_id and queue.
        """
        ticket_id = f"TKT-{uuid.uuid4().hex[:10].upper()}"
        queue = "vip" if priority.lower() == "high" else "standard"
        return HandoffResponse(
            session_id=session_id,
            ticket_id=ticket_id,
            queue=queue,
            status=f"queued ({reason})",
        )


async def _empty_async_iter() -> AsyncIterator[str]:
    """Empty async iterator for silent stream responses."""
    return
    yield  # makes this an async generator
