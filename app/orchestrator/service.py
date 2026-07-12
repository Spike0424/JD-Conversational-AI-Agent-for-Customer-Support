import logging
import uuid
from collections.abc import Iterator
from typing import Any

from app.llm.agent_runtime import ReActQAAgent
from app.orchestrator.intent_router import CUSTOMER_SCENE_LABELS, normalize_customer_scene
from app.orchestrator.scene_classifier import SceneClassifier
from app.schemas import ChatResponse, DocumentIngestResponse, HandoffResponse
from app.tools.search_knowledge import get_search_knowledge

logger = logging.getLogger(__name__)


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
        self, session_id: str, question: str, dependencies: dict[str, Any] | None = None
    ) -> ChatResponse:
        if dependencies is None:
            dependencies = {}
        trace_id = self._new_trace_id()

        raw_scene = await self._scene_classifier.classify(dependencies, question, session_id)
        scene = normalize_customer_scene(raw_scene) or raw_scene

        # Mixed orders: route to dedicated "mixed" prompt that asks for order number
        if self._scene_classifier.last_scene_hint == "mixed_orders":
            scene = "mixed"

        shop_id = dependencies.get("shop_id")
        goods_id = dependencies.get("goods_id")
        sk = get_search_knowledge()
        if sk:
            sk.set_context(shop_id=shop_id, scene=scene, goods_id=goods_id)

        answer = self._agent.ask(session_id=session_id, question=question, scene=scene)
        formatted = self._format_header(scene) + answer.strip()
        return ChatResponse(
            session_id=session_id,
            answer=formatted,
            trace_id=trace_id,
            intent=scene,
            need_handoff=False,
        )

    async def stream_chat(
        self, session_id: str, question: str, dependencies: dict[str, Any] | None = None
    ) -> tuple[str, str, list, list[str], Iterator[str], str]:
        if dependencies is None:
            dependencies = {}
        trace_id = self._new_trace_id()

        raw_scene = await self._scene_classifier.classify(dependencies, question, session_id)
        scene = normalize_customer_scene(raw_scene) or raw_scene

        if self._scene_classifier.last_scene_hint == "mixed_orders":
            scene = "mixed"

        shop_id = dependencies.get("shop_id")
        goods_id = dependencies.get("goods_id")
        sk = get_search_knowledge()
        if sk:
            sk.set_context(shop_id=shop_id, scene=scene, goods_id=goods_id)

        return (
            trace_id, scene, [], [],
            self._agent.ask_stream(session_id=session_id, question=question, scene=scene),
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
