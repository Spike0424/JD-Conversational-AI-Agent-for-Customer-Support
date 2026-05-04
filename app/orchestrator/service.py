import uuid
from collections.abc import Iterator

from app.llm.agent_runtime import ReActQAAgent
from app.orchestrator.intent_router import detect_intent
from app.schemas import ChatResponse, HandoffResponse


class ChatOrchestrator:
    def __init__(self) -> None:
        self._agent = ReActQAAgent()

    @staticmethod
    def _new_trace_id() -> str:
        return uuid.uuid4().hex

    def chat(self, session_id: str, question: str) -> ChatResponse:
        trace_id = self._new_trace_id()
        intent = detect_intent(question)
        answer = self._agent.ask(session_id=session_id, question=question)
        return ChatResponse(
            session_id=session_id,
            answer=answer,
            trace_id=trace_id,
            intent=intent,
            citations=[],
            actions=[],
            need_handoff=False,
        )

    def stream_chat(self, session_id: str, question: str) -> tuple[str, str, Iterator[str]]:
        trace_id = self._new_trace_id()
        intent = detect_intent(question)
        return trace_id, intent, self._agent.ask_stream(session_id=session_id, question=question)

    def handoff(self, session_id: str, reason: str, priority: str) -> HandoffResponse:
        ticket_id = f"TKT-{uuid.uuid4().hex[:10].upper()}"
        queue = "vip" if priority.lower() == "high" else "standard"
        return HandoffResponse(
            session_id=session_id,
            ticket_id=ticket_id,
            queue=queue,
            status=f"queued ({reason})",
        )
