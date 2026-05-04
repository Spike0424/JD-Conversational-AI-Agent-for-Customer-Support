import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_orchestrator
from app.orchestrator import ChatOrchestrator
from app.schemas import ChatRequest, ChatResponse, HandoffRequest, HandoffResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
@router.post("/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest, orchestrator: ChatOrchestrator = Depends(get_orchestrator)) -> ChatResponse:
    return orchestrator.chat(session_id=request.session_id, question=request.question)


@router.post("/chat/stream")
@router.post("/v1/chat/stream")
def chat_stream(request: ChatRequest, orchestrator: ChatOrchestrator = Depends(get_orchestrator)) -> StreamingResponse:
    trace_id, intent, chunks = orchestrator.stream_chat(session_id=request.session_id, question=request.question)

    def event_stream() -> Iterator[str]:
        try:
            meta = {"trace_id": trace_id, "intent": intent}
            yield f"data: {json.dumps({'meta': meta}, ensure_ascii=False)}\n\n"
            for chunk in chunks:
                yield f"data: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001
            error = json.dumps({"error": str(exc), "trace_id": trace_id}, ensure_ascii=False)
            yield f"data: {error}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/v1/handoff", response_model=HandoffResponse)
def handoff(
    request: HandoffRequest, orchestrator: ChatOrchestrator = Depends(get_orchestrator)
) -> HandoffResponse:
    return orchestrator.handoff(session_id=request.session_id, reason=request.reason, priority=request.priority)
