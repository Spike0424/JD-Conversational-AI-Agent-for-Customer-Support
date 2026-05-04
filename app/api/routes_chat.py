import json
import uuid
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.deps import get_orchestrator
from app.orchestrator import ChatOrchestrator
from app.schemas import ChatRequest, ChatResponse, ErrorResponse, HandoffRequest, HandoffResponse

router = APIRouter(tags=["chat"])


def _error_response(status_code: int, error_code: str, message: str, trace_id: str | None = None) -> JSONResponse:
    payload = ErrorResponse(
        error_code=error_code,
        message=message,
        trace_id=trace_id or uuid.uuid4().hex,
    ).model_dump()
    return JSONResponse(status_code=status_code, content=payload)


@router.post("/chat", response_model=ChatResponse)
@router.post("/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest, orchestrator: ChatOrchestrator = Depends(get_orchestrator)) -> ChatResponse | JSONResponse:
    try:
        return orchestrator.chat(session_id=request.session_id, question=request.question)
    except ValueError as exc:
        return _error_response(400, "BAD_REQUEST", str(exc))
    except Exception as exc:  # noqa: BLE001
        return _error_response(500, "INTERNAL_ERROR", str(exc))


@router.post("/chat/stream")
@router.post("/v1/chat/stream")
def chat_stream(request: ChatRequest, orchestrator: ChatOrchestrator = Depends(get_orchestrator)) -> StreamingResponse:
    trace_id = uuid.uuid4().hex
    try:
        trace_id, intent, citations, actions, chunks = orchestrator.stream_chat(
            session_id=request.session_id, question=request.question
        )
    except ValueError as exc:
        return StreamingResponse(
            iter(
                [
                    f"data: {json.dumps({'error_code': 'BAD_REQUEST', 'message': str(exc), 'trace_id': trace_id}, ensure_ascii=False)}\n\n"
                ]
            ),
            media_type="text/event-stream",
        )
    except Exception as exc:  # noqa: BLE001
        return StreamingResponse(
            iter(
                [
                    f"data: {json.dumps({'error_code': 'INTERNAL_ERROR', 'message': str(exc), 'trace_id': trace_id}, ensure_ascii=False)}\n\n"
                ]
            ),
            media_type="text/event-stream",
        )

    def event_stream() -> Iterator[str]:
        try:
            meta = {"trace_id": trace_id, "intent": intent, "citations": citations, "actions": actions}
            yield f"data: {json.dumps({'meta': meta}, ensure_ascii=False)}\n\n"
            for chunk in chunks:
                yield f"data: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001
            error = json.dumps(
                {"error_code": "STREAM_RUNTIME_ERROR", "message": str(exc), "trace_id": trace_id},
                ensure_ascii=False,
            )
            yield f"data: {error}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/v1/handoff", response_model=HandoffResponse)
def handoff(
    request: HandoffRequest, orchestrator: ChatOrchestrator = Depends(get_orchestrator)
) -> HandoffResponse | JSONResponse:
    try:
        return orchestrator.handoff(session_id=request.session_id, reason=request.reason, priority=request.priority)
    except ValueError as exc:
        return _error_response(400, "BAD_REQUEST", str(exc))
    except Exception as exc:  # noqa: BLE001
        return _error_response(500, "INTERNAL_ERROR", str(exc))
