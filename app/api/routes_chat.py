import asyncio
import json
import time
import uuid
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.deps import get_orchestrator
from app.config import get_settings
from app.orchestrator import ChatOrchestrator
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentIngestRequest,
    DocumentIngestResponse,
    ErrorResponse,
    HandoffRequest,
    HandoffResponse,
    PdfPathIngestRequest,
)

_T = TypeVar("_T")

router = APIRouter(tags=["chat"])

_GET_USAGE = {
    "message": (
        "此接口需使用 POST（JSON Body），或使用 GET 并附带查询参数 "
        "`session_id` 与 `question`。"
    ),
    "post_body_example": {"session_id": "demo", "question": "你好"},
    "get_example": "/chat?session_id=demo&question=你好",
}


def _error_response(status_code: int, error_code: str, message: str, trace_id: str | None = None) -> JSONResponse:
    payload = ErrorResponse(
        error_code=error_code,
        message=message,
        trace_id=trace_id or uuid.uuid4().hex,
    ).model_dump()
    return JSONResponse(status_code=status_code, content=payload)


def _with_retry(fn: Callable[[], _T], error_code: str) -> _T | JSONResponse:
    """Execute *fn* with linear-backoff retry on transient upstream failures."""
    settings = get_settings()
    max_retries = max(0, settings.chat_retries)
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries:
                return _error_response(
                    502,
                    f"{error_code}_FAILED",
                    f"Upstream request failed after {max_retries + 1} attempt(s): {exc}",
                )
            time.sleep(0.5 * (attempt + 1))
    return _error_response(502, f"{error_code}_FAILED", str(last_error))


async def _with_retry_async(coro_factory: Callable[[], Any], error_code: str) -> Any:
    """Execute async operation with linear-backoff retry."""
    settings = get_settings()
    max_retries = max(0, settings.chat_retries)
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries:
                return _error_response(
                    502,
                    f"{error_code}_FAILED",
                    f"Upstream request failed after {max_retries + 1} attempt(s): {exc}",
                )
            await asyncio.sleep(0.5 * (attempt + 1))
    return _error_response(502, f"{error_code}_FAILED", str(last_error))


async def _chat_async(
    session_id: str, question: str, orchestrator: ChatOrchestrator,
) -> ChatResponse | JSONResponse:
    try:
        return await _with_retry_async(
            lambda: orchestrator.chat(session_id=session_id, question=question), "CHAT"
        )
    except ValueError as exc:
        return _error_response(400, "BAD_REQUEST", str(exc))


async def _streaming_response_for_question(
    session_id: str, question: str, orchestrator: ChatOrchestrator
) -> StreamingResponse:
    trace_id = uuid.uuid4().hex
    try:
        trace_id, intent, citations, actions, chunks, _layer = await orchestrator.stream_chat(
            session_id=session_id, question=question
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
            meta = {
                "trace_id": trace_id,
                "intent": intent,
                "citations": [c.model_dump() for c in citations],
                "actions": actions,
            }
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


def _parse_get_chat_params(
    session_id: str | None, question: str | None
) -> tuple[str, str] | JSONResponse | None:
    """返回 (session_id, question)，缺参时返回说明 JSON；参数不齐返回 400。"""
    sid = (session_id or "").strip()
    q = (question or "").strip()
    if not sid and not q:
        return None
    if not sid or not q:
        return _error_response(
            400,
            "BAD_REQUEST",
            "GET 请求需要同时提供非空的 session_id 与 question 查询参数。",
        )
    return (sid, q)


@router.post("/chat", response_model=ChatResponse)
@router.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, orchestrator: ChatOrchestrator = Depends(get_orchestrator)) -> ChatResponse | JSONResponse:
    return await _chat_async(request.session_id, request.question, orchestrator)


@router.post("/rag/documents", response_model=DocumentIngestResponse)
@router.post("/v1/rag/documents", response_model=DocumentIngestResponse)
def ingest_document(
    request: DocumentIngestRequest, orchestrator: ChatOrchestrator = Depends(get_orchestrator)
) -> DocumentIngestResponse | JSONResponse:
    try:
        return _with_retry(
            lambda: orchestrator.ingest_document(source=request.source, content=request.content),
            "INGEST",
        )
    except ValueError as exc:
        return _error_response(400, "BAD_REQUEST", str(exc))


@router.post("/rag/documents/pdf", response_model=DocumentIngestResponse)
@router.post("/v1/rag/documents/pdf", response_model=DocumentIngestResponse)
async def ingest_pdf_file(
    file: UploadFile = File(...),
    source: str | None = Form(default=None),
    orchestrator: ChatOrchestrator = Depends(get_orchestrator),
) -> DocumentIngestResponse | JSONResponse:
    try:
        data = await file.read()
        chosen_source = (source or "").strip() or file.filename or "uploaded.pdf"
        filename = file.filename or "uploaded.pdf"
        return _with_retry(
            lambda: orchestrator.ingest_pdf_bytes(source=chosen_source, filename=filename, pdf_bytes=data),
            "INGEST_PDF",
        )
    except ValueError as exc:
        return _error_response(400, "BAD_REQUEST", str(exc))


@router.post("/rag/documents/pdf/path", response_model=DocumentIngestResponse)
@router.post("/v1/rag/documents/pdf/path", response_model=DocumentIngestResponse)
def ingest_pdf_path(
    request: PdfPathIngestRequest, orchestrator: ChatOrchestrator = Depends(get_orchestrator)
) -> DocumentIngestResponse | JSONResponse:
    try:
        return _with_retry(
            lambda: orchestrator.ingest_pdf_path(pdf_path=request.pdf_path, source=request.source),
            "INGEST_PDF_PATH",
        )
    except ValueError as exc:
        return _error_response(400, "BAD_REQUEST", str(exc))


@router.get("/chat", response_model=None)
@router.get("/v1/chat", response_model=None)
async def chat_get(
    session_id: str | None = Query(None),
    question: str | None = Query(None),
    orchestrator: ChatOrchestrator = Depends(get_orchestrator),
) -> ChatResponse | JSONResponse:
    parsed = _parse_get_chat_params(session_id, question)
    if parsed is None:
        return JSONResponse(status_code=200, content=_GET_USAGE)
    if isinstance(parsed, JSONResponse):
        return parsed
    sid, q = parsed
    return await _chat_async(sid, q, orchestrator)


@router.post("/chat/stream")
@router.post("/v1/chat/stream")
async def chat_stream(request: ChatRequest, orchestrator: ChatOrchestrator = Depends(get_orchestrator)) -> StreamingResponse:
    return await _streaming_response_for_question(request.session_id, request.question, orchestrator)


@router.get("/chat/stream", response_model=None)
@router.get("/v1/chat/stream", response_model=None)
async def chat_stream_get(
    session_id: str | None = Query(None),
    question: str | None = Query(None),
    orchestrator: ChatOrchestrator = Depends(get_orchestrator),
) -> StreamingResponse | JSONResponse:
    parsed = _parse_get_chat_params(session_id, question)
    if parsed is None:
        return JSONResponse(
            status_code=200,
            content={**_GET_USAGE, "get_example_stream": "/chat/stream?session_id=demo&question=你好"},
        )
    if isinstance(parsed, JSONResponse):
        return parsed
    sid, q = parsed
    return await _streaming_response_for_question(sid, q, orchestrator)


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
