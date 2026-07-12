import asyncio
import os
import uuid

from util.logger_setup import setup_logging

setup_logging()

# Chainlit treats DATABASE_URL as its own PostgreSQL data-layer switch.
# This project uses DATABASE_URL for the agent session store (SQLite by default),
# so hide it from Chainlit to avoid importing asyncpg / initializing its DB layer.
os.environ.pop("DATABASE_URL", None)

import chainlit as cl

from app.orchestrator.intent_router import CUSTOMER_SCENE_LABELS
from app.orchestrator.service import ChatOrchestrator
from app.schemas import Citation


INGEST_HELP = """用法：

普通聊天：直接输入问题。

文档入库：使用 `/ingest 来源名`，下一行开始粘贴文档内容，例如：

```text
/ingest faq.md
这里是需要进入向量库的文档内容
```

只有 `/ingest` 会对文档内容做 embedding 并写入向量库；普通聊天不会触发 embedding。
"""


def _get_orchestrator() -> ChatOrchestrator:
    orchestrator = cl.user_session.get("orchestrator")
    if orchestrator is None:
        orchestrator = ChatOrchestrator()
        cl.user_session.set("orchestrator", orchestrator)
    return orchestrator


def _get_session_id() -> str:
    session_id = cl.user_session.get("session_id")
    if not session_id:
        session_id = getattr(cl.context.session, "id", "") or f"chainlit-{uuid.uuid4().hex}"
        cl.user_session.set("session_id", session_id)
    return session_id


def _parse_ingest_command(content: str) -> tuple[str, str] | None:
    if not content.startswith("/ingest"):
        return None

    first_line, _, body = content.partition("\n")
    parts = first_line.split(maxsplit=1)
    if len(parts) != 2 or not parts[1].strip() or not body.strip():
        raise ValueError(INGEST_HELP)
    return parts[1].strip(), body.strip()


def _format_meta(scene: str, actions: list[str], layer: str) -> str:
    scene_label = CUSTOMER_SCENE_LABELS.get(scene, scene)
    action_text = "、".join(actions) if actions else "无"
    rag_text = "已启用" if "search_knowledge_base" in actions else "未启用"
    return f"**{layer}** | 场景：「{scene_label}」| RAG {rag_text} | 建议动作：{action_text}\n\n---\n"


def _format_citations(citations: list[Citation]) -> str:
    if not citations:
        return ""
    lines = ["", "", "依据来源："]
    for idx, citation in enumerate(citations, start=1):
        snippet = citation.snippet[:120].strip()
        lines.append(f"{idx}. {citation.source} (score={citation.score:.4f}) - {snippet}")
    return "\n".join(lines)


@cl.on_chat_start
async def on_chat_start() -> None:
    _get_orchestrator()
    _get_session_id()
    await cl.Message(content=f"你好，我是你的 LangChain QA Agent。\n\n{INGEST_HELP}").send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    orchestrator = _get_orchestrator()
    content = message.content.strip()

    try:
        ingest_payload = _parse_ingest_command(content)
    except ValueError as exc:
        await cl.Message(content=str(exc)).send()
        return

    if ingest_payload:
        source, document_content = ingest_payload
        try:
            result = await asyncio.to_thread(
                orchestrator.ingest_document,
                source=source,
                content=document_content,
            )
        except Exception as exc:  # noqa: BLE001
            await cl.Message(content=f"文档入库失败：{exc}").send()
            return

        await cl.Message(content=f"文档已入库：`{result.source}`，新增 chunks：{result.chunks_added}。").send()
        return

    session_id = _get_session_id()
    response = cl.Message(content="")

    try:
        trace_id, scene, citations, actions, chunks, layer = await orchestrator.stream_chat(
            session_id=session_id,
            question=content,
        )
        await response.stream_token(_format_meta(scene, actions, layer))
        for chunk in chunks:
            await response.stream_token(chunk)
        await response.stream_token(_format_citations(citations))
        await response.stream_token(f"\n\ntrace_id: `{trace_id}`")
        await response.send()
    except Exception as exc:  # noqa: BLE001
        await cl.Message(content=f"请求失败：{exc}").send()
