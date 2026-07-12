"""Generate Q&A pairs from RAG chunks using DeepSeek LLM."""

import json
import logging
import re
from typing import Any

import numpy as np
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.db import create_session as create_qa_session, create_session as create_rag_session, init_db as init_qa_db
from app.qa_models import QAPair
from app.rag_models import RagChunk

logger = logging.getLogger(__name__)

QA_GENERATION_PROMPT = """请基于以下官方售后规则文本，生成 5 个用户可能会问的高频真实问题，并给出极其精准的官方标准答案。
必须严格基于文本，不能编造。

格式要求为 JSON 列表，格式如下：
[
  {{"question": "用户问题", "answer": "官方标准回答", "source": "引用的原文片段"}}
]

规则文本：
{rule_text}"""


def _build_qa_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.model_name,
        temperature=0.3,
        request_timeout=120,
    )


def _get_embeddings():
    from langchain_community.embeddings import FastEmbedEmbeddings

    settings = get_settings()
    return FastEmbedEmbeddings(model_name=settings.local_embedding_model_name)


def _get_unique_sources() -> list[str]:
    """Fetch distinct document sources from rag_chunks."""
    session = create_rag_session()
    try:
        rows = session.query(RagChunk.source).distinct().all()
        return sorted(set(r[0] for r in rows if r[0]))
    finally:
        session.close()


def _get_chunks_by_source(source: str) -> list[str]:
    """Get all chunk snippets for a given source, ordered by chunk_index."""
    session = create_rag_session()
    try:
        rows = (
            session.query(RagChunk.snippet)
            .filter(RagChunk.source == source)
            .order_by(RagChunk.chunk_index)
            .all()
        )
        return [r[0] for r in rows if r[0]]
    finally:
        session.close()


def _parse_qa_response(raw: str) -> list[dict[str, str]]:
    """Extract JSON array from LLM response."""
    # Try to find a JSON array block
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        logger.warning("No JSON array found in LLM response: %s", raw[:200])
        return []
    try:
        data = json.loads(match.group(0))
        if not isinstance(data, list):
            return []
        return [
            {
                "question": str(item.get("question", "")),
                "answer": str(item.get("answer", "")),
                "source": str(item.get("source", "")),
            }
            for item in data
            if item.get("question") and item.get("answer")
        ]
    except json.JSONDecodeError:
        logger.warning("JSON parse failed for: %s", raw[:200])
        return []


def generate_qa_pairs(source_filter: str = "", questions_per_doc: int = 5) -> dict[str, Any]:
    """Generate Q&A pairs from RAG knowledge chunks using LLM.

    Args:
        source_filter: Only process sources containing this string (empty = all).
        questions_per_doc: Target questions per document source.

    Returns:
        {"generated": int, "sources_processed": int, "errors": list[str]}
    """
    init_qa_db()
    llm = _build_qa_llm()
    embeddings = _get_embeddings()
    session = create_qa_session()

    sources = _get_unique_sources()
    if source_filter:
        sources = [s for s in sources if source_filter in s]

    prompt_template = QA_GENERATION_PROMPT.replace("5 个", f"{questions_per_doc} 个")
    total_generated = 0
    errors = []

    for source in sources:
        chunks = _get_chunks_by_source(source)
        if not chunks:
            continue
        rule_text = "\n\n".join(chunks)

        # Truncate if too long (DeepSeek context window safety)
        if len(rule_text) > 6000:
            rule_text = rule_text[:6000] + "\n...(内容过长已截断)"

        prompt = prompt_template.format(rule_text=rule_text)
        logger.info("Generating QA for source=%s (text_len=%d)", source, len(rule_text))

        try:
            response = llm.invoke(prompt)
            raw = str(response.content) if hasattr(response, "content") else str(response)
            pairs = _parse_qa_response(raw)

            for pair in pairs:
                try:
                    vec = np.array(embeddings.embed_query(pair["question"]), dtype=np.float32)
                    qa = QAPair(
                        question=pair["question"],
                        answer=pair["answer"],
                        source=pair["source"],
                        source_doc=source,
                        embedding_bytes=vec.tobytes(),
                    )
                    session.add(qa)
                    total_generated += 1
                except Exception as exc:
                    errors.append(f"{source}: embed failed - {exc}")

            session.commit()
            logger.info("  Generated %d pairs for source=%s", len(pairs), source)
        except Exception as exc:
            msg = f"{source}: {exc}"
            errors.append(msg)
            logger.error("  Failed: %s", msg)

    session.close()
    return {
        "generated": total_generated,
        "sources_processed": len(sources),
        "errors": errors,
    }
